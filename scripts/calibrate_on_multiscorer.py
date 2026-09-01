#!/usr/bin/env python3
"""Train op MESA voor volume, kalibreer op PSG-IPA voor waarheid.

WAAROM DEZE SCHEIDING
=====================
Twee cohorten, twee rollen, en ze zijn niet uitwisselbaar.

MESA levert 2056 opnames met ÉÉN scoorder. Genoeg om te leren *wat* een event
is -- 150-300 patiënten is de schaal waarop een LightGBM op 32 features stabiel
wordt, en de geleverde arousalclassifier is op 653 proefpersonen getraind.

PSG-IPA levert 5 opnames met TWAALF scoorders. Veel te weinig om op te trainen
(de foldspreiding is ±0,08 bij leave-one-recording-out), maar het is de enige
data die zegt *hoe zeker* een event is. Gemeten: 22,8 % van de kandidaten is
omstreden, en menselijke scoorders halen onderling F1 0,556.

De truc is dat die tweede rol veel minder data vraagt. Een model schat 32
featuregewichten; een kalibratiecurve schat een monotone afbeelding van score
naar kans. Vijf opnames zijn voor het eerste onvoldoende en voor het tweede
genoeg.

WAT KALIBRATIE WEL EN NIET REPAREERT
====================================
WEL: een verschoven drempel. Traint het model op een regel die 4 % desaturatie
eist en wil je 3 %-of-arousal, dan verschuift dat de beslisgrens -- en een
monotone hertoewijzing corrigeert dat.

NIET: een klasse die nooit gelabeld is. MESA annoteert geen RERA's, dus RDI is
er niet uit te leren, en geen kalibratie verzint die. Dat is een
DEKKINGSprobleem, geen kalibratieprobleem, en het onderscheid is scherp.

Om die reden labelt `build_training_dataset.py` MESA met de `aasm15`-referentie
(3 % desaturatie OF arousal, alle apneus) en niet met MESA's eigen kopcijfer
(`oahi4`, 4 % desaturatie, geen arousal-tak). De v3-correctie gebeurt bij het
labelen, niet erna.

Gebruik
-------
    python scripts/calibrate_on_multiscorer.py \\
        --train mesa_train.parquet --calibrate psgipa_soft.parquet \\
        --output-model model_mesa.txt --report kalibratie.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _ece(y, p, n_bins=10):
    """Verwachte kalibratiefout: hoe ver ligt een voorspelde 0,7 van 70 %?"""
    randen = np.linspace(0, 1, n_bins + 1)
    fout, n = 0.0, len(p)
    bins = []
    for i in range(n_bins):
        m = (p >= randen[i]) & ((p < randen[i + 1]) if i < n_bins - 1 else (p <= 1.0))
        if m.sum() == 0:
            continue
        v, w = float(p[m].mean()), float(y[m].mean())
        fout += m.sum() * abs(v - w)
        bins.append({"bin": f"{randen[i]:.1f}-{randen[i+1]:.1f}",
                     "n": int(m.sum()), "voorspeld": round(v, 3),
                     "werkelijk": round(w, 3)})
    return fout / max(n, 1), bins


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, type=Path,
                    help="MESA-dataset (veel opnames, één scoorder)")
    ap.add_argument("--calibrate", required=True, type=Path,
                    help="PSG-IPA-dataset (weinig opnames, twaalf scoorders)")
    ap.add_argument("--output-model", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import lightgbm as lgb
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from psgscoring.ml_classifier import FEATURE_COLUMNS

    tr = pd.read_parquet(args.train)
    ca = pd.read_parquet(args.calibrate)
    kol = [c for c in FEATURE_COLUMNS if c in tr.columns and c in ca.columns]
    print(f"trainen op {len(tr)} kandidaten / {tr['group'].nunique()} opnames "
          f"({tr['n_scorers'].max()} scoorder(s))")
    print(f"kalibreren op {len(ca)} kandidaten / {ca['group'].nunique()} opnames "
          f"({ca['n_scorers'].max()} scoorders)")
    print(f"{len(kol)} gedeelde features")

    if ca["n_scorers"].max() <= 1:
        print("FOUT: de kalibratieset heeft één scoorder. Dan is er geen "
              "scoorderfractie om tegen te kalibreren en levert dit niets op "
              "wat de training niet al wist.")
        return 1

    # ── 1. Trainen op het grote cohort, hard label ────────────────────────
    Xtr, ytr = tr[kol].to_numpy(float), tr["label"].to_numpy(float)
    gtr = tr["group"].to_numpy()
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
              "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
              "seed": args.seed}
    folds = min(5, tr["group"].nunique())
    gkf = GroupKFold(n_splits=folds)
    rondes = []
    for k, (a, b) in enumerate(gkf.split(Xtr, ytr, groups=gtr), 1):
        d = lgb.Dataset(Xtr[a], label=ytr[a], feature_name=kol)
        v = lgb.Dataset(Xtr[b], label=ytr[b], feature_name=kol, reference=d)
        bst = lgb.train(params, d, num_boost_round=args.rounds, valid_sets=[v],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        p = bst.predict(Xtr[b], num_iteration=bst.best_iteration)
        auc = roc_auc_score(ytr[b], p) if len(set(ytr[b])) > 1 else float("nan")
        rondes.append(bst.best_iteration or args.rounds)
        print(f"  fold {k}: AUC {auc:.4f} ({bst.best_iteration} rondes)")
    n_final = int(np.mean(rondes))
    model = lgb.train(params, lgb.Dataset(Xtr, label=ytr, feature_name=kol),
                      num_boost_round=n_final)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_model))
    print(f"\n  model -> {args.output_model} ({n_final} bomen)")

    # ── 2. Kalibreren op de scoorderfractie ───────────────────────────────
    Xca = ca[kol].to_numpy(float)
    ruw = model.predict(Xca)
    zacht = ca["label_soft"].to_numpy(float)
    hard = (zacht >= 0.5).astype(float)

    ece_voor, bins_voor = _ece(zacht, ruw)
    auc_ruw = roc_auc_score(hard, ruw) if len(set(hard)) > 1 else float("nan")
    print(f"\n  VOOR kalibratie:  AUC {auc_ruw:.4f}   ECE {ece_voor:.4f}")

    # Isotoon: monotoon, dus de RANGORDE blijft en daarmee de AUC. Alleen de
    # afbeelding score -> kans verandert. Dat is precies wat een werkpunt nodig
    # heeft en wat een AUC niet meet.
    #
    # Leave-one-recording-out, anders kalibreert hij op zichzelf: met vijf
    # opnames zou dat een kromme geven die alleen deze vijf beschrijft.
    kal = np.full(len(ca), np.nan)
    groepen = ca["group"].to_numpy()
    for g in np.unique(groepen):
        m = groepen == g
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(ruw[~m], zacht[~m])
        kal[m] = iso.predict(ruw[m])

    ece_na, bins_na = _ece(zacht, kal)
    auc_kal = roc_auc_score(hard, kal) if len(set(hard)) > 1 else float("nan")
    print(f"  NA kalibratie:    AUC {auc_kal:.4f}   ECE {ece_na:.4f}"
          f"   ({100*(ece_voor-ece_na)/max(ece_voor,1e-9):+.0f} %)")

    print(f"\n  {'bin':>10s}{'n':>7s}{'voor':>10s}{'na':>10s}{'werkelijk':>12s}")
    voor_map = {b["bin"]: b for b in bins_voor}
    for b in bins_na:
        v = voor_map.get(b["bin"], {})
        print(f"  {b['bin']:>10s}{b['n']:>7d}"
              f"{v.get('voorspeld', float('nan')):>10.3f}"
              f"{b['voorspeld']:>10.3f}{b['werkelijk']:>12.3f}")

    # Eindmodel: kalibreer op ALLES, want dit gaat de deur uit.
    iso_final = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_final.fit(ruw, zacht)
    punten = {"x": [round(float(v), 5) for v in iso_final.X_thresholds_],
              "y": [round(float(v), 5) for v in iso_final.y_thresholds_]}
    print(f"\n  kalibratiekromme: {len(punten['x'])} knikpunten")

    if args.report:
        args.report.write_text(json.dumps({
            "train": str(args.train), "n_train": len(tr),
            "n_train_groups": int(tr["group"].nunique()),
            "calibrate": str(args.calibrate), "n_cal": len(ca),
            "n_cal_groups": int(ca["group"].nunique()),
            "features": kol, "n_trees": n_final,
            "auc_raw": float(auc_ruw), "auc_calibrated": float(auc_kal),
            "ece_raw": float(ece_voor), "ece_calibrated": float(ece_na),
            "bins_raw": bins_voor, "bins_calibrated": bins_na,
            "isotonic": punten,
        }, indent=2))
        print(f"  rapport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
