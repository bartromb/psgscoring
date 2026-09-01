#!/usr/bin/env python3
"""Train de event-classifier — met de labelruis als eersterangs grootheid.

WAT HIER ANDERS IS DAN IN DE VORIGE TRAINING
============================================
Het geleverde model is getraind met `objective="binary"` op een HARD label van
één scoorder. Dat is een aanname die op PSG-IPA meetbaar niet klopt: twaalf
scoorders halen onderling F1 0,556 (mediaan), en op SN4 scoorde de ene expert
1 event waar de andere er 38 zag, met kappa 0,000 op het subtype. Een model dat
daar hard op traint, leert het toeval van één scoorder.

Vier knoppen, elk met een reden:

1. `--objective cross_entropy` traint op `label_soft` (de FRACTIE scoorders die
   dit event markeerde) in plaats van op 0/1. LightGBM accepteert continue
   labels in [0,1] hiermee rechtstreeks. Een event dat 11 van de 12 zagen is
   dan iets anders dan een dat er 3 zagen -- met een hard label zijn die twee
   niet te onderscheiden.

2. `--weight-by-agreement` weegt elk voorbeeld met de lokale overeenstemming.
   De ruis is namelijk NIET uniform: hij schaalt omgekeerd met de ziektelast
   (SN3, 273-339 events: F1 0,948; SN4, 1-38 events: F1 0,553). Zonder weging
   trekt juist het regime waar mensen elkaar tegenspreken het hardst aan het
   model.

3. `--monotone` legt fysiologische richting op. Meer desaturatie mag de kans op
   een event nooit VERLAGEN, een langere duur evenmin. Dat is gratis
   regularisatie en het maakt het model klinisch verdedigbaar: een reviewer kan
   de richting narekenen zonder het model te openen.

4. `--drop-confidence` laat de uitvoer van de regelgebaseerde classifier weg.
   Die zit nu als feature in de set, waardoor het model deels een omhulsel om
   de regels is. Of dat helpt of juist verbergt, is een MEETBARE vraag; deze
   vlag maakt hem beantwoordbaar.

WAT DIT SCRIPT NIET DOET
========================
Het kiest geen werkpunt. Dat gebeurt gescheiden en vooraf vastgelegd, op een
aparte set -- zie `docs/arousal_drempel_herijking_preregistratie.md` voor de
vorm. Een script dat traint en tegelijk de drempel kiest, kiest hem op de
trainingsdata.

GROEPEN
=======
Alle splitsingen gaan over `group` (de patiënt), nooit over rijen. Twee events
uit dezelfde nacht delen fysiologie, montage en signaalkwaliteit; wie binnen een
opname splitst, meet geheugen in plaats van generalisatie.

Gebruik
-------
    python scripts/train_event_classifier.py --dataset psgipa_soft.parquet \\
        --objective cross_entropy --weight-by-agreement --monotone \\
        --output model_soft.txt --report model_soft.json
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

#: Fysiologische richting per feature: +1 = mag de kans alleen verhogen,
#: -1 = alleen verlagen, 0 = vrij. Alleen ingevuld waar de richting NIET
#: discutabel is; een verkeerde constraint is schadelijker dan geen.
MONOTOON = {
    "desaturation_pct": 1,      # dieper ontzadigd -> nooit minder waarschijnlijk
    "duration_s": 1,            # langer -> nooit minder waarschijnlijk
    "flow_reduction_pct": 1,    # meer flowreductie -> nooit minder waarschijnlijk
    "min_spo2": -1,             # lagere nadir -> nooit minder waarschijnlijk
}


def _features(df, drop_confidence: bool):
    from psgscoring.ml_classifier import FEATURE_COLUMNS

    kol = [c for c in FEATURE_COLUMNS if c in df.columns]
    if drop_confidence:
        kol = [c for c in kol if c != "confidence"]
    ontbreekt = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if ontbreekt:
        print(f"  LET OP: {len(ontbreekt)} features ontbreken in de dataset: "
              f"{ontbreekt[:6]}{'...' if len(ontbreekt) > 6 else ''}")
    return kol


def _kalibratie_rapport(y, p, n_bins=10):
    """Hoe goed komt een voorspelde 0,7 overeen met 70 % werkelijk?

    Zonder dit is een 'confidence' een volgnummer en geen kans, en dan is elk
    werkpunt willekeurig.
    """
    randen = np.linspace(0, 1, n_bins + 1)
    uit = []
    for i in range(n_bins):
        m = (p >= randen[i]) & (p < randen[i + 1] if i < n_bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        uit.append({"bin": f"{randen[i]:.1f}-{randen[i+1]:.1f}",
                    "n": int(m.sum()),
                    "voorspeld": round(float(p[m].mean()), 3),
                    "werkelijk": round(float(y[m].mean()), 3)})
    ece = sum(b["n"] * abs(b["voorspeld"] - b["werkelijk"]) for b in uit) / max(1, len(p))
    return uit, float(ece)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--objective", choices=("binary", "cross_entropy"),
                    default="binary",
                    help="cross_entropy traint op label_soft (fractie scoorders)")
    ap.add_argument("--weight-by-agreement", action="store_true",
                    help="weeg voorbeelden met de lokale overeenstemming")
    ap.add_argument("--monotone", action="store_true",
                    help="leg fysiologische richting op waar die vaststaat")
    ap.add_argument("--drop-confidence", action="store_true",
                    help="laat de regelgebaseerde confidence als feature weg")
    ap.add_argument("--split-on", choices=("group", "scorer"), default="group",
                    help="waarover te splitsen: opname (default) of SCOORDER. "
                         "Splitsen op scoorder meet of het model een "
                         "fysiologie heeft geleerd of een persoon")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import lightgbm as lgb
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    df = (pd.read_parquet(args.dataset) if args.dataset.suffix == ".parquet"
          else pd.read_csv(args.dataset))
    print(f"{len(df)} kandidaten over {df['group'].nunique()} opnames")

    if args.objective == "cross_entropy":
        if "label_soft" not in df.columns:
            print("FOUT: --objective cross_entropy vraagt kolom label_soft")
            return 1
        if df["n_scorers"].max() <= 1:
            print("FOUT: deze dataset heeft ÉÉN scoorder per opname. Dan is "
                  "label_soft gelijk aan label en meet je geen labelruis. "
                  "Gebruik een multi-scoorder cohort (PSG-IPA) of --objective "
                  "binary.")
            return 1
        y = df["label_soft"].to_numpy(dtype=float)
        print(f"  zacht label: gemiddeld {y.mean():.3f}; "
              f"omstreden (0<y<1): {int(((y > 0) & (y < 1)).sum())}")
    else:
        y = df["label"].to_numpy(dtype=float)
        print(f"  hard label: {y.mean():.3f} positief")

    kol = _features(df, args.drop_confidence)
    X = df[kol].to_numpy(dtype=float)
    if args.split_on == "scorer":
        if "scorer" not in df.columns or df["scorer"].isna().all():
            print("FOUT: --split-on scorer vraagt een gevulde kolom `scorer`. "
                  "PSG-IPA heeft twaalf scoorders per opname en dus geen "
                  "enkele scoorder-ID; MESA wel (`scorerid5`).")
            return 1
        # Rijen zonder scoorder kunnen niet in een scoordersplit: ze horen bij
        # geen enkele fold en zouden stil in de training belanden.
        weg = int(df["scorer"].isna().sum())
        if weg:
            print(f"  {weg} rijen zonder scoorder-ID uitgesloten")
            df = df[df["scorer"].notna()].reset_index(drop=True)
            X = df[kol].to_numpy(dtype=float)
            y = (df["label_soft"] if args.objective == "cross_entropy"
                 else df["label"]).to_numpy(dtype=float)
        groepen = df["scorer"].astype(int).astype(str).to_numpy()
        print(f"  splitsen op SCOORDER: {len(set(groepen))} scoorders "
              f"({dict(zip(*np.unique(groepen, return_counts=True)))})")
    else:
        groepen = df["group"].to_numpy()
    print(f"  {len(kol)} features"
          f"{' (confidence weggelaten)' if args.drop_confidence else ''}")

    w = None
    if args.weight_by_agreement:
        if "agreement" not in df.columns:
            print("FOUT: --weight-by-agreement vraagt kolom agreement")
            return 1
        w = df["agreement"].to_numpy(dtype=float)
        print(f"  gewichten: mediaan {np.median(w):.3f}, "
              f"{int((w < 0.5).sum())} rijen onder 0,5")

    params = {
        "objective": args.objective,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": args.seed,
    }
    if args.objective == "binary":
        params["metric"] = "auc"
    if args.monotone:
        params["monotone_constraints"] = [MONOTOON.get(c, 0) for c in kol]
        opgelegd = {c: MONOTOON[c] for c in kol if c in MONOTOON}
        print(f"  monotone constraints: {opgelegd}")

    # Het aantal folds volgt de GROEPEN waarop gesplitst wordt, niet het
    # aantal opnames. Met drie scoorders kan GroupKFold geen vijf folds maken,
    # en de fout die sklearn dan geeft noemt de oorzaak niet.
    n_groepen = len(set(groepen))
    folds = min(args.folds, n_groepen)
    if folds < 2:
        print(f"FOUT: {n_groepen} opname(s) — te weinig voor groepsgewijze CV. "
              "Een split binnen één opname meet geheugen, geen generalisatie.")
        return 1
    if folds < args.folds:
        print(f"  LET OP: {folds} folds i.p.v. {args.folds} "
              f"({n_groepen} opnames)")

    gkf = GroupKFold(n_splits=folds)
    scores, rondes = [], []
    oof = np.full(len(df), np.nan)
    for k, (tr, va) in enumerate(gkf.split(X, y, groups=groepen), 1):
        dtr = lgb.Dataset(X[tr], label=y[tr],
                          weight=(w[tr] if w is not None else None),
                          feature_name=kol)
        dva = lgb.Dataset(X[va], label=y[va],
                          weight=(w[va] if w is not None else None),
                          feature_name=kol, reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=args.rounds,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        p = b.predict(X[va], num_iteration=b.best_iteration)
        oof[va] = p
        yb = (y[va] >= 0.5).astype(int)
        s = (roc_auc_score(yb, p) if len(set(yb)) > 1 else float("nan"))
        scores.append(s)
        rondes.append(b.best_iteration or args.rounds)
        print(f"  fold {k}: AUC {s:.4f}  ({b.best_iteration} rondes, "
              f"{len(set(groepen[va]))} opnames in validatie)")

    print(f"\n  CV AUC {np.nanmean(scores):.4f} ± {np.nanstd(scores):.4f}")
    yb = (y >= 0.5).astype(int)
    ap_score = average_precision_score(yb, oof) if len(set(yb)) > 1 else float("nan")
    bins, ece = _kalibratie_rapport(yb, oof)
    print(f"  gemiddelde precisie {ap_score:.4f}   kalibratiefout (ECE) {ece:.4f}")

    n_final = int(np.mean(rondes))
    dall = lgb.Dataset(X, label=y, weight=w, feature_name=kol)
    final = lgb.train(params, dall, num_boost_round=n_final)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(args.output))
    print(f"\n  model -> {args.output}  ({n_final} bomen)")

    imp = sorted(zip(kol, final.feature_importance("gain")),
                 key=lambda t: -t[1])
    print("\n  belangrijkste features (gain):")
    for naam, g in imp[:8]:
        print(f"    {naam:32s} {g:12.1f}")

    if args.report:
        args.report.write_text(json.dumps({
            "dataset": str(args.dataset), "n_rows": len(df),
            "n_groups": int(n_groepen), "objective": args.objective,
            "weight_by_agreement": args.weight_by_agreement,
            "monotone": args.monotone, "drop_confidence": args.drop_confidence,
            "features": kol, "cv_auc": scores,
            "cv_auc_mean": float(np.nanmean(scores)),
            "average_precision": float(ap_score),
            "calibration_bins": bins, "ece": ece,
            "n_trees": n_final,
            "importance_gain": {n: float(g) for n, g in imp},
        }, indent=2))
        print(f"  rapport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
