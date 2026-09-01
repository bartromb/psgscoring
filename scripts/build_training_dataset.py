#!/usr/bin/env python3
"""Bouw een trainingsset voor de event-classifier — met de labelruis erin.

WAAROM DIT SCRIPT IN DE REPO STAAT
==================================
Het model `psgscoring/data/lightgbm_v06_q7holdout.txt` is getraind met een
script dat BUITEN versiebeheer stond (`~/MESA-ab-test/build_lightgbm_dataset.py`).
Het model was daarmee niet reproduceerbaar te hertrainen: wie de loss wilde
veranderen, moest eerst archeologie bedrijven. Dit is de opvolger, in de repo,
naast de code die hij traint.

DE FEATURES WORDEN NIET OPNIEUW GESCHREVEN
==========================================
Het origineel had TWEE implementaties van dezelfde features — één om te
trainen en één om te scoren, met in de docstring "this is the runtime mirror".
Twee implementaties van hetzelfde getal is precies waar dit project herhaaldelijk
op is gestrand (de stadium-AHI, de FRI-noemer, de PLM-teller vorige week). Dit
script importeert `_extract_candidate_features` uit het pakket zelf. Wijkt de
training af van de runtime, dan is dat per constructie onmogelijk in plaats van
alleen onwaarschijnlijk.

TWEE LABELVORMEN, EN DAAR ZIT HET PUNT
======================================
`--cohort mesa` levert een HARD label: één scoorder, 0 of 1.

`--cohort psgipa` levert daarnaast een ZACHT label: de fractie van de twaalf
scoorders die dit event markeerde. Dat is geen verfijning maar een andere
grootheid. Op PSG-IPA is gemeten wat menselijke scoorders onderling halen:

    F1 mens-mens        mediaan 0,556   (SN3 0,948  tot  SN4 0,553)
    kappa subtype       mediaan 0,561   (SN3 0,707  tot  SN4 0,000)
    events per scoorder SN4: van 1 tot 38 op dezelfde nacht

Een hard label doet alsof die spreiding niet bestaat. Een event dat 11 van de
12 scoorders zagen is fysiologisch iets anders dan een dat er 3 zagen, en met
`label` = 1 zijn ze niet te onderscheiden. `label_soft` bewaart het verschil,
en LightGBM kan er rechtstreeks op trainen met `objective="cross_entropy"`.

De ruis is bovendien NIET uniform: hij schaalt omgekeerd met de ziektelast.
Daarom staat `agreement` als kolom in de uitvoer — om er gewichten aan te
hangen, of om het regime waar mensen elkaar tegenspreken uit de training te
laten.

UITVOER
=======
Eén rij per kandidaat (geaccepteerd EN afgewezen), met:

    group           patiënt-ID  -- gebruik dit voor GroupKFold, nooit splitsen
                                   binnen een opname
    candidate_id    uniek per rij
    label           0/1, hard
    label_soft      [0,1], fractie scoorders (bij mesa gelijk aan label)
    n_scorers       hoeveel scoorders deze opname droeg
    agreement       lokale overeenstemming, zie `_agreement`
    subtype_soft    JSON: verdeling van menselijke subtypelabels
    <FEATURE_COLUMNS>

Gebruik
-------
    python scripts/build_training_dataset.py --cohort psgipa \\
        --data-dir ~/PSG-IPA --output psgipa_soft.parquet

    python scripts/build_training_dataset.py --cohort mesa \\
        --data-dir ~/MESA/mesa --limit 300 --workers 6 \\
        --output mesa_q7.parquet
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logging.getLogger("psgscoring").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

#: Overlap (s) waarbinnen een kandidaat als "dezelfde" telt als een menselijk
#: event. Gelijk aan het origineel; zie `overlaps()` daar.
MIN_OVERLAP_S = 2.0

#: De menselijke PSG-IPA-labels op onze typenamen. `Hypopnea` zonder meer is
#: bij AASM obstructief tenzij anders gescoord -- dezelfde aanname die onze
#: kale `hypopnea` maakt.
PSGIPA_LABELS = {
    "obstructive apnea": "obstructive",
    "central apnea": "central",
    "mixed apnea": "mixed",
    "hypopnea": "hypopnea",
    "central hypopnea": "hypopnea_central",
    "obstructive hypopnea": "hypopnea",
    "mixed hypopnea": "hypopnea_mixed",
}


# ── Referenties inlezen ────────────────────────────────────────────────────

def _psgipa_scorer_events(pad: Path):
    """(events, hypnogramregels) uit één PSG-IPA-scoordersbestand."""
    ev, stages = [], []
    for regel in pad.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        d = [x.strip() for x in regel.split(",")]
        if len(d) < 5:
            continue
        try:
            onset, dur = float(d[2]), float(d[3])
        except ValueError:
            continue
        naam = d[4].strip().lower()
        if naam.startswith("sleep stage"):
            stages.append((onset, naam.replace("sleep stage", "").strip().upper()))
        elif naam in PSGIPA_LABELS:
            ev.append({"onset_s": onset, "duration_s": dur,
                       "type": PSGIPA_LABELS[naam]})
    return sorted(ev, key=lambda e: e["onset_s"]), stages


def _hypno_from_stages(stages, dur_s, epoch_s=30.0):
    n = int(np.ceil(dur_s / epoch_s))
    hyp = ["W"] * n
    for onset, st in stages:
        i = int(onset // epoch_s)
        if 0 <= i < n and st in ("W", "N1", "N2", "N3", "R"):
            hyp[i] = st
    return hyp


def _overlaps(a, b, min_ov=MIN_OVERLAP_S):
    a0, a1 = a["onset_s"], a["onset_s"] + a["duration_s"]
    b0, b1 = b["onset_s"], b["onset_s"] + b["duration_s"]
    return max(0.0, min(a1, b1) - max(a0, b0)) >= min_ov


def _agreement(onset_s, scorer_sets, window_s=300.0):
    """Lokale overeenstemming rond dit moment, als spreidingsmaat.

    Niet globaal per opname: op een nacht met een rustige eerste helft en een
    zware tweede helft verschilt de betrouwbaarheid binnen dezelfde opname. Het
    getal is 1 - (spreiding / gemiddelde) van het aantal events per scoorder in
    een venster van vijf minuten rond dit punt, afgekapt op [0, 1].
    """
    tellingen = []
    for ev in scorer_sets:
        tellingen.append(sum(1 for e in ev
                             if abs(e["onset_s"] - onset_s) <= window_s))
    gem = float(np.mean(tellingen)) if tellingen else 0.0
    if gem <= 1e-9:
        return 1.0          # niemand ziet hier iets: daar zijn ze het over eens
    cv = float(np.std(tellingen)) / gem
    return float(max(0.0, min(1.0, 1.0 - cv)))


# ── Eén opname ─────────────────────────────────────────────────────────────

def _kandidaten_en_context(raw, hypno, profiel):
    """Draai de pijplijn en geef alle kandidaten terug, geaccepteerd of niet."""
    import psgscoring

    out = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile=profiel)
    resp = out.get("respiratory") or {}
    geaccepteerd = list(resp.get("events") or [])
    afgewezen = list(resp.get("rejected_hypopneas") or [])
    arousals = list((out.get("arousal") or {}).get("events") or [])
    sq = resp.get("signal_quality") or {}
    spo2_sum = (out.get("spo2") or {}).get("summary") or {}
    ctx = {
        "arousals": arousals,
        "hypno": hypno,
        "sig_dur_s": float(raw.times[-1]),
        "tst_h": float((resp.get("summary") or {}).get("index_denominator_h") or 0.0),
        "overall_qual5": int(sq.get("overall_score") or sq.get("overall_qual5") or 3),
        "median_spo2": float(spo2_sum.get("median_spo2")
                             or spo2_sum.get("mean_spo2") or 95.0),
        "thermistor_type": 0,
    }
    return geaccepteerd, afgewezen, ctx


def _rijen(rec_id, geaccepteerd, afgewezen, ctx, scorer_sets):
    """Eén rij per kandidaat, met harde en zachte labels."""
    from psgscoring.ml_classifier import _extract_candidate_features

    alle = [(c, True) for c in geaccepteerd] + [(c, False) for c in afgewezen]
    rijen = []
    n_sc = max(1, len(scorer_sets))
    for k, (cand, is_acc) in enumerate(alle):
        # Hoeveel scoorders markeerden hier iets, en als wat?
        treffers, subtypes = 0, collections.Counter()
        for ev in scorer_sets:
            match = next((e for e in ev if _overlaps(cand, e)), None)
            if match is not None:
                treffers += 1
                subtypes[match["type"]] += 1
        feats = _extract_candidate_features(
            candidate=cand, is_accepted=is_acc,
            all_candidates=[c for c, _ in alle],
            arousals=ctx["arousals"], hypno=ctx["hypno"],
            sig_dur_s=ctx["sig_dur_s"], tst_h=ctx["tst_h"],
            overall_qual5=ctx["overall_qual5"],
            median_spo2=ctx["median_spo2"],
            thermistor_type=ctx["thermistor_type"],
        )
        rij = {
            "group": rec_id,
            "candidate_id": f"{rec_id}:{k}",
            # HARD label: meerderheid. Bij één scoorder is dat gewoon die ene.
            "label": int(treffers * 2 >= n_sc) if n_sc > 1 else int(treffers > 0),
            # ZACHT label: de fractie. Hier zit de informatie die een hard
            # label weggooit.
            "label_soft": treffers / n_sc,
            "n_scorers": n_sc,
            "agreement": _agreement(cand["onset_s"], scorer_sets),
            "subtype_soft": json.dumps(dict(subtypes)),
            "our_type": cand.get("type", ""),
        }
        rij.update(feats)
        rijen.append(rij)
    return rijen


def _een_psgipa(sn_id, data_dir, profiel):
    import mne
    mne.set_log_level("ERROR")
    data_dir = Path(data_dir)
    psg = data_dir / "Resp_events" / "PSG" / f"{sn_id}_Respiration.edf"
    ann = data_dir / "Resp_events" / "Annotations" / "manual"
    raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
    dur = float(raw.times[-1])

    paden = sorted(ann.glob(f"{sn_id}_Respiration_manual_scorer*.txt"),
                   key=lambda p: int(re.search(r"scorer(\d+)", p.name).group(1)))
    scorer_sets, hypno = [], None
    for i, p in enumerate(paden):
        ev, stages = _psgipa_scorer_events(p)
        scorer_sets.append(ev)
        if i == 0:
            hypno = _hypno_from_stages(stages, dur)
    acc, rej, ctx = _kandidaten_en_context(raw, hypno, profiel)
    return _rijen(sn_id, acc, rej, ctx, scorer_sets)


def _een_mesa(rec_id, data_dir, profiel):
    import mne
    mne.set_log_level("ERROR")
    sys.path.insert(0, str(REPO / "scripts"))
    from validate_mesa import parse_nsrr

    data_dir = Path(data_dir)
    edf = data_dir / "polysomnography" / "edfs" / f"{rec_id}.edf"
    xml = (data_dir / "polysomnography" / "annotations-events-nsrr"
           / f"{rec_id}-nsrr.xml")
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    dur = float(raw.times[-1])
    hypno, refs, _tst = parse_nsrr(xml, dur)
    # Eén referentieset -> één "scoorder". Zie de moduledocstring: hiermee is
    # `label_soft` gelijk aan `label` en meet je geen ruis, alleen detectie.
    naam = "aasm15" if "aasm15" in refs else min(refs)
    menselijk = [{"onset_s": o, "duration_s": max(0.0, e - o), "type": t}
                 for (o, e, t) in refs[naam]]
    acc, rej, ctx = _kandidaten_en_context(raw, hypno, profiel)
    return _rijen(rec_id, acc, rej, ctx, [menselijk])


# ── Hoofdprogramma ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", choices=("mesa", "psgipa"), required=True)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--profile", default="aasm_v3_rec")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.cohort == "psgipa":
        ids = ["SN1", "SN2", "SN3", "SN4", "SN5"]
        fn = _een_psgipa
    else:
        xml_dir = args.data_dir / "polysomnography" / "annotations-events-nsrr"
        ids = [p.stem.replace("-nsrr", "") for p in sorted(xml_dir.glob("*.xml"))]
        fn = _een_mesa
    if args.limit:
        ids = ids[:args.limit]
    print(f"{len(ids)} opnames, profiel {args.profile}, cohort {args.cohort}")

    rijen = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fn, i, str(args.data_dir), args.profile): i
                    for i in ids}
            for f in as_completed(futs):
                try:
                    r = f.result()
                    rijen.extend(r)
                    print(f"  {futs[f]}: {len(r)} kandidaten", flush=True)
                except Exception as e:                        # noqa: BLE001
                    print(f"  {futs[f]}: MISLUKT — {e}", flush=True)
    else:
        for i in ids:
            try:
                r = fn(i, str(args.data_dir), args.profile)
                rijen.extend(r)
                print(f"  {i}: {len(r)} kandidaten", flush=True)
            except Exception as e:                            # noqa: BLE001
                print(f"  {i}: MISLUKT — {e}", flush=True)

    if not rijen:
        print("geen rijen — niets weggeschreven")
        return 1

    import pandas as pd
    df = pd.DataFrame(rijen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        df.to_parquet(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    print(f"\n{len(df)} rijen over {df['group'].nunique()} opnames "
          f"-> {args.output}")
    print(f"  positief (hard):     {df['label'].mean():.3f}")
    print(f"  gemiddeld zacht:     {df['label_soft'].mean():.3f}")
    if df["n_scorers"].max() > 1:
        rand = df[(df["label_soft"] > 0) & (df["label_soft"] < 1)]
        print(f"  OMSTREDEN kandidaten (0 < soft < 1): {len(rand)} "
              f"({100*len(rand)/len(df):.1f} %)")
        print("  -- die zijn met een hard label niet van de rest te "
              "onderscheiden; dat is precies wat label_soft bewaart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
