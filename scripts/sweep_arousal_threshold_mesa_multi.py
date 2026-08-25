#!/usr/bin/env python3
"""
sweep_arousal_threshold_mesa_multi.py — replicatiearm voor de werkpuntkeuze.

WAAROM NAAST PSG-IPA
--------------------
PSG-IPA levert de referentiekwaliteit (twaalf scoorders) maar heeft n = 5.
Het releasebeleid eist voor elke default-flip een MESA-meting met een vooraf
vastgelegde beslisregel. MESA heeft één scoorder, maar wel de aantallen -- en
het is het cohort waarop het huidige werkpunt 0,80 ooit gekozen is, dus een
verschuiving hoort daar zichtbaar te zijn.

Draait de PRODUCTIECONFIGURATIE: de afleidingsset zoals `_pick_eeg_multi` hem
kiest, door `detect_arousals_multi`. De eerdere 0,80-keuze is op MESA gemaakt
toen die picker nog de saturatiecurve kon aanwijzen; met een werkende union
stijgen de tellingen en schuift het optimum mee.

BESLISREGEL (vooraf)
--------------------
Uitkomstmaat is de count-ratio (onze telling / NSRR-referentie), mediaan over
de opnames. Doel is 1,0. Een kandidaat-werkpunt is alleen aanvaardbaar als de
mediane ratio **niet boven 1,10** uitkomt -- dat is de grens uit
`docs/arousal-recall-diagnose.md`, gezet om niet terug te vallen in de
1,47x-overtelling van vóór v0.26.

Gebruik:
    python scripts/sweep_arousal_threshold_mesa_multi.py --n 20
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median

import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
mne.set_log_level("ERROR")

from psgscoring.arousal import detect_arousals, detect_arousals_multi
from psgscoring.pipeline import _pick_eeg_multi

SLEEP_XML = {"Stage 1 sleep|1", "Stage 2 sleep|2", "Stage 3 sleep|3",
             "Stage 4 sleep|4", "REM sleep|5"}


def ref_arousals(xml_path: Path) -> tuple[int, float]:
    root = ET.parse(xml_path).getroot()
    n, tst = 0, 0.0
    for ev in root.iter("ScoredEvent"):
        c = ev.findtext("EventConcept") or ""
        if c in SLEEP_XML:
            tst += float(ev.findtext("Duration") or 0)
        if "arousal" in c.lower():
            n += 1
    return n, tst / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesa-dir", default="/home/bart/MESA/mesa")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sweep", default="0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    drempels = [float(x) for x in a.sweep.split(",")]

    from validate_mesa import parse_nsrr
    edfs = sorted((Path(a.mesa_dir) / "polysomnography" / "edfs")
                  .glob("mesa-sleep-*.edf"))[:a.n]
    xmls = Path(a.mesa_dir) / "polysomnography" / "annotations-events-nsrr"

    rijen = []
    for edf in edfs:
        xml = xmls / f"{edf.stem}-nsrr.xml"
        if not xml.exists():
            continue
        h = mne.io.read_raw_edf(edf, preload=False, verbose=False)
        if "EMG" not in h.ch_names:
            continue
        n_ref, tst_h = ref_arousals(xml)
        if n_ref == 0 or tst_h <= 0:
            continue
        dur = h.n_times / h.info["sfreq"]
        hypno, _r, _t = parse_nsrr(xml, dur)
        houden = {c for c in h.ch_names
                  if c.upper().startswith("EEG") and not c.upper().endswith("_OFF")}
        houden.add("EMG")
        raw = mne.io.read_raw_edf(
            edf, exclude=[c for c in h.ch_names if c not in houden],
            preload=True, verbose=False)
        sf = raw.info["sfreq"]
        emg = raw.get_data(picks=["EMG"])[0]
        derivs = _pick_eeg_multi(raw, {})
        if not derivs:
            continue

        rij = {"rec": edf.stem, "ref": n_ref, "ref_ai": n_ref / tst_h,
               "derivs": [n for n, _d, _s in derivs], "armen": {}}
        r0 = detect_arousals(derivs[0][1], sf, hypno, emg_data=emg)
        rij["armen"]["regels"] = len(r0.get("events") or [])
        for t in drempels:
            r = detect_arousals_multi(derivs, sf, hypno, emg_data=emg,
                                      lgbm=True, lgbm_threshold=t)
            rij["armen"][f"{t:.2f}"] = len(r.get("events") or [])
        rijen.append(rij)
        print(f"{edf.stem.replace('mesa-sleep-',''):<8} ref {n_ref:>4} "
              f"({rij['ref_ai']:>5.1f}/u) [{len(derivs)} afl] | "
              + " ".join(f"{k}={v}" for k, v in rij["armen"].items()),
              flush=True)

    if not rijen:
        return
    print(f"\nn = {len(rijen)}")
    print("── count-ratio (onze telling / NSRR-referentie) ──")
    armen = list(rijen[0]["armen"])
    for arm in armen:
        rr = [r["armen"][arm] / r["ref"] for r in rijen]
        binnen = sum(1 for x in rr if 0.9 <= x <= 1.1)
        vlag = "" if median(rr) <= 1.10 else "   << boven de grens 1,10"
        print(f"   {arm:>7} : mediaan {median(rr):.2f}   "
              f"binnen 0,9-1,1 op {binnen}/{len(rr)}{vlag}")
    print("\nVooraf: een kandidaat-werkpunt is alleen aanvaardbaar als de\n"
          "mediane ratio niet boven 1,10 uitkomt.")
    if a.out:
        Path(a.out).write_text(json.dumps(rijen, indent=2, default=float))


if __name__ == "__main__":
    main()
