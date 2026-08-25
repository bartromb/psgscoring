#!/usr/bin/env python3
"""
diag_arousal_burden_calibration.py — is één vast werkpunt houdbaar over
opnames met een verschillende arousallast?

WAAROM DIT DE VOLGENDE VRAAG IS
-------------------------------
D2 is weerlegd: de featureverdeling van een REFERENTIEEL kin-EMG is vrijwel
gelijk aan die van MESA's bipolaire EMG (mediaan 1,13 tegen 1,09; aandeel
<= 1: 42,6 % tegen 44,5 %). De montage is dus niet de as waarlangs het misgaat.

Wat de meting wel liet zien: het model houdt op werkpunt 0,80 ook op MESA maar
11,9 % van de kandidaten over. Die scherpte is dus algemeen, niet klinisch.

Op MESA leverde dat een telling van 0,83x de menselijke referentie -- prima.
Op de klinische opname (AHI 62) laat het regelgebaseerde pad 65,6 % van de
events in een arousal eindigen en de classifier 15,5 %. Het verschil tussen
die twee situaties is niet de montage maar de LAST: MESA is een
bevolkingscohort, deze patiënt heeft ernstig OSAS.

Hypothese die hier getoetst wordt: de count-ratio (onze telling / menselijke
telling) DAALT naarmate de arousallast stijgt. Is dat zo, dan is een vast
werkpunt per definitie te streng aan de zware kant en is dát de reparatie --
niet een montage-afhankelijke drempel.

Vooraf vastgelegd: de uitkomstmaat is Spearman rho tussen de menselijke
arousal-index en de count-ratio. Negatief en significant => hypothese
gesteund. Rond nul => weerlegd, en dan ligt de oorzaak elders.

Gebruik:
    python scripts/diag_arousal_burden_calibration.py --n 20
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
mne.set_log_level("ERROR")

from psgscoring.arousal import detect_arousals

EPOCH_S = 30.0
SLEEP = {"Stage 1 sleep|1", "Stage 2 sleep|2", "Stage 3 sleep|3",
         "Stage 4 sleep|4", "REM sleep|5"}


def ref_counts(xml_path: Path):
    """(n_arousals, n_resp_events, tst_h) uit de NSRR-annotatie."""
    root = ET.parse(xml_path).getroot()
    n_ar = n_resp = 0
    tst_s = 0.0
    for ev in root.iter("ScoredEvent"):
        c = (ev.findtext("EventConcept") or "")
        d = float(ev.findtext("Duration") or 0)
        if c in SLEEP:
            tst_s += d
        cl = c.lower()
        if "arousal" in cl:
            n_ar += 1
        elif "apnea" in cl or "hypopnea" in cl:
            n_resp += 1
    return n_ar, n_resp, tst_s / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesa-dir", default="/home/bart/MESA/mesa")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from validate_mesa import parse_nsrr
    edfs = sorted((Path(a.mesa_dir) / "polysomnography" / "edfs")
                  .glob("mesa-sleep-*.edf"))[:a.n]
    xmls = Path(a.mesa_dir) / "polysomnography" / "annotations-events-nsrr"

    kop = (f"{'opname':<12}{'ref AI':>8}{'ref n':>7}{'onze n':>8}"
           f"{'ratio':>8}{'regels n':>10}{'r-ratio':>9}{'resp/u':>8}")
    print(kop); print("-" * len(kop), flush=True)
    rijen = []
    for edf in edfs:
        xml = xmls / f"{edf.stem}-nsrr.xml"
        if not xml.exists():
            continue
        h = mne.io.read_raw_edf(edf, preload=False, verbose=False)
        if "EMG" not in h.ch_names or "EEG3" not in h.ch_names:
            continue
        dur = h.n_times / h.info["sfreq"]
        hypno, _r, _t = parse_nsrr(xml, dur)
        n_ar_ref, n_resp_ref, tst_h = ref_counts(xml)
        if tst_h <= 0 or n_ar_ref == 0:
            continue
        raw = mne.io.read_raw_edf(
            edf, exclude=[c for c in h.ch_names if c not in {"EEG3", "EMG"}],
            preload=True, verbose=False)
        sf = raw.info["sfreq"]
        eeg = raw.get_data(picks=["EEG3"])[0]
        emg = raw.get_data(picks=["EMG"])[0]

        aan = detect_arousals(eeg, sf, hypno, emg_data=emg,
                              lgbm=True, lgbm_threshold=a.threshold)
        uit = detect_arousals(eeg, sf, hypno, emg_data=emg)
        n_aan = len(aan.get("events") or [])
        n_uit = len(uit.get("events") or [])
        rij = {"rec": edf.stem, "ref_ai": n_ar_ref / tst_h,
               "ref_n": n_ar_ref, "n_aan": n_aan, "n_uit": n_uit,
               "ratio": n_aan / n_ar_ref, "ratio_regels": n_uit / n_ar_ref,
               "resp_per_h": n_resp_ref / tst_h, "tst_h": tst_h}
        rijen.append(rij)
        print(f"{edf.stem.replace('mesa-sleep-',''):<12}"
              f"{rij['ref_ai']:>8.1f}{n_ar_ref:>7}{n_aan:>8}"
              f"{rij['ratio']:>8.2f}{n_uit:>10}{rij['ratio_regels']:>9.2f}"
              f"{rij['resp_per_h']:>8.1f}", flush=True)

    if len(rijen) < 4:
        print("\nte weinig opnames voor een correlatie")
        return
    from scipy.stats import spearmanr
    ai = np.array([r["ref_ai"] for r in rijen])
    ratio = np.array([r["ratio"] for r in rijen])
    ratio_r = np.array([r["ratio_regels"] for r in rijen])
    resp = np.array([r["resp_per_h"] for r in rijen])
    rho, p = spearmanr(ai, ratio)
    rho2, p2 = spearmanr(resp, ratio)
    rho3, p3 = spearmanr(ai, ratio_r)
    print(f"\nn = {len(rijen)}")
    print(f"count-ratio met classifier : mediaan {np.median(ratio):.2f}")
    print(f"count-ratio regelgebaseerd : mediaan {np.median(ratio_r):.2f}")
    print(f"\nSpearman  arousallast  vs count-ratio (classifier): "
          f"rho = {rho:+.3f}  p = {p:.4f}")
    print(f"Spearman  eventlast    vs count-ratio (classifier): "
          f"rho = {rho2:+.3f}  p = {p2:.4f}")
    print(f"Spearman  arousallast  vs count-ratio (regels)     : "
          f"rho = {rho3:+.3f}  p = {p3:.4f}   [controle]")
    print("\nVooraf: negatief en significant => een vast werkpunt is te streng\n"
          "aan de zware kant. Rond nul => weerlegd, oorzaak ligt elders.")
    if a.out:
        Path(a.out).write_text(json.dumps(rijen, indent=2, default=float))


if __name__ == "__main__":
    main()
