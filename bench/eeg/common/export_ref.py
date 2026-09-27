#!/usr/bin/env python3
"""Referentie-arousals en hypnogram van PSG-IPA per opname, voor bench/evaluate.py.

Bron: /srv/DATA/PSG-IPA/Resp_events/Annotations/manual/SNx_Respiration_manual_scorer1.edf
  - arousals: annotaties met "eeg arousal" in de beschrijving uit het
    scoorder-1-bestand -- de gedeelde EEG-arousalannotatie van de Resp_events-
    subboom (scoorderkopieën verschillen <= 1 event per opname: SN1/SN5 12/12
    identiek, SN2 scoorder 7 heeft 95 i.p.v. 96, SN3 scoorder 1 heeft 241 waar
    tien scoorders 242 hebben, SN4 scoorder 10 heeft 59); identiek aan de
    referentie van de orakelmeting (docs/orakel_rule1a_psgipa.json); zelfde
    lezer als docs/meet_koppelvenster_psgipa.py
  - hypnogram: validate_psgipa.parse_scorer_file (scoorder 1)

Uitvoer (bench/eeg/ref/):
  SNx_ref_arousal.csv   onset_s,offset_s,type        (type = arousal)
  SNx_hypno.json        {"hypno": [...30-s-stadia...], "dur_s": ..., "tst_h": ...}
Leest alleen; schrijft niets onder /srv/DATA.
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
import mne
mne.set_log_level("ERROR")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from validate_psgipa import parse_scorer_file, find_scorer_files  # noqa: E402

ROOT = Path(os.environ.get("PSGSCORING_DATA_ROOT", "/srv/DATA")) / "PSG-IPA"
UIT = REPO / "bench" / "eeg" / "ref"
SNS = ["SN1", "SN2", "SN3", "SN4", "SN5"]


def main():
    UIT.mkdir(parents=True, exist_ok=True)
    overzicht = {}
    for sn in SNS:
        edf = ROOT / "Resp_events" / "PSG" / f"{sn}_Respiration.edf"
        hdr = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
        dur = hdr.n_times / hdr.info["sfreq"]
        f1 = next(f for f in find_scorer_files(ROOT, sn) if f.stem.endswith("scorer1"))
        ahi, tst_h, hypno = parse_scorer_file(f1, dur)
        ann = mne.read_annotations(str(f1))
        ev = sorted((float(o), float(o) + float(d)) for o, d, x in
                    zip(ann.onset, ann.duration, ann.description)
                    if "eeg arousal" in str(x).lower() and 0 <= float(o) < dur)
        with (UIT / f"{sn}_ref_arousal.csv").open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["onset_s", "offset_s", "type"])
            for a, b in ev:
                w.writerow([f"{a:.3f}", f"{b:.3f}", "arousal"])
        (UIT / f"{sn}_hypno.json").write_text(json.dumps(
            {"hypno": hypno, "dur_s": dur, "tst_h": tst_h, "sfreq": hdr.info["sfreq"],
             "ch_names": hdr.ch_names, "scorer_file": f1.name}))
        overzicht[sn] = {"n_arousal": len(ev), "tst_h": round(tst_h, 2),
                         "arousal_index": round(len(ev) / tst_h, 1), "dur_h": round(dur / 3600, 2),
                         "ref_ahi_scorer1": round(ahi, 1)}
        print(sn, overzicht[sn])
    (UIT / "overzicht.json").write_text(json.dumps(overzicht, indent=1))


if __name__ == "__main__":
    main()
