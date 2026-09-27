#!/usr/bin/env python3
"""psgscoring (geïnstalleerd) op MESA-nachten: arousal-events tegen NSRR.

    OMP_NUM_THREADS=1 .venv/bin/python bench/eeg/common/run_baseline_mesa.py mesa-sleep-4535

Zelfde aanroep als op PSG-IPA (run_pneumo_analysis, aasm_v3_rec), hypnogram
en referentie-arousals uit de NSRR-XML via bench/eeg/unet50/data.parse_mesa_xml
-- dezelfde parser als de U-Net en MSED gebruiken, zodat de drie armen exact
dezelfde referentie zien (het einde van een arousal wordt op de EDF-duur
geknipt; raakt 2 van 80 nachten met <= 1 event, gelijk voor alle armen). Schrijft
bench/eeg/baseline_mesa/<rec>_pred.csv en <rec>_ref.csv (NSRR-arousals,
onset/offset) — alleen eventtijden, geen signaal.
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import mne
mne.set_log_level("ERROR")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "bench" / "eeg" / "unet50"))
UIT = REPO / "bench" / "eeg" / "baseline_mesa"
EDF = Path("/srv/DATA/MESA/mesa/polysomnography/edfs"); XML = Path("/srv/DATA/MESA/mesa/polysomnography/annotations-events-nsrr")


def main(rec: str):
    import psgscoring
    from data import parse_mesa_xml
    UIT.mkdir(exist_ok=True)
    raw = mne.io.read_raw_edf(str(EDF / f"{rec}.edf"), preload=True, verbose=False)
    dur = raw.n_times / raw.info["sfreq"]
    hypno, ar = parse_mesa_xml(XML / f"{rec}-nsrr.xml", dur)
    t0 = time.time()
    res = psgscoring.run_pneumo_analysis(raw, hypno=hypno, scoring_profile="aasm_v3_rec")
    ev = (res.get("arousal") or {}).get("events") or []
    for name, rows in ((f"{rec}_pred.csv", [(float(e["onset_s"]), float(e["onset_s"]) + float(e["duration_s"])) for e in ev]),
                       (f"{rec}_ref.csv", ar)):
        with (UIT / name).open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["onset_s", "offset_s", "type"])
            for a, b in sorted(rows):
                w.writerow([f"{a:.3f}", f"{b:.3f}", "arousal"])
    (UIT / f"{rec}_summary.json").write_text(json.dumps({"rec": rec, "n_pred": len(ev), "n_ref": len(ar),
                                                          "runtime_s": round(time.time() - t0, 1),
                                                          "version": psgscoring.__version__}))
    print(rec, len(ev), len(ar), round(time.time() - t0), "s")


if __name__ == "__main__":
    main(sys.argv[1])
