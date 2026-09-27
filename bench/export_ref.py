#!/usr/bin/env python3
"""Exporteer referentie-events uit /srv/DATA naar het formaat van bench/evaluate.py.

    python bench/export_ref.py psgipa SN3 --scorer 1 -o uit/SN3_ref.csv
    python bench/export_ref.py psgipa SN3 --scorer all -o uit/          # 12 bestanden
    python bench/export_ref.py mesa mesa-sleep-0001 --ref aasm15 -o uit/mesa-sleep-0001_ref.csv

PSG-IPA (publiek, PhysioNet): respiratoire annotaties per scoorder uit
Resp_events/Annotations/manual/*.edf, via validate_psgipa.event_set — dezelfde
lezer als de validatieharnassen. MESA (NSRR, DUA — ter plekke gebruiken,
nooit kopiëren): gereconstrueerde referentie via validate_mesa.parse_nsrr
(`aasm15` primair; `desat3_all`, `oahi3`, `oahi4` beschikbaar). Dataroot:
$PSGSCORING_DATA_ROOT, default /srv/DATA.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
ROOT = Path(os.environ.get("PSGSCORING_DATA_ROOT", "/srv/DATA"))
# mne meldt per EDF verschillende filterinstellingen per kanaal; hier lezen we alleen de duur.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")


def schrijf(pad: Path, events):
    pad.parent.mkdir(parents=True, exist_ok=True)
    with pad.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["onset_s", "offset_s", "type"])
        for a, b, t in sorted(events):
            w.writerow([f"{a:.3f}", f"{b:.3f}", t])
    print(f"{pad}: {len(events)} events")


def psgipa(sn: str, scorer: str, uit: Path):
    import mne
    from validate_psgipa import event_set, find_scorer_files
    mne.set_log_level("ERROR")
    edf = ROOT / "PSG-IPA" / "Resp_events" / "PSG" / f"{sn}_Respiration.edf"
    dur = float(mne.io.read_raw_edf(str(edf), preload=False, verbose=False).times[-1])
    files = find_scorer_files(ROOT / "PSG-IPA", sn)
    if not files:
        raise SystemExit(f"geen scoorderbestanden voor {sn} onder {ROOT}")
    if scorer == "all":
        for f in files:
            k = f.stem.rsplit("scorer", 1)[-1]
            schrijf(uit / f"{sn}_ref_scorer{k}.csv", event_set(f, dur))
    else:
        f = next((f for f in files if f.stem.endswith(f"scorer{scorer}")), None)
        if f is None:
            raise SystemExit(f"scoorder {scorer} niet gevonden; beschikbaar: {[x.stem for x in files]}")
        schrijf(uit if uit.suffix else uit / f"{sn}_ref_scorer{scorer}.csv", event_set(f, dur))


def mesa(rec: str, ref: str, uit: Path):
    import mne
    from validate_mesa import parse_nsrr
    mne.set_log_level("ERROR")
    edf = ROOT / "MESA" / "mesa" / "polysomnography" / "edfs" / f"{rec}.edf"
    xml = ROOT / "MESA" / "mesa" / "polysomnography" / "annotations-events-nsrr" / f"{rec}-nsrr.xml"
    dur = float(mne.io.read_raw_edf(str(edf), preload=False, verbose=False).times[-1])
    _hyp, refs, tst_h = parse_nsrr(xml, dur)
    if ref not in refs:
        raise SystemExit(f"referentie {ref} onbekend; keuze: {sorted(refs)}")
    print(f"{rec}: TST {tst_h:.2f} h, referentie {ref}")
    schrijf(uit if uit.suffix else uit / f"{rec}_ref_{ref}.csv", refs[ref])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bron", choices=("psgipa", "mesa"))
    ap.add_argument("opname", help="SN1..SN5 of mesa-sleep-NNNN")
    ap.add_argument("--scorer", default="1", help="PSG-IPA: 1..12 of all")
    ap.add_argument("--ref", default="aasm15", help="MESA: aasm15|desat3_all|oahi3|oahi4")
    ap.add_argument("-o", "--out", type=Path, required=True, help="bestand (.csv) of map")
    a = ap.parse_args()
    (psgipa(a.opname, a.scorer, a.out) if a.bron == "psgipa" else mesa(a.opname, a.ref, a.out))


if __name__ == "__main__":
    main()
