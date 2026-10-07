#!/usr/bin/env python3
"""psgscoring-baselines voor de respiratoire U-Net-evaluatie: per nacht `run_pneumo_analysis` met het
NSRR-hypnogram (harnasconventie: artefact-epochs leeg), events als CSV + AHI in een jsonl.

    OMP_NUM_THREADS=1 ../eeg/_venv/bin/python baseline.py --cohort shhs1 --profiles aasm_v3_rec aasm_v3_breath --workers 20
    OMP_NUM_THREADS=1 ../eeg/_venv/bin/python baseline.py --cohort mesa_val --profiles aasm_v3_rec aasm_v3_breath_dual --workers 20

Draai NA de lopende MESA-run (CPU) en met de bibliotheek in /srv/CODE/psgscoring zoals hij dan staat;
de versie en git-SHA gaan mee in de log."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, json, subprocess, sys, time
from multiprocessing import get_context
from pathlib import Path
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER)); sys.path.insert(0, str(HIER.parents[1] / "scripts")); sys.path.insert(0, str(HIER.parents[1]))
OUT = HIER / "out"
MESA = Path("/srv/DATA/MESA/mesa/polysomnography"); SHHS = Path("/srv/DATA/SHHS/shhs/polysomnography")
# Zoals scripts/arousal_unet_replicatie.py (en eerder SHHS-validation/score_shhs.py): het thermokoppel op de drukplaats.
SHHS_CMAP = {"flow_pressure": "NEW AIR", "thorax": "THOR RES", "abdomen": "ABDO RES"}
APNEA = {"obstructive", "central", "mixed", "uncertain"}


def paden(cohort, rec):
    if cohort == "shhs1":
        return SHHS / "edfs/shhs1" / f"{rec}.edf", SHHS / "annotations-events-nsrr/shhs1" / f"{rec}-nsrr.xml", SHHS_CMAP
    return MESA / "edfs" / f"{rec}.edf", MESA / "annotations-events-nsrr" / f"{rec}-nsrr.xml", None


def een(werk):
    cohort, rec, profiles, force = werk
    import mne; mne.set_log_level("ERROR")
    import logging; logging.getLogger().setLevel(logging.ERROR)
    import psgscoring
    from validate_mesa import parse_nsrr
    edf, xml, cmap = paden(cohort, rec); d = OUT / cohort; d.mkdir(parents=True, exist_ok=True)
    rij = {"rec": rec, "profiles": {}}
    try:
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
        dur = float(raw.times[-1]); hypno, refs, tst_h = parse_nsrr(xml, dur)
        for prof in profiles:
            f_csv = d / f"{rec}_base_{prof}.csv"
            if f_csv.exists() and not force:
                continue
            t0 = time.time()
            res = psgscoring.run_pneumo_analysis(raw.copy(), hypno=hypno, scoring_profile=prof, channel_map=cmap, artifact_epochs=[])
            resp = res.get("respiratory") or {}; ev = resp.get("events") or []; summ = resp.get("summary") or {}
            with f_csv.open("w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["onset_s", "offset_s", "type"])
                for e in sorted(ev, key=lambda e: float(e.get("onset_s") or 0)):
                    a = float(e["onset_s"]); b = a + float(e.get("duration_s") or 0.0)
                    w.writerow([f"{a:.3f}", f"{b:.3f}", "apnea" if str(e.get("type")) in APNEA else "hypopnea"])
            rij["profiles"][prof] = {"n": len(ev), "ahi": summ.get("ahi_total"), "rdi": summ.get("rdi"), "t_s": round(time.time() - t0, 1),
                                     "flow_channels": {k: v for k, v in ((res.get("meta") or {}).get("flow_channels") or {}).items() if k in ("apnea_sensor", "hypopnea_sensor", "thermistor_rejected")}}
        rij.update(tst_h=tst_h, n_ref=len(refs.get("aasm15", [])))
    except Exception as e:  # noqa: BLE001
        rij["error"] = repr(e)
    return rij


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["shhs1", "mesa_val"])
    ap.add_argument("--profiles", nargs="+", default=["aasm_v3_rec", "aasm_v3_breath"])
    ap.add_argument("--workers", type=int, default=20); ap.add_argument("--force", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ids = (OUT / a.cohort / "ids.txt").read_text().split() if a.cohort == "shhs1" else (HIER / "ids_val.txt").read_text().split()
    if a.limit:
        ids = ids[: a.limit]
    sha = subprocess.run(["git", "-C", str(HIER.parents[1]), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    import psgscoring
    log = OUT / a.cohort / "baseline_log.jsonl"; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "psgscoring": psgscoring.__version__, "git": sha, "profiles": a.profiles, "n": len(ids)}) + "\n")
    werk = [(a.cohort, rec, a.profiles, a.force) for rec in ids]
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for i, rij in enumerate(pool.imap_unordered(een, werk), 1):
            with log.open("a") as fh:
                fh.write(json.dumps(rij, default=str) + "\n")
            print(f"[{i}/{len(ids)}]", rij["rec"], rij.get("error") or {p: (v["n"], v["ahi"]) for p, v in rij["profiles"].items()}, flush=True)


if __name__ == "__main__":
    main()
