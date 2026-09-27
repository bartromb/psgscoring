#!/usr/bin/env python3
"""Baseline: de huidige arousaldetector van psgscoring zoals geïnstalleerd.

    OMP_NUM_THREADS=1 .venv/bin/python bench/eeg/common/run_baseline.py SN3

Draait psgscoring.run_pneumo_analysis(raw, hypno, scoring_profile="aasm_v3_rec")
op de Resp_events-EDF met het hypnogram van scoorder 1 (bench/eeg/ref/) en
schrijft result["arousal"]["events"] als onset_s,offset_s,type naar
bench/eeg/baseline/SNx_pred.csv, plus de samenvatting/provenance als JSON.
Geen enkele parameter wordt aangepast: dit is de productiestand van de
bibliotheek (multi-derivatie-union + LightGBM op werkpunt 0,70, 10 s-interval,
onset-offset +2 s, re-ranker alleen waar Pleth/HR bestaat).
"""
from __future__ import annotations
import csv, json, os, sys, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
import mne
mne.set_log_level("ERROR")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
ROOT = Path(os.environ.get("PSGSCORING_DATA_ROOT", "/srv/DATA")) / "PSG-IPA"
REF = REPO / "bench" / "eeg" / "ref"
UIT = REPO / "bench" / "eeg" / "baseline"


def main(sn: str, profile: str = "aasm_v3_rec"):
    import psgscoring
    UIT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((REF / f"{sn}_hypno.json").read_text())
    raw = mne.io.read_raw_edf(str(ROOT / "Resp_events" / "PSG" / f"{sn}_Respiration.edf"),
                              preload=True, verbose=False)
    t0 = time.time()
    res = psgscoring.run_pneumo_analysis(raw, hypno=meta["hypno"], scoring_profile=profile)
    dt = time.time() - t0
    ar = res.get("arousal", {}) or {}
    events = ar.get("events") or []
    with (UIT / f"{sn}_pred.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["onset_s", "offset_s", "type"])
        for e in sorted(events, key=lambda e: float(e["onset_s"])):
            on = float(e["onset_s"]); off = on + float(e["duration_s"])
            w.writerow([f"{on:.3f}", f"{off:.3f}", "arousal"])
    summ = {k: v for k, v in ar.items() if k != "events"}
    def _safe(o):
        try:
            json.dumps(o); return o
        except TypeError:
            return str(o)
    out = {"sn": sn, "profile": profile, "psgscoring_version": psgscoring.__version__,
           "n_events": len(events), "runtime_s": round(dt, 1),
           "summary": json.loads(json.dumps(summ, default=str)),
           "meta_keys": sorted((res.get("meta") or {}).keys()),
           "meta_arousal": {k: _safe(v) for k, v in (res.get("meta") or {}).items()
                            if "arousal" in k.lower() or "derivation" in k.lower() or "rerank" in k.lower()}}
    (UIT / f"{sn}_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(sn, "events", len(events), "runtime", round(dt, 1), "s")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
