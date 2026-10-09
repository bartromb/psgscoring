#!/usr/bin/env python3
"""Doorwerking unet_v1 op PSG-IPA SN1–5: beide armen via run_pneumo_analysis (profiel aasm_v3_breath_dual,
hypnogram van de arousal-replicatie), respiratoire F1 tegen de 12 scoorders, AHI tegen de scoordermediaan,
arousal-F1 tegen de EEG-arousal-referentie van de replicatie (SN?_ref.csv)."""
import csv, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/srv/CODE/psgscoring"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import mne; mne.set_log_level("ERROR")
import logging; logging.getLogger().setLevel(logging.ERROR)
import psgscoring, validate_psgipa as vp
from evaluate import match
REP = Path("/srv/CODE/docs/arousal_unet_20260927/out/psgipa"); OUT = Path(__file__).parent
def lees(p):
    with open(p) as fh: return [(float(r["onset_s"]), float(r["offset_s"]), None) for r in csv.DictReader(fh)]
def f1(pred, ref):
    a, b, c, _ = match(pred, ref, "iou", 0.20); return 2*a/(2*a+b+c) if (2*a+b+c) else None
uit = []
for sn in ("SN1", "SN2", "SN3", "SN4", "SN5"):
    hyp = json.load(open(REP / f"{sn}_hypno.json")); hypno = hyp["hypno"] if isinstance(hyp, dict) and "hypno" in hyp else hyp
    raw = mne.io.read_raw_edf(f"/srv/DATA/PSG-IPA/Resp_events/PSG/{sn}_Respiration.edf", preload=True, verbose="ERROR"); dur = float(raw.times[-1])
    scorers = [vp.event_set(f, dur) for f in sorted(Path("/srv/DATA/PSG-IPA/Resp_events/Annotations/manual").glob(f"{sn}_Respiration_manual_scorer*.edf"))]
    ar_ref = lees(REP / f"{sn}_ref.csv") if (REP / f"{sn}_ref.csv").exists() else None
    tst_h = sum(1 for s_ in hypno if s_ in ("N1", "N2", "N3", "R")) * 30 / 3600
    for arm in ("lgbm", "unet_v1"):
        os.environ["PSGSCORING_AROUSAL_DETECTOR"] = arm; os.environ["PSGSCORING_UNET_THREADS"] = "2"
        t0 = time.time(); r = psgscoring.run_pneumo_analysis(raw.copy(), hypno=hypno, scoring_profile="aasm_v3_breath_dual")
        resp = (r.get("respiratory") or {}); summ = resp.get("summary") or {}; ev = resp.get("events") or []
        pred = [(float(e["onset_s"]), float(e["onset_s"]) + float(e.get("duration_s") or 0), None) for e in ev if e.get("onset_s") is not None]
        f1s = [f1(pred, [(a, b, None) for a, b, _ in sc]) for sc in scorers]; f1s = [x for x in f1s if x is not None]
        ar = (r.get("arousal") or {}); ar_ev = ar.get("events") or []; ar_sum = ar.get("summary") or {}
        ar_pred = [(float(e["onset_s"]), float(e.get("end_s") or (float(e["onset_s"]) + float(e.get("duration_s") or 0))), None) for e in ar_ev if e.get("onset_s") is not None]
        rij = {"rec": sn, "arm": arm, "ahi": summ.get("ahi_total"), "rdi": summ.get("rdi"), "n_hypopnea": summ.get("n_hypopnea"), "n_apnea": summ.get("n_apnea_total"), "n_rera": summ.get("n_rera"),
               "arousal_index": ar_sum.get("arousal_index"), "n_arousals": len(ar_ev), "detector": ar_sum.get("detector"), "fallback": ar_sum.get("unet_fallback_reason"),
               "resp_f1_scorer_median": float(np.median(f1s)) if f1s else None, "ahi_scorer_median": float(np.median([len(sc) for sc in scorers])) / tst_h,
               "arousal_f1_ref": f1(ar_pred, ar_ref) if ar_ref else None, "n_arousal_ref": len(ar_ref) if ar_ref else None, "t_s": round(time.time() - t0, 1)}
        uit.append(rij); print(rij, flush=True)
        json.dump(uit, open(OUT / "psgipa_rows.json", "w"), indent=1, default=str)
print("PSGIPA_KLAAR", flush=True)
