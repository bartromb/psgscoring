#!/usr/bin/env python
"""inkortingstoets_mesa.py — de stadium-basislijn op een te kort hypnogram (MESA, 30 nachten).

Zes armen per nacht: NSRR-hypnogram dekkend (H0), laatste epoch weg (H1, vloerlengte zoals YASA),
laatste 5 % weg (H5), elk met `stage_baseline_partial_epochs` True (new) en False (old; de
basislijn faalt dan stil zodra het hypnogram korter is dan het signaal). Tellingen en F1 tegen
NSRR `aasm15` binnen het gemeenschappelijke venster (onset < len(hypnogram_arm)·30 s).
Preregistratie: docs/inkortingstoets_preregistratie_20261026.md. Hervat uit rijen.partial.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
sys.path.insert(0, str(HIER.parent))
from validate_mesa import (  # noqa: E402
    APNEA_TYPES, LEGACY_MATCHER, PRIMARY_REFERENCE, _apply_fields, match_events, parse_nsrr,
    severity,
)

DATA = Path("/srv/DATA/MESA/mesa/polysomnography")
EDF_DIR = DATA / "edfs"
XML_DIR = DATA / "annotations-events-nsrr"
EPOCH_S = 30.0
VELD = "stage_baseline_partial_epochs"
ARMEN = {"H0_new": ("0", True), "H0_old": ("0", False), "H1_new": ("1", True), "H1_old": ("1", False),
         "H5_new": ("5pct", True), "H5_old": ("5pct", False)}


def _knip(hypno: list, hoe: str) -> list:
    n = len(hypno)
    k = 0 if hoe == "0" else (1 if hoe == "1" else int(round(0.05 * n)))
    return list(hypno[: n - k])


def _venster(evs, t_cut):
    return [e for e in evs if e[0] < t_cut]


def _record(res: dict, refs: dict, hypno: list, t_cut: float) -> dict:
    r = res.get("respiratory", {}) or {}
    summ = r.get("summary", {}) or {}
    ev = [e for e in r.get("events", []) if e.get("onset_s") is not None]
    algo = [(float(e["onset_s"]), float(e["onset_s"]) + float(e.get("duration_s") or 0.0), str(e.get("type"))) for e in ev]
    algo_w = _venster(algo, t_cut)
    ref = refs[PRIMARY_REFERENCE]; ref_w = _venster(ref, t_cut)
    slaap_w_h = sum(1 for s in hypno if s in ("N1", "N2", "N3", "R")) * EPOCH_S / 3600.0
    m = match_events(algo_w, ref_w, **LEGACY_MATCHER)
    apn_a = [e for e in algo_w if e[2] in APNEA_TYPES]; apn_r = [e for e in ref_w if e[2] in APNEA_TYPES]
    ma = match_events(apn_a, apn_r, **LEGACY_MATCHER) if (apn_a or apn_r) else {"f1": None, "precision": None, "recall": None, "tp": 0, "fp": 0, "fn": 0}
    dual = r.get("dual_sensor_apnea") if isinstance(r.get("dual_sensor_apnea"), dict) else {}
    return {
        "stage_baseline_used": r.get("stage_baseline_used"),
        "ahi_full": float(summ.get("ahi_total") or 0.0), "n_events_full": len(algo),
        "n_apnea_full": sum(1 for e in algo if e[2] in APNEA_TYPES),
        "t_cut_s": t_cut, "slaap_venster_h": slaap_w_h,
        "n_events_w": len(algo_w), "n_apnea_w": len(apn_a), "n_hyp_w": len(algo_w) - len(apn_a),
        "n_ref_w": len(ref_w), "n_ref_apnea_w": len(apn_r),
        "ahi_w": (len(algo_w) / slaap_w_h if slaap_w_h else None), "ahi_ref_w": (len(ref_w) / slaap_w_h if slaap_w_h else None),
        "severity_w": severity(len(algo_w) / slaap_w_h) if slaap_w_h else None,
        "severity_ref_w": severity(len(ref_w) / slaap_w_h) if slaap_w_h else None,
        "match": {k: m[k] for k in ("f1", "precision", "recall", "tp", "fp", "fn")},
        "match_apnea": {k: ma.get(k) for k in ("f1", "precision", "recall", "tp", "fp", "fn")},
        "dual": {k: dual.get(k) for k in ("n_both", "n_pressure_only", "n_thermistor_only", "n_kept")} if dual else None,
        "events": [(round(a, 2), round(b, 2), t) for a, b, t in algo],
    }


def analyse_one(args):
    rec_id, profile = args
    import mne
    mne.set_log_level("ERROR")
    import psgscoring
    t0 = time.time()
    try:
        raw = mne.io.read_raw_edf(str(EDF_DIR / f"{rec_id}.edf"), preload=True, verbose=False)
    except Exception as e:  # noqa: BLE001
        return {"recording": rec_id, "error": f"EDF: {e}"}
    dur = float(raw.times[-1])
    try:
        hypno, refs, tst_h = parse_nsrr(XML_DIR / f"{rec_id}-nsrr.xml", dur)
    except Exception as e:  # noqa: BLE001
        return {"recording": rec_id, "error": f"XML: {e}"}
    if tst_h < 1.0:
        return {"recording": rec_id, "error": f"TST {tst_h:.2f} h te kort"}
    out = {"recording": rec_id, "profile": profile, "psgscoring": psgscoring.__version__, "tst_h": tst_h,
           "duration_s": dur, "n_epochs_hypno": len(hypno), "n_samples": int(raw.n_times), "sfreq": float(raw.info["sfreq"]),
           "n_ref": len(refs[PRIMARY_REFERENCE]), "ahi_ref": len(refs[PRIMARY_REFERENCE]) / tst_h, "arms": {}}
    for arm, (hoe, vlag) in ARMEN.items():
        h = _knip(hypno, hoe); t_cut = len(h) * EPOCH_S
        t1 = time.time()
        try:
            _apply_fields(profile, {VELD: vlag})
            res = psgscoring.run_pneumo_analysis(raw.copy(), hypno=h, scoring_profile=profile)
            rec = _record(res, refs, h, t_cut)
            rec["veld_gelezen"] = bool(psgscoring.constants.SCORING_PROFILES[profile].get("STAGE_BASELINE_PARTIAL_EPOCHS"))
        except Exception as e:  # noqa: BLE001
            rec = {"error": str(e)}
        finally:
            _apply_fields(profile, None)
        rec["n_epochs"] = len(h); rec["seconds"] = round(time.time() - t1, 1)
        out["arms"][arm] = rec
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", default="aasm_v3_breath_dual")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.environ.setdefault("PSGSCORING_AROUSAL_DETECTOR", "lgbm")
    a.out.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in a.ids_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if a.only: ids = [i for i in ids if i in set(a.only)]
    partial = a.out / "rijen.partial.jsonl"; klaar = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            try:
                r = json.loads(line); klaar[r["recording"]] = r
            except Exception:  # noqa: BLE001
                pass
    todo = [i for i in ids if i not in klaar]
    print(f"start {time.strftime('%F %T')}: {len(ids)} nachten, {len(todo)} te doen, {a.workers} workers, armen {list(ARMEN)}", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex, partial.open("a") as fh:
        for r in ex.map(analyse_one, [(i, a.profile) for i in todo]):
            klaar[r["recording"]] = r; fh.write(json.dumps(r) + "\n"); fh.flush()
            arms = r.get("arms", {})
            print(f"  {r['recording']}: {'FOUT ' + r['error'] if r.get('error') else 'ok'} {r.get('seconds')} s  sbu=" +
                  ",".join(f"{k}:{v.get('stage_baseline_used')}" for k, v in arms.items()) + f"  {time.strftime('%T')}", flush=True)
    rows = [klaar[i] for i in ids if i in klaar]
    json.dump(rows, open(a.out / "rijen.json", "w"), indent=1)
    print("KLAAR", time.strftime("%F %T"), flush=True)


if __name__ == "__main__":
    main()
