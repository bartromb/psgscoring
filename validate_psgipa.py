#!/usr/bin/env python3
"""
validate_psgipa.py — PSG-IPA validation harness for psgscoring.

Reproduces paper v31 (Rombaut et al. 2026) on the PSG-IPA multi-scorer
reference subset (SN1-SN5, 12 scorers each), per the supplement S3.2
methodology: scorer-1 hypnogram and scorer-1 respiratory events are
read from the same {SN}_Respiration_manual_scorer1.edf file in the
Resp_events/ subtree (single time-axis, no meas_date alignment needed).

Runs all three clinical profiles (aasm_v3_strict / aasm_v3_rec /
aasm_v3_sensitive) on each recording in parallel, computes the
robustness-grade (A/B/C) per recording, aggregate Bland-Altman and
severity-concordance metrics on the standard (aasm_v3_rec) profile,
and SN3 event-level F1 / mean Δt against each scorer's event set.

Outputs:
  - human-readable summary table to stdout
  - JSON to /tmp/validation_results.json (consumed by validation_report.py)

Usage:
    python validate_psgipa.py --data-dir ~/PSG-IPA --workers 5
    python validate_psgipa.py --data-dir ~/PSG-IPA --output-json results.json

Requires: psgscoring, mne, numpy, scikit-learn.

Reference values (paper v31, scorer-median per recording):
    SN1: 5.96 [4.66-6.56]    Mild      12 scorers
    SN2: 4.33 [1.65-6.80]    Normal    12 scorers
    SN3: 53.98 [45.06-55.96] Severe    12 scorers
    SN4: 3.82 [0.17-6.32]    Normal    12 scorers
    SN5: 9.98 [3.56-14.39]   Mild      12 scorers
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median, stdev

import mne
import numpy as np

mne.set_log_level("ERROR")

CLINICAL_PROFILES = {
    "strict":    "aasm_v3_strict",
    "standard":  "aasm_v3_rec",
    "sensitive": "aasm_v3_sensitive",
}

STAGE_KEYS = [
    ("stage n3", "N3"), ("stage 3", "N3"), ("stage 4", "N3"),
    ("stage n2", "N2"), ("stage 2", "N2"),
    ("stage n1", "N1"), ("stage 1", "N1"),
    ("stage rem", "R"), ("stage r", "R"),
    ("stage w", "W"), ("wake", "W"),
]


def classify_stage(desc):
    d = desc.lower().strip()
    if "lights" in d or "stage ?" in d:
        return None
    for k, v in STAGE_KEYS:
        if k in d:
            return v
    return None


def is_resp_event(desc):
    d = desc.lower()
    if "arousal" in d:
        return False
    return any(k in d for k in [
        "obstructive apnea", "central apnea", "mixed apnea",
        "hypopnea", "hypopnoea", "apnea", "apnoea",
    ])


def event_type(desc):
    d = desc.lower()
    if "arousal" in d:
        return None
    if "obstructive" in d and ("apnea" in d or "apnoea" in d):
        return "obstructive"
    if "central" in d and ("apnea" in d or "apnoea" in d):
        return "central"
    if "mixed" in d and ("apnea" in d or "apnoea" in d):
        return "mixed"
    if "hypopnea" in d or "hypopnoea" in d:
        return "hypopnea"
    if "apnea" in d or "apnoea" in d:
        return "apnea_unspec"
    return None


def severity(ahi):
    if ahi is None or (isinstance(ahi, float) and np.isnan(ahi)):
        return "?"
    if ahi < 5:
        return "Normal"
    if ahi < 15:
        return "Mild"
    if ahi < 30:
        return "Moderate"
    return "Severe"


def grade_from_severities(strict_sev, std_sev, sens_sev):
    s = {strict_sev, std_sev, sens_sev}
    if len(s) == 1:
        return "A"
    if len(s) == 2:
        return "B"
    return "C"


def parse_scorer_file(scorer_edf, signal_duration_s):
    try:
        ann = mne.read_annotations(str(scorer_edf))
    except Exception:
        return None, None, None
    n_sleep_epochs = 0
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if onset < 0 or onset >= signal_duration_s:
            continue
        st = classify_stage(str(desc))
        if st in ("N1", "N2", "N3", "R"):
            n_eps = max(1, int(round(float(dur) / 30.0)))
            n_sleep_epochs += n_eps
    tst_h = n_sleep_epochs * 30.0 / 3600.0
    if tst_h < 0.1:
        return None, None, None
    n_events = sum(
        1 for onset, desc in zip(ann.onset, ann.description)
        if 0 <= onset < signal_duration_s and is_resp_event(str(desc))
    )
    ahi = n_events / tst_h
    n_epochs = int(np.ceil(signal_duration_s / 30.0))
    hypno = ["W"] * n_epochs
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if onset < 0 or onset >= signal_duration_s:
            continue
        st = classify_stage(str(desc))
        if st is None:
            continue
        ep_start = int(float(onset) // 30)
        n_eps = max(1, int(round(float(dur) / 30.0)))
        for i in range(n_eps):
            if 0 <= ep_start + i < n_epochs:
                hypno[ep_start + i] = st
    return ahi, tst_h, hypno


def find_scorer_files(data_dir, sn_id):
    d = Path(data_dir) / "Resp_events" / "Annotations" / "manual"
    return sorted(d.glob(f"{sn_id}_Respiration_manual_scorer*.edf"))


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def event_set(scorer_edf, signal_duration_s):
    try:
        ann = mne.read_annotations(str(scorer_edf))
    except Exception:
        return []
    out = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if onset < 0 or onset >= signal_duration_s:
            continue
        t = event_type(str(desc))
        if t is None:
            continue
        out.append((float(onset), float(onset) + float(dur), t))
    return out


def analyse_one(sn_id, data_dir):
    data_dir = Path(data_dir)
    psg_path = data_dir / "Resp_events" / "PSG" / f"{sn_id}_Respiration.edf"
    if not psg_path.exists():
        return {"recording": sn_id, "error": "PSG not found"}
    try:
        import psgscoring
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
        sig_dur_s = float(raw.times[-1])
        scorer_files = find_scorer_files(data_dir, sn_id)
        if not scorer_files:
            return {"recording": sn_id, "error": "No scorer files"}

        ref_ahis = []
        scorer1_hypno = None
        for i, f in enumerate(scorer_files):
            ahi, _tst, hypno = parse_scorer_file(f, sig_dur_s)
            if ahi is not None:
                ref_ahis.append(ahi)
                if i == 0:
                    scorer1_hypno = hypno
        if not ref_ahis or scorer1_hypno is None:
            return {"recording": sn_id, "error": "No usable scorer data"}

        ref_med = float(median(ref_ahis))
        ref_lo  = float(min(ref_ahis))
        ref_hi  = float(max(ref_ahis))

        algo_ahis = {}
        algo_events_std = None
        for prof_short, prof_canonical in CLINICAL_PROFILES.items():
            results = psgscoring.run_pneumo_analysis(
                raw, hypno=scorer1_hypno, scoring_profile=prof_canonical,
            )
            rsum = results.get("respiratory", {}).get("summary", {})
            ahi = rsum.get("ahi_total", rsum.get("ahi"))
            if ahi is None or (isinstance(ahi, float) and np.isnan(ahi)):
                return {"recording": sn_id, "error": f"No AHI for profile {prof_canonical}"}
            algo_ahis[prof_short] = float(ahi)
            if prof_short == "standard":
                algo_events_std = results.get("respiratory", {}).get("events", [])

        sev_strict = severity(algo_ahis["strict"])
        sev_std    = severity(algo_ahis["standard"])
        sev_sens   = severity(algo_ahis["sensitive"])

        out = {
            "recording": sn_id,
            "psgscoring_version": psgscoring.__version__,
            "signal_hours": round(sig_dur_s / 3600.0, 2),
            "n_scorers": len(ref_ahis),
            "ref_median": round(ref_med, 2),
            "ref_range":  [round(ref_lo, 2), round(ref_hi, 2)],
            "algo_strict":    round(algo_ahis["strict"], 2),
            "algo_standard":  round(algo_ahis["standard"], 2),
            "algo_sensitive": round(algo_ahis["sensitive"], 2),
            "delta_ahi_standard": round(algo_ahis["standard"] - ref_med, 2),
            "severity_ref":      severity(ref_med),
            "severity_strict":   sev_strict,
            "severity_standard": sev_std,
            "severity_sensitive": sev_sens,
            "severity_match_standard": severity(ref_med) == sev_std,
            "robustness_grade": grade_from_severities(sev_strict, sev_std, sev_sens),
        }

        if sn_id == "SN3" and algo_events_std:
            algo_events = [
                (float(e["onset_s"]), float(e["onset_s"]) + float(e["duration_s"]), e["type"])
                for e in algo_events_std
                if e.get("onset_s") is not None
                and e.get("duration_s") is not None
                and float(e["onset_s"]) < sig_dur_s
            ]
            f1_list, dt_list, tp_list, fp_list, fn_list = [], [], [], [], []
            for f in scorer_files:
                ref_events = event_set(f, sig_dur_s)
                if not ref_events:
                    continue
                matched_a, matched_r, onset_diffs = set(), set(), []
                for i, (a0, a1, _) in enumerate(algo_events):
                    best_j, best_v = -1, 0.0
                    for j, (r0, r1, _) in enumerate(ref_events):
                        if j in matched_r:
                            continue
                        v = iou(a0, a1, r0, r1)
                        if v >= 0.20 and v > best_v:
                            best_v, best_j = v, j
                    if best_j >= 0:
                        matched_a.add(i)
                        matched_r.add(best_j)
                        onset_diffs.append(abs(algo_events[i][0] - ref_events[best_j][0]))
                tp = len(matched_a)
                fp = len(algo_events) - tp
                fn = len(ref_events) - len(matched_r)
                prec = tp / (tp + fp) if (tp + fp) else 0
                rec_ = tp / (tp + fn) if (tp + fn) else 0
                f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0
                f1_list.append(f1)
                tp_list.append(tp); fp_list.append(fp); fn_list.append(fn)
                if onset_diffs:
                    dt_list.append(float(np.mean(onset_diffs)))
            if f1_list:
                out["sn3_f1_median"]   = round(float(median(f1_list)), 3)
                out["sn3_dt_median_s"] = round(float(median(dt_list)), 2) if dt_list else None
                out["sn3_tp_median"]   = int(median(tp_list))
                out["sn3_fp_median"]   = int(median(fp_list))
                out["sn3_fn_median"]   = int(median(fn_list))
        return out
    except Exception as e:
        import traceback
        return {
            "recording": sn_id,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def aggregate_metrics(results):
    refs   = [r["ref_median"]    for r in results if "error" not in r]
    algos  = [r["algo_standard"] for r in results if "error" not in r]
    deltas = [r["delta_ahi_standard"] for r in results if "error" not in r]
    if len(deltas) < 2:
        return {}
    bias = float(np.mean(deltas))
    mae  = float(np.mean(np.abs(deltas)))
    sd   = float(stdev(deltas))
    pearson_r = float(np.corrcoef(refs, algos)[0, 1])
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = float(cohen_kappa_score(
            [severity(x) for x in refs],
            [severity(x) for x in algos],
            weights="quadratic",
        ))
    except ImportError:
        kappa = float("nan")
    return {
        "n_recordings":  len(deltas),
        "bias":          round(bias, 3),
        "mae":           round(mae, 3),
        "sd":            round(sd, 3),
        "loa_low":       round(bias - 1.96 * sd, 3),
        "loa_high":      round(bias + 1.96 * sd, 3),
        "pearson_r":     round(pearson_r, 4),
        "weighted_kappa": round(kappa, 3),
    }


def to_report_json(results, agg):
    sn3 = next((r for r in results if r.get("recording") == "SN3"), None)
    payload = {
        "per_recording": {
            r["recording"]: {
                "ref_median":     r["ref_median"],
                "ref_range":      r["ref_range"],
                "algo_strict":    r["algo_strict"],
                "algo_standard":  r["algo_standard"],
                "algo_sensitive": r["algo_sensitive"],
                "robustness_grade": r["robustness_grade"],
            }
            for r in results
            if "error" not in r
        },
        "aggregate": {
            "n_recordings":  agg.get("n_recordings"),
            "bias":          agg.get("bias"),
            "mae":           agg.get("mae"),
            "sd":            agg.get("sd"),
            "loa_low":       agg.get("loa_low"),
            "loa_high":      agg.get("loa_high"),
            "pearson_r":     agg.get("pearson_r"),
            "weighted_kappa": agg.get("weighted_kappa"),
        },
    }
    if sn3 and "sn3_f1_median" in sn3:
        payload["sn3_event_level"] = {
            "f1":   sn3["sn3_f1_median"],
            "dt_s": sn3.get("sn3_dt_median_s"),
            "tp":   sn3.get("sn3_tp_median"),
            "fp":   sn3.get("sn3_fp_median"),
            "fn":   sn3.get("sn3_fn_median"),
        }
    return payload


def print_summary(results, agg):
    print(f"\n{'Rec':4s}  {'sig_h':>5s}  {'n':>2s}  {'Ref':>5s}  {'[range]':>12s}  "
          f"{'Strict':>6s}  {'Std':>5s}  {'Sens':>5s}  {'Grade':>5s}  Sev")
    print("─" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['recording']:4s}  ERROR: {r['error']}")
            continue
        rng = f"[{r['ref_range'][0]:.1f}-{r['ref_range'][1]:.1f}]"
        print(f"{r['recording']:4s}  {r['signal_hours']:>4.2f}h  {r['n_scorers']:>2d}  "
              f"{r['ref_median']:>4.1f}  {rng:>12s}  "
              f"{r['algo_strict']:>5.1f}  {r['algo_standard']:>4.1f}  {r['algo_sensitive']:>4.1f}  "
              f"{r['robustness_grade']:>5s}  {r['severity_ref'][:1]}/{r['severity_standard'][:1]}"
              f" {'✓' if r['severity_match_standard'] else '✗'}")
        if "sn3_f1_median" in r:
            print(f"        SN3 events: F1={r['sn3_f1_median']:.3f}  "
                  f"Δt={r['sn3_dt_median_s']:.2f}s  "
                  f"TP/FP/FN={r['sn3_tp_median']}/{r['sn3_fp_median']}/{r['sn3_fn_median']}")
    if agg:
        print()
        print("─── Aggregate (standard profile) ─────────────────")
        print(f"  Bias:       {agg['bias']:+.2f} /h")
        print(f"  MAE:        {agg['mae']:.2f} /h")
        print(f"  SD:         {agg['sd']:.2f} /h")
        print(f"  LoA:        [{agg['loa_low']:+.2f}, {agg['loa_high']:+.2f}] /h")
        print(f"  Pearson r:  {agg['pearson_r']:.3f}")
        print(f"  Weighted κ: {agg['weighted_kappa']:.3f}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", required=True,
                   help="PSG-IPA data root (must contain Resp_events/)")
    p.add_argument("--workers", type=int, default=5,
                   help="Parallel workers (default 5)")
    p.add_argument("--recordings", nargs="+",
                   default=["SN1", "SN2", "SN3", "SN4", "SN5"],
                   help="Subset of recordings to validate")
    p.add_argument("--output-json", type=Path,
                   default=Path("/tmp/validation_results.json"),
                   help="Where to write the report JSON")
    args = p.parse_args()

    import psgscoring
    print(f"\npsgscoring v{psgscoring.__version__} — clinical profile sweep "
          f"(strict / standard / sensitive)")
    print(f"Method: paper v31 conventie — stages + events from Resp_events/ "
          f"(single time-axis)\n")

    results = []
    if args.workers > 1 and len(args.recordings) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(analyse_one, sn, args.data_dir): sn
                       for sn in args.recordings}
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for sn in args.recordings:
            results.append(analyse_one(sn, args.data_dir))
    results.sort(key=lambda r: r["recording"])

    agg = aggregate_metrics(results)
    print_summary(results, agg)

    payload = to_report_json(results, agg)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON written: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
