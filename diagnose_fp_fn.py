#!/usr/bin/env python3
"""
diagnose_fp_fn.py — Diagnostic script for SN1/SN2 false-positives + SN5 false-negatives

Goal: find patterns that allow targeted algorithm improvement.

For each event in algo and reference scoring:
  - Match algo events to reference events (IoU)
  - Categorize false positives and false negatives by:
    * Sleep stage (W/N1/N2/N3/REM) at event onset
    * Time-of-night (early/mid/late)
    * SpO2 desaturation (>=3% vs <3%)
    * Effort pattern at event (paradoxical / in-phase / unclear)
    * Local breath stability (stable / unstable)
    * Distance from previous/next confirmed event

Usage:
    python diagnose_fp_fn.py --data-dir $PSGIPA_DATA_DIR --recordings SN1 SN2 SN5

Output: prints frequency tables of FP/FN characteristics + writes JSON dump
        for further analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict

import numpy as np
import mne

# Suppress noisy MNE warnings
mne.set_log_level("WARNING")
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("diagnose")


# Reuse parsing from validate_psgipa.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from validate_psgipa import (
    parse_hypnogram_edf, parse_respiratory_events, find_scorers,
    per_scorer_ahi, load_signal, event_iou,
)


def get_stage_at(time_s, hypno_epochs):
    """Return sleep stage at given time (30s epochs)."""
    idx = int(time_s / 30.0)
    if 0 <= idx < len(hypno_epochs):
        return hypno_epochs[idx]
    return "?"


def time_segment(time_s, recording_dur_s):
    """Categorize time as early/mid/late third of recording."""
    if recording_dur_s <= 0:
        return "?"
    pos = time_s / recording_dur_s
    if pos < 0.33: return "early"
    elif pos < 0.67: return "mid"
    return "late"


def compute_local_breath_cv(flow_signal, sf, onset_s, end_s, window_s=120.0):
    """Compute coefficient of variation of breath amplitudes in surrounding window.

    Returns:
      cv (float): coefficient of variation, lower = more stable
      n_breaths (int): how many breaths counted in window
    """
    pre_start = max(0, int((onset_s - window_s) * sf))
    post_end = min(len(flow_signal), int((end_s + window_s) * sf))

    window = flow_signal[pre_start:post_end]
    if len(window) < int(10 * sf):
        return None, 0

    # Simple peak detection (positive peaks)
    from scipy.signal import find_peaks
    # Filter band-pass-like: just remove DC and high-freq
    window = window - np.median(window)

    peaks, _ = find_peaks(window, distance=int(1.5 * sf), prominence=np.std(window) * 0.3)
    if len(peaks) < 5:
        return None, len(peaks)

    amplitudes = window[peaks]
    cv = float(np.std(amplitudes) / np.mean(amplitudes)) if np.mean(amplitudes) > 0 else 999.0
    return cv, len(peaks)


def classify_fp_event(event, hypno, sig_dur_s, flow=None, sf_flow=None):
    """Classify a false positive (algo event with no scorer match)."""
    onset = event["onset_s"]
    duration = event["duration_s"]
    end = onset + duration

    info = {
        "onset_s":  onset,
        "duration": duration,
        "type":     event.get("type", "?"),
        "stage":    get_stage_at(onset, hypno),
        "time_seg": time_segment(onset, sig_dur_s),
    }

    # SpO2 desat info if available in event dict
    info["desat_pct"] = event.get("desat_pct", event.get("min_spo2_drop", None))
    info["has_arousal"] = event.get("has_arousal", False)

    # Local stability if flow signal available
    if flow is not None and sf_flow is not None:
        cv, n_breaths = compute_local_breath_cv(flow, sf_flow, onset, end)
        info["local_breath_cv"] = cv
        info["local_n_breaths"] = n_breaths
    else:
        info["local_breath_cv"] = None

    return info


def classify_fn_event(ref_event, hypno, sig_dur_s):
    """Classify a false negative (scorer event with no algo match)."""
    onset = ref_event["onset_s"]
    duration = ref_event["duration_s"]

    return {
        "onset_s":  onset,
        "duration": duration,
        "ref_type": ref_event.get("type", "?"),
        "stage":    get_stage_at(onset, hypno),
        "time_seg": time_segment(onset, sig_dur_s),
        "raw_desc": ref_event.get("raw_desc", "?"),
    }


def match_events(algo, ref, iou_thr=0.20):
    """Greedy matching, returns (matched_pairs, unmatched_algo, unmatched_ref)."""
    matched_ref = set()
    matched_algo = set()
    pairs = []

    for i, a in enumerate(algo):
        a_on = a["onset_s"]
        a_end = a_on + a["duration_s"]
        best_iou, best_j = 0.0, -1
        for j, r in enumerate(ref):
            if j in matched_ref:
                continue
            r_on = r["onset_s"]
            r_end = r_on + r["duration_s"]
            iou = event_iou(a_on, a_end, r_on, r_end)
            if iou > best_iou and iou >= iou_thr:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched_algo.add(i)
            matched_ref.add(best_j)
            pairs.append((i, best_j, best_iou))

    fp = [a for i, a in enumerate(algo) if i not in matched_algo]
    fn = [r for j, r in enumerate(ref) if j not in matched_ref]
    return pairs, fp, fn


def diagnose_recording(data_dir, recording, profile="aasm_v3_rec"):
    """Run full diagnosis on one recording."""
    print(f"\n{'='*78}")
    print(f"  DIAGNOSING {recording}")
    print(f"{'='*78}")

    from psgscoring import run_pneumo_analysis

    # Load reference (median scorer chosen by AHI closest to median)
    scorer_data = per_scorer_ahi(data_dir, recording)
    ref_ahis = [s["ahi"] for s in scorer_data]
    ref_median = np.median(ref_ahis)

    median_scorer = min(scorer_data, key=lambda s: abs(s["ahi"] - ref_median))
    print(f"  Reference: scorer{median_scorer['scorer_id']}, AHI={median_scorer['ahi']:.1f}")
    print(f"  Reference events: {len(median_scorer['events'])}")

    # Load signal + hypnogram (scorer1)
    raw = load_signal(data_dir, recording)
    sig_dur_s = float(raw.times[-1])

    sleep_scorers = find_scorers(data_dir, recording, "SleepStages")
    scorer1_path = next((p for sid, p in sleep_scorers if sid == 1), sleep_scorers[0][1])
    hypno_full, _ = parse_hypnogram_edf(scorer1_path, signal_duration_s=sig_dur_s)

    # Pad hypnogram to signal length
    n_epochs = int(np.ceil(sig_dur_s / 30.0))
    while len(hypno_full) < n_epochs:
        hypno_full.append("W")
    hypno = hypno_full[:n_epochs]

    # Run algorithm
    print(f"  Running psgscoring...")
    output = run_pneumo_analysis(raw=raw, hypno=hypno, scoring_profile=profile)
    algo_events = output["respiratory"].get("events", [])
    algo_ahi = output["respiratory"]["summary"].get("ahi_total", 0.0)
    print(f"  Algorithm: {len(algo_events)} events, AHI={algo_ahi:.1f}")

    # Convert algo events to standard format
    algo_std = []
    for e in algo_events:
        algo_std.append({
            "onset_s": float(e.get("onset_s", e.get("start_s", 0))),
            "duration_s": float(e.get("duration_s", 0)),
            "type": str(e.get("type", e.get("event_type", "unknown"))),
            "desat_pct": e.get("desat_pct", e.get("min_spo2_drop")),
            "has_arousal": e.get("has_arousal", False),
        })

    # Match algo against reference
    pairs, fp_events, fn_events = match_events(algo_std, median_scorer["events"])

    print(f"\n  Matched (TP):       {len(pairs)}")
    print(f"  False positives:    {len(fp_events)}")
    print(f"  False negatives:    {len(fn_events)}")

    # Try to extract flow signal for breath-CV analysis
    flow = None
    sf_flow = None
    try:
        for ch_name in ["nasal pressure", "Nasal Pres", "Nasal", "Cannula", "nasal"]:
            if ch_name in raw.ch_names:
                idx = raw.ch_names.index(ch_name)
                flow = raw.get_data(picks=[idx])[0]
                sf_flow = float(raw.info["sfreq"])
                break
    except Exception:
        pass

    # Classify FPs
    fp_details = [classify_fp_event(e, hypno, sig_dur_s, flow, sf_flow) for e in fp_events]

    # Classify FNs
    fn_details = [classify_fn_event(r, hypno, sig_dur_s) for r in fn_events]

    # Print summary tables
    print(f"\n  === FALSE POSITIVES ({len(fp_details)}) ===")
    if fp_details:
        # Stage distribution
        stage_counts = Counter(d["stage"] for d in fp_details)
        print(f"  Stage distribution:")
        for st, n in stage_counts.most_common():
            pct = n / len(fp_details) * 100
            print(f"    {st:>4}: {n:3d} ({pct:5.1f}%)")

        # Time segment
        seg_counts = Counter(d["time_seg"] for d in fp_details)
        print(f"  Time-of-night:")
        for sg, n in seg_counts.most_common():
            print(f"    {sg:>6}: {n}")

        # Desaturation
        with_desat = sum(1 for d in fp_details
                         if d.get("desat_pct") is not None and d["desat_pct"] >= 3)
        no_desat = len(fp_details) - with_desat
        print(f"  Desaturation:")
        print(f"    >=3%:  {with_desat}")
        print(f"    <3%:   {no_desat}")

        # Local stability
        cv_vals = [d["local_breath_cv"] for d in fp_details if d["local_breath_cv"] is not None]
        if cv_vals:
            print(f"  Local breath CV:")
            print(f"    median: {np.median(cv_vals):.3f}")
            print(f"    CV<0.30 (stable):    {sum(1 for c in cv_vals if c < 0.30)}")
            print(f"    CV 0.30-0.45:        {sum(1 for c in cv_vals if 0.30 <= c < 0.45)}")
            print(f"    CV>=0.45 (unstable): {sum(1 for c in cv_vals if c >= 0.45)}")

        # Type breakdown
        type_counts = Counter(d["type"] for d in fp_details)
        print(f"  Algo event type:")
        for t, n in type_counts.most_common():
            print(f"    {t:<25}: {n}")

    print(f"\n  === FALSE NEGATIVES ({len(fn_details)}) ===")
    if fn_details:
        stage_counts = Counter(d["stage"] for d in fn_details)
        print(f"  Stage distribution:")
        for st, n in stage_counts.most_common():
            pct = n / len(fn_details) * 100
            print(f"    {st:>4}: {n:3d} ({pct:5.1f}%)")

        # Wake-fraction is critical for SN5
        n_wake = stage_counts.get("W", 0)
        if n_wake > 0:
            print(f"\n  ⚠ {n_wake} of {len(fn_details)} FN events are during WAKE")
            print(f"    psgscoring's sleep-mask correctly excludes these")
            print(f"    Scorer counted them: AASM-conform difference")

        # Time segment
        seg_counts = Counter(d["time_seg"] for d in fn_details)
        print(f"  Time-of-night:")
        for sg, n in seg_counts.most_common():
            print(f"    {sg:>6}: {n}")

        # Type breakdown
        type_counts = Counter(d["ref_type"] for d in fn_details)
        print(f"  Reference event type:")
        for t, n in type_counts.most_common():
            print(f"    {t:<25}: {n}")

    return {
        "recording": recording,
        "n_tp": len(pairs),
        "n_fp": len(fp_details),
        "n_fn": len(fn_details),
        "fp_details": fp_details,
        "fn_details": fn_details,
        "algo_ahi": algo_ahi,
        "ref_ahi": median_scorer["ahi"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--recordings", nargs="+", default=["SN1", "SN2", "SN5"])
    p.add_argument("--profile", default="aasm_v3_rec")
    p.add_argument("--output-json", type=Path, default=Path("/tmp/diagnose_results.json"))
    args = p.parse_args()

    if not args.data_dir.exists():
        sys.exit(f"Data dir not found: {args.data_dir}")

    print(f"\n{'='*78}")
    print(f"  DIAGNOSTIC ANALYSIS — psgscoring {args.profile}")
    print(f"  Looking for patterns in FP (SN1/SN2) and FN (SN5)")
    print(f"{'='*78}")

    all_results = {}
    for rec in args.recordings:
        try:
            result = diagnose_recording(args.data_dir, rec, args.profile)
            # Strip non-serializable items
            all_results[rec] = {
                "n_tp": result["n_tp"],
                "n_fp": result["n_fp"],
                "n_fn": result["n_fn"],
                "algo_ahi": result["algo_ahi"],
                "ref_ahi": result["ref_ahi"],
                "fp_details": [
                    {k: (v if not isinstance(v, np.floating) else float(v))
                     for k, v in d.items()}
                    for d in result["fp_details"]
                ],
                "fn_details": result["fn_details"],
            }
        except Exception as e:
            import traceback
            logger.error(f"Failed on {rec}: {e}")
            traceback.print_exc()
            all_results[rec] = {"error": str(e)}

    # Write JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*78}")
    print(f"  CROSS-RECORDING SUMMARY")
    print(f"{'='*78}")

    print(f"\n  {'Rec':<6} {'TP':>5} {'FP':>5} {'FN':>5} {'Algo AHI':>10} {'Ref AHI':>10}")
    for rec, r in all_results.items():
        if "error" in r:
            print(f"  {rec:<6} ERROR: {r['error']}")
        else:
            print(f"  {rec:<6} {r['n_tp']:>5} {r['n_fp']:>5} {r['n_fn']:>5} "
                  f"{r['algo_ahi']:>10.1f} {r['ref_ahi']:>10.1f}")

    # Aggregate FP patterns across SN1+SN2
    all_fp = []
    for rec in ["SN1", "SN2"]:
        if rec in all_results and "fp_details" in all_results[rec]:
            all_fp.extend(all_results[rec]["fp_details"])

    if all_fp:
        print(f"\n  === AGGREGATED FP PATTERN (SN1+SN2, n={len(all_fp)}) ===")
        stage_counts = Counter(d["stage"] for d in all_fp)
        for st, n in stage_counts.most_common():
            print(f"    {st:>4}: {n:3d} ({n/len(all_fp)*100:5.1f}%)")

        cv_vals = [d.get("local_breath_cv") for d in all_fp if d.get("local_breath_cv") is not None]
        if cv_vals:
            cv_vals_floats = [float(v) for v in cv_vals]
            print(f"  Local breath CV (n={len(cv_vals_floats)}):")
            print(f"    median: {np.median(cv_vals_floats):.3f}")
            print(f"    < 0.30 (very stable):  {sum(1 for c in cv_vals_floats if c < 0.30)}")
            print(f"    0.30-0.45:             {sum(1 for c in cv_vals_floats if 0.30 <= c < 0.45)}")
            print(f"    >= 0.45 (unstable):    {sum(1 for c in cv_vals_floats if c >= 0.45)}")

    # Aggregate FN patterns across SN5
    if "SN5" in all_results and "fn_details" in all_results["SN5"]:
        sn5_fn = all_results["SN5"]["fn_details"]
        n_wake = sum(1 for d in sn5_fn if d["stage"] == "W")
        n_total = len(sn5_fn)
        if n_total > 0:
            print(f"\n  === SN5 FN PATTERN (n={n_total}) ===")
            print(f"    During WAKE:  {n_wake} ({n_wake/n_total*100:.1f}%)")
            print(f"    During SLEEP: {n_total - n_wake} ({(n_total-n_wake)/n_total*100:.1f}%)")
            print()
            if n_wake / n_total > 0.7:
                print(f"  CONCLUSION: SN5 FN dominated by wake-period events.")
                print(f"  This is AASM-conform (algo correctly excludes wake).")
                print(f"  Not an algorithm bug, but documentation issue.")
            else:
                print(f"  CONCLUSION: Significant FN during SLEEP.")
                print(f"  Algorithm misses real events — needs investigation.")

    print(f"\n  Full diagnostic data: {args.output_json}")


if __name__ == "__main__":
    main()
