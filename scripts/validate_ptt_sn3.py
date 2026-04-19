#!/usr/bin/env python3
"""
validate_ptt_sn3.py
===================

Sanity-check PTT module on a real PSG-IPA recording (SN3 = severe, overwegend
obstructief, ~54 events/hour). This confirms that real-data PTT amplitudes
reach the expected 0.7-0.9 range for obstructive events before v0.3.001 release.

Expected output if PTT is working on real data:
  - PTT series: 25000+ valid beats over the full night
  - Mean PTT: 180-260 ms (adult range)
  - Obstructive apneas: mean effort_score > 0.6, amplitude > 15 ms
  - Central apneas (30 from post-processing): mean effort_score < 0.3
  - Separation: clear bimodal distribution

Run:
    python validate_ptt_sn3.py /path/to/PSG-IPA/SN3/SN3.edf

Exit code 0 = OK to release v0.3.001
Exit code 1 = unexpected results, investigate before release
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mne

from psgscoring import run_pneumo_analysis
from psgscoring.ptt import (
    compute_ptt_series,
    ptt_effort_score,
)


# --- Channel name candidates for DOMINO / Medatec / Compumedics exports ---
PPG_NAMES = ["Pleth", "PLETH", "PPG", "Plethysmogram", "SpO2-DC", "SPO2 DC",
             "SPO2-DC", "Pleth Wave", "PLETH_WAVE", "Plethysmograph"]
ECG_NAMES = ["ECG", "EKG", "ECG1", "ECG I", "ECG II", "ECG_II",
             "EKG1", "EKG 1", "ECG1-ECG2"]


def find_channel(raw, candidates):
    for c in candidates:
        for ch in raw.ch_names:
            if ch.strip().upper() == c.upper():
                return ch
    # Fallback: substring match
    for c in candidates:
        for ch in raw.ch_names:
            if c.upper() in ch.strip().upper():
                return ch
    return None


def summarize_effort_scores(events, label):
    """Compute summary stats on PTT effort scores for a subset of events."""
    scores = [e.get("ptt_effort_score") for e in events
              if e.get("ptt_effort_score") is not None]
    amps = [e.get("ptt_amplitude_ms") for e in events
            if e.get("ptt_amplitude_ms") is not None
            and not np.isnan(e.get("ptt_amplitude_ms"))]
    if not scores:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": len(scores),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_q25": float(np.percentile(scores, 25)),
        "score_q75": float(np.percentile(scores, 75)),
        "amp_mean_ms": float(np.mean(amps)) if amps else None,
        "amp_median_ms": float(np.median(amps)) if amps else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edf_path", type=str,
                        help="Path to SN3.edf or equivalent severe recording")
    parser.add_argument("--profile", type=str, default="standard",
                        choices=["strict", "standard", "sensitive"])
    parser.add_argument("--output", type=str, default="sn3_ptt_validation.json")
    args = parser.parse_args()

    edf = Path(args.edf_path)
    if not edf.exists():
        print(f"ERROR: EDF not found: {edf}", file=sys.stderr)
        sys.exit(2)

    print(f"=== PTT validation on {edf.name} (profile: {args.profile}) ===\n")

    # --- Step 1: load EDF and locate ECG + PPG channels ---
    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose="ERROR")
    print(f"Available channels: {raw.ch_names}\n")

    ecg_ch = find_channel(raw, ECG_NAMES)
    ppg_ch = find_channel(raw, PPG_NAMES)

    if ecg_ch is None or ppg_ch is None:
        print(f"ERROR: required channels missing (ECG={ecg_ch}, PPG={ppg_ch})")
        print("Check channel naming; add aliases to PPG_NAMES/ECG_NAMES if needed.")
        sys.exit(2)

    print(f"Using ECG channel: '{ecg_ch}'")
    print(f"Using PPG channel: '{ppg_ch}'")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz\n")

    # --- Step 2: compute PTT series ---
    raw.load_data(verbose="ERROR")
    ecg_data = raw.get_data(picks=[ecg_ch])[0]
    ppg_data = raw.get_data(picks=[ppg_ch])[0]
    fs = raw.info["sfreq"]

    print("Computing PTT series (this may take 30-60 seconds)...")
    ptt_series = compute_ptt_series(ecg_data, ppg_data, fs, fs)
    print(f"  Total beats: {ptt_series.n_beats}")
    print(f"  Valid beats: {ptt_series.n_valid} "
          f"({100*ptt_series.n_valid/max(ptt_series.n_beats,1):.1f}%)")
    print(f"  Mean PTT: {ptt_series.mean_ptt_ms:.1f} ms")
    print(f"  Rejected (artifact): {ptt_series.n_rejected_artifact}")

    # Sanity checks on PTT series
    errors = []
    if ptt_series.n_valid < 1000:
        errors.append(f"Too few valid beats ({ptt_series.n_valid}). "
                      f"Expected >10,000 for overnight recording.")
    if ptt_series.mean_ptt_ms < 140 or ptt_series.mean_ptt_ms > 320:
        errors.append(f"Mean PTT {ptt_series.mean_ptt_ms:.1f} ms outside "
                      f"physiological range (140-320 ms). Check signal quality.")

    # --- Step 3: run full analysis with PTT enabled ---
    print(f"\nRunning run_pneumo_analysis (profile: {args.profile})...")
    results = run_pneumo_analysis(
        str(edf),
        profile=args.profile,
        enable_ptt=True,
    )

    events = results.get("apnea_events", [])
    ahi = results.get("ahi", None)
    print(f"  Total events: {len(events)}")
    print(f"  AHI: {ahi:.1f}/h" if ahi else "  AHI: N/A")

    # --- Step 4: compute per-event PTT scores ---
    print("\nComputing per-event PTT effort scores...")
    for e in events:
        result = ptt_effort_score(
            ptt_series,
            e["onset"],
            e["onset"] + e["duration"],
        )
        e["ptt_effort_score"] = result.effort_score
        e["ptt_amplitude_ms"] = result.amplitude_ms
        e["ptt_n_beats"] = result.n_beats_in_event
        e["ptt_reason"] = result.reason

    # --- Step 5: stratify by type and summarise ---
    obstructive = [e for e in events if e.get("type") == "obstructive"]
    central = [e for e in events if e.get("type") == "central"]
    mixed = [e for e in events if e.get("type") == "mixed"]
    hypopnea = [e for e in events if e.get("type") == "hypopnea"]

    obs_summary = summarize_effort_scores(obstructive, "obstructive")
    cen_summary = summarize_effort_scores(central, "central")
    mix_summary = summarize_effort_scores(mixed, "mixed")
    hyp_summary = summarize_effort_scores(hypopnea, "hypopnea")

    print("\n=== Effort score distribution by event type ===")
    for s in [obs_summary, cen_summary, mix_summary, hyp_summary]:
        if s.get("n", 0) == 0:
            print(f"  {s['label']}: no events")
            continue
        print(f"  {s['label']:12s} n={s['n']:4d}  "
              f"median={s['score_median']:.2f}  "
              f"IQR=[{s['score_q25']:.2f}, {s['score_q75']:.2f}]  "
              f"amp_median={s['amp_median_ms']:.1f} ms")

    # --- Step 6: release-gate criteria ---
    print("\n=== Release-gate criteria (expected on SN3) ===")

    criteria = {}

    # Gate 1: obstructive events should have high scores
    if obs_summary.get("n", 0) > 5:
        obs_ok = obs_summary["score_median"] > 0.5
        print(f"  [{'PASS' if obs_ok else 'FAIL'}] Obstructive median score "
              f"{obs_summary['score_median']:.2f} > 0.50")
        criteria["obstructive_score_high"] = obs_ok
    else:
        print("  [N/A]  Too few obstructive events for test")
        criteria["obstructive_score_high"] = None

    # Gate 2: central events should have low scores
    if cen_summary.get("n", 0) > 3:
        cen_ok = cen_summary["score_median"] < 0.4
        print(f"  [{'PASS' if cen_ok else 'FAIL'}] Central median score "
              f"{cen_summary['score_median']:.2f} < 0.40")
        criteria["central_score_low"] = cen_ok
    else:
        print("  [N/A]  Too few central events for test")
        criteria["central_score_low"] = None

    # Gate 3: separation between obstructive and central
    if obs_summary.get("n", 0) > 5 and cen_summary.get("n", 0) > 3:
        sep = obs_summary["score_median"] - cen_summary["score_median"]
        sep_ok = sep > 0.25
        print(f"  [{'PASS' if sep_ok else 'FAIL'}] Separation (obs-cen) "
              f"{sep:.2f} > 0.25")
        criteria["separation"] = sep_ok
    else:
        criteria["separation"] = None

    # Gate 4: amplitude magnitudes physiologically plausible
    if obs_summary.get("amp_median_ms"):
        amp_ok = 5 < obs_summary["amp_median_ms"] < 60
        print(f"  [{'PASS' if amp_ok else 'FAIL'}] Obstructive amplitude "
              f"{obs_summary['amp_median_ms']:.1f} ms in range 5-60 ms")
        criteria["amplitude_range"] = amp_ok
    else:
        criteria["amplitude_range"] = None

    # Gate 5: PTT series quality
    q_ok = ptt_series.n_valid > 1000 and 140 < ptt_series.mean_ptt_ms < 320
    print(f"  [{'PASS' if q_ok else 'FAIL'}] PTT series quality (n_valid "
          f"{ptt_series.n_valid}, mean {ptt_series.mean_ptt_ms:.0f} ms)")
    criteria["series_quality"] = q_ok

    # --- Step 7: write JSON output ---
    output = {
        "edf_path": str(edf),
        "profile": args.profile,
        "ecg_channel": ecg_ch,
        "ppg_channel": ppg_ch,
        "ptt_series": {
            "n_beats": ptt_series.n_beats,
            "n_valid": ptt_series.n_valid,
            "mean_ptt_ms": ptt_series.mean_ptt_ms,
            "n_rejected_artifact": ptt_series.n_rejected_artifact,
        },
        "ahi": ahi,
        "by_type": {
            "obstructive": obs_summary,
            "central": cen_summary,
            "mixed": mix_summary,
            "hypopnea": hyp_summary,
        },
        "release_criteria": criteria,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nOutput written to {args.output}")

    # --- Step 8: exit code ---
    failed = [k for k, v in criteria.items() if v is False]
    if failed:
        print(f"\n*** FAIL: {len(failed)} gate(s) failed: {failed}")
        print("*** Do NOT release v0.3.001 until investigated.")
        sys.exit(1)
    elif errors:
        print(f"\n*** WARNING: {errors}")
        sys.exit(1)
    else:
        print("\n*** PASS: all release gates satisfied. OK to release v0.3.001.")
        sys.exit(0)


if __name__ == "__main__":
    main()
