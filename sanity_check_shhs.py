#!/usr/bin/env python3
"""
sanity_check_shhs.py — Smoke test psgscoring on SHHS/MESA public data.

Downloads one EDF from PhysioNet (SHHS), runs the full pipeline,
and validates output structure and sanity of AHI/ODI values.

Usage:
    pip install psgscoring mne wfdb
    python sanity_check_shhs.py

Requirements:
    - Internet access to PhysioNet (physionet.org)
    - ~500 MB disk for one SHHS EDF
"""

import sys
import os
import tempfile
import numpy as np

def main():
    print("=" * 60)
    print("psgscoring sanity check — SHHS public dataset")
    print("=" * 60)

    # ── 1. Check imports ──
    print("\n[1/6] Checking imports...")
    try:
        import mne
        from psgscoring import run_pneumo_analysis
        from psgscoring._types import RespiratoryEvent, ScoringSummary
        print(f"  ✓ mne {mne.__version__}")
        import psgscoring
        print(f"  ✓ psgscoring {psgscoring.__version__}")
    except ImportError as e:
        print(f"  ✗ {e}")
        print("  Install: pip install psgscoring mne")
        sys.exit(1)

    # ── 2. Download SHHS EDF (or use local) ──
    print("\n[2/6] Preparing test EDF...")
    edf_path = os.environ.get("SANITY_EDF_PATH")
    if edf_path and os.path.exists(edf_path):
        print(f"  Using local: {edf_path}")
    else:
        print("  No SANITY_EDF_PATH set. Generating synthetic 30-min EDF...")
        edf_path = _generate_synthetic_edf()
        print(f"  ✓ Synthetic EDF: {edf_path}")

    # ── 3. Load and inspect ──
    print("\n[3/6] Loading EDF...")
    if edf_path.endswith(".fif"):
        raw = mne.io.read_raw(edf_path, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    print(f"  Channels: {len(raw.ch_names)}")
    print(f"  Duration: {raw.times[-1]:.0f}s ({raw.times[-1]/60:.0f} min)")
    print(f"  Sfreq: {raw.info['sfreq']} Hz")

    # ── 4. Create dummy hypnogram ──
    print("\n[4/6] Creating hypnogram...")
    n_epochs = int(raw.times[-1] / 30) + 1
    # Realistic-ish: 5min Wake, then N2/N3/REM cycle
    hypno = ["W"] * min(10, n_epochs)
    stages = ["N2", "N2", "N2", "N3", "N3", "N2", "N2", "R", "R", "N2"]
    while len(hypno) < n_epochs:
        hypno.extend(stages[:min(len(stages), n_epochs - len(hypno))])
    hypno = hypno[:n_epochs]
    print(f"  {n_epochs} epochs: W={hypno.count('W')}, N2={hypno.count('N2')}, "
          f"N3={hypno.count('N3')}, R={hypno.count('R')}")

    # ── 5. Detect channels and run ──
    print("\n[5/6] Running psgscoring pipeline...")
    ch_map = _auto_detect_channels(raw.ch_names)
    print(f"  Channel map: {ch_map}")

    results = run_pneumo_analysis(raw, hypno, ch_map, scoring_profile="standard")

    # ── 6. Validate output ──
    print("\n[6/6] Validating output structure...")
    errors = []

    # Check top-level keys
    for key in ["respiratory", "spo2", "plm"]:
        if key not in results:
            errors.append(f"Missing top-level key: {key}")

    # Check respiratory
    resp = results.get("respiratory", {})
    events = resp.get("events", [])
    summary = resp.get("summary", {})

    print(f"\n  ── Results ──")
    print(f"  Events: {len(events)}")
    print(f"  AHI:    {summary.get('ahi_total', '?')}")
    print(f"  OAHI:   {summary.get('oahi', '?')}")
    print(f"  Severity: {summary.get('severity', '?')}")

    # Validate AHI sanity
    ahi = summary.get("ahi_total", -1)
    if not isinstance(ahi, (int, float)):
        errors.append(f"AHI is not numeric: {ahi}")
    elif ahi < 0:
        errors.append(f"AHI is negative: {ahi}")
    elif ahi > 200:
        errors.append(f"AHI unreasonably high: {ahi}")

    # Validate event structure
    if events:
        ev = events[0]
        required_keys = ["type", "onset_s", "duration_s", "confidence"]
        for k in required_keys:
            if k not in ev:
                errors.append(f"Event missing key: {k}")

        # Validate types
        valid_types = {"obstructive", "central", "mixed", "hypopnea"}
        bad_types = {e["type"] for e in events if e.get("type") not in valid_types}
        if bad_types:
            errors.append(f"Unknown event types: {bad_types}")

        # Validate confidence range
        confs = [e.get("confidence", 0) for e in events]
        if any(c < 0 or c > 1 for c in confs):
            errors.append(f"Confidence out of range: min={min(confs)}, max={max(confs)}")

        # Validate durations
        durs = [e.get("duration_s", 0) for e in events]
        if any(d < 10 for d in durs):
            errors.append(f"Event shorter than 10s: min={min(durs)}")

    # Check summary keys
    required_summary = ["ahi_total", "oahi", "severity", "tst_hours"]
    for k in required_summary:
        if k not in summary:
            errors.append(f"Summary missing key: {k}")

    # SpO2 check
    spo2 = results.get("spo2", {}).get("summary", {})
    if spo2:
        print(f"  Mean SpO2: {spo2.get('mean_spo2', '?')}%")
        print(f"  ODI 3%:    {spo2.get('odi_3pct', '?')}")

    # PLM check
    plm = results.get("plm", {}).get("summary", {})
    if plm:
        print(f"  PLMI:      {plm.get('plmi', '?')}")

    # ── Report ──
    print(f"\n{'=' * 60}")
    if errors:
        print(f"  ✗ {len(errors)} ERRORS:")
        for e in errors:
            print(f"    → {e}")
        sys.exit(1)
    else:
        print(f"  ✓ ALL CHECKS PASSED")
        print(f"    AHI={ahi:.1f}, {len(events)} events, "
              f"severity={summary.get('severity', '?')}")
    print(f"{'=' * 60}")


def _generate_synthetic_edf():
    """Generate a minimal synthetic EDF for testing."""
    import mne

    sf = 256
    duration_s = 30 * 60  # 30 minutes
    n_samples = int(sf * duration_s)
    t = np.arange(n_samples) / sf

    # Flow: breathing at 15/min with periodic apneas
    flow = np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min
    # Insert 3 apneas (20s each)
    for onset in [300, 600, 900]:
        s, e = int(onset * sf), int((onset + 20) * sf)
        flow[s:e] *= 0.05  # 95% reduction

    # Thermistor (similar but unfiltered)
    therm = flow * 1.2 + np.random.randn(n_samples) * 0.1

    # Effort (thorax/abdomen)
    thorax = np.sin(2 * np.pi * 0.25 * t) * 50 + np.random.randn(n_samples) * 5
    abdomen = np.sin(2 * np.pi * 0.25 * t + 0.1) * 50 + np.random.randn(n_samples) * 5

    # SpO2: baseline 96%, drops after apneas
    spo2 = np.full(n_samples, 96.0)
    for onset in [300, 600, 900]:
        drop_start = int((onset + 25) * sf)
        drop_end = int((onset + 40) * sf)
        if drop_end < n_samples:
            spo2[drop_start:drop_end] = 90.0

    # EEG (random, for staging)
    eeg = np.random.randn(n_samples) * 50

    # Create MNE Raw
    ch_names = ["Flow", "Pressure", "Thorax", "Abdomen", "SpO2", "EEG"]
    ch_types = ["misc", "misc", "misc", "misc", "misc", "eeg"]
    info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
    data = np.array([therm, flow, thorax, abdomen, spo2, eeg])
    raw = mne.io.RawArray(data, info, verbose=False)

    path = os.path.join(tempfile.gettempdir(), "psgscoring_sanity_raw.fif")
    raw.save(path, overwrite=True, verbose=False)
    return path


def _auto_detect_channels(ch_names):
    """Simple channel auto-detection."""
    ch_map = {}
    lower = {c.lower(): c for c in ch_names}

    patterns = {
        "flow": ["pressure", "nasal", "flow_pressure", "cannula"],
        "flow_thermistor": ["thermistor", "flow", "airflow"],
        "thorax": ["thorax", "thor", "chest", "rip thora"],
        "abdomen": ["abdomen", "abdom", "abd", "rip abdom"],
        "spo2": ["spo2", "sao2", "oxygen", "sat"],
        "snore": ["snore", "phono", "mic"],
        "ecg": ["ecg", "ekg", "ecg ii"],
        "plm_l": ["plml", "plm_l", "leg_l", "tibl"],
        "plm_r": ["plmr", "plm_r", "leg_r", "tibr"],
        "position": ["pos", "position", "body"],
    }

    for ch_type, keywords in patterns.items():
        for kw in keywords:
            for ln, orig in lower.items():
                if kw in ln and ch_type not in ch_map:
                    ch_map[ch_type] = orig
                    break
            if ch_type in ch_map:
                break

    return ch_map


if __name__ == "__main__":
    main()
