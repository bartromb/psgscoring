"""
example_basic.py — Basic usage of psgscoring

Demonstrates the core signal processing algorithms on simulated data.
For real PSG data, replace the simulated signals with MNE channel data.
"""

import numpy as np
import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from psgscoring import (
    linearize_nasal_pressure,
    compute_mmsd,
    preprocess_flow,
    compute_dynamic_baseline,
    bandpass_flow,
    detect_breaths,
    compute_breath_amplitudes,
    compute_flattening_index,
)

# ── Simulate 5 minutes of nasal pressure data ──────────────────
sf = 256.0  # Hz
duration_s = 300  # 5 minutes
t = np.arange(0, duration_s, 1/sf)
n_samples = len(t)

# Normal breathing: ~0.25 Hz (15 breaths/min)
breathing = np.sin(2 * np.pi * 0.25 * t)

# Add noise (realistic)
noise = 0.15 * np.random.randn(n_samples)

# Simulate an apnea at t=60-75s (15s, flow drops to ~5%)
apnea_mask = np.ones(n_samples)
apnea_start = int(60 * sf)
apnea_end = int(75 * sf)
apnea_mask[apnea_start:apnea_end] = 0.05

# Simulate a hypopnea at t=120-135s (15s, flow drops to ~40%)
hypop_start = int(120 * sf)
hypop_end = int(135 * sf)
apnea_mask[hypop_start:hypop_end] = 0.40

# Final signal (squared to simulate nasal pressure physics)
flow_raw = (breathing * apnea_mask + noise) ** 2 * np.sign(breathing * apnea_mask + noise)

print("=" * 60)
print("psgscoring — Demo with simulated data")
print("=" * 60)
print(f"Duration: {duration_s}s, Sample rate: {sf} Hz, Samples: {n_samples}")
print()

# ── A. Square-root linearization ───────────────────────────────
print("A. NASAL PRESSURE LINEARIZATION")
print("   (Thurnheer et al., AJRCCM 2001)")
flow_lin = linearize_nasal_pressure(flow_raw)
print(f"   Raw range:        [{flow_raw.min():.3f}, {flow_raw.max():.3f}]")
print(f"   Linearized range: [{flow_lin.min():.3f}, {flow_lin.max():.3f}]")
print()

# ── B. MMSD computation ───────────────────────────────────────
print("B. MMSD APNEA VALIDATION")
print("   (Lee et al., Physiol Meas 2008)")
flow_filt = bandpass_flow(flow_lin, sf)
mmsd = compute_mmsd(flow_filt, sf)
mmsd_normal = np.mean(mmsd[int(30*sf):int(50*sf)])  # normal segment
mmsd_apnea = np.mean(mmsd[int(63*sf):int(72*sf)])    # during apnea
mmsd_hypop = np.mean(mmsd[int(123*sf):int(132*sf)])   # during hypopnea
print(f"   Normal segment MMSD:   {mmsd_normal:.6f}")
print(f"   Apnea segment MMSD:    {mmsd_apnea:.6f} ({100*mmsd_apnea/mmsd_normal:.1f}% of normal)")
print(f"   Hypopnea segment MMSD: {mmsd_hypop:.6f} ({100*mmsd_hypop/mmsd_normal:.1f}% of normal)")
print(f"   → Apnea confirmed: MMSD < 40% of baseline = {mmsd_apnea/mmsd_normal < 0.40}")
print()

# ── C. Flow envelope + baseline ────────────────────────────────
print("C. FLOW ENVELOPE + DYNAMIC BASELINE")
envelope = preprocess_flow(flow_raw, sf, is_nasal_pressure=True)
baseline = compute_dynamic_baseline(envelope, sf, window_s=60)  # shorter for demo
flow_norm = np.clip(envelope / baseline, 0, 2)

print(f"   Normal flow (t=30-50s):  {np.mean(flow_norm[int(30*sf):int(50*sf)]):.3f}")
print(f"   Apnea flow (t=63-72s):   {np.mean(flow_norm[int(63*sf):int(72*sf)]):.3f}")
print(f"   Hypopnea flow (t=123-132s): {np.mean(flow_norm[int(123*sf):int(132*sf)]):.3f}")
print(f"   → Apnea threshold: <0.10, Hypopnea threshold: 0.10-0.70")
print()

# ── D. Breath-by-breath analysis ──────────────────────────────
print("D. BREATH-BY-BREATH ANALYSIS")
breaths = detect_breaths(flow_filt, sf)
print(f"   Detected {len(breaths)} breaths in {duration_s}s")
if breaths:
    ratios = compute_breath_amplitudes(breaths, sf)
    print(f"   Amplitude ratios: min={ratios.min():.3f}, max={ratios.max():.3f}, mean={ratios.mean():.3f}")

    # Flattening index for first 5 breaths
    print("   First 5 breaths:")
    for i, b in enumerate(breaths[:5]):
        fi = compute_flattening_index(b["insp_segment"])
        print(f"     Breath {i+1}: onset={b['onset_s']:.1f}s, "
              f"amp={b['amplitude']:.3f}, flat_idx={fi:.2f}")
print()

print("=" * 60)
print("All algorithms executed successfully.")
print("For real PSG data, use: run_full_analysis(raw, hypno)")
print("=" * 60)
