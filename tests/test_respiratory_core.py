"""
Numeric unit tests for the core signal primitives (breath.py, signal.py).

These run on every `pytest tests/` (unlike the env-gated golden test) and lock
breath segmentation, per-breath amplitude ratios, and the dynamic baseline on
deterministic synthetic signals. Expected values characterised against v0.7.3.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring.breath import detect_breaths, compute_breath_amplitudes
from psgscoring.signal import compute_dynamic_baseline, linearize_nasal_pressure


# ── Breath segmentation ─────────────────────────────────────────────────────

def test_detect_breaths_on_sinusoid():
    sf = 10.0
    t = np.arange(0, 60, 1 / sf)
    flow = np.sin(2 * np.pi * 0.25 * t)          # 0.25 Hz → ~15 cycles in 60 s
    breaths = detect_breaths(flow, sf)
    assert 13 <= len(breaths) <= 15               # zero-crossing segmentation
    amps = np.array([b["amplitude"] for b in breaths])
    assert amps.mean() == pytest.approx(2.0, abs=0.05)   # peak-to-trough of unit sine
    for b in breaths:
        assert b["end"] > b["start"]
        assert 1.0 <= b["duration_s"] <= 15.0


def test_detect_breaths_short_signal_returns_empty():
    assert detect_breaths(np.zeros(5), 10.0) == []
    assert detect_breaths(np.array([]), 10.0) == []


def test_breath_amplitude_ratios_are_unity_for_constant_amplitude():
    sf = 10.0
    t = np.arange(0, 60, 1 / sf)
    breaths = detect_breaths(np.sin(2 * np.pi * 0.25 * t), sf)
    ratios = compute_breath_amplitudes(breaths, sf)
    assert len(ratios) == len(breaths)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.02)


def test_breath_amplitudes_empty():
    assert compute_breath_amplitudes([], 10.0).size == 0


# ── Dynamic baseline ────────────────────────────────────────────────────────

def test_dynamic_baseline_of_constant_is_constant():
    bl = compute_dynamic_baseline(np.full(2000, 2.0), 100.0)
    assert bl.min() == pytest.approx(2.0, abs=1e-6)
    assert bl.max() == pytest.approx(2.0, abs=1e-6)


def test_dynamic_baseline_floor_is_positive():
    # all-zero envelope must not yield a zero baseline (division-by-zero guard)
    bl = compute_dynamic_baseline(np.zeros(2000), 100.0)
    assert np.all(bl > 0)


# ── Nasal-pressure linearisation (AASM Rule 3) ──────────────────────────────

def test_linearize_nasal_pressure_is_sign_preserving_sqrt():
    x = np.array([-4.0, -1.0, 0.0, 1.0, 9.0])
    out = linearize_nasal_pressure(x)
    np.testing.assert_allclose(out, [-2.0, -1.0, 0.0, 1.0, 3.0])
