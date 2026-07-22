"""
Regression guards for the graceful-degradation fixes (v0.7.4).

These pin the crash-safety contracts so a future refactor can't silently
re-introduce a hard crash from a single bad channel / low-rate ECG / malformed
candidate. All run on every `pytest tests/`.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from psgscoring.ecg_effort import detect_r_peaks, compute_tecg
from psgscoring.ml_classifier import apply_ml_reclassification
from psgscoring.pipeline import _run_step

_MODEL = os.path.join(os.path.dirname(__file__), "..", "psgscoring",
                      "data", "lightgbm_v06_q7holdout.txt")


# ── ecg_effort: low sample rate must not raise ──────────────────────────────

def test_detect_r_peaks_low_samplerate_returns_empty():
    # sf <= 60 → 5–30 Hz bandpass has cutoff at/above Nyquist; must degrade
    peaks = detect_r_peaks(np.random.RandomState(0).randn(5000), 50.0)
    assert peaks.size == 0


def test_detect_r_peaks_short_signal_returns_empty():
    assert detect_r_peaks(np.zeros(100), 200.0).size == 0


def test_detect_r_peaks_finds_spikes_at_valid_rate():
    sf = 200.0
    x = np.zeros(int(sf * 20))
    for i in range(1, 20):
        x[int(i * sf)] = 10.0
    peaks = detect_r_peaks(x, sf)
    assert 15 <= peaks.size <= 20      # ~one per second


def test_compute_tecg_low_samplerate_does_not_crash():
    tecg = compute_tecg(np.random.RandomState(1).randn(3000), 50.0)
    assert len(tecg) == 3000           # returns a signal, no exception


# ── ml_classifier: any error keeps the rule-based result ────────────────────

def test_ml_reclass_missing_key_keeps_rulebased():
    if not os.path.exists(_MODEL):
        pytest.skip("shipped LightGBM model not present")
    accepted = [{"onset_s": 10.0, "type": "apnea", "conf": 0.9}]
    rejected = [{"type": "hypopnea", "conf": 0.4}]            # no 'onset_s'
    a, r, meta = apply_ml_reclassification(
        accepted, rejected, [], ["N2"] * 40, 1200.0, 0.33, 3, 95.0, 0, _MODEL, 0.65)
    assert meta["status"].startswith("error")
    assert a == accepted and r == rejected                   # untouched


def test_ml_reclass_load_error_keeps_rulebased():
    accepted = [{"onset_s": 10.0, "type": "apnea", "conf": 0.9}]
    a, r, meta = apply_ml_reclassification(
        accepted, [], [], ["N2"] * 40, 1200.0, 0.33, 3, 95.0, 0,
        "/no/such/model.txt", 0.65)
    assert meta["status"].startswith("load_error")
    assert a == accepted


# ── pipeline._run_step helper ───────────────────────────────────────────────

def test_run_step_passes_through_on_success():
    sentinel = {"success": True, "summary": {"x": 1}}
    assert _run_step("demo", lambda: sentinel) is sentinel


def test_run_step_degrades_on_exception():
    def boom():
        raise ValueError("bad channel")
    out = _run_step("demo", boom)
    assert out["success"] is False
    assert "demo_failed" in out["error"]
    assert out["summary"] == {}
