"""
Numeric unit tests for psgscoring.spo2.

Unlike the golden test (env-gated behind PSGSCORING_GOLDEN, exercised only in a
dedicated CI job), these run on every `pytest tests/` and lock the SpO2 metrics
— ODI, T90, and hypoxic burden — on small, fully-deterministic synthetic inputs.
The expected values were characterised against the v0.7.3 (validated) code.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring.spo2 import analyze_spo2, compute_hypoxic_burden


def _sleep_hypno(n_samples: int, sf: float) -> list:
    """All-N2 hypnogram spanning n_samples at rate sf."""
    n_epochs = max(1, int(n_samples / sf / 30))
    return ["N2"] * n_epochs


def _spo2_with_dips(baseline=97.0, dip=91.0, onsets=(100, 300, 500),
                    dip_len=15, n=600):
    spo2 = np.full(n, baseline)
    for o in onsets:
        spo2[o:o + dip_len] = dip
    return spo2


# ── ODI ────────────────────────────────────────────────────────────────────

def test_odi_counts_clear_desaturations():
    sf = 1.0
    spo2 = _spo2_with_dips()               # three clean 6% dips
    res = analyze_spo2(spo2, sf, _sleep_hypno(len(spo2), sf))
    assert res["success"] is True
    s = res["summary"]
    assert s["n_desaturations"] == 3
    assert s["baseline_spo2"] == pytest.approx(97.0, abs=0.5)
    # 3 desaturations over 600 s of sleep = 3 / (600/3600) = 18/h
    assert s["odi_3pct"] == pytest.approx(18.0, abs=0.1)
    assert s["odi_4pct"] == pytest.approx(18.0, abs=0.1)  # 6% dip qualifies for both


def test_flat_spo2_has_no_desaturations():
    sf = 1.0
    spo2 = np.full(600, 97.0)
    s = analyze_spo2(spo2, sf, _sleep_hypno(len(spo2), sf))["summary"]
    assert s["n_desaturations"] == 0
    assert s["odi_3pct"] == 0.0
    assert s["odi_4pct"] == 0.0
    assert s["time_below_90"] == "00:00:00"


def test_analyze_spo2_empty_is_graceful():
    res = analyze_spo2(np.array([]), 1.0, [])
    assert res["success"] is False


# ── Hypoxic burden ──────────────────────────────────────────────────────────

def test_hypoxic_burden_positive_on_desaturations():
    sf = 1.0
    spo2 = _spo2_with_dips()
    events = [{"onset_s": float(o), "duration_s": 15.0, "desaturation_pct": 6.0}
              for o in (100, 300, 500)]
    hb = compute_hypoxic_burden(spo2, sf, events, _sleep_hypno(len(spo2), sf))
    assert hb["unit"] == "%·min/h"
    assert hb["n_events_with_burden"] == 3
    # area of three ~6%×15s dips normalised per hour of sleep — locked value
    assert hb["hypoxic_burden"] == pytest.approx(26.1, abs=1.0)


def test_hypoxic_burden_zero_events():
    hb = compute_hypoxic_burden(np.full(600, 97.0), 1.0, [], ["N2"] * 20)
    assert hb["hypoxic_burden"] is None
    assert hb["n_events_with_burden"] == 0
