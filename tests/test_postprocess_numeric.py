"""
Numeric unit tests for psgscoring.postprocess.

Locks the central-instability index and mixed-apnea decomposition on
deterministic inputs. Runs on every `pytest tests/`.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring.postprocess import (
    compute_central_instability_index,
    decompose_mixed_apneas,
)


# ── Central instability index ───────────────────────────────────────────────

def test_cii_high_when_oahi_spreads_across_profiles():
    cii = compute_central_instability_index(
        ahi_strict=10.0, ahi_standard=12.0, ahi_sensitive=15.0,
        oahi_strict=5.0, oahi_standard=6.0, oahi_sensitive=8.0,
    )
    assert cii["central_instability_index"] == pytest.approx(0.474, abs=0.01)
    assert cii["ahi_range"] == pytest.approx(3.0, abs=1e-6)


def test_cii_zero_when_profiles_agree():
    cii = compute_central_instability_index(
        ahi_strict=10.0, ahi_standard=10.0, ahi_sensitive=10.0,
        oahi_strict=5.0, oahi_standard=5.0, oahi_sensitive=5.0,
    )
    assert cii["central_instability_index"] == pytest.approx(0.0, abs=1e-6)


def test_cii_handles_missing_values():
    # None inputs must not raise
    out = compute_central_instability_index(None, None, None)
    assert isinstance(out, dict)
    assert "central_instability_index" in out


# ── Mixed-apnea decomposition ───────────────────────────────────────────────

def test_decompose_mixed_no_mixed_events_is_noop():
    events = [{"type": "obstructive", "onset_s": 10.0, "duration_s": 20.0},
              {"type": "central", "onset_s": 60.0, "duration_s": 15.0}]
    sf = 10.0
    thorax = np.random.RandomState(0).randn(2000)
    out = decompose_mixed_apneas(events, thorax, thorax.copy(), sf)
    # non-mixed events are returned unchanged in count
    assert len(out) == len(events)
    assert [e["type"] for e in out] == ["obstructive", "central"]


def test_decompose_mixed_missing_effort_is_graceful():
    events = [{"type": "mixed", "onset_s": 10.0, "duration_s": 30.0}]
    out = decompose_mixed_apneas(events, None, None, 10.0)
    assert len(out) == 1   # cannot decompose without effort → returned as-is
