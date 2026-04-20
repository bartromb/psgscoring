"""
tests/test_bug2_classifier_quality_gate.py

Regression test suite documenting Bug 2: the apnea-type classifier does
not yet consume the RIP-pair quality gate's ``recommended_mode`` output.

Context
-------
v0.2.962 introduced ``compare_rip_pair()`` which correctly detects
single-sensor failures (e.g. dead thorax RIP) and returns
``recommended_mode="single-channel"`` with the ``working_channel``
identifier.

v0.2.963 fixed the 2D-shape regression that was silently disabling that
gate in the real deployment pipeline.

BUG 2 (still open as of v0.2.963): despite the RIP-pair gate producing
correct output, ``classify_apnea_type()`` does not read it. With a dead
thorax RIP, the classifier sees absent paradoxical movement, falls
through to the 'truly flat → central' rule, and may misclassify events
that the RIP gate has flagged as untrustworthy.

Clinical case: Loos (AZORG, April 2026). Thorax RIP dead (energy ratio
6861×). The gate correctly reports single-channel + working=abdomen and
the dashboard shows 'Abdomen-only' badge. But the event-type breakdown
on the backend still uses bilateral analysis → clinically unreliable
classification.

This test file:
    1. Tests that ``single_channel_fallback_classify`` works correctly in
       isolation (it already does).
    2. Tests that ``classify_apnea_type`` is NOT YET quality-gate aware
       (failing tests that document the bug — they will start passing
       when Bug 2 is fixed in v0.3.001).

When Bug 2 is implemented in v0.3.001, the @pytest.mark.xfail tests here
should become passing tests — an empirical signal that the integration
is complete.

Run:
    pytest tests/test_bug2_classifier_quality_gate.py -v

Expected output on current codebase:
    - 4 tests PASS (fallback classifier works in isolation)
    - 2 tests XFAIL (expected failure, bug not yet fixed)
    - 0 tests FAIL

Once v0.3.001 implements the classifier gate integration, the XFAIL
tests will turn into XPASS (unexpected pass) → change @pytest.mark.xfail
to regular test → they become regression tests.
"""
import os
import sys

import numpy as np
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_TEST_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from psgscoring.classify import classify_apnea_type
from psgscoring.signal_quality import (
    compare_rip_pair,
    single_channel_fallback_classify,
)


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────


def _synth_eupnea(duration_s: float, sf: float, amplitude: float = 1.0) -> np.ndarray:
    """Synthetic normal breathing: 15 breaths/min sinusoid."""
    t = np.arange(int(duration_s * sf)) / sf
    return amplitude * np.sin(2 * np.pi * 0.25 * t)


def _synth_flat(duration_s: float, sf: float, noise: float = 1e-4) -> np.ndarray:
    """Synthetic dead sensor: near-zero flat with tiny noise."""
    return np.random.RandomState(42).randn(int(duration_s * sf)) * noise


def _synth_apnea_abdomen(duration_s: float, sf: float, apnea_start_s: float,
                          apnea_end_s: float) -> np.ndarray:
    """
    Synthetic abdomen signal: normal breathing interrupted by clear
    effort-absent apnea (→ central pattern when viewed single-channel).
    """
    signal = _synth_eupnea(duration_s, sf, amplitude=1.0)
    i0 = int(apnea_start_s * sf)
    i1 = int(apnea_end_s * sf)
    # Suppress to 5% amplitude during the apnea window → looks central
    signal[i0:i1] *= 0.05
    return signal


# ───────────────────────────────────────────────────────────────────────
# Group 1: fallback classifier in isolation (WORKS — passing tests)
# ───────────────────────────────────────────────────────────────────────


class TestFallbackClassifierInIsolation:
    """These tests verify single_channel_fallback_classify() works
    correctly when called directly. This is NOT the bug — this is the
    baseline showing the correct behaviour that Bug 2 fails to invoke.
    """

    def test_fallback_detects_central_on_single_abdomen(self):
        """Abdomen with effort-absent apnea → 'central'."""
        sf = 32.0
        signal = _synth_apnea_abdomen(
            duration_s=600, sf=sf, apnea_start_s=300, apnea_end_s=315
        )
        result = single_channel_fallback_classify(
            apnea_start_s=300, apnea_end_s=315,
            effort_signal=signal, sf=sf,
        )
        assert result == "central", (
            f"Expected 'central' for effort-absent apnea on single-channel "
            f"fallback, got '{result}'"
        )

    def test_fallback_detects_obstructive_on_effort_present(self):
        """Abdomen with continued effort during apnea → 'obstructive'."""
        sf = 32.0
        signal = _synth_eupnea(duration_s=600, sf=sf, amplitude=1.0)
        # No amplitude reduction → effort clearly still present
        result = single_channel_fallback_classify(
            apnea_start_s=300, apnea_end_s=315,
            effort_signal=signal, sf=sf,
        )
        assert result == "obstructive", (
            f"Expected 'obstructive' when effort is preserved, got '{result}'"
        )

    def test_fallback_returns_uncertain_on_too_short_event(self):
        """Events <2s cannot be reliably classified → 'uncertain'."""
        sf = 32.0
        signal = _synth_eupnea(duration_s=600, sf=sf, amplitude=1.0)
        result = single_channel_fallback_classify(
            apnea_start_s=300, apnea_end_s=301,  # 1 second, too short
            effort_signal=signal, sf=sf,
        )
        assert result == "uncertain"

    def test_rip_gate_correctly_identifies_single_channel(self):
        """Sanity check: compare_rip_pair already flags dead thorax
        correctly (this is the v0.2.962 + v0.2.963 fix working)."""
        sf = 32.0
        thorax_dead = _synth_flat(duration_s=600, sf=sf)
        abdomen_ok = _synth_eupnea(duration_s=600, sf=sf, amplitude=1.0)

        gate = compare_rip_pair(thorax_dead, abdomen_ok, sf)

        assert gate["recommended_mode"] in ("single-channel", "unreliable"), (
            f"RIP gate should flag dead thorax, got mode="
            f"{gate['recommended_mode']}"
        )
        # Energy ratio should be astronomical (abdomen >> thorax)
        assert gate.get("energy_ratio", 0) > 100, (
            f"Energy ratio should be >>100 for dead thorax, "
            f"got {gate.get('energy_ratio')}"
        )


# ───────────────────────────────────────────────────────────────────────
# Group 2: the actual Bug 2 — classifier does NOT consume the gate
# ───────────────────────────────────────────────────────────────────────


class TestBug2ClassifierDoesNotConsumeGate:
    """
    These tests document Bug 2: classify_apnea_type() does NOT accept or
    consume the RIP-pair quality gate output.

    Currently marked @pytest.mark.xfail — they fail on the current
    codebase because the integration is not yet implemented.

    When Bug 2 is fixed in v0.3.001:
        1. classify_apnea_type() gains a signal_quality parameter
        2. When recommended_mode=='single-channel', it routes to
           single_channel_fallback_classify() internally
        3. These xfail tests become xpass
        4. Remove @pytest.mark.xfail decorator
        5. Tests become regression protection

    The tests below use the Loos-like scenario: dead thorax, live
    abdomen with an effort-absent apnea. The EXPECTED correct behaviour
    (documented here, not yet achieved) is that the classifier should
    return 'central' because that is what the abdomen shows — it should
    NOT return 'obstructive' (the current 7-rule chain's default when
    effort looks absent but is actually unmeasurable).
    """

    @pytest.mark.xfail(
        reason="Bug 2: classify_apnea_type does not accept signal_quality "
        "parameter. Fix planned for v0.3.001.",
        strict=True,
    )
    def test_classifier_accepts_signal_quality_parameter(self):
        """
        EXPECTED in v0.3.001: classify_apnea_type accepts a
        signal_quality parameter (dict from compare_rip_pair).

        This test uses inspect to check the function signature.
        """
        import inspect
        sig = inspect.signature(classify_apnea_type)
        assert "signal_quality" in sig.parameters, (
            "classify_apnea_type should accept a 'signal_quality' parameter "
            "in v0.3.001. Currently missing — Bug 2 not yet implemented."
        )

    @pytest.mark.xfail(
        reason="Bug 2: classifier does not route to fallback on dead thorax. "
        "Fix planned for v0.3.001.",
        strict=True,
    )
    def test_classifier_uses_fallback_on_dead_thorax(self):
        """
        EXPECTED in v0.3.001: when the RIP-pair gate reports
        recommended_mode=='single-channel' and working_channel=='abdomen',
        classify_apnea_type routes to single_channel_fallback_classify
        using the abdomen signal.

        Loos-like scenario: abdomen shows effort-absent apnea (central
        pattern) while thorax is dead.

        Currently fails because classify_apnea_type does not accept the
        signal_quality parameter at all.
        """
        sf = 32.0
        duration_s = 600
        apnea_start_s = 300
        apnea_end_s = 315

        thorax_dead = _synth_flat(duration_s, sf)
        abdomen_apnea = _synth_apnea_abdomen(
            duration_s, sf, apnea_start_s, apnea_end_s
        )

        gate = compare_rip_pair(thorax_dead, abdomen_apnea, sf)
        assert gate["recommended_mode"] == "single-channel", (
            "Precondition: RIP gate must flag single-channel"
        )

        # Attempted call to hypothetical v0.3.001 API (WILL FAIL NOW):
        event_type, confidence, details = classify_apnea_type(
            onset_idx=int(apnea_start_s * sf),
            end_idx=int(apnea_end_s * sf),
            thorax_env=np.abs(thorax_dead),
            abdomen_env=np.abs(abdomen_apnea),
            thorax_raw=thorax_dead,
            abdomen_raw=abdomen_apnea,
            effort_baseline=1.0,
            sf=sf,
            signal_quality=gate,  # ← v0.3.001 new parameter, not yet supported
        )

        # Expected correct behaviour in v0.3.001:
        assert event_type == "central", (
            f"With dead thorax and effort-absent abdomen, classifier "
            f"should return 'central' via single-channel fallback. "
            f"Got '{event_type}'."
        )
        assert details.get("classification_source") == "single-channel-fallback", (
            "Classification should document it used the fallback path"
        )


# ───────────────────────────────────────────────────────────────────────
# Group 3: current behaviour documentation (passing tests)
# ───────────────────────────────────────────────────────────────────────


class TestCurrentClassifierBehaviour:
    """
    Documents how classify_apnea_type currently behaves in the Bug 2
    scenario. These tests pass because they describe the CURRENT
    (buggy) behaviour, not the desired behaviour.

    If v0.3.001 changes this behaviour, these tests should be updated
    or removed — they are empirical documentation, not regression
    protection.
    """

    def test_current_classifier_does_not_accept_signal_quality_kwarg(self):
        """Document that adding signal_quality as kwarg currently fails."""
        import inspect
        sig = inspect.signature(classify_apnea_type)
        assert "signal_quality" not in sig.parameters, (
            "As of v0.2.963 classify_apnea_type does not have "
            "signal_quality parameter. If this assertion starts failing, "
            "Bug 2 has been implemented and the xfail tests above "
            "should be updated to regular tests."
        )


if __name__ == "__main__":
    # Allow running as a standalone script for quick inspection
    pytest.main([__file__, "-v"])
