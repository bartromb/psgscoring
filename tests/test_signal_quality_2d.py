"""
Regression test for v0.2.963 SQUEEZE2D fix.

When MNE's raw.get_data(picks=[ch]) returns a 2D array with shape (1, N),
assess_rip_channel() previously produced 2D PSD output from welch(),
breaking the 1D boolean masking downstream. The fix squeezes input to
1D at the top of the function.

Clinical impact: without this fix, signal quality assessment silently
failed on the real deployment pipeline, leaving the Loos case
undetectable via the RIP pair quality gate.
"""
import numpy as np
import pytest

from psgscoring.signal_quality import assess_rip_channel, compare_rip_pair


SF = 32.0  # Hz, typical RIP sample rate


def _synthetic_breathing(duration_s=120, sf=SF, amplitude=1.0):
    """Generate a simple sinusoid at 0.25 Hz (15 breaths/min)."""
    t = np.arange(int(duration_s * sf)) / sf
    return amplitude * np.sin(2 * np.pi * 0.25 * t)


def test_assess_rip_channel_1d_input():
    """Baseline: 1D input (as the library always assumed)."""
    signal = _synthetic_breathing()
    result = assess_rip_channel(signal, SF, label="test")
    assert result["status"] == "ok"
    assert result["mad"] > 0
    assert result["breath_energy"] > 0


def test_assess_rip_channel_2d_input_regression():
    """Regression test: 2D input from MNE raw.get_data(picks=[ch])."""
    # MNE returns shape (n_channels, n_samples) even for single channel
    signal_1d = _synthetic_breathing()
    signal_2d = signal_1d.reshape(1, -1)  # (1, N) as MNE produces
    assert signal_2d.ndim == 2
    assert signal_2d.shape[0] == 1

    result = assess_rip_channel(signal_2d, SF, label="test_2d")
    # Must not crash and must produce valid result
    assert result["status"] in ("ok", "weak", "failed")
    assert isinstance(result["mad"], float)
    assert isinstance(result["breath_energy"], float)
    # Result should match 1D version since squeeze() makes them equivalent
    result_1d = assess_rip_channel(signal_1d, SF, label="test_1d")
    assert result["status"] == result_1d["status"]
    assert abs(result["mad"] - result_1d["mad"]) < 1e-9


def test_assess_rip_channel_higher_dim_rejected():
    """Defensive: 3D input should fail gracefully, not crash."""
    bogus_3d = np.ones((2, 2, 100))  # cannot squeeze to 1D
    result = assess_rip_channel(bogus_3d, SF)
    assert result["status"] == "failed"
    assert "Expected 1D signal" in result["reason"]


def test_compare_rip_pair_with_2d_inputs():
    """End-to-end: compare_rip_pair should work with 2D MNE inputs."""
    thorax_2d = _synthetic_breathing(amplitude=1.0).reshape(1, -1)
    abdomen_2d = _synthetic_breathing(amplitude=1.0).reshape(1, -1)

    result = compare_rip_pair(thorax_2d, abdomen_2d, SF)
    assert result["recommended_mode"] in (
        "bilateral", "single-channel", "unreliable"
    )
    # Both signals equal → should be bilateral
    assert result["recommended_mode"] == "bilateral"


def test_loos_like_case_2d():
    """Loos scenario: thorax essentially flat, abdomen normal, 2D input."""
    thorax_dead_2d = (np.random.RandomState(42).randn(int(120 * SF)) * 1e-4).reshape(1, -1)
    abdomen_ok_2d = _synthetic_breathing(amplitude=1.0).reshape(1, -1)

    result = compare_rip_pair(thorax_dead_2d, abdomen_ok_2d, SF)
    # With thorax dead, mode should be single-channel or unreliable
    assert result["recommended_mode"] in ("single-channel", "unreliable")
    # Energy ratio should be large (abdomen >> thorax)
    assert result.get("energy_ratio", 0) > 10
