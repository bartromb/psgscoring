"""
test_psgscoring.py — Unit tests for core signal processing algorithms.

Run with: pytest tests/ -v
"""

import importlib
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

from psgscoring import (
    linearize_nasal_pressure,
    compute_mmsd,
    preprocess_flow,
    preprocess_effort,
    compute_dynamic_baseline,
    compute_stage_baseline,
    bandpass_flow,
    detect_breaths,
    compute_breath_amplitudes,
    compute_flattening_index,
    detect_breath_events,
    detect_position_changes,
    reset_baseline_at_position_changes,
    build_sleep_mask,
    hypno_to_numeric,
    channel_map_from_user,
    is_nrem, is_rem, is_sleep,
)

# ---------------------------------------------------------------------------
# Import v0.2.0-only functions (compute_anchor_baseline, detect_desaturations,
# get_desaturation) from root-level modules via a package shim.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_root_package(pkg_name: str) -> types.ModuleType:
    """Register root-level modules as a temporary package for relative imports."""
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [_ROOT]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    for mod in ("constants", "utils", "signal", "spo2"):
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{mod}", os.path.join(_ROOT, f"{mod}.py"),
            submodule_search_locations=[_ROOT],
        )
        m = importlib.util.module_from_spec(spec)
        m.__package__ = pkg_name
        sys.modules[f"{pkg_name}.{mod}"] = m
    for mod in ("constants", "utils", "signal", "spo2"):
        sys.modules[f"{pkg_name}.{mod}"].__spec__.loader.exec_module(
            sys.modules[f"{pkg_name}.{mod}"]
        )
    return pkg

_psg = _load_root_package("_psg_root")
compute_anchor_baseline = sys.modules["_psg_root.signal"].compute_anchor_baseline
detect_desaturations    = sys.modules["_psg_root.spo2"].detect_desaturations
get_desaturation        = sys.modules["_psg_root.spo2"].get_desaturation


SF = 256.0  # Standard sample frequency


# ═══════════════════════════════════════════════════════════════
# A. linearize_nasal_pressure
# ═══════════════════════════════════════════════════════════════

class TestLinearize:
    def test_preserves_sign(self):
        x = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
        y = linearize_nasal_pressure(x)
        assert y[0] < 0  # negative stays negative
        assert y[3] > 0  # positive stays positive
        assert y[2] == 0  # zero stays zero

    def test_sqrt_magnitude(self):
        x = np.array([4.0])
        y = linearize_nasal_pressure(x)
        assert abs(y[0] - 2.0) < 1e-10  # √4 = 2

    def test_corrects_pressure_flow_relationship(self):
        """50% flow reduction = 75% pressure reduction.
        After linearization, should show ~50% amplitude reduction."""
        flow_100 = np.array([1.0])  # 100% flow → pressure = 1²  = 1.0
        flow_50 = np.array([0.25])  # 50% flow  → pressure = 0.5² = 0.25

        lin_100 = linearize_nasal_pressure(flow_100)  # √1.0 = 1.0
        lin_50 = linearize_nasal_pressure(flow_50)    # √0.25 = 0.5

        ratio = lin_50[0] / lin_100[0]
        assert abs(ratio - 0.5) < 1e-10, f"Expected 0.5, got {ratio}"

    def test_empty_input(self):
        y = linearize_nasal_pressure(np.array([]))
        assert len(y) == 0

    def test_large_array(self):
        x = np.random.randn(100000)
        y = linearize_nasal_pressure(x)
        assert y.shape == x.shape


# ═══════════════════════════════════════════════════════════════
# B. compute_mmsd
# ═══════════════════════════════════════════════════════════════

class TestMMSD:
    def _make_signal(self, duration_s=60):
        t = np.arange(0, duration_s, 1/SF)
        flow = np.sin(2 * np.pi * 0.25 * t)
        # Apnea at t=20-30s
        flow[int(20*SF):int(30*SF)] *= 0.02
        return flow

    def test_shape(self):
        flow = self._make_signal()
        mmsd = compute_mmsd(flow, SF)
        assert mmsd.shape == flow.shape

    def test_apnea_lower_than_normal(self):
        flow = self._make_signal()
        filt = bandpass_flow(flow, SF)
        mmsd = compute_mmsd(filt, SF)
        normal = np.mean(mmsd[int(5*SF):int(15*SF)])
        apnea = np.mean(mmsd[int(22*SF):int(28*SF)])
        assert apnea < normal * 0.40, \
            f"MMSD during apnea ({apnea:.6f}) should be <40% of normal ({normal:.6f})"

    def test_constant_signal_near_zero(self):
        """A constant signal has zero 2nd derivative → MMSD ≈ 0."""
        flat = np.ones(int(10 * SF))
        mmsd = compute_mmsd(flat, SF)
        assert np.max(mmsd) < 1e-10


# ═══════════════════════════════════════════════════════════════
# C. preprocess_flow
# ═══════════════════════════════════════════════════════════════

class TestPreprocessFlow:
    def test_output_shape(self):
        x = np.random.randn(int(30 * SF))
        env = preprocess_flow(x, SF)
        assert env.shape == x.shape

    def test_envelope_nonnegative(self):
        x = np.random.randn(int(30 * SF))
        env = preprocess_flow(x, SF)
        assert np.all(env >= 0)

    def test_nasal_pressure_flag(self):
        """With is_nasal_pressure=True, output should differ from False."""
        x = np.abs(np.random.randn(int(30 * SF))) + 0.1
        env_raw = preprocess_flow(x, SF, is_nasal_pressure=False)
        env_lin = preprocess_flow(x, SF, is_nasal_pressure=True)
        # They should differ (sqrt changes the signal)
        assert not np.allclose(env_raw, env_lin)


# ═══════════════════════════════════════════════════════════════
# D. compute_dynamic_baseline
# ═══════════════════════════════════════════════════════════════

class TestDynamicBaseline:
    def test_positive(self):
        env = np.abs(np.sin(np.arange(0, 300, 1/SF) * 0.25)) + 0.1
        bl = compute_dynamic_baseline(env, SF)
        assert np.all(bl > 0)

    def test_length(self):
        n = int(60 * SF)
        env = np.random.rand(n) + 0.1
        bl = compute_dynamic_baseline(env, SF)
        assert len(bl) == n


# ═══════════════════════════════════════════════════════════════
# E. detect_breaths
# ═══════════════════════════════════════════════════════════════

class TestDetectBreaths:
    def test_regular_breathing(self):
        """15 breaths/min for 2 min → expect ~28-32 breaths."""
        t = np.arange(0, 120, 1/SF)
        flow = np.sin(2 * np.pi * 0.25 * t)  # 0.25 Hz = 15/min
        filt = bandpass_flow(flow, SF)
        breaths = detect_breaths(filt, SF)
        assert 20 < len(breaths) < 40, f"Expected ~30 breaths, got {len(breaths)}"

    def test_breath_has_required_keys(self):
        t = np.arange(0, 30, 1/SF)
        flow = np.sin(2 * np.pi * 0.25 * t)
        filt = bandpass_flow(flow, SF)
        breaths = detect_breaths(filt, SF)
        if breaths:
            b = breaths[0]
            for key in ["start", "end", "onset_s", "duration_s",
                        "amplitude", "insp_segment"]:
                assert key in b, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════
# F. compute_flattening_index
# ═══════════════════════════════════════════════════════════════

class TestFlatteningIndex:
    def test_sinusoidal_low(self):
        """A sinusoidal inspiratory segment has low flattening."""
        seg = np.sin(np.linspace(0, np.pi, 100))
        fi = compute_flattening_index(seg)
        assert fi <= 0.4, f"Sinusoidal should be low, got {fi}"

    def test_flat_plateau_high(self):
        """A plateau signal has high flattening."""
        seg = np.ones(100)
        fi = compute_flattening_index(seg)
        assert fi > 0.8, f"Plateau should be high, got {fi}"


# ═══════════════════════════════════════════════════════════════
# G. Utility functions
# ═══════════════════════════════════════════════════════════════

class TestUtils:
    def test_is_nrem(self):
        assert is_nrem("N1") and is_nrem("N2") and is_nrem("N3")
        assert is_nrem(1) and is_nrem(2) and is_nrem(3)
        assert not is_nrem("W") and not is_nrem("R")

    def test_is_rem(self):
        assert is_rem("R") and is_rem(4)
        assert not is_rem("N2") and not is_rem("W")

    def test_is_sleep(self):
        assert is_sleep("N1") and is_sleep("R")
        assert not is_sleep("W")

    def test_build_sleep_mask(self):
        hypno = ["W", "N1", "N2", "N3", "R", "W"]
        mask = build_sleep_mask(hypno, SF, int(6 * 30 * SF))
        # First epoch (W) = False, epochs 1-4 (sleep) = True, epoch 5 (W) = False
        assert not mask[0]
        assert mask[int(1.5 * 30 * SF)]  # middle of N1
        assert not mask[int(5.5 * 30 * SF)]  # middle of last W


# ═══════════════════════════════════════════════════════════════
# H. hypno_to_numeric
# ═══════════════════════════════════════════════════════════════

class TestHypnoToNumeric:
    def test_standard_labels(self):
        result = hypno_to_numeric(["W", "N1", "N2", "N3", "R"])
        assert list(result) == [0, 1, 2, 3, 4]

    def test_unknown_label(self):
        result = hypno_to_numeric(["W", "?", "N2"])
        assert result[1] == -1

    def test_empty(self):
        assert list(hypno_to_numeric([])) == []


# ═══════════════════════════════════════════════════════════════
# I. channel_map_from_user
# ═══════════════════════════════════════════════════════════════

class TestChannelMapFromUser:
    _CH = ["Nasal Pressure", "Thorax", "SpO2", "ECG II", "Leg L EMG"]

    def test_auto_detection(self):
        ch_map = channel_map_from_user(None, self._CH)
        assert "flow_pressure" in ch_map
        assert ch_map["flow_pressure"] == "Nasal Pressure"

    def test_manual_override(self):
        ch_map = channel_map_from_user({"thorax": "Thorax"}, self._CH)
        assert ch_map["thorax"] == "Thorax"

    def test_invalid_override_ignored(self):
        ch_map = channel_map_from_user({"flow": "NonExistent"}, self._CH)
        assert ch_map.get("flow") != "NonExistent"


# ═══════════════════════════════════════════════════════════════
# J. preprocess_effort
# ═══════════════════════════════════════════════════════════════

class TestPreprocessEffort:
    def test_output_shape(self):
        x = np.random.randn(int(30 * SF))
        env = preprocess_effort(x, SF)
        assert env.shape == x.shape

    def test_envelope_nonnegative(self):
        x = np.random.randn(int(30 * SF))
        env = preprocess_effort(x, SF)
        assert np.all(env >= 0)

    def test_differs_from_flow_preprocessing(self):
        """Effort uses 0.03–2 Hz vs flow 0.05–3 Hz — outputs must differ."""
        x = np.abs(np.random.randn(int(60 * SF))) + 0.1
        assert not np.allclose(preprocess_flow(x, SF), preprocess_effort(x, SF))


# ═══════════════════════════════════════════════════════════════
# K. compute_stage_baseline
# ═══════════════════════════════════════════════════════════════

class TestComputeStageBaseline:
    def _make_hypno_and_env(self):
        n_epochs = 20
        hypno = ["W"] * 4 + ["N2"] * 8 + ["R"] * 4 + ["N3"] * 4
        n = n_epochs * int(30 * SF)
        env = np.abs(np.sin(np.linspace(0, 10 * np.pi, n))) + 0.5
        return env, hypno, n

    def test_output_length(self):
        env, hypno, n = self._make_hypno_and_env()
        bl = compute_stage_baseline(env, SF, hypno)
        assert len(bl) == n

    def test_positive(self):
        env, hypno, n = self._make_hypno_and_env()
        bl = compute_stage_baseline(env, SF, hypno)
        assert np.all(bl > 0)


# ═══════════════════════════════════════════════════════════════
# L. detect_position_changes
# ═══════════════════════════════════════════════════════════════

class TestDetectPositionChanges:
    def test_no_change_returns_empty(self):
        pos = np.ones(int(5 * 60 * SF))  # 5 min of constant position 1
        changes = detect_position_changes(pos, SF)
        assert changes == []

    def test_detects_single_change(self):
        n_samp = int(10 * 60 * SF)  # 10 min total
        pos = np.ones(n_samp)
        # Switch to position 2 after 5 min, stay there
        pos[int(5 * 60 * SF):] = 2
        changes = detect_position_changes(pos, SF)
        assert len(changes) == 1
        assert changes[0]["from"] == 1
        assert changes[0]["to"] == 2

    def test_change_dict_keys(self):
        n_samp = int(10 * 60 * SF)
        pos = np.ones(n_samp)
        pos[int(5 * 60 * SF):] = 3
        changes = detect_position_changes(pos, SF)
        if changes:
            for key in ("sample", "time_s", "from", "to"):
                assert key in changes[0], f"Missing key: {key}"

    def test_too_short_returns_empty(self):
        pos = np.array([1, 2, 1, 2], dtype=float)
        assert detect_position_changes(pos, SF) == []


# ═══════════════════════════════════════════════════════════════
# M. reset_baseline_at_position_changes
# ═══════════════════════════════════════════════════════════════

class TestResetBaselineAtPositionChanges:
    def test_no_changes_returns_copy(self):
        bl = np.ones(1000)
        env = np.ones(1000) * 0.8
        result = reset_baseline_at_position_changes(bl, env, SF, [])
        np.testing.assert_array_equal(result, bl)

    def test_baseline_updated_after_change(self):
        n = int(5 * 60 * SF)
        bl = np.ones(n)
        # Second half of signal has higher amplitude → new baseline should rise
        env = np.ones(n)
        env[n // 2:] = 2.0
        change = {"sample": n // 2, "time_s": n // 2 / SF}
        result = reset_baseline_at_position_changes(bl, env, SF, [change])
        # Check 30 s into the recalc window (default 60 s) where baseline rises
        check_idx = n // 2 + int(30 * SF)
        assert result[check_idx] > bl[check_idx]


# ═══════════════════════════════════════════════════════════════
# N. detect_breath_events
# ═══════════════════════════════════════════════════════════════

class TestDetectBreathEvents:
    def _make_breaths_and_ratios(self, n=40, apnea_at=15, apnea_len=5):
        """Simulate 40 breaths; breaths 15-19 are apneic (ratio < 0.10)."""
        t = np.arange(0, n * 4, 1 / SF)
        flow = np.sin(2 * np.pi * 0.25 * t)
        filt = bandpass_flow(flow, SF)
        breaths = detect_breaths(filt, SF)
        ratios = np.ones(len(breaths))
        # Mark a stretch of breaths as apneic
        for i in range(min(apnea_at, len(ratios)),
                       min(apnea_at + apnea_len, len(ratios))):
            ratios[i] = 0.05
        hypno = ["N2"] * 200
        return breaths, ratios, hypno

    def test_detects_apnea(self):
        breaths, ratios, hypno = self._make_breaths_and_ratios()
        apneas, hypopneas = detect_breath_events(breaths, ratios, SF, hypno)
        assert len(apneas) >= 1

    def test_apnea_event_keys(self):
        breaths, ratios, hypno = self._make_breaths_and_ratios()
        apneas, _ = detect_breath_events(breaths, ratios, SF, hypno)
        if apneas:
            for key in ("onset_s", "duration_s", "stage", "min_ratio"):
                assert key in apneas[0], f"Missing key: {key}"

    def test_empty_breaths(self):
        apneas, hypopneas = detect_breath_events([], np.array([]), SF, ["N2"])
        assert apneas == [] and hypopneas == []


# ═══════════════════════════════════════════════════════════════
# O. detect_desaturations  (root-level v0.2.0)
# ═══════════════════════════════════════════════════════════════

class TestDetectDesaturations:
    def _make_spo2(self, n_s=600, sf=1.0, drop_at=200, drop_mag=5.0):
        spo2 = np.full(int(n_s * sf), 96.0)
        start = int(drop_at * sf)
        end   = int((drop_at + 60) * sf)
        spo2[start:end] -= drop_mag
        sleep_mask = np.ones(len(spo2), dtype=bool)
        return spo2, sleep_mask, sf

    def test_detects_drop(self):
        spo2, mask, sf = self._make_spo2()
        events = detect_desaturations(spo2, sf, mask)
        assert len(events) >= 1

    def test_event_keys(self):
        spo2, mask, sf = self._make_spo2()
        events = detect_desaturations(spo2, sf, mask)
        if events:
            for key in ("onset_s", "duration_s", "nadir_spo2", "drop_pct"):
                assert key in events[0], f"Missing key: {key}"

    def test_no_drop_no_events(self):
        spo2 = np.full(600, 96.0)
        mask = np.ones(600, dtype=bool)
        events = detect_desaturations(spo2, 1.0, mask, drop_pct=3.0)
        assert events == []


# ═══════════════════════════════════════════════════════════════
# P. get_desaturation  (root-level v0.2.0)
# ═══════════════════════════════════════════════════════════════

class TestGetDesaturation:
    def test_detects_3pct_drop(self):
        sf = 1.0
        spo2 = np.full(300, 96.0)
        # 5 % nadir starting 10 s after event onset (within 30 s post-event)
        spo2[40:80] = 91.0
        desat, min_spo2 = get_desaturation(spo2, onset_s=30.0, dur_s=10.0,
                                            sf_spo2=sf)
        assert desat is not None and desat >= 3.0
        assert min_spo2 is not None and min_spo2 <= 91.5

    def test_returns_none_for_none_input(self):
        desat, min_spo2 = get_desaturation(None, onset_s=0, dur_s=10, sf_spo2=1.0)
        assert desat is None and min_spo2 is None

    def test_no_desat_returns_small_value(self):
        spo2 = np.full(300, 96.0)
        desat, _ = get_desaturation(spo2, onset_s=10.0, dur_s=10.0, sf_spo2=1.0)
        # Flat signal → desaturation should be ~0
        assert desat is None or desat < 3.0


# ═══════════════════════════════════════════════════════════════
# Q. compute_anchor_baseline  (root-level v0.2.0)
# ═══════════════════════════════════════════════════════════════

class TestComputeAnchorBaseline:
    def _make_data(self, n_epochs=40, sf=SF):
        hypno = ["W"] * 4 + ["N2"] * 20 + ["R"] * 8 + ["N3"] * 8
        n = n_epochs * int(30 * sf)
        t = np.linspace(0, n / sf, n)
        env = np.abs(np.sin(2 * np.pi * 0.25 * t)) + 0.5
        return env, hypno, sf

    def test_reliable_with_sufficient_n2(self):
        env, hypno, sf = self._make_data()
        result = compute_anchor_baseline(env, sf, hypno)
        assert result["anchor_reliable"] is True
        assert result["anchor_value"] is not None
        assert result["anchor_value"] > 0

    def test_unreliable_without_n2(self):
        n = int(10 * 30 * SF)
        env = np.ones(n) * 0.7
        hypno = ["W"] * 10
        result = compute_anchor_baseline(env, SF, hypno)
        assert result["anchor_reliable"] is False

    def test_result_keys(self):
        env, hypno, sf = self._make_data()
        result = compute_anchor_baseline(env, sf, hypno)
        for key in ("anchor_value", "anchor_epochs_used", "anchor_reliable",
                    "anchor_ratio", "mouth_breathing_suspected"):
            assert key in result, f"Missing key: {key}"

    def test_mouth_breathing_flag(self):
        """6 N2 epochs at high amplitude, rest at very low → ratio < 0.60."""
        sf = SF
        # 6 N2 epochs (high) + 34 wake/other epochs (very low)
        hypno = ["W"] * 17 + ["N2"] * 6 + ["W"] * 17
        n = len(hypno) * int(30 * sf)
        env = np.full(n, 0.05)
        for ep_i, stage in enumerate(hypno):
            if stage == "N2":
                s = ep_i * int(30 * sf)
                e = min(s + int(30 * sf), n)
                env[s:e] = 1.0
        result = compute_anchor_baseline(env, sf, hypno)
        if result["anchor_reliable"]:
            assert result["mouth_breathing_suspected"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
