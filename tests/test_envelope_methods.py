"""
tests/test_envelope_methods.py — the `envelope_method` / `envelope_fs` axis.

v0.19.0 made "how the amplitude envelope is built" a profile field, with four
exploratory arms next to the full-array Hilbert transform that produced every
published result so far.

What these tests are actually guarding
--------------------------------------
The golden harness cannot referee this axis. Its cases are 600 s at 32 Hz --
shorter than a single 30-minute chunk -- so `hilbert_chunked` never engages and
every golden case passes unchanged whether the chunking is correct, subtly
wrong, or off by a whole block. A green golden run is not evidence here. The
signals below are eight hours long for that reason: long enough to cross
several chunk boundaries.

The other silent failure is the preprocessing cache in `respiratory.py`. It
exists because the AHI-interval reruns share their envelopes, which was sound
while the envelope was profile-independent. It no longer is. If the envelope
settings drop out of the cache key, a group mixing envelope arms scores every
arm on the first arm's envelope and reports an interval that looks reassuringly
tight. `test_cache_does_not_leak_between_envelope_methods` is that test.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp_signal

from psgscoring.profiles import PROFILE_GROUPS, PROFILES
from psgscoring.signal import (
    ENVELOPE_METHODS,
    bandpass_flow,
    breath_amplitude_envelope,
    compute_envelope,
    hilbert_envelope,
    hilbert_envelope_chunked,
    preprocess_effort,
    preprocess_flow,
    rectify_lowpass_envelope,
)

SF = 64.0                      # low enough to keep an 8 h array cheap in CI
HOURS = 8.0
N = int(SF * 3600 * HOURS)

# The four arms added in v0.19.0, and the single field that distinguishes each.
ENVELOPE_PROFILES = {
    "aasm_v3_env_chunked":   ("hilbert_chunked", None),
    "aasm_v3_env_rectify":   ("rectify_lowpass", None),
    "aasm_v3_env_breath":    ("breath_amplitude", None),
    "aasm_v3_env_decimated": ("hilbert", 10.0),
}


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

def _breathing(n: int, sf: float, seed: int = 20260816) -> np.ndarray:
    """AM-modulated breathing at 0.25 Hz with periodic apneic dropouts."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sf
    x = np.sin(2 * np.pi * 0.25 * t) * (1.0 + 0.45 * np.sin(2 * np.pi * 0.012 * t))
    for start in range(60, int(n / sf) - 60, 240):        # 20 s dropout every 4 min
        x[int(start * sf):int((start + 20) * sf)] *= 0.08
    return x + 0.02 * rng.standard_normal(n)


@pytest.fixture(scope="module")
def long_flow():
    """Eight hours of filtered flow — long enough to cross chunk boundaries."""
    return bandpass_flow(_breathing(N, SF), SF)


# ─────────────────────────────────────────────────────────────────────────
# The default must not have moved
# ─────────────────────────────────────────────────────────────────────────

class TestDefaultUnchanged:

    def test_preprocess_flow_default_is_the_old_expression(self):
        """
        Bit-identical to the pre-v0.19.0 body. Not `allclose`: the whole point
        of routing through a dispatcher is that the default path is the same
        arithmetic in the same order, and `allclose` would hide a reordering
        that shifts the last digits of every published number.
        """
        x = _breathing(int(SF * 600), SF)
        want = np.convolve(
            np.abs(sp_signal.hilbert(bandpass_flow(x, SF))),
            np.ones(max(1, int(SF))) / max(1, int(SF)), mode="same")
        got = preprocess_flow(x, SF)
        assert np.array_equal(got, want), (
            "the default flow envelope changed; every published result assumes "
            "it did not")

    def test_preprocess_effort_default_is_the_old_expression(self):
        x = _breathing(int(SF * 600), SF, seed=7)
        nyq = SF / 2
        b, a = sp_signal.butter(3, [max(0.03 / nyq, 0.001), min(2.0 / nyq, 0.99)],
                                btype="band")
        win = max(1, int(SF * 2))
        want = np.convolve(np.abs(sp_signal.hilbert(sp_signal.filtfilt(b, a, x))),
                           np.ones(win) / win, mode="same")
        assert np.array_equal(preprocess_effort(x, SF), want)

    def test_every_other_profile_still_uses_the_full_hilbert(self):
        """
        A new arm must never be reached by a profile that did not ask for it.
        `mesa_shhs` and `chicago_1999` matter most: they reproduce published
        cohorts and a changed envelope would break that silently.
        """
        for name, prof in PROFILES.items():
            if name in ENVELOPE_PROFILES:
                continue
            pp = prof.post_processing
            assert pp.envelope_method == "hilbert", (
                f"{name} moved off the default envelope")
            assert pp.envelope_fs is None, f"{name} decimates its envelope"

    @pytest.mark.parametrize("name,expected", sorted(ENVELOPE_PROFILES.items()))
    def test_each_arm_sets_exactly_its_own_field(self, name, expected):
        method, fs = expected
        pp = PROFILES[name].post_processing
        assert (pp.envelope_method, pp.envelope_fs) == (method, fs)

    def test_arms_differ_from_the_reference_only_in_the_envelope(self):
        """
        The axis is only interpretable if nothing else varies along it. Compare
        every other field of every arm against `aasm_v3_rec`.
        """
        import dataclasses

        ref = PROFILES["aasm_v3_rec"]
        for name in ENVELOPE_PROFILES:
            arm = PROFILES[name]
            for section in ("hypopnea", "apnea", "spo2", "post_processing"):
                r = dataclasses.asdict(getattr(ref, section))
                a = dataclasses.asdict(getattr(arm, section))
                diff = {k for k in r if r[k] != a[k]}
                assert diff <= {"envelope_method", "envelope_fs"}, (
                    f"{name}.{section} also differs from aasm_v3_rec in "
                    f"{sorted(diff - {'envelope_method', 'envelope_fs'})}; the "
                    f"envelope axis would no longer measure the envelope")

    def test_envelope_group_is_the_reference_plus_the_four_arms(self):
        group = PROFILE_GROUPS["envelope"]
        assert group[0] == "aasm_v3_rec", "the reference arm must come first"
        assert set(group[1:]) == set(ENVELOPE_PROFILES)


# ─────────────────────────────────────────────────────────────────────────
# Option 1 — chunked Hilbert
# ─────────────────────────────────────────────────────────────────────────

class TestChunkedHilbert:

    def test_short_signal_takes_the_exact_path(self):
        """Below one chunk the function IS the full transform, not an approximation."""
        x = bandpass_flow(_breathing(int(SF * 300), SF), SF)
        assert np.array_equal(hilbert_envelope_chunked(x, SF), hilbert_envelope(x))

    def test_it_is_close_to_the_full_transform_but_not_equal(self, long_flow):
        """
        The specification called this free. It is not, and the test pins both
        halves of that: close enough to be useful, different enough that it
        cannot be swapped in under the default.

        If a future change makes it exact, this test fails — and that is the
        correct outcome, because it would mean the arm no longer needs to be a
        separate profile.
        """
        ref = hilbert_envelope(long_flow)
        got = hilbert_envelope_chunked(long_flow, SF)
        scale = float(np.percentile(ref, 95))

        assert not np.array_equal(got, ref)
        # Interior: exclude two minutes at each end, where the difference is
        # about the FFT wrapping circularly rather than about chunking.
        edge = int(120 * SF)
        rel = np.abs(got - ref)[edge:-edge].max() / scale
        assert rel < 5e-3, f"interior deviation grew to {rel:.2e} of p95"
        assert rel > 1e-8, (
            f"interior deviation is {rel:.2e}: the chunking no longer engages, "
            f"or the fixture became shorter than one chunk")

    def test_the_record_edges_are_where_it_differs_most(self, long_flow):
        """
        Documented, measured, and deliberately not hidden: an FFT over the
        whole array wraps the end of the night onto the beginning, and a block
        wraps within its own window. The first samples are the price.
        """
        ref = hilbert_envelope(long_flow)
        got = hilbert_envelope_chunked(long_flow, SF)
        scale = float(np.percentile(ref, 95))
        first_second = np.abs(got - ref)[:int(SF)].max() / scale
        edge = int(120 * SF)
        interior = np.abs(got - ref)[edge:-edge].max() / scale
        assert first_second > interior, (
            "the edge is no longer the worst case; the chunk seams have become "
            "the dominant error, which means the discard is misaligned")

    def test_no_step_discontinuity_at_the_chunk_seams(self, long_flow):
        """
        An off-by-one in the discard shows up as a jump exactly at a seam. Any
        such jump would be large compared with the envelope's own
        sample-to-sample movement, so compare against that.
        """
        got = hilbert_envelope_chunked(long_flow, SF, chunk_s=1800.0, pad_s=60.0)
        steps = np.abs(np.diff(got))
        typical = float(np.percentile(steps, 99.9))
        chunk = int(1800.0 * SF)
        for seam in range(chunk, len(got), chunk):
            assert steps[seam - 1] < 20 * typical, (
                f"discontinuity at the seam at sample {seam} "
                f"({steps[seam - 1]:.3e} vs typical {typical:.3e})")

    def test_last_block_shorter_than_a_chunk_is_still_covered(self):
        """A signal of 1.5 chunks must come back fully populated, not truncated."""
        n = int(SF * 2700)                      # 45 min, chunk = 30 min
        x = bandpass_flow(_breathing(n, SF), SF)
        got = hilbert_envelope_chunked(x, SF, chunk_s=1800.0, pad_s=60.0)
        assert len(got) == n
        assert np.all(np.isfinite(got))
        assert got[-int(SF * 60):].max() > 0, "the tail block was never written"

    def test_transform_memory_stops_scaling_with_the_recording(self):
        """
        The reason the arm exists, measured rather than asserted.

        Peak allocation is compared *above the returned envelope*, because the
        output array is the same size either way and is not what the arm
        controls. The claim being tested is not "smaller" but "no longer a
        function of recording length": double the night and the full transform
        doubles, while the block transform does not move.
        """
        import tracemalloc

        def overhead(fn, n):
            x = bandpass_flow(_breathing(n, SF), SF)
            tracemalloc.start()
            tracemalloc.reset_peak()
            fn(x)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return peak - n * 8          # minus the float64 envelope it must return

        n4 = int(SF * 3600 * 4)
        n8 = int(SF * 3600 * 8)
        full4 = overhead(hilbert_envelope, n4)
        full8 = overhead(hilbert_envelope, n8)
        chunk4 = overhead(lambda x: hilbert_envelope_chunked(x, SF), n4)
        chunk8 = overhead(lambda x: hilbert_envelope_chunked(x, SF), n8)

        assert full8 > 1.5 * full4, (
            f"the full transform stopped scaling with length "
            f"({full4/1e6:.0f} -> {full8/1e6:.0f} MB); the comparison is void")
        assert chunk8 < 1.3 * chunk4, (
            f"the block transform grew with the recording "
            f"({chunk4/1e6:.0f} -> {chunk8/1e6:.0f} MB): a block is holding on "
            f"to something it should have released")
        assert chunk8 < full8 / 5, (
            f"block overhead {chunk8/1e6:.0f} MB vs full {full8/1e6:.0f} MB — "
            f"the arm no longer buys what it costs in accuracy")


# ─────────────────────────────────────────────────────────────────────────
# Option 2 — rectify + lowpass
# ─────────────────────────────────────────────────────────────────────────

class TestRectifyLowpass:

    def test_a_pure_tone_recovers_its_own_amplitude(self):
        """
        Without the pi/2 correction the envelope lands 36 % low, and since the
        baseline percentile is computed from the same envelope the error would
        be invisible in a ratio — right up to the point where a fixed reduction
        threshold fires early.
        """
        t = np.arange(int(SF * 600)) / SF
        amp = 2.5
        env = rectify_lowpass_envelope(amp * np.sin(2 * np.pi * 0.25 * t), SF)
        mid = env[int(SF * 60):-int(SF * 60)]
        assert abs(float(np.median(mid)) - amp) < 0.1 * amp

    def test_it_tracks_a_dropout(self, long_flow):
        env = rectify_lowpass_envelope(long_flow, SF)
        # dropout at 60-80 s in the fixture, normal breathing at 150-200 s
        during = float(np.median(env[int(65 * SF):int(75 * SF)]))
        outside = float(np.median(env[int(150 * SF):int(200 * SF)]))
        assert during < 0.5 * outside

    def test_it_is_a_different_envelope_not_a_cheaper_one(self, long_flow):
        """Correlated with the analytic envelope, but not a substitute for it."""
        ref = hilbert_envelope(long_flow)
        got = rectify_lowpass_envelope(long_flow, SF)
        r = float(np.corrcoef(ref, got)[0, 1])
        assert r > 0.85, f"lost the shape entirely (r={r:.3f})"
        assert r < 0.999, "identical to the analytic envelope, which it is not"


# ─────────────────────────────────────────────────────────────────────────
# Option 3 — breath amplitude
# ─────────────────────────────────────────────────────────────────────────

class TestBreathAmplitude:

    def test_it_tracks_a_dropout(self, long_flow):
        env = breath_amplitude_envelope(long_flow, SF)
        during = float(np.median(env[int(65 * SF):int(75 * SF)]))
        outside = float(np.median(env[int(150 * SF):int(200 * SF)]))
        assert during < 0.5 * outside

    def test_amplitude_is_on_the_same_scale_as_the_analytic_envelope(self, long_flow):
        """
        Half the peak-to-trough excursion, not the whole excursion. Getting this
        wrong doubles the envelope; against a baseline drawn from the same
        envelope the ratio survives, but every absolute figure downstream
        (hypoxic burden anchoring, effort comparison) doubles with it.
        """
        ref = hilbert_envelope(long_flow)
        got = breath_amplitude_envelope(long_flow, SF)
        ratio = float(np.percentile(got, 95)) / float(np.percentile(ref, 95))
        assert 0.5 < ratio < 2.0, f"scale is off by {ratio:.2f}x"

    def test_too_few_breaths_returns_zeros_not_a_plausible_flat_line(self):
        """
        A flat non-zero envelope reads downstream as steady breathing, i.e. as
        a normal recording. Zeros cannot be mistaken for that.
        """
        flat = np.zeros(int(SF * 300))
        assert np.array_equal(breath_amplitude_envelope(flat, SF),
                              np.zeros(int(SF * 300)))

    def test_effort_channels_never_use_it(self):
        """
        Zero-crossings on a RIP belt are a property of the belt's baseline
        drift, not of the breath. The fallback to the analytic signal is
        deliberate; this pins it so a later refactor cannot quietly "fix" the
        inconsistency.
        """
        x = _breathing(int(SF * 600), SF, seed=11)
        assert np.array_equal(
            preprocess_effort(x, SF, envelope_method="breath_amplitude"),
            preprocess_effort(x, SF, envelope_method="hilbert"))


# ─────────────────────────────────────────────────────────────────────────
# Option 4 — decimated Hilbert
# ─────────────────────────────────────────────────────────────────────────

class TestDecimatedHilbert:

    def test_it_returns_the_original_sample_grid(self, long_flow):
        """
        The arm deliberately does not export a second sample rate: every
        downstream consumer indexes the envelope in original samples.
        """
        got = compute_envelope(long_flow, SF, "hilbert", envelope_fs=10.0)
        assert len(got) == len(long_flow)

    def test_it_stays_close_to_the_full_rate_transform(self, long_flow):
        ref = hilbert_envelope(long_flow)
        got = compute_envelope(long_flow, SF, "hilbert", envelope_fs=10.0)
        scale = float(np.percentile(ref, 95))
        edge = int(60 * SF)
        rel = np.abs(got - ref)[edge:-edge].max() / scale
        assert rel < 0.35, f"decimation moved the envelope by {rel:.2f} of p95"
        assert not np.array_equal(got, ref)

    def test_it_is_not_time_shifted(self, long_flow):
        """
        A decimation filter without `zero_phase` delays the envelope, and a
        delayed envelope moves every event boundary in the same direction --
        the one failure mode that would look like a real physiological finding.
        """
        ref = hilbert_envelope(long_flow)
        got = compute_envelope(long_flow, SF, "hilbert", envelope_fs=10.0)
        seg = slice(int(600 * SF), int(1800 * SF))
        a = ref[seg] - ref[seg].mean()
        b = got[seg] - got[seg].mean()
        lags = np.arange(-int(2 * SF), int(2 * SF) + 1)
        corr = [float(np.dot(a, np.roll(b, int(k)))) for k in lags]
        assert abs(int(lags[int(np.argmax(corr))])) <= 1, "the envelope is time-shifted"

    def test_a_meaningless_rate_falls_back_instead_of_failing(self, long_flow):
        """An envelope_fs at or above the acquisition rate cannot decimate."""
        assert np.array_equal(
            compute_envelope(long_flow, SF, "hilbert", envelope_fs=SF * 2),
            hilbert_envelope(long_flow))


# ─────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────

class TestDispatcher:

    def test_unknown_method_raises(self):
        """
        Silently falling back to the default would let a typo in a profile
        score under different rules than the profile name advertises.
        """
        x = np.zeros(int(SF * 60))
        with pytest.raises(ValueError, match="unknown envelope_method"):
            compute_envelope(x, SF, "hilbert_chunk")          # plausible typo

    @pytest.mark.parametrize("method", ENVELOPE_METHODS)
    def test_every_declared_method_is_reachable(self, method):
        x = bandpass_flow(_breathing(int(SF * 600), SF), SF)
        got = compute_envelope(x, SF, method)
        assert len(got) == len(x)
        assert np.all(np.isfinite(got))
        assert np.all(got >= 0), "an amplitude envelope cannot be negative"

    @pytest.mark.parametrize("method", ENVELOPE_METHODS)
    def test_each_method_produces_a_distinct_envelope(self, method):
        """
        Four arms that all quietly compute the same thing would still pass
        every test above. Compare each against the default.
        """
        x = bandpass_flow(_breathing(int(SF * 3600 * 2), SF), SF)
        got = preprocess_flow(x, SF, envelope_method=method)
        ref = preprocess_flow(x, SF)
        if method == "hilbert":
            assert np.array_equal(got, ref)
        else:
            assert not np.allclose(got, ref, rtol=1e-6, atol=1e-9), (
                f"{method} is indistinguishable from the default envelope")

    def test_decimation_is_ignored_by_the_o_n_methods(self, long_flow):
        """
        `rectify_lowpass` and `breath_amplitude` never build an analytic
        signal, so decimating them would cost accuracy for no memory saving.
        """
        for method in ("rectify_lowpass", "breath_amplitude"):
            assert np.array_equal(
                compute_envelope(long_flow, SF, method, envelope_fs=10.0),
                compute_envelope(long_flow, SF, method))
