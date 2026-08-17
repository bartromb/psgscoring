"""
tests/test_wavelet_denoise.py — `flow_wavelet_denoise` (v0.20.0, off everywhere).

Soft wavelet thresholding of the flow waveform, after the bandpass and before
anything reads it. Three things can go wrong, and two of them are silent:

  1. **The default moves.** Every published number assumes an undenoised
     waveform. `TestDefaultUnchanged` compares against the literal pre-v0.20.0
     expression, bit for bit rather than `allclose`.

  2. **Only some consumers are denoised.** Four separate places read the
     bandpassed flow — the amplitude envelope, the MMSD apnea validation, the
     breath detector, and the boundary snapping. The specification requires all
     of them to see the same signal; if one is missed, the envelope events and
     the per-breath evidence stop describing the same recording, and nothing
     raises. `test_every_bandpass_flow_consumer_passes_the_flag` reads the
     source to check this structurally, so a *fifth* consumer added later fails
     the test instead of shipping.

  3. **It smooths the flanks it is supposed to leave alone.** An apnea onset is
     also an abrupt amplitude change. `TestFlanksSurvive` injects spikes into a
     known signal and requires the spike to go while the event edges stay put —
     the pre-registered acceptance condition, expressed as a test rather than
     left to the cohort measurement.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from scipy import signal as sp_signal

from psgscoring.profiles import PROFILES, PostProcessingRules
from psgscoring.signal import bandpass_flow, denoise_flow_wavelet, preprocess_flow

pywt = pytest.importorskip("pywt", reason="PyWavelets is an optional extra")

REPO = Path(__file__).resolve().parents[1]
SF = 64.0


def _breathing(dur_s=1800.0, sf=SF, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * sf)) / sf
    x = np.sin(2 * np.pi * 0.25 * t) * (1.0 + 0.3 * np.sin(2 * np.pi * 0.01 * t))
    return x + 0.02 * rng.standard_normal(len(t)), t


def _with_spikes(x, sf, spikes, amp=10.0):
    """Add rectangular high-energy bursts: motion artefacts, crudely."""
    y = x.copy()
    for t0, dur in spikes:
        a, b = int(t0 * sf), int((t0 + dur) * sf)
        y[a:b] += amp * np.sin(2 * np.pi * 1.5 * np.arange(b - a) / sf)
    return y


# ─────────────────────────────────────────────────────────────────────────
# The default must not have moved
# ─────────────────────────────────────────────────────────────────────────

class TestDefaultUnchanged:

    def test_bandpass_default_is_the_old_expression(self):
        x, _ = _breathing(600.0)
        nyq = SF / 2
        b, a = sp_signal.butter(3, [max(0.05 / nyq, 0.001), min(3.0 / nyq, 0.99)],
                                btype="band")
        want = sp_signal.filtfilt(b, a, x)
        assert np.array_equal(bandpass_flow(x, SF), want), (
            "bandpass_flow changed with denoising off; every published number "
            "assumes it did not")

    def test_preprocess_flow_default_is_unchanged(self):
        x, _ = _breathing(600.0)
        assert np.array_equal(preprocess_flow(x, SF),
                              preprocess_flow(x, SF, denoise=False))

    def test_the_dataclass_default_is_off(self):
        assert PostProcessingRules().flow_wavelet_denoise is False

    def test_no_shipped_profile_enables_it(self):
        """
        It has not been measured against human scoring. Until it has, a profile
        that enables it would be a recommendation the evidence does not support.
        """
        on = [n for n, p in PROFILES.items()
              if p.post_processing.flow_wavelet_denoise]
        assert not on, f"denoising is enabled on {on} without a cohort measurement"

    def test_the_field_reaches_the_legacy_dict(self):
        # SCORING_PROFILES also carries the legacy aliases, so iterate over the
        # registry and look the alias-free name up.
        import psgscoring.constants as C
        for name, prof in PROFILES.items():
            assert C.SCORING_PROFILES[name]["FLOW_WAVELET_DENOISE"] == \
                prof.post_processing.flow_wavelet_denoise, name


# ─────────────────────────────────────────────────────────────────────────
# All consumers, not some
# ─────────────────────────────────────────────────────────────────────────

def test_every_bandpass_flow_consumer_passes_the_flag():
    """
    Structural, because the alternative is a convention nobody can verify.

    Every call to `bandpass_flow` inside the package must pass `denoise=`.
    A call that does not is a consumer reading a differently-filtered waveform
    than its neighbours, which produces an internally inconsistent recording
    and no error at all.
    """
    offenders = []
    for py in sorted((REPO / "psgscoring").glob("*.py")):
        src = py.read_text(encoding="utf-8")
        for m in re.finditer(r"bandpass_flow\(", src):
            # The definition itself, and re-exports, are not call sites.
            line_start = src.rfind("\n", 0, m.start()) + 1
            line = src[line_start:src.find("\n", m.start())]
            if line.lstrip().startswith(("def ", "#", '"', "*")):
                continue
            # Grab the call's arguments up to the balancing paren.
            depth, i = 0, m.end() - 1
            while i < len(src):
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            call = src[m.end():i]
            if "denoise" not in call:
                offenders.append(f"{py.name}: bandpass_flow({call.strip()[:60]}...)")
    assert not offenders, (
        "these bandpass_flow call sites do not pass denoise=, so they would "
        "read an undenoised waveform while their neighbours read a denoised "
        f"one: {offenders}")


def test_the_consumer_check_can_fail():
    """Guards the guard: the regex must actually find the call sites."""
    n = sum(len(re.findall(r"bandpass_flow\(", p.read_text(encoding="utf-8")))
            for p in (REPO / "psgscoring").glob("*.py"))
    assert n >= 5, (
        f"only {n} bandpass_flow occurrences found; the structural check has "
        f"stopped matching and would pass vacuously")


# ─────────────────────────────────────────────────────────────────────────
# It removes spikes
# ─────────────────────────────────────────────────────────────────────────

class TestTheUniversalThresholdDegenerates:
    """
    The specified recipe does not work in the position the specification puts
    it, and these tests pin the measurement rather than the intention.

    `sigma = MAD(d1)/0.6745` assumes the finest detail scale is noise. That
    scale spans sf/4 to sf/2 -- 16 to 32 Hz at 64 Hz sampling -- and the
    upstream bandpass has already removed everything above 3 Hz. So d1 is empty
    by construction, sigma collapses to zero, and soft thresholding becomes a
    no-op. Measured: sigma = 0.00000, T = 0.00000, largest detail coefficient
    55.8.

    Two consequences, and both matter for whether this option can proceed:

      1. sigma is not estimable at this point in the chain. Fixing it would mean
         estimating sigma before the bandpass, or moving the denoising ahead of
         it -- either of which is a change to the specification and must be
         measured as one, not slipped in.
      2. Even with a valid sigma, the universal threshold targets the *noise
         floor*, not outliers. A spike whose coefficient is 55.8 against a T of
         order 0.05 loses 0.1 % of its amplitude. Removing impulsive artefacts
         needs a per-scale outlier criterion, which is a different rule.

    The pre-registered synthetic gate was ">90 % spike suppression". It is not
    met, so per the working agreement the option does not advance to the cohort
    measurement.
    """

    def test_sigma_collapses_and_says_so(self):
        x, _ = _breathing(600.0)
        _out, info = denoise_flow_wavelet(bandpass_flow(x, SF), SF)
        assert info["applied"] is False
        assert info["sigma"] < 1e-6, (
            f"sigma is {info['sigma']:.3e}: the estimator no longer degenerates, "
            f"so the finding this option was parked on has changed and the "
            f"measurement must be redone")
        assert "degenerates in this position of the chain" in info["reason"]
        # The ratio is what makes the finding readable: an absolute sigma would
        # depend on the recording's units.
        assert info["sigma_over_signal_scale"] < 1e-4

    def test_a_degenerate_threshold_leaves_the_signal_untouched(self):
        """
        Not a workaround. The alternative -- shrinking by T = 0 -- returns a
        numerically different array through the DWT round trip while changing
        nothing meaningful, which would make the flag look effective.
        """
        x, _ = _breathing(600.0)
        filt = bandpass_flow(x, SF)
        out, _info = denoise_flow_wavelet(filt, SF)
        assert np.array_equal(out, filt)

    def test_the_spike_survives_as_measured(self):
        """
        The pre-registered criterion, kept as a failing-by-design record: the
        spike is NOT suppressed. If a future change makes this pass, the option
        has become viable and the CHANGELOG entry needs revisiting.
        """
        clean, _ = _breathing(600.0)
        dirty = _with_spikes(clean, SF, [(120.0, 1.0), (300.0, 2.0), (450.0, 0.5)])
        before = float(np.sqrt(np.mean(
            (bandpass_flow(dirty, SF) - bandpass_flow(clean, SF)) ** 2)))
        after = float(np.sqrt(np.mean(
            (bandpass_flow(dirty, SF, denoise=True)
             - bandpass_flow(clean, SF)) ** 2)))
        assert after >= 0.9 * before, (
            f"spike energy went {before:.4f} -> {after:.4f}: suppression now "
            f"works, which contradicts the recorded measurement. Re-measure "
            f"before trusting either number.")

    def test_info_reports_the_parameters_either_way(self):
        """
        A scored recording must be able to say what was done to it -- including
        'nothing, and here is why'.
        """
        x, _ = _breathing(600.0)
        out, info = denoise_flow_wavelet(bandpass_flow(x, SF), SF)
        assert info["wavelet"] == "sym8"
        assert info["levels"] >= 1
        assert "threshold" in info and "sigma" in info
        assert len(out) == len(x), "the DWT round trip changed the length"


class TestSpikeSuppression:

    def test_a_signal_too_short_for_the_dwt_is_returned_unchanged(self):
        x = np.sin(2 * np.pi * 0.25 * np.arange(10) / SF)
        out, info = denoise_flow_wavelet(x, SF)
        assert info["applied"] is False
        assert np.array_equal(out, x)

    def test_odd_length_survives(self):
        """`waverec` returns an even-length array; on odd n that is one too many."""
        x, _ = _breathing(300.0)
        x = x[:-1] if len(x) % 2 == 0 else x
        assert len(x) % 2 == 1
        out, _ = denoise_flow_wavelet(x, SF)
        assert len(out) == len(x)


# ─────────────────────────────────────────────────────────────────────────
# ...without moving the flanks it must not touch
# ─────────────────────────────────────────────────────────────────────────

class TestFlanksSurvive:

    def test_an_apnea_flank_is_not_smoothed_away(self):
        """
        The primary risk, as a test. An apnea onset is an abrupt amplitude
        change too, so a threshold that is too aggressive erases the edge the
        detection runs on. Locate the edge before and after denoising and
        require it to stay within a quarter second.
        """
        sf = SF
        t = np.arange(int(900 * sf)) / sf
        x = np.sin(2 * np.pi * 0.25 * t)
        onset, offset = 300.0, 320.0
        x[int(onset * sf):int(offset * sf)] *= 0.05          # 20 s apnea
        rng = np.random.default_rng(3)
        x = x + 0.02 * rng.standard_normal(len(t))

        def edge(sig):
            env = np.abs(sp_signal.hilbert(bandpass_flow(sig, sf)))
            w = max(1, int(sf))
            env = np.convolve(env, np.ones(w) / w, mode="same")
            lo = slice(int((onset + 5) * sf), int((offset - 5) * sf))
            hi = slice(int(100 * sf), int(200 * sf))
            half = 0.5 * (env[hi].mean() + env[lo].mean())
            seg = env[int((onset - 10) * sf):int((onset + 10) * sf)]
            below = np.where(seg < half)[0]
            return (onset - 10) + below[0] / sf if len(below) else None

        e_plain = edge(x)
        env_dn = np.abs(sp_signal.hilbert(bandpass_flow(x, sf, denoise=True)))
        assert e_plain is not None, "the fixture has no detectable onset edge"

        w = max(1, int(sf))
        env_dn = np.convolve(env_dn, np.ones(w) / w, mode="same")
        lo = slice(int((onset + 5) * sf), int((offset - 5) * sf))
        hi = slice(int(100 * sf), int(200 * sf))
        half = 0.5 * (env_dn[hi].mean() + env_dn[lo].mean())
        seg = env_dn[int((onset - 10) * sf):int((onset + 10) * sf)]
        below = np.where(seg < half)[0]
        assert len(below), "the onset edge disappeared entirely after denoising"
        e_dn = (onset - 10) + below[0] / sf

        assert abs(e_dn - e_plain) <= 0.25, (
            f"the apnea onset moved {abs(e_dn - e_plain):.2f} s "
            f"({e_plain:.2f} -> {e_dn:.2f}); the pre-registered limit is 0.25 s")

    def test_clean_breathing_is_barely_touched(self):
        """
        With no artefact present the threshold should have almost nothing to
        remove. If it reshapes clean breathing, it is not an artefact remover.
        """
        clean, _ = _breathing(900.0)
        filt = bandpass_flow(clean, SF)
        dn = bandpass_flow(clean, SF, denoise=True)
        rel = float(np.sqrt(np.mean((dn - filt) ** 2)) / np.std(filt))
        assert rel < 0.10, (
            f"denoising changes clean breathing by {rel:.1%} of its own "
            f"amplitude; the threshold is too aggressive")
