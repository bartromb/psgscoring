"""
psgscoring.signal
=================
Signal preprocessing and baseline estimation for respiratory channels.

Functions in this module accept plain NumPy arrays and a sample rate; they do
NOT depend on MNE or the rest of the YASAFlaskified stack.

Dependencies: numpy, scipy, psgscoring.constants, psgscoring.utils
"""

from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import median_filter

from .breath import detect_breaths
from .constants import BASELINE_WINDOW_S, EPOCH_LEN_S
from .utils import is_nrem, is_rem, safe_r


# ---------------------------------------------------------------------------
# Nasal pressure linearization  (AASM Rule 3)
# ---------------------------------------------------------------------------

def linearize_nasal_pressure(data: np.ndarray) -> np.ndarray:
    """
    Square-root transformation of a nasal pressure signal.

    Nasal pressure is proportional to flow² (Bernoulli).  Without correction
    a true 50 % flow reduction appears as a 75 % amplitude reduction, leading
    to systematic hypopnea over-scoring.

    Formula: sign(x) * sqrt(|x|)  — sign-preserving, maintains
    inspiration-positive / expiration-negative polarity.

    References
    ----------
    Montserrat et al., AJRCCM 2001 (r²=0.88–0.96 vs pneumotachography).
    Thurnheer et al., AJRCCM 2001.
    AASM Scoring Manual 2.6 Rule 3.
    """
    return np.sign(data) * np.sqrt(np.abs(data))


# ---------------------------------------------------------------------------
# MMSD  — Mean Magnitude of Second Derivative
# ---------------------------------------------------------------------------

def compute_mmsd(
    flow_data: np.ndarray,
    sf: float,
    window_s: float = 1.0,
) -> np.ndarray:
    """
    Drift-independent measure of respiratory effort.

    The second derivative amplifies rapid oscillations (breathing) while
    suppressing slow baseline drift.  Useful to reject false-positive apneas
    caused by sensor drift rather than true airflow cessation.

    References
    ----------
    Lee et al., Physiol Meas 2008 — 92 % agreement, κ=0.78 on 24 PSGs.
    """
    d2  = np.diff(flow_data, n=2)
    d2  = np.concatenate([[0], d2, [0]])          # restore original length
    win = max(1, int(sf * window_s))
    return np.convolve(np.abs(d2), np.ones(win) / win, mode="same")


# ---------------------------------------------------------------------------
# Flow preprocessing pipeline
# ---------------------------------------------------------------------------

def denoise_flow_wavelet(
    x: np.ndarray,
    sf: float,
    wavelet: str = "sym8",
    coarsest_s: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """
    Remove impulsive artefacts by soft wavelet thresholding.

    Returns ``(denoised, info)``; ``info`` carries the mother wavelet, the
    number of levels, the estimated noise scale and the threshold actually
    applied, so a scored recording can say what was removed from it.

    Why this is not the bandpass again
    ----------------------------------
    The upstream bandpass removes what lies spectrally outside 0.05–3 Hz. A
    motion artefact lasting 1–2 s has power *inside* that band and passes
    straight through. Wavelet thresholding acts locally in time **and** scale,
    so it can take out the spike without touching the breathing around it. The
    scope is correspondingly narrow: short, high-energy disturbances within the
    respiratory band. It is not a general-purpose cleaner.

    The threshold
    -------------
    Donoho & Johnstone's universal threshold, ``T = sigma * sqrt(2 ln N)``,
    with ``sigma = MAD(d1) / 0.6745`` estimated from the finest detail scale.
    That is an analytically derived quantity rather than a tuned constant,
    which is the same footing the RIP quality gate stands on. It assumes white
    noise and therefore *over*-estimates T on coloured noise — conservative in
    the direction that matters here, because an over-estimated threshold removes
    less, not more.

    Soft, not hard: soft thresholding shrinks every coefficient continuously
    toward zero, while hard thresholding introduces discontinuities that
    reappear in the envelope as phantom flanks — exactly what must not happen
    at an event boundary.

    The risk, stated plainly
    ------------------------
    An apnea onset is also an abrupt amplitude change. A threshold that is too
    aggressive smooths precisely the flanks the detection runs on and shifts
    event boundaries, which is the axis block 1B measures. That is why the
    profile field defaults to off and why the boundary-offset measurement is
    the primary endpoint rather than event-F1.

    Raises
    ------
    ImportError
        If PyWavelets is absent. A profile that asks for denoising must not
        silently score without it; install ``psgscoring[denoise]``.
    """
    try:
        import pywt
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(
            "wavelet denoising requires PyWavelets. Install it with "
            "`pip install psgscoring[denoise]`. Scoring was NOT performed: a "
            "profile that requests denoising cannot fall back to not doing it."
        ) from exc

    n = len(x)
    # Level so the coarsest detail scale spans roughly `coarsest_s` seconds:
    # detail level k covers ~2**k / sf seconds.
    want = int(np.floor(np.log2(max(2.0, coarsest_s * sf))))
    cap = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    level = max(1, min(want, cap))
    info = {"wavelet": wavelet, "levels": level, "coarsest_s": coarsest_s}

    if n < 2 ** (level + 1) or cap < 1:
        info.update({"applied": False, "reason": "signal too short for the DWT"})
        return x, info

    coeffs = pywt.wavedec(x, wavelet, level=level)
    # MAD of the FINEST detail scale: the estimator assumes that at that scale
    # the signal is essentially noise.
    d1 = coeffs[-1]
    sigma = float(np.median(np.abs(d1)) / 0.6745)
    thr = sigma * float(np.sqrt(2.0 * np.log(n)))

    # The estimator degenerates HERE, and it does so structurally rather than
    # occasionally. The finest detail scale covers sf/4 to sf/2 -- 16 to 32 Hz
    # at 64 Hz sampling -- and the upstream bandpass already removed everything
    # above 3 Hz. So d1 is empty by construction, MAD(d1) -> 0, and T -> 0,
    # which makes soft thresholding an expensive no-op.
    #
    # DO NOT "FIX" THIS BY MOVING THE ESTIMATOR. That reading is too narrow and
    # it was measured on 2026-09-01 (docs/wavelet_denoise_geen_schaalscheiding.md).
    # Estimating sigma from the finest scale that actually lies INSIDE the
    # passband (1-2 Hz) gives a real noise floor of 4.8e-02 and raises spike
    # suppression from 0.0 % to 0.7 %. The gate asks for 90 %. A per-scale
    # outlier rule instead of the universal threshold does not help either
    # (-79 % across all scales; 0.6 % when restricted to the scales above the
    # breathing frequency, with flanks intact at 0.016 s).
    #
    # The binding constraint is not the threshold but the premise. Energy per
    # scale, breathing against a 0.5-2 s artefact:
    #
    #     scale 7 (0.25-0.50 Hz):  breathing 99.9 %   artefact 57.5 %
    #     scale 6 (0.50-1.00 Hz):  breathing  0.1 %   artefact 28.5 %
    #     scale 5 (1.00-2.00 Hz):  breathing  0.0 %   artefact 10.7 %
    #
    # 58 % of the artefact energy sits in the scale that carries 99.9 % of the
    # breathing. An artefact of 0.5-2 s and a breath of 3-6 s are ONE OCTAVE
    # apart, so no scale contains one without the other. Wavelet thresholding
    # separates by scale; here there is nothing to separate. A method that
    # could work would separate by SHAPE (an artefact is aperiodic, breathing
    # is not), which is a different proposal with its own gate.
    #
    # Measured on 10 min of synthetic breathing at 64 Hz: sigma = 5.5e-07
    # against a signal scale of 9.7e-01, so the estimated noise floor is six
    # orders of magnitude below the signal it is meant to sit in, and T is
    # 2.5e-07 of the largest detail coefficient. Nothing is removed.
    #
    # The test is a RATIO, not an absolute floor: "the estimated noise level is
    # a negligible fraction of the signal's own robust scale" is the statement
    # that means "d1 measured an empty band". An absolute cutoff would depend on
    # the recording's units, which is the mistake the RIP gate already made once
    # (v0.17.0, where an absolute MAD threshold turned out to be measuring the
    # EDF's unit declaration).
    #
    # Reporting this is the whole point of the branch. A profile that asks for
    # denoising and silently does nothing is worse than one that refuses: the
    # flag would be set, the provenance would look populated, and the signal
    # would be untouched.
    sig_scale = float(np.median(np.abs(x - np.median(x))) / 0.6745)
    if sig_scale <= 0 or sigma < 1e-4 * sig_scale:
        info.update({
            "applied": False,
            "sigma": sigma,
            "threshold": thr,
            "sigma_over_signal_scale": (sigma / sig_scale) if sig_scale > 0 else None,
            "reason": (
                "the universal threshold degenerates in this position of the "
                "chain: the finest detail scale spans sf/4-sf/2 Hz, which the "
                "3 Hz low-pass has already emptied, so MAD(d1) estimates an "
                "empty band rather than the noise floor and T collapses. "
                "Estimating sigma here is invalid; it would have to come from "
                "before the bandpass, which is a change to the specification "
                "and has to be measured as one."
            ),
        })
        return x, info

    out = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(out, wavelet)
    # waverec returns an even-length array, so it is one sample long on odd n.
    rec = np.asarray(rec[:n], dtype=float)

    info.update({
        "applied": True,
        "sigma": sigma,
        "threshold": thr,
        "removed_rms": float(np.sqrt(np.mean((x - rec) ** 2))),
    })
    return rec, info


def bandpass_flow(
    flow_data: np.ndarray,
    sf: float,
    denoise: bool = False,
    denoise_info: dict | None = None,
) -> np.ndarray:
    """
    3rd-order Butterworth bandpass 0.05–3 Hz, zero-phase.

    Retains the raw waveform (no envelope) for zero-crossing breath
    segmentation.  The 3 Hz upper cutoff removes snoring vibrations
    (50–200 Hz) before any downstream flattening-index computation.

    Parameters
    ----------
    denoise      : v0.20.0. Apply :func:`denoise_flow_wavelet` to the filtered
                   waveform. Default ``False`` = the pre-v0.20.0 result,
                   bit-for-bit.
    denoise_info : optional dict, updated in place with the provenance of the
                   denoising so the caller can put it in ``meta``.

    Denoising lives *here*, rather than at each caller, on purpose. Four
    separate consumers read the bandpassed flow — the amplitude envelope, the
    MMSD apnea validation, the breath detector, and the boundary snapping — and
    the specification requires all of them to see the same signal, or the
    envelope events and the per-breath evidence stop describing one recording.
    Putting the switch in the shared function makes that structural instead of
    a thing to remember; `tests/test_wavelet_denoise.py` fails if a fifth
    consumer is added without it.
    """
    nyq = sf / 2
    lo  = max(0.05 / nyq, 0.001)
    hi  = min(3.0  / nyq, 0.99)
    b, a = sp_signal.butter(3, [lo, hi], btype="band")
    filtered = sp_signal.filtfilt(b, a, flow_data)
    if not denoise:
        return filtered
    filtered, info = denoise_flow_wavelet(filtered, sf)
    if denoise_info is not None:
        denoise_info.update(info)
    return filtered


# ---------------------------------------------------------------------------
# Amplitude envelopes  (the `envelope_method` / `envelope_fs` profile axis)
# ---------------------------------------------------------------------------
#
# `scipy.signal.hilbert` FFTs the WHOLE signal at once and returns complex128
# (16 B/sample). One 8 h channel at 256 Hz is ~7.4 M samples, so the transform
# alone peaks around half a gigabyte per channel, on top of the preloaded raw
# data. Multiply that by the worker count of a sweep and the machine swaps.
#
# The three alternatives below trade that peak for a different envelope. They
# are NOT interchangeable: each one shifts event boundaries, and therefore the
# measured reductions, in its own way. That is why they sit on a profile axis
# with `hilbert` as the default rather than being swapped in globally --
# including `hilbert_chunked`, which was specified as a free implementation
# detail but measured as a behaviour change (see its docstring).

ENVELOPE_METHODS = ("hilbert", "hilbert_chunked", "rectify_lowpass",
                    "breath_amplitude")


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Amplitude envelope via the analytic signal over the full array."""
    return np.abs(sp_signal.hilbert(x))


def hilbert_envelope_chunked(
    x: np.ndarray,
    sf: float,
    chunk_s: float = 1800.0,
    pad_s: float = 60.0,
) -> np.ndarray:
    """
    Analytic-signal envelope computed blockwise with overlap-discard.

    Peak memory drops from "one complex128 copy of the whole night" to "one
    complex128 copy of ``chunk_s + 2 * pad_s`` seconds" -- roughly 30 MB per
    channel instead of ~500 MB, independent of recording length.

    NOT identical to :func:`hilbert_envelope`
    ----------------------------------------
    This was specified as a free optimisation, on the reasoning that the
    Hilbert transform is a linear convolution and a generous pad therefore
    reproduces the full-array result up to numerical noise. Measurement says
    otherwise, and the difference is structural rather than a tuning problem:

      * The Hilbert kernel is 1/(pi t). It decays as 1/t and has no compact
        support, so any finite pad truncates a tail that never vanishes. On an
        8 h synthetic breathing signal at 256 Hz the interior residual sits
        around 1e-4 of the p95 envelope and does NOT converge as the pad grows
        (60 s -> 2.8e-4, 120 s -> 1.1e-3, 600 s -> 5.5e-5): it is a floor, not
        a decay curve. Widening the pad buys memory back without buying
        accuracy.
      * At the very first and last samples the two disagree by ~30 % of the
        p95 envelope. `scipy.signal.hilbert` is an FFT, so the full-array
        version wraps the end of the night onto the beginning; a chunked
        version wraps within its own window instead. Neither is more correct,
        but they are not the same number.

    1e-4 is far below any clinical threshold, yet the envelope is compared
    against a baseline sample by sample, so it can still move a boundary and,
    at the margin, flip a borderline event. That is a behaviour change, and it
    belongs on the profile axis with the default untouched.

    Note that the golden harness cannot referee this: its cases are 600 s at
    32 Hz, far shorter than one chunk, so chunking never engages and every
    case passes unchanged. Absence of a golden diff is not evidence here --
    :mod:`tests.test_envelope_methods` uses a signal long enough to cross
    several chunk boundaries instead.

    Parameters
    ----------
    chunk_s : block length in seconds.
    pad_s   : overlap discarded on each side. The slowest component that
              survives the upstream bandpass has a 20 s period (0.05 Hz), so
              60 s covers three of them. Larger pads are not better, per the
              measurement above.
    """
    n     = len(x)
    chunk = max(1, int(chunk_s * sf))
    pad   = max(0, int(pad_s * sf))

    # Short signal: one block is the whole array, so this IS the full transform.
    if n <= chunk + 2 * pad:
        return hilbert_envelope(x)

    out = np.empty(n, dtype=float)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)          # last block may be short: keep it
        lo   = max(0, start - pad)
        hi   = min(n, stop + pad)
        seg  = np.abs(sp_signal.hilbert(x[lo:hi]))
        out[start:stop] = seg[start - lo: start - lo + (stop - start)]
    return out


def rectify_lowpass_envelope(
    x: np.ndarray,
    sf: float,
    cutoff_hz: float = 0.5,
) -> np.ndarray:
    """
    Envelope by AM demodulation: ``|x|`` through a zero-phase lowpass.

    O(n), streamable, negligible memory. The price is a genuinely different
    envelope: full-wave rectification puts energy at twice the breathing
    frequency, so what survives the lowpass carries a ripple at the breathing
    rate and the flanks of a reduction are shaped by the filter rather than by
    the analytic signal. Event boundaries -- and therefore the measured
    percentage reduction -- move.

    ``|x|`` of a sinusoid has mean 2A/pi, so the result is scaled back by pi/2
    to land on the same amplitude as the analytic envelope. Without that the
    envelope sits ~36 % low against an unchanged baseline percentile, and every
    reduction threshold fires early.

    Reference: standard AM demodulation; e.g. Oppenheim & Schafer,
    *Discrete-Time Signal Processing*, on envelope detection.
    """
    nyq = sf / 2
    hi  = min(cutoff_hz / nyq, 0.99)
    b, a = sp_signal.butter(2, hi, btype="low")
    return sp_signal.filtfilt(b, a, np.abs(x)) * (np.pi / 2.0)


def breath_amplitude_envelope(
    x: np.ndarray,
    sf: float,
    min_breath_s: float = 1.0,
    max_breath_s: float = 15.0,
) -> np.ndarray:
    """
    Envelope from per-breath peak-to-trough amplitudes, interpolated.

    Each detected breath contributes one point -- half its peak-to-trough
    excursion, placed at the breath midpoint -- and the points are joined by
    linear interpolation onto the original sample grid. Memory is negligible:
    one value per breath instead of one complex number per sample.

    Why this is a method change and not an optimisation
    ---------------------------------------------------
    AASM defines reductions per breath, not against a continuous envelope, so
    a breath-granular envelope is arguably closer to the rule. It also makes
    event boundaries fall on breaths by construction. But it moves the
    sensitivity into peak detection at low amplitudes -- which is exactly when
    events happen -- and when breaths are missed the envelope interpolates
    straight across the gap instead of dipping. On a signal where detection
    fails outright this returns zeros rather than a wrong answer.

    Origin
    ------
    The approach (per-breath peak detection, then interpolation to a
    continuous envelope) is described for CAISR's respiratory module. This is
    an independent implementation written from that description under
    ``CAISR_CLEANROOM_BRIEF.md``: no CAISR code, names, or constants were
    used. Linear interpolation is used here rather than cubic -- a cubic fit
    can overshoot below zero between a normal breath and an apneic one, and a
    negative envelope has no meaning against a baseline ratio.

    Notes
    -----
    Expects an already bandpass-filtered waveform (zero-crossing segmentation
    needs the raw oscillation, not an envelope).
    """
    n = len(x)
    breaths = detect_breaths(x, sf, min_breath_s=min_breath_s,
                             max_breath_s=max_breath_s)
    if len(breaths) < 2:
        # Too few breaths to interpolate between. Returning zeros is honest:
        # a flat non-zero envelope would read downstream as steady breathing.
        return np.zeros(n, dtype=float)

    centres = np.array([b["mid"] for b in breaths], dtype=float)
    # Half the peak-to-trough excursion is the amplitude of the equivalent
    # sinusoid, which is what the analytic envelope reports.
    amps = np.array([b["amplitude"] for b in breaths], dtype=float) / 2.0

    order   = np.argsort(centres)
    centres = centres[order]
    amps    = amps[order]
    # np.interp holds the first/last value beyond the ends, which is the right
    # behaviour here: no extrapolated trend into a region with no breaths.
    return np.interp(np.arange(n, dtype=float), centres, amps)


def _decimated_hilbert_envelope(
    x: np.ndarray,
    sf: float,
    envelope_fs: float,
    chunked: bool = False,
    chunk_s: float = 1800.0,
    pad_s: float = 60.0,
) -> np.ndarray:
    """
    Decimate to ``envelope_fs``, take the envelope there, interpolate back.

    Everything the upstream bandpass passes sits below 3 Hz, so a 10 Hz
    envelope rate keeps Nyquist comfortably above the band while cutting the
    transform by a factor of 20-50.

    The result is returned on the ORIGINAL sample grid. That is deliberate:
    every downstream consumer indexes the envelope in original samples, and
    handing them a second sample rate would push an fs-aware conversion into
    the baseline, the boundary refinement, and the breath coupling all at
    once. The saving is in the transform, not in the returned array.

    The decimation anti-alias filter and the coarser sample raster both leave
    a fingerprint, so this is a small but real deviation -- hence a profile
    field rather than a default.
    """
    if envelope_fs is None or envelope_fs <= 0 or envelope_fs >= sf:
        return (hilbert_envelope_chunked(x, sf, chunk_s, pad_s) if chunked
                else hilbert_envelope(x))

    factor = int(sf // envelope_fs)
    if factor < 2:
        return (hilbert_envelope_chunked(x, sf, chunk_s, pad_s) if chunked
                else hilbert_envelope(x))

    # ftype="fir" with zero_phase keeps the envelope aligned in time; an IIR
    # decimator would shift it, and a shifted envelope moves every boundary.
    small    = sp_signal.decimate(x, factor, ftype="fir", zero_phase=True)
    sf_small = sf / factor
    env_small = (hilbert_envelope_chunked(small, sf_small, chunk_s, pad_s)
                 if chunked else hilbert_envelope(small))

    # Map decimated index k back to original index k * factor.
    src = np.arange(len(env_small), dtype=float) * factor
    return np.interp(np.arange(len(x), dtype=float), src, env_small)


def compute_envelope(
    filtered: np.ndarray,
    sf: float,
    method: str = "hilbert",
    envelope_fs: float | None = None,
) -> np.ndarray:
    """
    Amplitude envelope of an already-filtered signal, per the profile axis.

    Parameters
    ----------
    filtered    : bandpass-filtered waveform (NOT an envelope).
    method      : one of :data:`ENVELOPE_METHODS`. ``"hilbert"`` is the
                  default and reproduces every result published to date.
    envelope_fs : if set, decimate to this rate before the analytic-signal
                  transform. Ignored by the two methods that never build one
                  (``rectify_lowpass``, ``breath_amplitude``) -- both are
                  already O(n) in memory, so decimating would cost accuracy
                  for nothing.

    Raises
    ------
    ValueError
        On an unknown method. Falling back to the default would make a typo in
        a profile silently score with different rules than its name claims.
    """
    if method not in ENVELOPE_METHODS:
        raise ValueError(
            f"unknown envelope_method {method!r}; expected one of "
            f"{', '.join(ENVELOPE_METHODS)}")

    if method == "rectify_lowpass":
        return rectify_lowpass_envelope(filtered, sf)
    if method == "breath_amplitude":
        return breath_amplitude_envelope(filtered, sf)

    chunked = (method == "hilbert_chunked")
    if envelope_fs:
        return _decimated_hilbert_envelope(filtered, sf, envelope_fs,
                                           chunked=chunked)
    return (hilbert_envelope_chunked(filtered, sf) if chunked
            else hilbert_envelope(filtered))


def preprocess_flow(
    flow_data: np.ndarray,
    sf: float,
    is_nasal_pressure: bool = False,
    envelope_method: str = "hilbert",
    envelope_fs: float | None = None,
    denoise: bool = False,
    denoise_info: dict | None = None,
) -> np.ndarray:
    """
    Full flow preprocessing: [linearize ->] bandpass -> envelope -> 1 s smooth.

    Parameters
    ----------
    flow_data         : raw signal array
    sf                : sample rate (Hz)
    is_nasal_pressure : if True, apply sqrt-linearization before filtering
                        (AASM Rule 3; use for hypopnea channel only)
    envelope_method   : see :func:`compute_envelope`. Default reproduces the
                        pre-v0.19.0 behaviour exactly.
    envelope_fs       : see :func:`compute_envelope`.
    """
    if is_nasal_pressure:
        flow_data = linearize_nasal_pressure(flow_data)
    filtered = bandpass_flow(flow_data, sf, denoise=denoise,
                             denoise_info=denoise_info)
    envelope = compute_envelope(filtered, sf, envelope_method, envelope_fs)
    win      = max(1, int(sf))                        # 1-second smoothing
    return np.convolve(envelope, np.ones(win) / win, mode="same")


def preprocess_effort(
    effort_data: np.ndarray,
    sf: float,
    envelope_method: str = "hilbert",
    envelope_fs: float | None = None,
) -> np.ndarray:
    """
    Thorax / abdomen RIP preprocessing: bandpass 0.05–2 Hz -> amplitude envelope.

    The envelope axis applies here too, with one exception: ``breath_amplitude``
    falls back to the analytic signal. That method segments breaths by
    zero-crossing on a flow waveform; a RIP belt measures excursion, and its
    zero-crossings are a property of the belt's own baseline drift rather than
    of the breath. Effort would silently become the noisiest channel in the
    montage exactly where the flow envelope got sharper.
    """
    nyq = sf / 2
    lo  = max(0.03 / nyq, 0.001)
    hi  = min(2.0  / nyq, 0.99)
    b, a = sp_signal.butter(3, [lo, hi], btype="band")
    filtered = sp_signal.filtfilt(b, a, effort_data)
    method = ("hilbert" if envelope_method == "breath_amplitude"
              else envelope_method)
    envelope = compute_envelope(filtered, sf, method, envelope_fs)
    win      = max(1, int(sf * 2))
    return np.convolve(envelope, np.ones(win) / win, mode="same")


# ---------------------------------------------------------------------------
# Baseline estimation
# ---------------------------------------------------------------------------

def compute_dynamic_baseline(
    flow_env: np.ndarray,
    sf: float,
    window_s: int = BASELINE_WINDOW_S,
    percentile: float = 95.0,
) -> np.ndarray:
    """
    Per-sample dynamic baseline via a sliding-window percentile.

    Sampled every 10 s then linearly interpolated: ~2 500× faster than
    per-sample computation (2 880 vs 7.4 M iterations for 8 h @ 256 Hz).

    Segments at or below 30 % of the local high-percentile envelope are
    excluded from the baseline computation to prevent apnea periods from
    suppressing it.

    Parameters
    ----------
    window_s   : sliding-window length in seconds (default 300 = 5 min).
                 Shorter windows (e.g. 120) track local quiet-breathing
                 more responsively at the cost of more variable baseline.
    percentile : envelope percentile used as the baseline anchor. The
                 default 95 captures the upper end of breathing
                 amplitude; 80-90 reduces inflation by transient peaks
                 (Lazazzera et al. 2020; Koley & Dey 2014). Profile-
                 tunable as of v0.5.1.
    """
    win  = int(window_s * sf)
    n    = len(flow_env)
    step = max(1, int(sf * 10))          # anchor every 10 s

    sample_points   = np.arange(0, n, step)
    baseline_sparse = np.empty(len(sample_points))

    for idx, center in enumerate(sample_points):
        start = max(0, center - win // 2)
        end   = min(n, center + win // 2)
        seg   = flow_env[start:end]
        anchor = np.percentile(seg, percentile)
        stable = seg[seg > 0.30 * anchor]
        baseline_sparse[idx] = (
            np.percentile(stable, percentile) if len(stable) > 10 else anchor
        )

    baseline = np.interp(np.arange(n), sample_points, baseline_sparse)
    return np.maximum(baseline, 1e-6)



def compute_pre_event_baseline(
    onset_s: float,
    breaths: list,
    sf: float,
    window_s: float = 120.0,
    stability_cv: float = 0.25,
    n_largest: int = 3,
    hypno: list | None = None,
    epoch_len_s: float = 30.0,
) -> float | None:
    """AASM-conforme pre-event baseline (v0.12.3+, achter ``baseline_mode``).

    De AASM meet de >=30% daling ten opzichte van de ademhaling **vóór** het
    event. ``compute_dynamic_baseline`` gebruikt een *gecentreerd* venster van
    5 minuten en neemt daarmee de recovery-hyperpnea ná het event mee, wat de
    baseline verhoogt en de gemeten daling verkleint. Fix 1 en Fix 6 zijn
    patches op precies dat ontwerp.

    Neemt de ademteugen in ``window_s`` seconden vóór ``onset_s``:
      * stabiele ademhaling (CV < ``stability_cv``) -> gemiddelde amplitude
        van die ademteugen;
      * anders -> gemiddelde van de ``n_largest`` grootste amplitudes, de
        operationalisering die de AASM aangeeft wanneer stabiele ademhaling
        niet te bepalen is.

    Returns
    -------
    float, of ``None`` wanneer het venster geen bruikbare ademhaling bevat.
    ``None`` is een expliciet signaal aan de aanroeper om terug te vallen op
    de rolling baseline — beter dan een verzonnen getal, want aan het begin
    van een opname of na een lange gap is er domweg geen pre-event ademhaling.

    Randgevallen (bewust afgehandeld, niet impliciet):
      * te weinig ademteugen in het venster -> ``None``
      * venster volledig in wake -> ``None`` wanneer ``hypno`` meegegeven is;
        wake-ademhaling is geen geldige referentie voor een slaapevent
      * amplitudes <= 0 -> genegeerd
    """
    if not breaths or onset_s is None:
        return None

    lo = max(0.0, float(onset_s) - float(window_s))
    hi = float(onset_s)

    amps, in_sleep = [], 0
    for b in breaths:
        b_on = b.get("onset_s")
        if b_on is None or not (lo <= b_on < hi):
            continue
        amp = b.get("amplitude")
        if amp is None or not np.isfinite(amp) or amp <= 0:
            continue
        if hypno is not None:
            ep = int(b_on // epoch_len_s)
            if 0 <= ep < len(hypno) and hypno[ep] in ("N1", "N2", "N3", "R"):
                in_sleep += 1
            else:
                continue
        amps.append(float(amp))

    # Minimaal een handvol ademteugen; onder ~4 is een CV betekenisloos.
    if len(amps) < 4:
        return None
    if hypno is not None and in_sleep == 0:
        return None

    a = np.asarray(amps, dtype=float)
    mean = float(a.mean())
    if mean <= 0:
        return None

    cv = float(a.std() / mean)
    if cv < float(stability_cv):
        return mean
    k = max(1, min(int(n_largest), a.size))
    return float(np.sort(a)[-k:].mean())


def compute_anchor_baseline(
    flow_env: np.ndarray,
    sf: float,
    hypno: list,
    events: list | None = None,
    artifact_epochs: list | None = None,
    min_stable_epochs: int = 6,
) -> dict:
    """
    v0.8.11 — Baseline Anchoring.

    Zoek periodes van stabiele, event-vrije N2-slaap en bereken het
    absolute RMS-vermogen als "Gouden Standaard Basislijn" voor deze
    specifieke patiënt.

    Dit lost het probleem op van de mond-ademer: als de neusbril-flow
    structureel lager ligt dan de anker-basislijn (>40% daling), geeft
    dit een waarschuwing en verlaagt het de hypopnea-confidence.

    Parameters
    ----------
    flow_env        : preprocessed flow envelope
    sf              : sample rate
    hypno           : slaapstadia per epoch
    events          : gedetecteerde events (voor event-vrij masker)
    artifact_epochs : te vermijden epochs

    Returns
    -------
    dict met:
        anchor_value        : float — absolute RMS anker-basislijn
        anchor_epochs_used  : int   — aantal N2 epochs gebruikt
        anchor_reliable     : bool  — True als >= min_stable_epochs
        anchor_ratio        : float — verhouding huidig signaal / anker
        mouth_breathing_suspected : bool
    """
    artifact_set  = set(artifact_epochs or [])
    spe           = int(sf * EPOCH_LEN_S)
    n             = len(flow_env)

    # Bouw event-masker: samples binnen 30s van een event worden uitgesloten
    event_mask = np.zeros(n, dtype=bool)
    for ev in (events or []):
        onset  = int(ev.get("onset_s", 0) * sf)
        end    = int((ev.get("onset_s", 0) + ev.get("duration_s", 0)) * sf)
        margin = int(30 * sf)
        event_mask[max(0, onset - margin) : min(n, end + margin)] = True

    # Zoek stabiele N2-epochs zonder events en artefacten
    anchor_rms_values: list[float] = []
    for ep_i, stage in enumerate(hypno):
        if stage not in ("N2", 2):
            continue
        if ep_i in artifact_set:
            continue
        sl = ep_i * spe
        el = min(sl + spe, n)
        if np.any(event_mask[sl:el]):
            continue
        seg = flow_env[sl:el]
        if len(seg) < spe // 2:
            continue
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms > 1e-6:
            anchor_rms_values.append(rms)

    if len(anchor_rms_values) < min_stable_epochs:
        return {
            "anchor_value":             None,
            "anchor_epochs_used":       len(anchor_rms_values),
            "anchor_reliable":          False,
            "anchor_ratio":             None,
            "mouth_breathing_suspected": False,
        }

    # Gebruik mediaan (robuust tegen uitschieters)
    anchor_val = float(np.median(anchor_rms_values))

    # Huidig gemiddeld signaalvermogen over gehele opname
    valid = flow_env[flow_env > 1e-6]
    current_rms = float(np.sqrt(np.mean(valid ** 2))) if len(valid) > 0 else anchor_val
    anchor_ratio = current_rms / max(anchor_val, 1e-9)

    # Mond-ademer: signaal structureel >40% lager dan anker
    mouth_breathing_suspected = anchor_ratio < 0.60

    return {
        "anchor_value":              safe_r(anchor_val, 4),
        "anchor_epochs_used":        len(anchor_rms_values),
        "anchor_reliable":           True,
        "anchor_ratio":              safe_r(anchor_ratio, 3),
        "mouth_breathing_suspected": mouth_breathing_suspected,
    }

def compute_stage_baseline(
    flow_env: np.ndarray,
    sf: float,
    hypno: list,
    artifact_epochs: list | None = None,
    dynamic_baseline: np.ndarray | None = None,
) -> np.ndarray:
    """
    Stage-specific baseline: separate 90th-percentile estimates for
    NREM and REM (REM is physiologically more variable).

    Falls back to the dynamic baseline when insufficient stage data exist.
    A 5 s cosine-ramp smooths transitions between adjacent stage epochs.

    Parameters
    ----------
    dynamic_baseline : voorberekende dynamische basislijn (optioneel).
        Als opgegeven wordt compute_dynamic_baseline() niet opnieuw aangeroepen.
    """
    artifact_set = set(artifact_epochs or [])
    spe = int(sf * EPOCH_LEN_S)
    n   = len(flow_env)
    n_epochs = len(hypno)
    hypno_arr = np.array(hypno)
    ep_indices = np.arange(n_epochs)

    artifact_mask_ep = np.zeros(n_epochs, dtype=bool)
    if artifact_set:
        valid_art = [i for i in artifact_set if i < n_epochs]
        if valid_art:
            artifact_mask_ep[valid_art] = True

    stage_bl: dict[str, float] = {}
    for stage in ("N1", "N2", "N3", "R"):
        stage_ep = (hypno_arr == stage) & ~artifact_mask_ep
        if not stage_ep.any():
            continue
        sample_mask = np.repeat(stage_ep, spe)[:n]
        samples = flow_env[sample_mask]
        if len(samples) > int(sf * 30):
            p30    = float(np.percentile(samples, 30))
            stable = samples[samples > p30]
            if len(stable) > 10:
                stage_bl[stage] = float(np.percentile(stable, 90))

    global_bl = dynamic_baseline if dynamic_baseline is not None else compute_dynamic_baseline(flow_env, sf)

    if not stage_bl:
        return global_bl.copy()

    ep_values   = np.empty(n_epochs)
    use_global  = np.ones(n_epochs, dtype=bool)
    for stage, val in stage_bl.items():
        mask = hypno_arr == stage
        ep_values[mask] = val
        use_global[mask] = False

    if use_global.any():
        ug_idx = ep_indices[use_global]
        ep_values[use_global] = np.array([
            float(np.median(global_bl[i*spe : min((i+1)*spe, n)]))
            for i in ug_idx
        ])

    baseline = np.repeat(ep_values, spe)[:n]
    win      = max(1, int(sf * 5))
    baseline = np.convolve(baseline, np.ones(win) / win, mode="same")
    return np.maximum(baseline, 1e-6)


# ---------------------------------------------------------------------------
# Position-aware baseline reset
# ---------------------------------------------------------------------------

def detect_position_changes(
    pos_data: np.ndarray,
    sf: float,
    min_stable_s: float = 30.0,
) -> list[dict]:
    """
    Detect body-position changes in the position channel.

    Uses a median filter to remove momentary flicker, then requires that the
    new position be stable for at least *min_stable_s* seconds before the
    change is recorded.

    Returns
    -------
    list of dicts: {sample, time_s, from, to}
    """
    if pos_data is None or len(pos_data) < int(sf * 60):
        return []

    pos_q      = np.round(pos_data).astype(int)
    win        = max(3, int(sf * 5)) | 1           # odd window for median
    pos_smooth = median_filter(pos_q, size=win)

    changes  = []
    prev_pos = pos_smooth[0]

    for i in range(1, len(pos_smooth)):
        if pos_smooth[i] != prev_pos:
            check_end   = min(i + int(min_stable_s * sf), len(pos_smooth))
            stable_seg  = pos_smooth[i:check_end]
            if (
                len(stable_seg) > int(sf * 10) and
                np.sum(stable_seg == pos_smooth[i]) > 0.8 * len(stable_seg)
            ):
                changes.append({
                    "sample": i,
                    "time_s": i / sf,
                    "from":   int(prev_pos),
                    "to":     int(pos_smooth[i]),
                })
                prev_pos = pos_smooth[i]

    return changes


def reset_baseline_at_position_changes(
    baseline: np.ndarray,
    flow_env: np.ndarray,
    sf: float,
    pos_changes: list[dict],
    recalc_window_s: float = 60.0,
) -> np.ndarray:
    """
    After a body-position change, recompute the local baseline from the
    first *recalc_window_s* seconds in the new position.

    A 10 s linear ramp smooths the transition to prevent artefactual
    step-changes in the normalised flow signal.
    """
    if not pos_changes:
        return baseline

    result = baseline.copy()
    n      = len(flow_env)

    for change in pos_changes:
        sample      = change["sample"]
        recalc_end  = min(sample + int(recalc_window_s * sf), n)
        seg         = flow_env[sample:recalc_end]
        if len(seg) < int(sf * 10):
            continue

        stable   = seg[seg > np.percentile(seg, 30)]
        new_bl   = float(
            np.percentile(stable, 90) if len(stable) > 10
            else np.percentile(seg, 90)
        )

        ramp_samp = min(int(sf * 10), recalc_end - sample)
        for i in range(ramp_samp):
            idx = sample + i
            if idx < n:
                alpha       = i / ramp_samp
                result[idx] = (1 - alpha) * result[idx] + alpha * new_bl

        result[sample + ramp_samp : recalc_end] = new_bl

    return result
