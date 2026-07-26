"""
ventilation.py — Ventilatory burden (VB).

Ventilatory burden quantifies the amount of reduced airflow across the night,
independent of desaturation or arousal. As defined and validated by the Terrill/Sands
group (*Ventilatory Burden*, Am J Respir Crit Care Med 2023; 208(11):1153 & 1216,
Parekh et al.),

    VB = the proportion of overnight breaths with < 50% of the normalized
         (eupneic) breathing amplitude — i.e. the percentage of "small breaths".

It is a bounded 0–100% measure that independently predicts cardiovascular and
all-cause mortality. A commonly cited **normative value is ≈ 25% or lower**; higher
values indicate a greater ventilatory burden.

**Breath-based** implementation: each breath's amplitude is read as the *peak* of the
normalized flow envelope within that breath. ``_compute_flow_norm`` already divides the
Hilbert amplitude envelope by the adaptive (dynamic, ~95th-percentile) eupneic baseline,
so 1.0 = eupneic; a breath whose peak is < ``threshold`` is a "small breath". Sampling
the per-breath *peak* (rather than a time-fraction over the whole envelope) is essential:
the smoothed envelope dips between breaths even during normal breathing, so a raw
time-fraction over-counts heavily.

References:
  - Parekh A, et al. Ventilatory Burden … Predictive of Cardiovascular and All-Cause
    Mortality. AJRCCM 2023;208(11):1216.
  - Ventilatory Burden: Development of a New Approach. AJRCCM 2023;208(11):1153.
"""
from __future__ import annotations

import numpy as np

VB_NORMATIVE_MAX = 25.0   # ≈ normal upper bound (% of breaths), AJRCCM 2023


def compute_ventilatory_burden(flow_norm: np.ndarray | None,
                               sf_flow: float,
                               breaths: list | None,
                               hypno: list | None = None,
                               threshold: float = 0.5) -> float | None:
    """Ventilatory burden: percentage of (sleep) breaths whose peak airflow amplitude
    is below ``threshold`` of the eupneic baseline.

    Parameters
    ----------
    flow_norm : normalized flow envelope where 1.0 = eupneic baseline
                (as produced by the pipeline's ``_compute_flow_norm``).
    sf_flow   : sampling frequency of ``flow_norm`` (Hz).
    breaths   : detected breaths (each with ``onset_s`` / ``duration_s``), e.g.
                ``output["respiratory"]["_breaths"]``.
    hypno     : per-epoch sleep stages; when given, only breaths starting in a sleep
                epoch (N1/N2/N3/R) are counted. If None, all breaths are used.
    threshold : "small breath" cut-off as a fraction of the eupneic amplitude
                (default 0.5 = <50%, per the AJRCCM 2023 definition).

    Returns
    -------
    float | None : ventilatory burden in **percent (0–100)**, or None if it cannot be
                   computed. Normative ≈ ``VB_NORMATIVE_MAX`` (25%) or lower.
    """
    if flow_norm is None or len(flow_norm) == 0 or not breaths:
        return None
    fn = np.asarray(flow_norm, dtype=float)
    n = len(fn)
    sleep_eps = None
    if hypno:
        from .constants import EPOCH_LEN_S
        sleep_eps = {i for i, s in enumerate(hypno) if s in ("N1", "N2", "N3", "R")}
        epoch_len = EPOCH_LEN_S
    peaks = []
    for b in breaths:
        onset = b.get("onset_s")
        if onset is None:
            continue
        if sleep_eps is not None and int(onset // epoch_len) not in sleep_eps:
            continue
        o = max(0, int(onset * sf_flow))
        e = int((onset + (b.get("duration_s") or 0.0)) * sf_flow)
        e = min(n, e if e > o else o + 1)
        if e <= o:
            continue
        seg = fn[o:e]
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            continue
        peaks.append(float(np.max(seg)))            # breath peak amplitude / eupneic baseline
    if not peaks:
        return None
    return round(float(np.mean(np.asarray(peaks) < threshold)) * 100.0, 1)
