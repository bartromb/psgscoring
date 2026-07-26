"""
ventilation.py — Ventilatory burden (VB).

Ventilatory burden quantifies the amount of reduced airflow across the night,
independent of desaturation or arousal. As defined and validated by the Terrill/Sands
group (*Ventilatory Burden*, Am J Respir Crit Care Med 2023; 208(11):1153 & 1216),

    VB = the proportion of overnight breaths with < 50% of the normalized
         (eupneic) breathing amplitude — i.e. the percentage of "small breaths".

It is a bounded 0–100% measure that independently predicts cardiovascular and
all-cause mortality. A commonly cited **normative value is ≈ 25% or lower**; higher
values indicate a greater ventilatory burden.

This implementation is the envelope-based (time-fraction) surrogate of the
breath-based definition: it is computed on the normalized flow *envelope*
(``preprocess_flow`` gives a Hilbert amplitude envelope; ``_compute_flow_norm``
divides it by the dynamic eupneic baseline so 1.0 = eupneic), restricted to sleep.
Because the envelope is amplitude-smoothed, the fraction of sleep time below the
threshold tracks the fraction of small breaths closely.

References:
  - Ventilatory Burden: Development of a New Approach. AJRCCM 2023;208(11):1153.
  - Ventilatory Burden … Predictive of Cardiovascular and All-Cause Mortality.
    AJRCCM 2023;208(11):1216.
"""
from __future__ import annotations

import numpy as np

VB_NORMATIVE_MAX = 25.0   # ≈ normal upper bound (% of breaths), AJRCCM 2023


def compute_ventilatory_burden(flow_norm: np.ndarray | None,
                               sf_flow: float,
                               hypno: list | None = None,
                               threshold: float = 0.5) -> float | None:
    """Ventilatory burden: percentage of the (sleep) night spent below `threshold`
    of the eupneic breathing amplitude.

    Parameters
    ----------
    flow_norm : normalized flow envelope where 1.0 = eupneic baseline
                (as produced by the pipeline's ``_compute_flow_norm``).
    sf_flow   : sampling frequency of ``flow_norm`` (Hz).
    hypno     : per-epoch sleep stages; when given, VB is restricted to sleep
                epochs (N1/N2/N3/R). If None, the whole signal is used.
    threshold : "small breath" cut-off as a fraction of the eupneic amplitude
                (default 0.5 = <50%, per the AJRCCM 2023 definition).

    Returns
    -------
    float | None : ventilatory burden in **percent (0–100)**, or None if it cannot
                   be computed. Normative ≈ ``VB_NORMATIVE_MAX`` (25%) or lower.
    """
    if flow_norm is None or len(flow_norm) == 0:
        return None
    fn = np.asarray(flow_norm, dtype=float)
    n = len(fn)
    if hypno:
        from .constants import EPOCH_LEN_S
        spe = max(1, int(round(sf_flow * EPOCH_LEN_S)))
        mask = np.zeros(n, dtype=bool)
        for ep_i, stage in enumerate(hypno):
            if stage in ("N1", "N2", "N3", "R"):
                s = ep_i * spe
                if s >= n:
                    break
                mask[s:min(s + spe, n)] = True
        seg = fn[mask]
    else:
        seg = fn
    seg = seg[~np.isnan(seg)]
    if len(seg) == 0:
        return None
    return round(float(np.mean(seg < threshold)) * 100.0, 1)
