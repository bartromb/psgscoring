"""
ventilation.py — Ventilatory burden (airflow-deficit) metric.

Ventilatory burden (VB) quantifies the total event-associated reduction in airflow
relative to the eupneic baseline, analogous to the hypoxic burden but on the flow
signal instead of SpO2. Together with the hypoxic burden it forms a cardiovascular-
risk pair that goes beyond the AHI (which only counts events).

Method (after Labarca et al. 2023): for each scored respiratory event the airflow
deficit vs the eupneic baseline is integrated over the event; the deficits are summed
and normalized to total sleep time → %·min per hour of sleep.

NOTE: VB definitions vary between publications. This implementation integrates the
event-associated deficit on the normalized flow envelope (1.0 = eupneic baseline).
Confirm the exact parameterization against Labarca et al. 2023 before clinical use.
"""
from __future__ import annotations

import numpy as np


def compute_ventilatory_burden(flow_norm: np.ndarray | None,
                               sf_flow: float,
                               events: list,
                               tst_h: float | None) -> float | None:
    """Event-associated ventilatory (airflow) burden, %·min per hour of sleep.

    Parameters
    ----------
    flow_norm : normalized flow envelope where 1.0 = eupneic baseline
                (as produced by the pipeline's ``_compute_flow_norm``).
    sf_flow   : sampling frequency of ``flow_norm`` (Hz).
    events    : scored respiratory events (each with ``onset_s`` / ``duration_s``).
    tst_h     : total sleep time in hours (normalization denominator).

    Returns
    -------
    float | None : ventilatory burden in %·min/h, or None if it cannot be computed.
    """
    if flow_norm is None or len(flow_norm) == 0 or not tst_h or tst_h <= 0:
        return None
    n = len(flow_norm)
    total_pct_s = 0.0
    for e in events:
        o = int((e.get("onset_s") or 0.0) * sf_flow)
        end = int(((e.get("onset_s") or 0.0) + (e.get("duration_s") or 0.0)) * sf_flow)
        o = max(0, o)
        end = min(n, end)
        if end <= o:
            continue
        # fraction of eupneic baseline missing during the event (0..1), as %
        deficit = np.clip(1.0 - flow_norm[o:end], 0.0, 1.0)
        total_pct_s += float(np.sum(deficit)) / sf_flow * 100.0   # %·s
    return round(total_pct_s / 60.0 / tst_h, 2)                    # %·min / h
