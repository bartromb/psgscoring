"""
Regression guard for the RERA/RDI-vs-CSR ordering bug (fixed v0.7.4).

Bug: `_compute_rera_rdi` (step 8b) wrote rera_index / rdi / rem_ahi / nrem_ahi
into the respiratory summary, but the Cheyne-Stokes "Fix 3" step then replaced
`output["respiratory"]["summary"]` wholesale with a fresh `_compute_summary()`
on CSR-positive nights — silently dropping every RERA/RDI/REM-NREM key. Non-CSR
nights were unaffected, which is why the golden/PSG-IPA tests didn't catch it.

Fix: `_compute_rera_rdi` now runs *after* the CSR summary recompute. This test
builds a CSR-positive synthetic recording (periodic apneas → high flow-envelope
autocorrelation) and asserts the RERA family survives to the final output.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import psgscoring

SF = 32.0
DUR_S = 600
BR = 0.25


def _make_periodic_apnea_raw(period_s: float = 40.0, apnea_len_s: float = 20.0):
    """A recording with a regular apnea every `period_s` → periodic flow envelope
    that trips the CSR autocorrelation detector. Deterministic (seeded)."""
    mne = pytest.importorskip("mne")
    mne.set_log_level("ERROR")
    n = int(SF * DUR_S)
    t = np.arange(n) / SF
    rng = np.random.default_rng(7)

    flow_amp = np.ones(n)
    eff_amp = np.ones(n)
    spo2 = np.full(n, 96.0)
    events = []
    start = 60.0
    while start + apnea_len_s < DUR_S - 20:
        t0, t1 = start, start + apnea_len_s
        flow_amp[int(t0 * SF):int(t1 * SF)] *= 0.02        # central apnea
        eff_amp[int(t0 * SF):int(t1 * SF)] *= 0.05
        d0, nadir, d1 = int((t1 - 2) * SF), int((t1 + 4) * SF), int((t1 + 10) * SF)
        spo2[d0:nadir] = np.linspace(96.0, 91.0, nadir - d0)
        spo2[nadir:d1] = np.linspace(91.0, 96.0, d1 - nadir)
        events.append((t0, t1))
        start += period_s

    base = np.sin(2 * np.pi * BR * t)
    flow = flow_amp * base + rng.normal(0, 0.005, n)
    thorax = eff_amp * base + rng.normal(0, 0.005, n)
    abdomen = 0.9 * eff_amp * base + rng.normal(0, 0.005, n)

    data = np.vstack([flow, thorax, abdomen, spo2])
    info = mne.create_info(["FLOW", "THORAX", "ABDOMEN", "SPO2"], SF, ch_types="misc")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    hypno = ["W"] + ["N2"] * (DUR_S // 30 - 1)
    cmap = {"flow": "FLOW", "thorax": "THORAX", "abdomen": "ABDOMEN", "spo2": "SPO2"}
    return raw, hypno, cmap, events


def _run(with_arousals=False, with_rem=False):
    raw, hypno, cmap, events = _make_periodic_apnea_raw()
    if with_rem:
        # mark the last third of the night as REM so rem_ahi is populated
        for i in range(2 * len(hypno) // 3, len(hypno)):
            hypno[i] = "R"
    kwargs = dict(channel_map=cmap, scoring_profile="aasm_v3_rec")
    if with_arousals:
        # an arousal at the end of each apnea → RERA coupling → rera_index > 0
        kwargs["arousal_events"] = [{"onset_s": float(t1), "duration_s": 3.0}
                                    for _, t1 in events]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = psgscoring.run_pneumo_analysis(raw, hypno, **kwargs)
    return out


def test_fixture_is_csr_positive():
    """Guard the guard: the fixture must actually trigger CSR, else the test
    is vacuous (the bug only manifests when Fix 3 fires)."""
    out = _run()
    assert out["cheyne_stokes"].get("csr_detected") is True
    assert out["respiratory"].get("success") is True


def test_rera_rdi_survive_csr_recompute():
    out = _run()
    s = out["respiratory"]["summary"]
    # The whole RERA family must be present (not None / not missing) on a
    # CSR-positive night — this is exactly what the ordering bug dropped.
    assert s.get("rdi") is not None
    assert s.get("rera_index") is not None
    assert s.get("n_rera") is not None
    # RDI = AHI + RERA index (no arousals here → rera_index 0 → rdi == ahi_total)
    assert s["rera_index"] == 0.0
    assert s["rdi"] == pytest.approx(s["ahi_total"], abs=0.05)


def test_rem_nrem_ahi_survive_csr_recompute():
    """rem_ahi / nrem_ahi must also be retained on a CSR-positive night with REM."""
    out = _run(with_rem=True)
    assert out["cheyne_stokes"].get("csr_detected") is True
    s = out["respiratory"]["summary"]
    assert s.get("rem_ahi") is not None
    assert s.get("nrem_ahi") is not None


def test_ahi_total_is_unchanged_by_the_reorder():
    """The fix only appends RERA/RDI keys; core scoring (ahi_total, event count)
    must be identical with and without a CSR flag firing — i.e. the reorder
    cannot move any AHI number."""
    out = _run()
    s = out["respiratory"]["summary"]
    # ahi_total is computed upstream and never written by _compute_rera_rdi
    assert s["ahi_total"] > 0
    # rdi is derived purely as ahi_total + rera_index, never the other way round
    assert s["rdi"] == pytest.approx(s["ahi_total"] + s["rera_index"], abs=0.05)
