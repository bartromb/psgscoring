"""Analysevenster op signaalgeldigheid voor de polygrafieroute (`psgscoring.polygraphy`).

Preregistratie `docs/waakpoort_preregistratie_20261026.md`: per 30 s-epoch `flow_dead` (IQR ≤ 2 % van
de mediane epoch-IQR van de opname, of 0), `spo2_invalid` (mediaan buiten 50–100 of < 50 % van de
samples in bereik), `sensor_off = flow_dead ∨ spo2_invalid`; venster = eerste..laatste run van ≥ 4
geldige epochs, dode runs ≥ 6 epochs binnen het venster eruit; hypnogram "N2" binnen, "W" buiten;
handmatig venster (lights off/on) overschrijft naar buiten toe. Geen tweede poortmechanisme: de
slaappoort en TST-noemer van de keten doen de rest.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring import polygraphy as pg

FS = 32.0
EP = 30.0


def _breathing(n, fs=FS, amp=1.0, seed=0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * 0.25 * t) + 0.02 * np.random.default_rng(seed).normal(size=n)


def _night(minutes=120, lead_dead_min=10, tail_spo2_min=20, mid_dead=(60, 5), apnea=(90, 40)):
    """Flow + SpO2 van `minutes` minuten: kop vlak (sensor los), staart SpO2 ongeldig,
    een dode run in het midden (min), en een 'apneu' van 40 s die NIET mag wegvallen."""
    n = int(minutes * 60 * FS)
    flow = _breathing(n)
    spo2 = np.full(n, 95.0)
    lead = int(lead_dead_min * 60 * FS)
    flow[:lead] = 0.0
    spo2[:lead] = 0.0
    tail = int(tail_spo2_min * 60 * FS)
    spo2[-tail:] = 0.0
    m0, mlen = mid_dead
    a, b = int(m0 * 60 * FS), int((m0 + mlen) * 60 * FS)
    flow[a:b] = 0.0; spo2[a:b] = 0.0
    a0, dur = apnea
    flow[int(a0 * 60 * FS):int(a0 * 60 * FS + dur * FS)] = 0.0
    return {"flow_pressure": (flow, FS), "spo2": (spo2, FS)}, int(minutes * 60 // EP)


def test_epoch_validity_flags_dead_flow_and_invalid_spo2():
    sig, n_ep = _night()
    v = pg.epoch_validity(sig, n_ep)
    assert v["flow_dead"].shape == (n_ep,) and v["spo2_invalid"].shape == (n_ep,)
    assert v["flow_dead"][:20].all()                 # kop: 10 min vlak
    assert not v["flow_dead"][25:118].any() or v["flow_dead"][120:130].all()
    assert v["spo2_invalid"][-40:].all()             # staart: 20 min SpO2 0
    assert v["flow_dead"][120:130].all()             # dode run 60–65 min
    # de 40 s-'apneu' op 90 min ligt in epoch 180: niet dood (IQR over 30 s met 40 s plat
    # over twee epochs — hoogstens één epoch volledig plat)
    assert v["flow_dead"][178:183].sum() <= 1
    assert v["sensor_off"].dtype == bool
    assert v["reference_iqr"]["flow_pressure"] > 0


def test_analysis_window_rule_b():
    sig, n_ep = _night()
    v = pg.epoch_validity(sig, n_ep)
    w = pg.analysis_window(v["sensor_off"])
    assert w["start_epoch"] == 20 and w["end_epoch"] == n_ep - 40
    assert not w["analysed"][:20].any() and not w["analysed"][-40:].any()
    assert not w["analysed"][120:130].any()          # 5 min dood binnen het venster: eruit
    assert w["analysed"][130:160].all()
    assert w["analysed"][178:183].all()              # de 40 s-apneu blijft binnen
    assert w["excluded_runs"] == [(120, 130)]


def test_short_dead_runs_inside_the_window_stay():
    sig, n_ep = _night(mid_dead=(60, 2))             # 2 min = 4 epochs < 6
    v = pg.epoch_validity(sig, n_ep)
    w = pg.analysis_window(v["sensor_off"])
    assert w["analysed"][120:124].all() and w["excluded_runs"] == []


def test_polygraphy_hypnogram_marks_outside_as_wake_with_provenance():
    sig, n_ep = _night()
    hyp, prov = pg.polygraphy_hypnogram(sig, n_ep)
    assert len(hyp) == n_ep and set(hyp) <= {"N2", "W"}
    assert hyp[:20] == ["W"] * 20 and hyp[-40:] == ["W"] * 40 and hyp[120:130] == ["W"] * 10
    assert hyp[30] == "N2"
    assert prov["method"] == "signal_validity_window"
    assert prov["window_start_s"] == 20 * EP and prov["window_end_s"] == (n_ep - 40) * EP
    assert prov["excluded_min"]["before_window"] == 10.0
    assert prov["excluded_min"]["after_window"] == 20.0
    assert prov["excluded_min"]["inside_window"] == 5.0
    assert prov["analysed_min"] == (n_ep - 20 - 40 - 10) * 0.5
    assert prov["channels"]["flow"] == ["flow_pressure"] and prov["channels"]["spo2"] is True
    assert prov["thresholds"]["dead_frac"] == 0.02 and prov["thresholds"]["min_dead_run"] == 6
    assert prov["manual_window_s"] is None


def test_manual_window_overrides_outward_only():
    sig, n_ep = _night()
    hyp, prov = pg.polygraphy_hypnogram(sig, n_ep, manual_window_s=(40 * 60, 100 * 60))
    assert hyp[:80] == ["W"] * 80                     # vóór 40 min: W, ook waar het signaal geldig is
    assert hyp[200:] == ["W"] * (n_ep - 200)
    assert hyp[120:130] == ["W"] * 10                 # het automatische venster blijft gelden
    assert hyp[81] == "N2"
    assert prov["manual_window_s"] == [2400.0, 6000.0]
    assert prov["excluded_min"]["manual"] == pytest.approx(40.0 + 20.0)


def test_all_valid_recording_keeps_everything():
    n = int(60 * 60 * FS)
    sig = {"flow_thermistor": (_breathing(n), FS), "spo2": (np.full(n, 96.0), FS)}
    hyp, prov = pg.polygraphy_hypnogram(sig, 120)
    assert hyp == ["N2"] * 120 and prov["excluded_min"]["total"] == 0.0


def test_all_dead_recording_is_all_wake_with_reason():
    n = int(60 * 60 * FS)
    sig = {"flow_pressure": (np.zeros(n), FS), "spo2": (np.zeros(n), FS)}
    hyp, prov = pg.polygraphy_hypnogram(sig, 120)
    assert hyp == ["W"] * 120 and "geen" in prov["reason"]


def test_flow_channel_with_zero_reference_is_ignored_not_fatal():
    sig, n_ep = _night()
    n = len(sig["flow_pressure"][0])
    sig["flow_thermistor"] = (np.zeros(n), FS)       # dode thermistor de hele nacht
    v = pg.epoch_validity(sig, n_ep)
    assert v["flow_dead"][:20].all() and not v["flow_dead"][30:60].any()
    _, prov = pg.polygraphy_hypnogram(sig, n_ep)
    assert prov["channels"]["flow"] == ["flow_pressure"]
    assert "flow_thermistor" in prov["channels"]["ignored"]


def test_without_spo2_only_flow_counts_and_without_flow_window_is_off():
    sig, n_ep = _night()
    only_flow = {"flow_pressure": sig["flow_pressure"]}
    hyp, prov = pg.polygraphy_hypnogram(only_flow, n_ep)
    assert hyp[:20] == ["W"] * 20 and hyp[-40:] == ["N2"] * 40   # staart was alleen SpO2
    assert prov["channels"]["spo2"] is False
    hyp2, prov2 = pg.polygraphy_hypnogram({}, n_ep)
    assert hyp2 == ["N2"] * n_ep and prov2["method"] == "none" and "geen" in prov2["reason"]


def test_from_raw_uses_channel_roles():
    import mne
    sig, n_ep = _night(minutes=60, lead_dead_min=5, tail_spo2_min=5, mid_dead=(30, 0), apnea=(40, 10))
    flow, spo2 = sig["flow_pressure"][0], sig["spo2"][0]
    info = mne.create_info(["Nasal Pressure", "SpO2", "Thorax"], FS, ch_types=["misc", "misc", "misc"])
    raw = mne.io.RawArray(np.vstack([flow, spo2, _breathing(len(flow), seed=3)]), info, verbose=False)
    hyp, prov = pg.polygraphy_hypnogram_from_raw(raw)
    assert len(hyp) == n_ep and hyp[:10] == ["W"] * 10 and hyp[-10:] == ["W"] * 10 and hyp[20] == "N2"
    assert prov["channels"]["flow"] == ["flow_pressure"] and prov["channels"]["spo2"] is True
    hyp2, _ = pg.polygraphy_hypnogram_from_raw(raw, channel_map={"flow_pressure": "Nasal Pressure", "spo2": "SpO2"})
    assert hyp2 == hyp
