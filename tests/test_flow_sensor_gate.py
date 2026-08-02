"""De thermistor wordt alleen apneu-sensor als hij de ademhaling volgt.

Aanleiding: v0.13.2 wees die rol toe zodra het kanaal herkend werd. Op een
montage waar de thermistor aanwezig is maar geen bruikbaar ademsignaal draagt
zakte het aantal apneus op een echte opname van 93 naar 0, en verschoof de
uitkomst van matig CSAS — bevestigd door menselijke scoring — naar mild SAS.
De AASM-sensor is alleen de juiste sensor als hij werkt.
"""
import numpy as np
import pytest

from psgscoring.pipeline import _resolve_flow_channels
from psgscoring.signal_quality import (
    THERMISTOR_AGREEMENT_MIN,
    assess_flow_sensor_agreement,
)

SF = 32.0
T = np.arange(0, 900, 1 / SF)


def _breathing(rate_hz=0.25, amp=1.0, phase=0.0, seed=None):
    """Ademsignaal met wat amplitude-modulatie, zoals een echte opname."""
    rng = np.random.default_rng(seed)
    mod = 1.0 + 0.5 * np.sin(2 * np.pi * 0.01 * T)      # trage variatie
    return amp * mod * np.sin(2 * np.pi * rate_hz * T + phase) \
        + 0.05 * amp * rng.normal(size=T.size)


def _apnea(sig, start_s, dur_s, sf=SF):
    """Zet een stuk ademhaling op vrijwel nul."""
    out = sig.copy()
    out[int(start_s * sf):int((start_s + dur_s) * sf)] *= 0.02
    return out


# ─────────────────────────────────────────────────────────────────
#  De maat zelf
# ─────────────────────────────────────────────────────────────────

def test_two_sensors_on_the_same_breathing_agree():
    p = _breathing(seed=1)
    t = _breathing(amp=0.4, phase=0.3, seed=2)      # andere schaal en fase
    r = assess_flow_sensor_agreement(p, SF, t, SF)
    assert r["usable"] is True
    assert r["agreement"] > THERMISTOR_AGREEMENT_MIN


def test_noise_is_rejected():
    p = _breathing(seed=1)
    noise = np.random.default_rng(3).normal(size=T.size)
    r = assess_flow_sensor_agreement(p, SF, noise, SF)
    assert r["usable"] is False
    assert "volgt de ademhaling niet" in r["reason"]


def test_agreement_tracks_shared_apneas():
    """De maat moet meten wat we ervan nodig hebben: samen wegvallen."""
    base = _breathing(seed=4)
    both = _apnea(_apnea(base, 200, 30), 400, 30)
    only_one = _apnea(base, 200, 30)
    r_shared = assess_flow_sensor_agreement(both, SF, both * 0.3, SF)
    r_split = assess_flow_sensor_agreement(
        only_one, SF, np.random.default_rng(5).normal(size=T.size), SF)
    assert r_shared["agreement"] > r_split["agreement"]


@pytest.mark.parametrize("kwargs,expect", [
    (dict(pressure=None), "ontbreekt"),
    (dict(thermistor=None), "ontbreekt"),
])
def test_missing_channel_is_not_usable(kwargs, expect):
    args = dict(pressure=_breathing(seed=6), sf_pressure=SF,
                thermistor=_breathing(seed=7), sf_thermistor=SF)
    args.update(kwargs)
    r = assess_flow_sensor_agreement(**args)
    assert r["usable"] is False
    assert expect in r["reason"]


def test_mismatched_sample_rates_are_not_usable():
    r = assess_flow_sensor_agreement(_breathing(seed=8), SF,
                                     _breathing(seed=9), SF / 2)
    assert r["usable"] is False
    assert "samplefrequenties" in r["reason"]


# ─────────────────────────────────────────────────────────────────
#  De poort in de sensorkeuze
# ─────────────────────────────────────────────────────────────────

def _resolve(pressure, thermistor):
    out = {"meta": {}}
    ch = {"flow_pressure": "PRESS", "flow_thermistor": "THERM"}
    apnea, hypop, _a, _h = _resolve_flow_channels(
        None, SF, pressure, SF, thermistor, SF, ch, out)
    return out["meta"]["flow_channels"], apnea, hypop


def test_a_working_thermistor_becomes_the_apnea_sensor():
    p = _breathing(seed=10)
    t = _breathing(amp=0.4, phase=0.2, seed=11)
    fc, apnea, hypop = _resolve(p, t)
    assert fc["apnea_sensor"] == "THERM"
    assert fc["hypopnea_sensor"] == "PRESS"
    assert fc["dual_sensor"] is True
    assert fc["thermistor_check"]["usable"] is True
    assert apnea is not hypop


def test_an_unusable_thermistor_is_rejected_and_pressure_is_kept():
    """Dit is het geval dat 93 apneus liet verdwijnen."""
    p = _breathing(seed=12)
    noise = np.random.default_rng(13).normal(size=T.size)
    fc, apnea, hypop = _resolve(p, noise)
    assert fc["apnea_sensor"] == "PRESS", "apneus horen op de neusdruk te blijven"
    assert fc["hypopnea_sensor"] == "PRESS"
    assert fc["dual_sensor"] is False
    assert apnea is hypop


def test_a_rejected_thermistor_is_reported_not_silently_dropped():
    """Aanwezig-maar-afgekeurd is iets anders dan afwezig."""
    p = _breathing(seed=14)
    noise = np.random.default_rng(15).normal(size=T.size)
    fc, _a, _h = _resolve(p, noise)
    assert fc["thermistor_rejected"] == "THERM"
    assert fc["thermistor_check"]["usable"] is False
    assert fc["thermistor_check"]["agreement"] is not None


def test_no_thermistor_at_all_reports_nothing_rejected():
    fc, _a, _h = _resolve(_breathing(seed=16), None)
    assert fc["thermistor_rejected"] is None
    assert fc["dual_sensor"] is False
