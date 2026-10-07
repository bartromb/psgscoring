"""Voorwaardelijke vereniging van apneus op twee flowsensoren (opt-in).

Onder `dual_sensor_apnea` is de tweede sensor zuiver additief: een apneu die
alleen de neusdruk ziet blijft staan, ook als de thermistor tijdens dat
venster gewoon doorademt (mond/canule). Gemeten op 20 eigen PSG's
(docs/breath_vs_breath_dual_eigen_psg_20261007.md): op R18 kwamen er zo 426
druk-apneus bij, de meeste zonder desaturatie of arousal.

`dual_sensor_confirmation = "thermistor_or_consequence"` laat een
alleen-druk-apneu alleen staan als de thermistor minstens `drop_min` zakt
(0,72 — de gekalibreerde thermistordrempel), óf er een desaturatie ≥ 3 %
is, óf — na de arousalstap — een arousal in [t0, t1 + 15 s]. Default None:
gedragsidentiek.
"""
import logging

import numpy as np

from psgscoring.postprocess import (confirm_single_sensor_apneas,
                                    flow_envelope, resolve_pending_apneas)

SF = 32.0
T = np.arange(0, 600, 1 / SF)
_RNG = np.random.default_rng(1)


def _breathing(amp_fn):
    """Ademsignaal op 0,25 Hz met een tijdsafhankelijke amplitude."""
    return amp_fn(T) * np.sin(2 * np.pi * 0.25 * T) + 0.01 * _RNG.normal(size=T.size)


def _event(t0, dur, corr, desat=None, typ="obstructive"):
    e = {"type": typ, "onset_s": float(t0), "duration_s": float(dur),
         "corroboration": corr}
    if desat is not None:
        e["desaturation_pct"] = desat
    return e


def _confirm(events, therm, usable=True):
    env = flow_envelope(therm, SF)
    return confirm_single_sensor_apneas(
        events, therm_env=env, sf_therm=SF, thermistor_usable=usable,
        drop_min=0.72, desat_pct=3.0)


def test_envelope_drop_measures_the_thermistor_window():
    """Thermistor zakt 80 % in [300, 320): de maat moet dat zien, en de
    doorademende controle niet."""
    therm = _breathing(lambda t: np.where((t >= 300) & (t < 320), 0.2, 1.0))
    ev, cc = _confirm([_event(300, 20, "pressure_only")], therm)
    assert ev[0]["dual_confirmation"] == "thermistor"
    assert 0.7 <= ev[0]["thermistor_drop"] <= 0.9
    assert cc["n_thermistor"] == 1 and cc["n_pending"] == 0


def test_pressure_only_with_breathing_thermistor_is_pending():
    therm = _breathing(lambda t: np.ones_like(t))
    ev, cc = _confirm([_event(300, 20, "pressure_only")], therm)
    assert ev[0]["dual_confirmation"] == "pending"
    assert ev[0]["thermistor_drop"] < 0.3
    assert cc["n_pending"] == 1


def test_desaturation_confirms_without_thermistor_drop():
    therm = _breathing(lambda t: np.ones_like(t))
    ev, cc = _confirm([_event(300, 20, "pressure_only", desat=3.5)], therm)
    assert ev[0]["dual_confirmation"] == "desat"
    assert cc["n_desat"] == 1


def test_both_and_usable_thermistor_only_are_untouched():
    therm = _breathing(lambda t: np.ones_like(t))
    ev, cc = _confirm([_event(100, 15, "both"), _event(300, 20, "thermistor_only")],
                      therm, usable=True)
    assert all("dual_confirmation" not in e for e in ev)
    assert cc["n_pending"] == 0


def test_thermistor_only_on_rejected_thermistor_needs_consequence():
    therm = _breathing(lambda t: np.ones_like(t))
    ev, cc = _confirm([_event(300, 20, "thermistor_only"),
                       _event(400, 20, "thermistor_only", desat=4.0)],
                      therm, usable=False)
    assert ev[0]["dual_confirmation"] == "pending"
    assert ev[1]["dual_confirmation"] == "desat"


def test_pending_is_kept_by_arousal_in_window_and_dropped_otherwise():
    events = [_event(300, 20, "pressure_only"), _event(400, 20, "pressure_only"),
              _event(500, 20, "both")]
    for e in events[:2]:
        e["dual_confirmation"] = "pending"
    kept, dropped, cc = resolve_pending_apneas(events, arousal_onsets=[330.0],
                                               window_s=15.0)
    assert [e["onset_s"] for e in kept] == [300.0, 500.0]
    assert kept[0]["dual_confirmation"] == "arousal"
    assert [e["onset_s"] for e in dropped] == [400.0]
    assert cc["n_arousal"] == 1 and cc["n_dropped"] == 1


def test_arousal_just_outside_window_does_not_confirm():
    events = [_event(300, 20, "pressure_only")]
    events[0]["dual_confirmation"] = "pending"
    kept, dropped, _ = resolve_pending_apneas(events, arousal_onsets=[335.5],
                                              window_s=15.0)
    assert kept == [] and len(dropped) == 1


def test_every_registered_profile_has_confirmation_off():
    """Default None op elk profiel: de golden-uitvoer mag niet bewegen."""
    from psgscoring.profiles import PROFILES
    assert all(p.post_processing.dual_sensor_confirmation is None
               for p in PROFILES.values())


def test_env_override_selects_policy(monkeypatch, caplog):
    from psgscoring.pipeline import _dual_sensor_confirmation
    monkeypatch.delenv("PSGSCORING_DUAL_SENSOR_CONFIRMATION", raising=False)
    assert _dual_sensor_confirmation({"DUAL_SENSOR_CONFIRMATION": None}) is None
    monkeypatch.setenv("PSGSCORING_DUAL_SENSOR_CONFIRMATION", "thermistor_or_consequence")
    assert (_dual_sensor_confirmation({"DUAL_SENSOR_CONFIRMATION": None})
            == "thermistor_or_consequence")
    monkeypatch.setenv("PSGSCORING_DUAL_SENSOR_CONFIRMATION", "off")
    assert _dual_sensor_confirmation(
        {"DUAL_SENSOR_CONFIRMATION": "thermistor_or_consequence"}) is None
    monkeypatch.setenv("PSGSCORING_DUAL_SENSOR_CONFIRMATION", "nonsense")
    with caplog.at_level(logging.WARNING, logger="psgscoring.pipeline"):
        assert _dual_sensor_confirmation({"DUAL_SENSOR_CONFIRMATION": None}) is None
    assert "PSGSCORING_DUAL_SENSOR_CONFIRMATION" in caplog.text
