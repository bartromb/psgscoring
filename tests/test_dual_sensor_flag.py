"""`dual_sensor` moet betekenen dat er twee sensoren gebruikt zijn.

Hij werd in ``respiratory.py`` op True gezet zodra er überhaupt een
hypopnee-kanaal was, zonder te vergelijken met het apneu-kanaal. Omdat
``_resolve_flow_channels()`` bij een ontbrekende thermistor terugvalt op
hetzelfde kanaal, stond de vlag praktisch altijd aan. De enige consument is
de rapportnoot "apneu op thermistor, hypopneu op nasale druk (AASM)", die
daardoor een methodiek claimde die niet was toegepast — ook op opnames met
één enkel flowkanaal.
"""
import numpy as np
import pytest

from psgscoring.pipeline import _resolve_flow_channels


def _resolve(pressure, thermistor, generic=None):
    """Draai alleen de kanaalresolutie en geef (meta, apnea, hypop) terug."""
    out = {"meta": {}}
    ch = {}
    if pressure is not None:
        ch["flow_pressure"] = "PRESS"
    if thermistor is not None:
        ch["flow_thermistor"] = "THERM"
    if generic is not None:
        ch["flow"] = "FLOW"
    apnea, hypop, _sfa, _sfh = _resolve_flow_channels(
        generic, 32.0, pressure, 32.0, thermistor, 32.0, ch, out)
    return out["meta"]["flow_channels"], apnea, hypop


SIG = np.ones(320)


def test_two_distinct_sensors_is_dual():
    fc, apnea, hypop = _resolve(SIG, SIG * 2)
    assert fc["dual_sensor"] is True
    assert fc["apnea_sensor"] == "THERM"
    assert fc["hypopnea_sensor"] == "PRESS"
    assert apnea is not hypop


def test_pressure_only_is_not_dual():
    """Somnomedics zonder herkende thermistor: één kanaal voor allebei."""
    fc, apnea, hypop = _resolve(SIG, None)
    assert fc["dual_sensor"] is False
    assert fc["apnea_sensor"] == "PRESS"
    assert fc["hypopnea_sensor"] == "PRESS"
    assert apnea is hypop


def test_thermistor_only_is_not_dual():
    fc, apnea, hypop = _resolve(None, SIG)
    assert fc["dual_sensor"] is False
    assert fc["apnea_sensor"] == "THERM"
    assert fc["hypopnea_sensor"] == "THERM"
    assert apnea is hypop


def test_generic_flow_only_is_not_dual():
    fc, apnea, hypop = _resolve(None, None, SIG)
    assert fc["dual_sensor"] is False
    assert apnea is hypop


@pytest.mark.parametrize("pressure,thermistor,expected", [
    (SIG, SIG, True),
    (SIG, None, False),
    (None, SIG, False),
])
def test_flag_equals_both_channels_present(pressure, thermistor, expected):
    fc, _a, _h = _resolve(pressure, thermistor)
    assert fc["dual_sensor"] is expected
