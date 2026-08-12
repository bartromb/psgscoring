"""Provenance van de flowketen: welke poort draaide, en werd er
gelineariseerd.

Geen van deze toetsen raakt een gescoorde waarde. Ze pinnen drie dingen die
tot nu toe alleen uit de broncode af te leiden waren en die een
cross-cohortvergelijking ongeldig kunnen maken zonder dat er iets aan de
scoring mankeert.
"""

from __future__ import annotations

import numpy as np
import pytest

from psgscoring.pipeline import _resolve_flow_channels
from psgscoring.utils import detect_channels


def _breathing(n=60_000, sf=100.0, f=0.25, amp=1.0, seed=0):
    t = np.arange(n) / sf
    rng = np.random.default_rng(seed)
    return amp * np.sin(2 * np.pi * f * t) + 0.01 * rng.standard_normal(n)


def _fresh_output():
    return {"meta": {}}


# ---------------------------------------------------------------------------
# 1. De poort meldt welke toets draaide en of die mocht blokkeren
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("additive,expected_mode", [
    (False, "blocking"),
    (True, "informational"),
])
def test_gate_mode_is_recorded(additive, expected_mode):
    """Onder een additief profiel meet de poort wel en weigert hij niet.

    Zonder dit veld ziet een lezer alleen `agreement: 0.31` en concludeert
    ten onrechte dat de thermistor geweigerd is.
    """
    out = _fresh_output()
    pres = _breathing(seed=1)
    therm = _breathing(seed=2)
    _resolve_flow_channels(
        None, None, pres, 100.0, therm, 100.0,
        {"flow_pressure": "Pres", "flow_thermistor": "Therm"}, out,
        additive_thermistor=additive,
    )
    gate = out["meta"]["flow_channels"]["thermistor_gate"]
    assert gate["mode"] == expected_mode
    assert gate["gate"] == "envelope_agreement"
    assert gate["threshold"] == pytest.approx(0.40)


def test_gate_name_and_threshold_follow_the_profile_choice():
    out = _fresh_output()
    pres = _breathing(seed=1)
    therm = _breathing(seed=2)
    _resolve_flow_channels(
        None, None, pres, 100.0, therm, 100.0,
        {"flow_pressure": "Pres", "flow_thermistor": "Therm"}, out,
        additive_thermistor=True, thermistor_gate="respiratory_band",
    )
    gate = out["meta"]["flow_channels"]["thermistor_gate"]
    assert gate["gate"] == "respiratory_band"
    assert gate["threshold"] == pytest.approx(0.70)


def test_gate_meta_present_on_single_channel_montage():
    """Ook zonder tweede sensor hoort het veld te bestaan, met een lege meting.

    Een ontbrekend veld en een veld met `null` zijn voor een consument niet
    hetzelfde: het eerste leest als "niet ondersteund", het tweede als
    "niets te meten".
    """
    out = _fresh_output()
    flow = _breathing(seed=3)
    _resolve_flow_channels(
        flow, 100.0, None, None, None, None, {"flow": "Flow"}, out,
    )
    fc = out["meta"]["flow_channels"]
    assert "thermistor_gate" in fc
    assert fc["thermistor_check"] is None


# ---------------------------------------------------------------------------
# 2. De NSRR-hulpkanaalnaam staat uit tenzij expliciet gevraagd
# ---------------------------------------------------------------------------

_NSRR_NEW = ["EEG", "EOG-L", "Pres", "Aux_AC", "SpO2", "HR"]
_NSRR_OLD = ["EEG", "EOG-L", "Pres", "Therm", "SpO2", "HR"]


def test_aux_ac_is_not_claimed_by_default(monkeypatch):
    """Default uit: elk bestaand profiel blijft byte-identiek."""
    monkeypatch.delenv("PSGSCORING_NSRR_AUX_AC", raising=False)
    ch = detect_channels(_NSRR_NEW)
    assert ch.get("flow_pressure") == "Pres"
    assert "flow_thermistor" not in ch


def test_aux_ac_is_claimed_when_enabled(monkeypatch):
    monkeypatch.setenv("PSGSCORING_NSRR_AUX_AC", "1")
    ch = detect_channels(_NSRR_NEW)
    assert ch.get("flow_pressure") == "Pres"
    assert ch.get("flow_thermistor") == "Aux_AC"


def test_enabling_aux_ac_does_not_disturb_the_old_montage(monkeypatch):
    """Een expliciete thermistornaam wint van het generieke hulpkanaal."""
    monkeypatch.setenv("PSGSCORING_NSRR_AUX_AC", "1")
    ch = detect_channels(_NSRR_OLD + ["Aux_AC"])
    assert ch.get("flow_thermistor") == "Therm"


def test_aux_ac_never_takes_the_pressure_channel(monkeypatch):
    monkeypatch.setenv("PSGSCORING_NSRR_AUX_AC", "1")
    ch = detect_channels(["Nasal Pressure", "Aux_AC"])
    assert ch.get("flow_pressure") == "Nasal Pressure"
    assert ch.get("flow_thermistor") == "Aux_AC"
