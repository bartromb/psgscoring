"""
tests/test_channel_detection.py — kanaalherkenning voor de AASM-sensoren.

AASM wijst twee verschillende sensoren aan: oronasale thermistor voor
APNEUS (cessatie) en neusdruk voor HYPOPNEEËN (gevoeliger). pipeline.py
kiest ze op de rollen `flow_thermistor` en `flow_pressure`; valt er één weg,
dan schuift de detectie door naar wat er nog is.

Op NSRR (MESA/SHHS) heet de neusdruk simpelweg "Pres". Die matchte geen
enkel flow_pressure-patroon, terwijl de pulse-rol het patroon "pr" had —
substring van "pres" — en het kanaal opeiste. Gevolg op MESA:
`flow_pressure` bleef leeg, waardoor ZOWEL apneu- als hypopnee-detectie op
de thermistor terechtkwam (amplitude ~1500x kleiner dan de neusdruk), en
de pulse-rol de neusdruk kreeg in plaats van "HR".

De matching is substring-based en per rol first-match-wins, dus de VOLGORDE
binnen een patroonlijst is semantiek, geen cosmetica.
"""
import pytest

from psgscoring.constants import CHANNEL_PATTERNS
from psgscoring.utils import detect_channels

# Zoals ze werkelijk in de datasets staan.
MESA = ["EKG", "EOG-L", "EOG-R", "EMG", "EEG1", "EEG2", "EEG3", "Pres", "Flow",
        "Snore", "Thor", "Abdo", "Leg", "Therm", "Pos", "Pleth", "OxStatus",
        "SpO2", "HR", "DHR"]
PSG_IPA = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EMG chin", "EOG E1-M2",
           "EOG E2-M2", "ECG", "EMG LAT", "EMG RAT", "Resp nasal",
           "Resp abdomen", "Resp chest", "SaO2"]


def test_mesa_nasal_pressure_is_flow_pressure_not_pulse():
    ch = detect_channels(MESA)
    assert ch.get("flow_pressure") == "Pres"
    assert ch.get("flow_thermistor") == "Therm"
    assert ch.get("pulse") == "HR", "pulse hoort HR te zijn, niet de neusdruk"


def test_mesa_both_aasm_sensors_are_distinct():
    """Apneu- en hypopneesensor mogen niet op hetzelfde kanaal uitkomen."""
    ch = detect_channels(MESA)
    assert ch["flow_pressure"] != ch["flow_thermistor"]


def test_psgipa_is_unaffected():
    """Geen Pres, geen HR: de herkenning moet identiek blijven."""
    ch = detect_channels(PSG_IPA)
    assert ch.get("flow") == "Resp nasal"
    assert ch.get("flow_pressure") is None
    assert ch.get("flow_thermistor") is None
    assert ch.get("pulse") is None
    assert ch.get("spo2") == "SaO2"


def test_explicit_nasal_pressure_still_wins_over_bare_pres():
    """Staat er een expliciet benoemd kanaal, dan heeft dat voorrang."""
    ch = detect_channels(["Nasal Pressure", "Pres", "Therm"])
    assert ch["flow_pressure"] == "Nasal Pressure"


def test_pr_pattern_is_last_in_pulse():
    """
    Volgorde-invariant: "pr" is een substring van "pres" en moet achteraan
    staan, anders eist de pulse-rol de neusdruk op vóór "hr" geprobeerd is.
    """
    pulse = CHANNEL_PATTERNS["pulse"]
    assert pulse[-1] == "pr"
    assert pulse.index("hr") < pulse.index("pr")


@pytest.mark.parametrize("name", ["PR", "Pulse", "Heart Rate", "HR"])
def test_pulse_channels_still_detected(name):
    ch = detect_channels([name, "Flow"])
    assert ch.get("pulse") == name
