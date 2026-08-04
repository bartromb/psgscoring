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


# ─────────────────────────────────────────────────────────────────
#  Somnomedics/SOMNO: "Flow Th." als thermistor
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["Flow Th.", "Flow Th", "flow th.", "FLOW TH."])
def test_somnomedics_thermistor_is_detected(name):
    """Zonder dit patroon blijft flow_thermistor leeg en scoort de pijplijn
    apneus op de neusdruk, terwijl de thermistor in de EDF zit."""
    ch = detect_channels(["Pressure Flow", name, "RIP Thora", "SpO2"])
    assert ch.get("flow_thermistor") == name
    assert ch.get("flow_pressure") == "Pressure Flow"


def test_somnomedics_montage_resolves_to_two_distinct_sensors():
    """De hele reden voor de fix: apneu en hypopneu op verschillende kanalen."""
    ch = detect_channels(["Snore", "Pressure Flow", "Flow Th.", "RIP Thora",
                          "RIP Abdom", "Sum RIP", "SpO2", "Pulse", "ECG II"])
    apnea = ch.get("flow_thermistor") or ch.get("flow_pressure") or ch.get("flow")
    hypop = ch.get("flow_pressure") or ch.get("flow_thermistor") or ch.get("flow")
    assert apnea == "Flow Th."
    assert hypop == "Pressure Flow"
    assert apnea != hypop


def test_pressure_flow_is_not_claimed_by_the_thermistor_role():
    """'flow th' mag 'Pressure Flow' niet opeisen — dat zou de rollen omdraaien."""
    ch = detect_channels(["Pressure Flow", "SpO2"])
    assert ch.get("flow_thermistor") is None
    assert ch.get("flow_pressure") == "Pressure Flow"


@pytest.mark.parametrize("montage,pressure,thermistor", [
    (["Pres", "Flow", "Therm", "Thor", "SpO2"], "Pres", "Therm"),          # MESA
    (["Resp nasal", "Resp abdomen", "SaO2"], None, None),                  # PSG-IPA
])
def test_existing_montages_are_unchanged(montage, pressure, thermistor):
    """PSG-IPA heeft geen thermistor; als die er ineens wel is, breekt het paper."""
    ch = detect_channels(montage)
    assert ch.get("flow_pressure") == pressure
    assert ch.get("flow_thermistor") == thermistor


# ─────────────────────────────────────────────────────────────────
#  Beenkanalen — PLM
# ─────────────────────────────────────────────────────────────────

SOMNO_PLM = ["Snore", "Pressure Flow", "Flow Th.", "RIP Thora", "RIP Abdom",
             "SpO2", "Pulse", "Pos.", "ECG II", "C3:A2", "C4:A1",
             "EMG1", "EMG2", "EMG3", "PLMl", "PLMr"]


def test_somnomedics_leg_channels_are_auto_detected():
    """"PLMl" bevat geen scheidingsteken, dus "plm l" en "plm-l" matchten niet.

    De PLM-resultaten in de rapporten kwamen daardoor uit de handmatige
    kanaalkeuze in de UI; zonder die keuze bleef de analyse leeg zonder dat
    iemand dat zag.
    """
    ch = detect_channels(SOMNO_PLM)
    assert ch.get("leg_l") == "PLMl"
    assert ch.get("leg_r") == "PLMr"


def test_the_two_legs_are_never_the_same_channel():
    """Eén kanaal dat zichzelf als beide benen aanwijst, verdubbelt de telling."""
    for montage in (SOMNO_PLM, ["PLMl", "SpO2"], ["EMG Tib L", "SpO2"]):
        ch = detect_channels(montage)
        if ch.get("leg_l") and ch.get("leg_r"):
            assert ch["leg_l"] != ch["leg_r"], montage


def test_a_single_leg_channel_does_not_fill_both_roles():
    ch = detect_channels(["PLMl", "SpO2", "Pres"])
    assert ch.get("leg_l") == "PLMl"
    assert ch.get("leg_r") is None


def test_the_somnomedics_montage_maps_completely():
    """Regressie per fabrikantmontage: de hele mapping in één assertie, zodat
    een patroon dat elders wint hier meteen zichtbaar wordt."""
    ch = detect_channels(SOMNO_PLM)
    assert {k: ch.get(k) for k in
            ("flow_pressure", "flow_thermistor", "thorax", "abdomen",
             "spo2", "pulse", "position", "snore", "ecg", "leg_l", "leg_r")} == {
        "flow_pressure":  "Pressure Flow",
        "flow_thermistor": "Flow Th.",
        "thorax":         "RIP Thora",
        "abdomen":        "RIP Abdom",
        "spo2":           "SpO2",
        "pulse":          "Pulse",
        "position":       "Pos.",
        "snore":          "Snore",
        "ecg":            "ECG II",
        "leg_l":          "PLMl",
        "leg_r":          "PLMr",
    }


def test_without_a_pulse_channel_the_nasal_pressure_is_not_used_as_heart_rate():
    """"pr" is substring van "Pressure Flow". Zonder pulse-kanaal eist de
    pulse-rol dan de neusdruk op en rekent analyze_heart_rate op een
    flowsignaal — met een "minimale hartfrequentie" als resultaat."""
    ch = detect_channels([c for c in SOMNO_PLM if c != "Pulse"])
    assert ch.get("pulse") != "Pressure Flow"
