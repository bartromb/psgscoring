"""Een tweede thermistorpoort: draagt DIT kanaal ademhaling?

Aanleiding, gemeten op negen onderscheiden Somnomedics-montages. De bestaande
poort (`assess_flow_sensor_agreement`) keurde er acht af. Drie daarvan hebben
98% van hun vermogen in de ademband en delen hun ademfrequentie tot op
0,002 Hz met de neusdruk — die thermistors werken aantoonbaar.

De oorzaak is niet een verkeerd afgestelde drempel maar een maat die een
andere vraag beantwoordt. Zes synthetische signalen die ALLEMAAL op 0,25 Hz
ademen krijgen agreement tussen -0,985 en +1,000, uitsluitend bepaald door hun
trage amplitudemodulatie. Dat is wat `test_the_old_gate_ignores_whether_they
_breathe_together` hieronder vastlegt: geen bewering over de nieuwe poort,
maar de meting die hem rechtvaardigt.

Waarom de bestaande unit-test dit nooit ving: hij bouwt beide kanalen uit
dezelfde term `1 + 0,5·sin(2π·0,01·t)`, en juist die term ÍS de envelope.
Twee kanalen die hem delen correleren per constructie 1,000.

De nieuwe poort staat UIT tenzij een profiel hem kiest. Hij verschuift op elke
opname of apneus op de thermistor of op de neusdruk gescoord worden, en dus de
AHI — dat raakt mesa_shhs en de paper.
"""

import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.pipeline import _resolve_flow_channels
from psgscoring.profiles import PROFILES, get_profile
from psgscoring.signal_quality import (
    THERMISTOR_BAND_POWER_MIN,
    assess_flow_sensor_agreement,
    assess_thermistor_band_power,
    respiratory_band_power,
)

SF = 32.0
DUR_S = 1800                      # drie vensters van 600 s
T = np.arange(0, DUR_S, 1 / SF)


def _breathing(rate=0.25, amp=1.0, phase=0.0, mod_ph=0.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    mod = 1.0 + 0.5 * np.sin(2 * np.pi * 0.01 * T + mod_ph)
    return amp * mod * np.sin(2 * np.pi * rate * T + phase) \
        + noise * amp * rng.normal(size=T.size)


# ─────────────────────────────────────────────────────────────────
#  De maat
# ─────────────────────────────────────────────────────────────────

def test_a_breathing_channel_puts_its_power_in_the_breathing_band():
    m = respiratory_band_power(_breathing(seed=1), SF)
    assert m["fraction"] > THERMISTOR_BAND_POWER_MIN
    assert m["peak_hz"] == pytest.approx(0.25, abs=0.02)
    assert m["n_windows"] >= 2


def test_noise_does_not():
    m = respiratory_band_power(np.random.default_rng(2).normal(size=T.size), SF)
    assert m["fraction"] < THERMISTOR_BAND_POWER_MIN


def test_a_flat_channel_scores_exactly_zero():
    """Dit is de bug in de oude poort, hier niet herhaald.

    Een constant kanaal kreeg daar een BEREKENDE correlatie over numerieke
    ruis toebedeeld (0,026 op een echte opname) omdat de bewaking `std == 0`
    exact toetst en filtfilt op een constante ruis van orde 1e-15 oplevert.
    """
    m = respiratory_band_power(np.full(T.size, 0.015259), SF)
    assert m["fraction"] == 0.0


def test_a_channel_that_is_too_short_says_so_instead_of_guessing():
    r = assess_thermistor_band_power(_breathing(seed=3)[:int(SF * 60)], SF)
    assert r["usable"] is False
    assert r["band_power"] is None
    assert "te kort" in r["reason"]


def test_the_measure_is_a_median_over_the_night_not_one_window():
    """Eén venster uit het midden is niet representatief: op echte opnames
    scheelde dat een factor twee. Een goede nacht met één slecht uur hoort
    nog steeds bruikbaar te zijn, en omgekeerd."""
    x = _breathing(seed=4)
    bad = int(SF * 600)
    x[:bad] = np.random.default_rng(5).normal(size=bad)   # eerste venster stuk
    assert respiratory_band_power(x, SF)["fraction"] > THERMISTOR_BAND_POWER_MIN


# ─────────────────────────────────────────────────────────────────
#  Waarom deze poort bestaat — de meting, niet de bewering
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mod_ph,label", [
    (0.0, "identieke modulatie"),
    (np.pi, "modulatie in tegenfase"),
    (np.pi / 2, "modulatie 90 graden verschoven"),
])
def test_the_old_gate_ignores_whether_they_breathe_together(mod_ph, label):
    """Beide kanalen ademen op 0,25 Hz. Alleen de modulatie verschilt.

    De oude poort zwaait van door naar af; de nieuwe blijft staan, want die
    kijkt naar wat het kanaal draagt en niet naar wat de andere sensor doet.
    """
    p = _breathing(seed=10)
    t = _breathing(amp=0.4, phase=0.3, mod_ph=mod_ph, seed=11)
    old = assess_flow_sensor_agreement(p, SF, t, SF)["agreement"]
    new = assess_thermistor_band_power(t, SF)
    assert new["usable"] is True, f"{label}: {new['reason']} (oud: {old})"
    assert new["peak_hz"] == pytest.approx(0.25, abs=0.02)


def test_the_old_gate_really_does_swing_across_the_threshold():
    """Zonder deze assertie zou de test hierboven ook slagen als de oude poort
    helemaal niet bewoog, en dan toont hij niets."""
    p = _breathing(seed=12)
    same = assess_flow_sensor_agreement(
        p, SF, _breathing(amp=0.4, phase=0.3, seed=13), SF)["agreement"]
    anti = assess_flow_sensor_agreement(
        p, SF, _breathing(amp=0.4, phase=0.3, mod_ph=np.pi, seed=13), SF)["agreement"]
    assert same > 0.40 > anti, (same, anti)


# ─────────────────────────────────────────────────────────────────
#  De drempel tegen de gemeten werkelijkheid
# ─────────────────────────────────────────────────────────────────

#: Ademband-vermogen van `Flow Th.` op negen onderscheiden montages,
#: mediaan over tien vensters per nacht. De drempel is het midden van het gat.
MEASURED_GOOD = [0.982, 0.981, 0.977, 0.970]
MEASURED_BAD = [0.441, 0.396, 0.318, 0.036, 0.000]


def test_the_threshold_separates_the_recordings_it_was_derived_from():
    assert min(MEASURED_GOOD) > THERMISTOR_BAND_POWER_MIN > max(MEASURED_BAD)


def test_the_threshold_sits_in_the_middle_of_the_gap():
    """Grootste marge naar beide klassen. Schuift hij richting een van beide,
    dan is dat een keuze die uitgelegd hoort te worden."""
    assert THERMISTOR_BAND_POWER_MIN == pytest.approx(
        (min(MEASURED_GOOD) + max(MEASURED_BAD)) / 2, abs=0.06)


# ─────────────────────────────────────────────────────────────────
#  De vlag
# ─────────────────────────────────────────────────────────────────

def test_only_the_two_new_dual_profiles_use_the_new_gate():
    """Deze as verschuift op elke opname welke sensor apneus scoort, en dus de
    AHI. Alleen de twee nieuwe exploratieve profielen mogen hem hebben: die
    zijn nieuw, dus er is geen bestaande uitkomst om te breken. Elk klinisch
    profiel en elk datasetprofiel houdt de bestaande poort — daar zou dit
    mesa_shhs en de gepubliceerde cijfers raken."""
    moved = {n for n, p in PROFILES.items()
             if p.post_processing.thermistor_gate != "envelope_agreement"}
    assert moved == {"aasm_v3_breath_dual", "aasm_v3_prob_dual"}, moved


def test_no_clinical_or_dataset_profile_moved():
    for n, p in PROFILES.items():
        if p.family in ("clinical", "dataset", "legacy"):
            assert p.post_processing.thermistor_gate == "envelope_agreement", n


def test_the_flag_reaches_the_dict_the_pipeline_reads():
    """Twee keer eerder is een veld gepatcht dat niemand leest — de
    dataclass-naam is niet de legacy-sleutel."""
    assert SCORING_PROFILES["aasm_v3_rec"]["THERMISTOR_GATE"] == "envelope_agreement"
    assert get_profile("aasm_v3_rec").post_processing.thermistor_gate \
        == "envelope_agreement"


def _resolve(pressure, thermistor, gate):
    out = {"meta": {}}
    ch = {"flow_pressure": "PRESS", "flow_thermistor": "THERM"}
    apnea, hypop, _a, _h = _resolve_flow_channels(
        None, SF, pressure, SF, thermistor, SF, ch, out, thermistor_gate=gate)
    return out["meta"]["flow_channels"], apnea, hypop


def test_the_gate_choice_actually_changes_the_apnea_sensor():
    """Het geval uit de kliniek: de thermistor ademt schoon mee, maar zijn
    amplitude moduleert anders. Oude poort wijst af, nieuwe houdt hem."""
    p = _breathing(seed=20)
    t = _breathing(amp=0.4, phase=0.3, mod_ph=np.pi, seed=21)

    fc_old, ap_old, hy_old = _resolve(p, t, "envelope_agreement")
    assert fc_old["apnea_sensor"] == "PRESS"
    assert ap_old is hy_old
    assert fc_old["thermistor_rejected"] == "THERM"

    fc_new, ap_new, hy_new = _resolve(p, t, "respiratory_band")
    assert fc_new["apnea_sensor"] == "THERM"
    assert fc_new["hypopnea_sensor"] == "PRESS"
    assert ap_new is not hy_new
    assert fc_new["thermistor_check"]["gate"] == "respiratory_band"


def test_the_new_gate_still_rejects_a_dead_thermistor():
    """Het geval waarvoor de poort bestaat: 93 apneus werden er 0 doordat een
    dood kanaal de apneu-sensor werd. Dat mag niet terugkomen."""
    fc, apnea, hypop = _resolve(
        _breathing(seed=22), np.full(T.size, 0.015259), "respiratory_band")
    assert fc["apnea_sensor"] == "PRESS"
    assert apnea is hypop
    assert fc["thermistor_rejected"] == "THERM"
    assert fc["thermistor_check"]["band_power"] == 0.0


def test_the_check_keeps_the_shape_the_report_reads():
    """Het rapport en de meta-blokken hoeven niet te weten welke poort draaide.

    `agreement` blijft None: dit is geen overeenstemmingsmaat, en er een getal
    in schrijven dat op een andere schaal leeft zou het rapport laten liegen.
    """
    r = assess_thermistor_band_power(_breathing(seed=23), SF)
    assert {"agreement", "usable", "reason"} <= set(r)
    assert r["agreement"] is None
    assert isinstance(r["band_power"], float)
    assert isinstance(r["reason"], str) and r["reason"]
