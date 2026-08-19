"""tests/test_rip_pair_scale_free.py — de paarpoort mag geen versterking meten.

`rip_quality_scale_free` (v0.17.0) maakte de PER-KANAAL drempel schaalvrij; de
paarregel bleef absoluut. `breath_energy` is de som van de PSD in de ademband
en schaalt kwadratisch met de amplitude, dus twee banden met een andere
versterking halen `ratio > 100` zonder dat er iets mis is -- en dan verklaart
de poort de zwakste "disconnected" en valt de effortclassificatie terug op één
kanaal.

Op een klinische opname kostte dat 73 van 142 herlabelde events (49 central ->
obstructive, 16 uncertain -> obstructive) en draaide het de diagnose om.

De gevallen hieronder zijn synthetisch en gepind: twee identiek ademende
banden met alleen een amplitudeverschil, tegenover een werkelijk dode band.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring.signal_quality import ENERGY_RATIO_FAIL, compare_rip_pair

SF = 32.0
DUR = 300.0


def breathing(amplitude=1.0, freq=0.22, seed=0):
    """Een ademend effortsignaal: sinus plus wat ruis."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, DUR, 1 / SF)
    sig = np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(t.size)
    return amplitude * sig


def dead(seed=1):
    """Een losgekoppelde band: ruis zonder ademcomponent."""
    rng = np.random.default_rng(seed)
    return 1e-4 * rng.standard_normal(int(DUR * SF))


# ── het geval dat de fout blootlegde ────────────────────────────────────

def test_amplitude_difference_alone_must_not_disconnect_a_channel():
    """Beide banden ademen; alleen de versterking verschilt.

    Dit is de gemeten situatie: ratio ver boven de drempel, beide kanalen
    `ok`. Met de vlag AAN hoort de classificatie bilateraal te blijven.
    """
    thor = breathing(amplitude=1.0, seed=0)
    abd = breathing(amplitude=25.0, seed=2)      # zelfde ademhaling, 25x groter

    r_off = compare_rip_pair(thor, abd, SF, scale_free=True)
    r_on = compare_rip_pair(thor, abd, SF, scale_free=True,
                            pair_scale_free=True)

    assert r_off["energy_ratio"] > ENERGY_RATIO_FAIL, (
        "de opzet klopt niet: zonder ratio boven de drempel meet deze test "
        "niets")
    assert r_off["thorax"]["status"] == "ok"
    assert r_off["abdomen"]["status"] == "ok"

    # ZONDER de vlag: het bestaande gedrag, en dat is precies het defect.
    assert r_off["recommended_mode"] == "single-channel"
    assert r_off["working_channel"] == "abdomen"

    # MET de vlag: geen kanaal afgekeurd dat zijn eigen toets doorstaat.
    assert r_on["recommended_mode"] == "bilateral"
    assert r_on["working_channel"] is None

    # `classification_reliable` blijft False, en dat is juist: het veld is
    # gedefinieerd als "bilateraal EN geen waarschuwingen", en er IS hier iets
    # te melden -- een asymmetrie van meer dan honderd maal. Alleen de
    # rapportagelaag leest dit veld; geen classificatiepad vertakt erop. De
    # bilaterale analyse draait dus, en het rapport zegt erbij dat er naar
    # gekeken mag worden. Dat is eerlijker dan "reliable" met een ratio van
    # 1186x eronder.
    assert r_on["classification_reliable"] is False
    assert r_on["warnings"], "bilateraal doorgaan zonder melding zou de "\
        "asymmetrie verbergen"


def test_the_warning_explains_itself_instead_of_going_silent():
    """Bilateraal doorgaan zonder iets te zeggen zou de asymmetrie verbergen."""
    r = compare_rip_pair(breathing(1.0, seed=0), breathing(25.0, seed=2), SF,
                         scale_free=True, pair_scale_free=True)
    joined = " ".join(r["warnings"]).lower()
    assert "versterkings" in joined or "eenheidsverschil" in joined
    assert "bilaterale classificatie blijft actief" in joined


# ── het geval dat de poort MOET blijven vangen ──────────────────────────

def test_a_genuinely_dead_channel_is_still_caught():
    """Een losgekoppelde band draagt geen ademhaling en valt op zijn eigen toets.

    Zou de vlag ook dit doorlaten, dan was de reparatie een verwijdering.
    """
    thor = dead(seed=1)
    abd = breathing(amplitude=1.0, seed=2)
    r = compare_rip_pair(thor, abd, SF, scale_free=True, pair_scale_free=True)
    assert r["recommended_mode"] in ("single-channel", "unreliable")
    if r["recommended_mode"] == "single-channel":
        assert r["working_channel"] == "abdomen"


def test_two_healthy_similar_channels_stay_bilateral_either_way():
    """Controle: zonder asymmetrie verandert de vlag niets."""
    thor = breathing(amplitude=1.0, seed=0)
    abd = breathing(amplitude=1.1, seed=2)
    for flag in (False, True):
        r = compare_rip_pair(thor, abd, SF, scale_free=True,
                             pair_scale_free=flag)
        assert r["recommended_mode"] == "bilateral"


# ── de vlag zelf ────────────────────────────────────────────────────────

def test_the_flag_is_off_on_every_profile_a_patient_can_get():
    """Dit verschuift OAHI/CAHI en hoort gemeten te worden, niet stil aan te gaan.

    De vlag mag ALLEEN aan staan op de exploratieve meetarm. Klinische
    profielen niet -- daar moet de meting eerst uitwijzen dat het beter is.
    Bevroren families al helemaal niet: die reproduceren gepubliceerde cijfers.
    """
    from psgscoring.profiles import PROFILES
    allowed = {"aasm_v3_pair_scalefree"}
    on = {n for n, p in PROFILES.items()
          if p.post_processing.rip_pair_scale_free}
    assert on <= allowed, (
        f"rip_pair_scale_free staat aan op profielen die dat niet horen: "
        f"{sorted(on - allowed)}")
    for n in on:
        assert PROFILES[n].family == "exploratory", (
            f"{n} draagt de vlag maar is familie "
            f"{PROFILES[n].family!r}, niet exploratory")


def test_the_measurement_arm_differs_in_exactly_one_field():
    """Een arm die meer dan één ding verandert meet meer dan één ding.

    Zonder deze test kan er ongemerkt een tweede verschil bijkomen en zou de
    meting niet meer aan de paarpoort toe te schrijven zijn.
    """
    import dataclasses as dc

    from psgscoring.profiles import PROFILES
    arm, anchor = PROFILES["aasm_v3_pair_scalefree"], PROFILES["aasm_v3_rec"]
    assert dc.asdict(arm.hypopnea) == dc.asdict(anchor.hypopnea)
    assert dc.asdict(arm.apnea) == dc.asdict(anchor.apnea)
    assert dc.asdict(arm.spo2) == dc.asdict(anchor.spo2)
    diff = {k for k, v in dc.asdict(arm.post_processing).items()
            if v != dc.asdict(anchor.post_processing)[k]}
    assert diff == {"rip_pair_scale_free"}, (
        f"de meetarm verschilt ook in: {sorted(diff - {'rip_pair_scale_free'})}")


def test_the_flag_reaches_the_legacy_constants():
    """Een vlag die de pijplijn niet bereikt is decoratie."""
    from psgscoring.constants import _profile_to_legacy_dict
    from psgscoring.profiles import PROFILES
    d = _profile_to_legacy_dict(PROFILES["aasm_v3_rec"])
    assert "RIP_PAIR_SCALE_FREE" in d


def test_the_pipeline_passes_the_flag():
    """Leest de bron, want een vergeten doorgifte faalt nergens luid."""
    import inspect

    import psgscoring.pipeline as P
    src = inspect.getsource(P)
    assert "pair_scale_free=bool(profile.get(\"RIP_PAIR_SCALE_FREE\"" in src, (
        "compare_rip_pair krijgt de vlag niet mee")


@pytest.mark.parametrize("amp", [5.0, 12.0, 60.0, 200.0])
def test_the_repair_holds_across_gain_differences(amp):
    """Niet gepind op één amplitude: elk versterkingsverschil hoort te werken."""
    r = compare_rip_pair(breathing(1.0, seed=0), breathing(amp, seed=2), SF,
                         scale_free=True, pair_scale_free=True)
    assert r["recommended_mode"] == "bilateral"


# ── de waarschuwing: zichtbaar maken zonder iets te veranderen ──────────

def test_a_suspect_gate_is_flagged_but_changes_nothing():
    """De poort blijft afgaan; alleen wordt er nu bij gezegd dat dat vreemd is.

    Dit is de weg die WEL zonder cohortvalidatie mag: geen enkele gescoorde
    waarde beweegt, er komt alleen uitleg bij. Op MESA gaat de poort af bij 6
    van 150 opnames en is precies 1 daarvan verdacht; zonder deze melding
    leest zo'n rapport als "89 centrale apneus" zonder dat iemand kan zien dat
    de bilaterale analyse uitstond.
    """
    thor = breathing(amplitude=1.0, seed=0)
    abd = breathing(amplitude=25.0, seed=2)
    r = compare_rip_pair(thor, abd, SF, scale_free=True)

    # Gedrag ONVERANDERD.
    assert r["recommended_mode"] == "single-channel"
    assert r["working_channel"] == "abdomen"

    # Maar nu machineleesbaar gemarkeerd, plus uitleg.
    assert r["pair_gate_suspect"] is True
    joined = " ".join(r["warnings"]).lower()
    assert "doorstaat zijn eigen kwaliteitstoets" in joined
    assert "controleer de subtypering" in joined


def test_a_genuinely_dead_channel_is_not_flagged_as_suspect():
    """Guard op de guard: zou alles verdacht heten, dan zegt de vlag niets."""
    r = compare_rip_pair(dead(seed=1), breathing(1.0, seed=2), SF,
                         scale_free=True)
    assert r["pair_gate_suspect"] is False


def test_the_suspect_field_exists_even_when_the_gate_does_not_fire():
    """Een veld dat soms ontbreekt dwingt elke lezer tot .get() met een default.

    En een default is precies waar een stille False vandaan komt.
    """
    r = compare_rip_pair(breathing(1.0, seed=0), breathing(1.1, seed=2), SF,
                         scale_free=True)
    assert r["recommended_mode"] == "bilateral"
    assert r["pair_gate_suspect"] is False


def test_the_flag_and_the_warning_are_independent():
    """Met pair_scale_free AAN is er geen afgekeurd kanaal, dus niets verdachts."""
    r = compare_rip_pair(breathing(1.0, seed=0), breathing(25.0, seed=2), SF,
                         scale_free=True, pair_scale_free=True)
    assert r["recommended_mode"] == "bilateral"
    assert r["pair_gate_suspect"] is False
