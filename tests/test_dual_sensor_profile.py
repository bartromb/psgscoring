"""`aasm_v3_dual` — apneus op beide flowsensoren.

Het profiel bestaat naast de andere zodat dezelfde opname onder meerdere
algoritmes beoordeeld kan worden. Twee regels bepalen het gedrag, en beide
komen uit wat er in productie misging:

  niet dubbel tellen   dezelfde apneu op twee sensoren is één apneu
  bij twijfel goedkeuren
                       de tweede sensor mag geen events WEGNEMEN, want een
                       onbetrouwbare sensor die dat wel doet kostte een echte
                       opname 83 centrale apneus
"""
import pytest

from psgscoring.profiles import PROFILES, get_profile


def test_the_profile_exists_alongside_the_others():
    assert "aasm_v3_dual" in PROFILES
    p = get_profile("aasm_v3_dual")
    assert p.family == "clinical"
    assert "v3" in p.aasm_version, "anders valt hij uit de v3-groep in de UI"


def test_it_uses_both_sensors():
    pp = get_profile("aasm_v3_dual").post_processing
    assert pp.dual_sensor_apnea is True


def test_falsifying_is_off():
    """De klinische kern: overdetectie tegenhouden mag alleen bij een
    betrouwbaar signaal, en die betrouwbaarheidsmaat is er niet.

    Op MESA bevestigt 19% van 1785 apneu-detecties op beide sensoren, en de
    envelope-overeenstemming voorspelt dat niet (r = +0,07). Aanzetten zou
    81% weggooien op een detector die al onderdetecteert.
    """
    pp = get_profile("aasm_v3_dual").post_processing
    assert pp.dual_sensor_corroboration is False


def test_the_other_profiles_do_not_use_two_sensors():
    """Naast elkaar, niet in plaats van: de rest verandert niet."""
    for name in ("aasm_v3_rec", "aasm_v3_breath", "aasm_v3_strict",
                 "aasm_v3_sensitive", "mesa_shhs", "cms_medicare",
                 "aasm_v1_rec", "aasm_v2_rec", "chicago_1999"):
        pp = get_profile(name).post_processing
        assert pp.dual_sensor_apnea is False, name
        assert pp.dual_sensor_corroboration is False, name


def test_hypopnea_rules_follow_aasm_rule_1a():
    """Apneus veranderen; hypopneeen blijven op de neusdruk per AASM."""
    h = get_profile("aasm_v3_dual").hypopnea
    assert h.flow_reduction_threshold == 0.30
    assert h.desat_threshold == 0.03
    assert h.desat_or_arousal is True
    assert h.sensor == "nasal_pressure"
    assert h.square_root_linearisation is True


def test_the_legacy_dict_carries_both_switches():
    from psgscoring.constants import SCORING_PROFILES
    d = SCORING_PROFILES["aasm_v3_dual"]
    assert d["DUAL_SENSOR_APNEA"] is True
    assert d["DUAL_SENSOR_CORROBORATION"] is False


@pytest.mark.parametrize("alias,expected", [
    ("strict", "aasm_v3_strict"),
    ("standard", "aasm_v3_rec"),
    ("sensitive", "aasm_v3_sensitive"),
])
def test_legacy_aliases_are_untouched(alias, expected):
    """Het nieuwe profiel mag niets kapen dat het paper reproduceert."""
    from psgscoring.constants import SCORING_PROFILES
    assert SCORING_PROFILES[alias]["_PROFILE_NAME"] == expected


def test_the_default_profile_is_still_the_recommended_one():
    import inspect

    from psgscoring import run_pneumo_analysis
    default = inspect.signature(
        run_pneumo_analysis).parameters["scoring_profile"].default
    assert default == "aasm_v3_rec"
