"""
`arousal_derivation_mode` is een echt profielveld, geen bijproduct van `family`.

Tot 11-10-2026 rekende constants.py de sleutel `AROUSAL_DERIVATION_MODE` af
uit `profile.family == "dataset"`: alleen `mesa_shhs` kreeg 'single', al het
andere 'multi'. `chicago_1999` (family 'legacy') draaide daardoor de
multi-afleidingsvariant terwijl al zijn andere pins van het enkele tijdperk
zijn -- en niets in zijn registry-blok zei dat.

Nu is het een optioneel veld: `None` = de bestaande familieregel (dus elk
profiel rendert wat het rendert), een expliciete waarde wint. De twee
bevroren profielen pinnen hem expliciet op wat ze VANDAAG renderen:
`mesa_shhs` 'single', `chicago_1999` 'multi'. Dat laatste is bewust NIET
'single': dat zou het gedrag van een bevroren profiel veranderen. De pin
legt de tegenspraak vast in plaats van haar stil op te lossen.
"""
from __future__ import annotations

from dataclasses import fields

import pytest


def _mk(family: str, mode=None):
    from psgscoring.profiles import HypopneaRules, PostProcessingRules, Profile
    return Profile(
        name="x", display_name="x", family=family, aasm_version="", aasm_rule="",
        description="", hypopnea=HypopneaRules(flow_reduction_threshold=0.30),
        post_processing=PostProcessingRules(arousal_derivation_mode=mode),
    )


def test_veld_bestaat_en_is_default_none():
    from psgscoring.profiles import PostProcessingRules
    assert "arousal_derivation_mode" in {f.name for f in fields(PostProcessingRules)}
    assert PostProcessingRules().arousal_derivation_mode is None


def test_none_volgt_de_familieregel():
    from psgscoring.constants import _profile_to_legacy_dict as render
    assert render(_mk("clinical"))["AROUSAL_DERIVATION_MODE"] == "multi"
    assert render(_mk("exploratory"))["AROUSAL_DERIVATION_MODE"] == "multi"
    assert render(_mk("legacy"))["AROUSAL_DERIVATION_MODE"] == "multi"
    assert render(_mk("dataset"))["AROUSAL_DERIVATION_MODE"] == "single"


def test_expliciete_waarde_wint_van_de_familieregel():
    from psgscoring.constants import _profile_to_legacy_dict as render
    assert render(_mk("clinical", "single"))["AROUSAL_DERIVATION_MODE"] == "single"
    assert render(_mk("dataset", "multi"))["AROUSAL_DERIVATION_MODE"] == "multi"


def test_ongeldige_waarde_faalt_bij_het_renderen():
    """Een typefout in de registry hoort bij import te klappen, niet in de
    pijplijn stil naar 'single' te degraderen."""
    from psgscoring.constants import _profile_to_legacy_dict as render
    with pytest.raises(ValueError):
        render(_mk("clinical", "mutli"))


def test_bevroren_profielen_pinnen_expliciet_op_wat_ze_renderen():
    from psgscoring.profiles import get_profile
    assert get_profile("mesa_shhs").post_processing.arousal_derivation_mode == "single"
    # Rationale van de andere pins is 'single-era'; de gerenderde waarde is en
    # blijft 'multi'. Zie het commentaar bij de pin in profiles.py.
    assert get_profile("chicago_1999").post_processing.arousal_derivation_mode == "multi"


def test_rendering_van_de_bevroren_profielen_is_onveranderd():
    from psgscoring.constants import SCORING_PROFILES
    assert SCORING_PROFILES["mesa_shhs"]["AROUSAL_DERIVATION_MODE"] == "single"
    assert SCORING_PROFILES["chicago_1999"]["AROUSAL_DERIVATION_MODE"] == "multi"


def test_alle_andere_profielen_laten_het_veld_op_none():
    """Geen stille pins elders: de familieregel blijft daar de bron."""
    from psgscoring.profiles import PROFILES
    expliciet = {n for n, p in PROFILES.items()
                 if p.post_processing.arousal_derivation_mode is not None}
    assert expliciet == {"mesa_shhs", "chicago_1999"}, expliciet
