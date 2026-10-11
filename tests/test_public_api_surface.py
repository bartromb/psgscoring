"""
Het publieke oppervlak zegt wat het is.

Vier kleine onwaarheden in één hoek van de bibliotheek, elk op zich
onschuldig en samen een patroon:

  * `psgscoring/__init__.py` importeerde de profiel-API (`Profile`,
    `get_profile`, ...) ná `__all__` en liet haar daar weg, zodat
    `from psgscoring import *` haar niet meenam en linters haar als
    'ongebruikte import' zagen;
  * `resolve_profile_name` waarschuwde dat een alias "will be removed in
    v0.5.0" terwijl de bibliotheek op 0.35.x staat en de aliassen er nog zijn;
  * het voorbeeld in de moduledocstring van `profiles.py` toonde 8 van de 22
    geregistreerde profielen;
  * `profile_metadata` beweerde "Used by pipeline.py", en pipeline.py gebruikt
    het niet.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "psgscoring"

PROFIEL_API = (
    "Profile", "get_profile", "list_profiles", "list_profile_groups",
    "PROFILES", "PROFILE_GROUPS", "profile_metadata",
)


def test_de_profiel_api_staat_in_all():
    import psgscoring
    ontbreekt = [n for n in PROFIEL_API if n not in psgscoring.__all__]
    assert not ontbreekt, ontbreekt


def test_alles_in_all_bestaat_ook():
    import psgscoring
    kapot = [n for n in psgscoring.__all__ if not hasattr(psgscoring, n)]
    assert not kapot, kapot


def test_profiles_all_noemt_profile_metadata():
    from psgscoring import profiles
    assert "profile_metadata" in profiles.__all__


def test_deprecatiewaarschuwing_noemt_geen_verwijderversie():
    from psgscoring.profiles import resolve_profile_name
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert resolve_profile_name("standard") == "aasm_v3_rec"
    berichten = [str(x.message) for x in w
                 if issubclass(x.category, DeprecationWarning)]
    assert berichten, "geen DeprecationWarning voor een legacy-alias"
    tekst = berichten[0]
    assert not re.search(r"removed in v?\d", tekst), tekst
    assert "aasm_v3_rec" in tekst


def test_moduledocstring_voorbeeld_noemt_elk_geregistreerd_profiel():
    from psgscoring import profiles
    doc = profiles.__doc__ or ""
    ontbreekt = [n for n in profiles.PROFILES if f"'{n}'" not in doc]
    assert not ontbreekt, (
        "het list_profiles()-voorbeeld in de moduledocstring van profiles.py "
        f"is verouderd; ontbreekt: {ontbreekt}")


def test_profile_metadata_docstring_claimt_geen_pipeline_gebruik():
    from psgscoring.profiles import profile_metadata
    doc = profile_metadata.__doc__ or ""
    assert "Used by pipeline.py" not in doc
    # En de claim is ook feitelijk onwaar: pipeline.py bouwt output["meta"]
    # zelf uit de gerenderde dict.
    assert "profile_metadata" not in (PKG / "pipeline.py").read_text(encoding="utf-8")


def test_kopcommentaar_boven_aasm_v3_dual_beschrijft_aasm_v3_dual():
    regels = (PKG / "profiles.py").read_text(encoding="utf-8").splitlines()
    i = next(n for n, r in enumerate(regels) if r.startswith("_aasm_v3_dual = Profile("))
    kop = "\n".join(regels[max(0, i - 3):i])
    assert "STRICT" not in kop and "Conservative" not in kop, kop
    assert "dual" in kop.lower(), kop
