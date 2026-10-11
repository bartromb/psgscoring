"""
Geen twee geregistreerde profielen delen een regelobject.

`_ENVELOPE_ARM_HYPOPNEA` was één `HypopneaRules`-instantie die de vier
envelope-armen (`aasm_v3_env_chunked`, `_rectify`, `_breath`, `_decimated`)
deelden. De dataclasses zijn mutabel. Een `replace()` is veilig, maar een
attribuuttoekenning op het ene profiel -- een harnas dat 'even' de drempel
zet, een sweep, een test zonder teardown -- verandert dan stil de drie andere.
`_with_dual_apneas` legt in zijn docstring uit waarom het elke geneste
dataclass met `replace()` opnieuw bouwt; dit is dezelfde regel voor de hele
registry, afgedwongen in plaats van aangenomen.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass

REGELOBJECTEN = ("hypopnea", "apnea", "spo2", "post_processing")


def _geneste_objecten(profiel):
    """(pad, object) voor elk dataclass-attribuut, recursief."""
    for attr in REGELOBJECTEN:
        obj = getattr(profiel, attr)
        yield attr, obj
        if is_dataclass(obj):
            for f in fields(obj):
                sub = getattr(obj, f.name)
                if is_dataclass(sub):
                    yield f"{attr}.{f.name}", sub


def test_geen_twee_profielen_delen_een_regelobject():
    from psgscoring.profiles import PROFILES

    gezien: dict = {}
    for naam, p in PROFILES.items():
        for pad, obj in _geneste_objecten(p):
            gezien.setdefault(id(obj), []).append(f"{naam}.{pad}")
    gedeeld = sorted(v for v in gezien.values() if len(v) > 1)
    assert not gedeeld, (
        "deze profielen delen één en hetzelfde (mutabele) regelobject; een "
        "toekenning op het ene verandert stil de andere. Geef elk zijn eigen "
        f"instantie met dataclasses.replace(): {gedeeld}")


def test_profielobjecten_zelf_zijn_uniek():
    from psgscoring.profiles import PROFILES
    ids = [id(p) for p in PROFILES.values()]
    assert len(ids) == len(set(ids))


def test_registernaam_en_profielnaam_vallen_samen():
    """Een gedeeld object zou hier ook opvallen: `name` is per profiel."""
    from psgscoring.profiles import PROFILES
    for naam, p in PROFILES.items():
        assert p.name == naam, f"registry-sleutel {naam!r} draagt profiel {p.name!r}"
