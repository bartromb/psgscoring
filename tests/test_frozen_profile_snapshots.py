"""
Bevroren profielen renderen byte-identiek -- en een nieuw veld breekt dat.

`mesa_shhs` en `chicago_1999` reproduceren gepubliceerde cijfers (paper
v31/v37, NSRR-conventie, Chicago 1999). Elke default die sindsdien in
`PostProcessingRules` veranderde is op die twee profielen GEPIND op het oude
gedrag -- vijf keer hetzelfde blok commentaar in profiles.py getuigt ervan.
Wat ontbrak was een poort die dat AFDWINGT: een nieuw veld met een nieuwe
default verschijnt in de gerenderde dict van beide profielen zonder dat iets
het meldt, en de ontbrekende pin valt pas een meting later op.

Dit bestand pint de volledige gerenderde dict (`SCORING_PROFILES[naam]`, de
vorm die respiratory.py en pipeline.py lezen) tegen
tests/golden/profile_<naam>.json: sleutels gesorteerd, floats op 6 decimalen.

ELKE afwijking is een fout, ook een nieuwe sleutel. Dat is de bedoeling: wie
een veld toevoegt, pint het eerst op de bevroren profielen op de waarde die de
code VÓÓR het veld effectief hanteerde, en zegent dan opnieuw:

    python tests/test_frozen_profile_snapshots.py bless

Opnieuw zegenen zonder pin is precies de fout die deze test bestaat om te
vangen. Lees het verschil voor je zegent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tests" / "golden"

#: De profielen die gepubliceerde resultaten reproduceren en daarom niet mee
#: mogen bewegen met een nieuwe default. Zie de vijf pin-blokken in profiles.py.
BEVROREN = ("mesa_shhs", "chicago_1999")

#: Floats worden op dit aantal decimalen vastgelegd. Ruim fijner dan elke
#: drempel in de registry (twee decimalen), dus een echte wijziging valt
#: altijd op; grof genoeg om geen ruis uit een platformafhankelijke repr te
#: pinnen.
DECIMALEN = 6


def _pad(naam: str) -> Path:
    return GOLDEN_DIR / f"profile_{naam}.json"


def _normaliseer(x):
    """JSON-vriendelijke, deterministische vorm: sleutels gesorteerd, floats
    afgerond, tuples als lijsten. Bool vóór int, want bool is een int."""
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return round(x, DECIMALEN)
    if isinstance(x, dict):
        return {str(k): _normaliseer(v)
                for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_normaliseer(v) for v in x]
    return str(x)


def _render(naam: str) -> dict:
    from psgscoring.constants import SCORING_PROFILES
    return _normaliseer(SCORING_PROFILES[naam])


def _tekst(d: dict) -> str:
    return json.dumps(d, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _verschil(gezegend: dict, huidig: dict) -> str:
    nieuw = sorted(set(huidig) - set(gezegend))
    weg = sorted(set(gezegend) - set(huidig))
    anders = sorted(
        k for k in set(huidig) & set(gezegend)
        if huidig[k] != gezegend[k] or type(huidig[k]) is not type(gezegend[k]))
    regels = []
    if nieuw:
        regels.append("  NIEUWE sleutels (geen pin op het bevroren profiel): "
                      + ", ".join(f"{k}={huidig[k]!r}" for k in nieuw))
    if weg:
        regels.append("  VERDWENEN sleutels: " + ", ".join(weg))
    if anders:
        regels.append("  GEWIJZIGDE waarden: " + ", ".join(
            f"{k}: {gezegend[k]!r} -> {huidig[k]!r}" for k in anders))
    return "\n".join(regels) or "  (alleen een verschil in serialisatie)"


@pytest.mark.parametrize("naam", BEVROREN)
def test_bevroren_profiel_rendert_byte_identiek(naam):
    pad = _pad(naam)
    assert pad.exists(), (
        f"{pad.relative_to(REPO)} ontbreekt. Dit is een BEVROREN profiel; zijn "
        f"gerenderde dict hoort vastgelegd te zijn. Zegen hem bewust:\n"
        f"    python tests/test_frozen_profile_snapshots.py bless")
    gezegend = json.loads(pad.read_text(encoding="utf-8"))
    huidig = _render(naam)
    if _tekst(gezegend) == _tekst(huidig):
        return
    pytest.fail(
        f"Het gerenderde profiel `{naam}` wijkt af van "
        f"{pad.relative_to(REPO)}:\n{_verschil(gezegend, huidig)}\n\n"
        f"`{naam}` is BEVROREN: het reproduceert gepubliceerde cijfers en mag "
        f"niet meebewegen met een nieuwe default. Een nieuw profielveld hoort "
        f"op `{naam}` (en op de andere bevroren profielen) EXPLICIET GEPIND te "
        f"worden op het gedrag dat de code vóór dat veld had -- in de registry "
        f"in profiles.py, naast de bestaande pin-blokken -- en pas DAARNA wordt "
        f"deze snapshot opnieuw gezegend met\n"
        f"    python tests/test_frozen_profile_snapshots.py bless\n"
        f"Zegen niet opnieuw om deze test groen te krijgen: lees het verschil "
        f"hierboven en beslis per sleutel of de waarde op het bevroren profiel "
        f"inderdaad de oude is.")


@pytest.mark.parametrize("naam", BEVROREN)
def test_snapshot_hoort_bij_het_juiste_profiel(naam):
    """Een gekopieerd bestand onder de verkeerde naam zou stil 'passen'."""
    pad = _pad(naam)
    if not pad.exists():
        pytest.skip("snapshot ontbreekt; de hoofdtest meldt dat")
    d = json.loads(pad.read_text(encoding="utf-8"))
    assert d.get("_PROFILE_NAME") == naam
    assert list(d) == sorted(d), "snapshot is niet met gesorteerde sleutels geschreven"


def _bless() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for naam in BEVROREN:
        pad = _pad(naam)
        pad.write_text(_tekst(_render(naam)), encoding="utf-8")
        print(f"gezegend: {naam} -> {pad.relative_to(REPO)}")


if __name__ == "__main__":
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if sys.argv[1:] == ["bless"]:
        _bless()
    else:
        print("gebruik: python tests/test_frozen_profile_snapshots.py bless")
        sys.exit(2)
