"""
Een profielveld dat niemand leest is geen instelling maar een belofte.

`profiles.py` heet in de code "the single source of truth" voor het
scoringsgedrag. Op 31-08-2026 bleken twaalf van de 101 velden dat niet te zijn:
ze staan in de dataclass, profielen zetten ze, en geen enkele consument leest
ze. Twee ervan VARIEREN bovendien over de profielen, dus die beweren vandaag al
iets dat niet gebeurt.

Hoe dat kon blijven staan: `csr_reclassification` was er zo een, en hij kwam
niet boven door een test maar doordat een duplicaat-lint aansloeg toen hij per
ongeluk een tweede keer gedeclareerd werd. Zonder die vergissing was hij nog
jaren blijven liggen. `tests/test_env_overrides_reachable.py` dekt hetzelfde
gat voor omgevingsvariabelen; dit is de profielkant.

DE POORT IS EEN ALLOW-LIST, GEEN VERBOD. Wie een veld toevoegt dat (nog) geen
consument heeft, zet het hier neer met een reden. Dat dwingt een BESLISSING af
in plaats van een ontdekking tijdens een meting.
"""
import ast
import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "psgscoring"

#: Velden die per definitie geen gedragsconsument hebben: naamgeving,
#: documentatie, herkomst, en de dataclasses die de andere dragen.
BESCHRIJVEND = {
    "name", "display_name", "description", "citation", "aasm_rule",
    "aasm_version", "family", "deprecated", "deprecated_alias_for",
    "output_variants", "hypopnea", "apnea", "spo2", "post_processing",
}

#: BEKENDE SCHULD, geïnventariseerd 31-08-2026. Elk veld hier staat in de
#: registry zonder dat iets het leest. Ze staan er niet omdat het goed is.
#:
#: De twee met (VARIEERT) beweren vandaag al iets dat niet gebeurt: profielen
#: zetten er verschillende waarden voor, en het maakt geen verschil.
ZONDER_CONSUMENT = {
    # `unsure_as_hypopnea` en `HypopneaRules.sensor` STONDEN hier. Ze zijn op
    # 31-08-2026 VERWIJDERD in plaats van bedraad, want ze beweerden allebei
    # iets onwaars in plaats van iets ongebruikts:
    #   * `sensor` dupliceerde `flow_fallback_strategy`, en bij chicago_1999
    #     spraken de twee elkaar tegen (`nasal_pressure_or_flow` naast
    #     strategie `none`);
    #   * `unsure_as_hypopnea` beschreef hoe de NSRR-ANNOTATIE gelezen moet
    #     worden, niet hoe wij scoren -- dat hoort in het meetharnas, waar het
    #     ook al geïmplementeerd is.
    # Wat overblijft is `ApneaRules.sensor`, die niet varieert.
    "sensor",
    # De >=10 s-eis. De detectoren lezen de MODULECONSTANTE; `pipeline.py`
    # doet `profile.get("HYPOPNEA_MIN_DUR_S", 10.0)` op een sleutel die niet
    # in de dict zit, dus die get faalt altijd naar de default.
    "min_duration_s",
    "arousal_required",
    "nasal_pressure_fallback",
    # Superseded door `hypopnea_force_linearisation`, die wel bedraad is.
    "square_root_linearisation",
    "global_p95_fallback",
    # Het filter is alleen uit te zetten via cv=0, niet via deze vlag.
    "stability_filter_enabled",
    "local_baseline_validation",
    # Draait op `sf_effort > 0`, niet op deze vlag.
    "mixed_apnea_decomposition",
}


@pytest.fixture(scope="module")
def analyse():
    velden = {}
    for n in ast.walk(ast.parse((PKG / "profiles.py").read_text(encoding="utf-8"))):
        if isinstance(n, ast.ClassDef):
            for x in n.body:
                if isinstance(x, ast.AnnAssign) and isinstance(x.target, ast.Name):
                    velden[x.target.id] = n.name

    const_regels = (PKG / "constants.py").read_text(encoding="utf-8").splitlines()

    def sleutels(veld):
        """Welke legacy-sleutels dit veld voedt, via de renderer."""
        out = set()
        for i, r in enumerate(const_regels):
            if re.search(rf"\.{veld}\b", r):
                for j in range(i, max(-1, i - 4), -1):
                    m = re.search(r'"([A-Z0-9_]+)"\s*:', const_regels[j])
                    if m:
                        out.add(m.group(1))
                        break
        return out

    bron = {p.name: p.read_text(encoding="utf-8") for p in PKG.glob("*.py")}

    def consument(veld, ks):
        for f, txt in bron.items():
            if f in ("profiles.py", "constants.py"):
                continue
            if re.search(rf"\.{veld}\b", txt):
                return f
            for k in ks:
                if f'"{k}"' in txt or f"'{k}'" in txt:
                    return f
        return None

    zonder = {v for v in velden
              if not v.startswith("_") and v not in BESCHRIJVEND
              and consument(v, sleutels(v)) is None}
    return velden, zonder


def test_geen_nieuw_veld_zonder_consument(analyse):
    _velden, zonder = analyse
    nieuw = zonder - ZONDER_CONSUMENT
    assert not nieuw, (
        "deze profielvelden worden door niets gelezen en staan niet in "
        "ZONDER_CONSUMENT. Een veld dat niemand leest is geen instelling maar "
        "een belofte: het zetten verandert niets, en dat ziet er in een meting "
        f"precies zo uit als 'geen effect'. Beslis expliciet: {sorted(nieuw)}")


def test_de_lijst_bevat_geen_velden_die_inmiddels_wel_gelezen_worden(analyse):
    """Anders groeit de schuldenlijst en krimpt hij nooit."""
    _velden, zonder = analyse
    opgelost = ZONDER_CONSUMENT - zonder
    assert not opgelost, (
        f"deze staan als schuld genoteerd maar hebben nu een consument; "
        f"haal ze uit ZONDER_CONSUMENT: {sorted(opgelost)}")


def test_de_lijst_verwijst_alleen_naar_bestaande_velden(analyse):
    velden, _zonder = analyse
    spook = ZONDER_CONSUMENT - set(velden)
    assert not spook, f"ZONDER_CONSUMENT noemt niet-bestaande velden: {sorted(spook)}"


def test_de_gewired_velden_van_vannacht_staan_er_niet_meer_in(analyse):
    """`csr_reclassification` en `desat_global_baseline_min_local_pct` waren
    dezelfde fout en zijn op 31-08 bedraad."""
    _velden, zonder = analyse
    for veld in ("csr_reclassification", "desat_global_baseline_min_local_pct",
                 "arousal_min_interval_s", "event_gap_tolerance_breaths"):
        assert veld not in zonder, f"{veld} heeft weer geen consument"


# ══════════════════════════════════════════════════════════════════════
#  De twee die niet sliepen maar logen
# ══════════════════════════════════════════════════════════════════════

def test_hypopnea_regels_dragen_geen_sensorveld_meer():
    """`HypopneaRules.sensor` dupliceerde `flow_fallback_strategy`.

    Erger dan ongebruikt: bij `chicago_1999` spraken ze elkaar tegen. Het veld
    zei `nasal_pressure_or_flow` -- dus met terugval -- terwijl de bedrade
    strategie `none` was. Eén profiel, twee tegengestelde uitspraken over
    dezelfde vraag, en alleen de tweede deed iets.
    """
    from psgscoring.profiles import get_profile
    assert not hasattr(get_profile("aasm_v3_rec").hypopnea, "sensor")


def test_de_sensorkeuze_leeft_in_de_bedrade_vlag():
    """Wat er wél is, en dat het onderscheid nog steeds draagt."""
    from psgscoring.profiles import get_profile
    assert get_profile("mesa_shhs").post_processing.flow_fallback_strategy == \
        "ripsum_on_nasal_failure"
    assert get_profile("aasm_v3_rec").post_processing.flow_fallback_strategy == "none"


def test_unsure_as_hypopnea_is_geen_profielveld_meer():
    """Het beschreef de NSRR-ANNOTATIE, niet onze scoring.

    Een profiel zegt hoe wíj scoren; hoe een dataset zijn labels bedoelt hoort
    in het harnas dat die dataset leest. `validate_mesa.py` implementeert het
    daar ook al.
    """
    from psgscoring.profiles import get_profile
    assert not hasattr(get_profile("mesa_shhs").post_processing,
                       "unsure_as_hypopnea")


def test_de_nsrr_conventie_is_niet_verloren():
    """Verwijderen mag geen kennis weggooien -- inclusief de open tegenspraak."""
    import inspect

    from psgscoring import profiles
    bron = inspect.getsource(profiles)
    assert "HYPOPNEA_CONCEPTS" in bron, "de verwijzing naar het harnas ontbreekt"
    assert ">50 %" in bron and ">=30 %" in bron, (
        "de openstaande tegenspraak over wat `Unsure` betekent is niet vastgelegd")
