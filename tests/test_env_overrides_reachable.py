"""
tests/test_env_overrides_reachable.py — een vlag is niets waard tot iemand hem leest.

Op 29-08-2026 sprong dezelfde val vier keer op één dag:

  1. `PSGSCORING_RULE1A_AROUSAL` in `compare_baseline_arousal.py` — het harnas
     dat over de arousal-tak moest beslissen, zette één van de TWEE vereiste
     vlaggen en leverde een "aan"-kolom die byte-identiek was aan "uit".
  2. `PSGSCORING_EVENT_MIN_QUALIFYING_FRACTION` bestond nog niet toen de
     tweede arm gemeten moest worden.
  3. `PSGSCORING_AROUSAL_MIN_INTERVAL_S` werd alleen in `pipeline.py` gelezen,
     terwijl `sweep_arousal_threshold_psgipa.py` `detect_arousals_multi`
     RECHTSTREEKS aanroept — twee identieke armen, en bijna gerapporteerd als
     nulmeting.
  4. Eerder al: de arousal-tak zelf (issue #16), jarenlang stil op nul.

Elke keer zag de run er geslaagd uit. Dat is het gevaarlijke: een vlag die zijn
consument niet bereikt levert geen fout, maar een meting die niets meet.

Deze module dekt twee dingen af:

  A. elke `PSGSCORING_*` die in de broncode GENOEMD wordt, wordt ook ergens
     werkelijk GELEZEN — geen gedocumenteerde dode knop;
  B. welke vlaggen ALLEEN via `pipeline.py` leesbaar zijn, staat expliciet in
     een lijst. Dat is geen verbod: het is een bewuste keuze die iemand moet
     máken. Een nieuwe detectorvlag die alleen daar gelezen wordt, laat deze
     test omvallen tot ze in de lijst staat of alsnog in de detector gelezen
     wordt.
"""
import ast
import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "psgscoring"
PATROON = re.compile(r"PSGSCORING_[A-Z0-9_]+")


def _gelezen(pad: Path) -> set[str]:
    """Vlaggen die deze module ECHT uitleest — os.environ.get/getenv/[...]."""
    boom = ast.parse(pad.read_text(encoding="utf-8"))
    uit: set[str] = set()
    for n in ast.walk(boom):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("get", "getenv")):
            for a in n.args:
                if (isinstance(a, ast.Constant) and isinstance(a.value, str)
                        and a.value.startswith("PSGSCORING_")):
                    uit.add(a.value)
        if (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, str)
                and n.slice.value.startswith("PSGSCORING_")):
            uit.add(n.slice.value)
    return uit


def _genoemd(pad: Path) -> set[str]:
    return set(PATROON.findall(pad.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def kaart():
    return {p.name: (_gelezen(p), _genoemd(p)) for p in sorted(PKG.glob("*.py"))}


# ══════════════════════════════════════════════════════════════════════
#  A. Geen gedocumenteerde dode knoppen
# ══════════════════════════════════════════════════════════════════════

def test_elke_genoemde_vlag_wordt_ergens_gelezen(kaart):
    gelezen = set().union(*(g for g, _ in kaart.values())) or set()
    genoemd = set().union(*(m for _, m in kaart.values())) or set()
    dood = genoemd - gelezen
    assert not dood, (
        "deze vlaggen staan in de broncode maar worden nergens uitgelezen — "
        f"een knop die niets doet: {sorted(dood)}")


# ══════════════════════════════════════════════════════════════════════
#  B. Alleen-via-de-pipeline is een KEUZE, geen ongeluk
# ══════════════════════════════════════════════════════════════════════
#
# Deze vlaggen zijn uitsluitend leesbaar wanneer de aanroeper via
# `run_pneumo_analysis` gaat. Elk meetharnas dat de detectoren rechtstreeks
# aanroept — en dat doen ze bijna allemaal — ziet ze NIET.
#
# Ze staan hier omdat ze vandaag zo zijn, niet omdat het goed is. Wie een van
# deze in een meting gebruikt: controleer eerst of het harnas de pipeline
# gebruikt, of geef de waarde expliciet als argument mee.
PIPELINE_ONLY = {
    # Kiezen WELKE detector/afleidingen draaien; een harnas dat zelf
    # afleidingen samenstelt bepaalt dit al met zijn eigen argumenten.
    "PSGSCORING_AROUSAL_DERIVATION",
    "PSGSCORING_AROUSAL_USES_ARTIFACT_EPOCHS",
    # Detectorgedrag, doorgegeven als ARGUMENT aan detect_arousals(_multi).
    # Een harnas dat die functie direct aanroept moet ze zelf meegeven; de
    # env bereikt hem niet. Dit is precies de val van 29-08 en de reden dat
    # PSGSCORING_AROUSAL_MIN_INTERVAL_S er niet meer in staat.
    "PSGSCORING_AROUSAL_EOG_REJECT",
    "PSGSCORING_AROUSAL_EVENT_LOCKED_THRESHOLD",
    "PSGSCORING_AROUSAL_HYSTERESIS",
    "PSGSCORING_AROUSAL_ONSET_OFFSET_S",
    "PSGSCORING_AROUSAL_SPECTRAL_SHIFT",
    # Stappen die alleen IN de pipeline bestaan: er is geen losse functie
    # waar een harnas ze aan voorbij zou kunnen komen.
    "PSGSCORING_AROUSAL_LIMB_WIRED",
    "PSGSCORING_RULE1A_AROUSAL",
    # Stap die alleen in de pipeline bestaat; het meetharnas gaat via
    # run_pneumo_analysis, dus de env bereikt hem daar.
    "PSGSCORING_CSR_RECLASSIFICATION",
    # Doorgegeven als ARGUMENT aan analyze_plm(bilateral_window_s=...). Een
    # harnas dat analyze_plm rechtstreeks aanroept moet het venster zelf
    # meegeven -- de env bereikt hem daar niet. De bilaterale meting gaat via
    # run_pneumo_analysis, dus daar werkt hij wel. Bewuste keuze, zelfde vorm
    # als de arousalvlaggen hierboven.
    "PSGSCORING_PLM_BILATERAL_WINDOW_S",
    # Doorgegeven als ARGUMENT aan detect_respiratory_events(). Een harnas dat
    # die functie direct aanroept moet de vlag zelf meegeven.
    "PSGSCORING_HYPOPNEA_SUBTYPE_AASM",
    # Idem: argument aan detect_respiratory_events(), doorgegeven aan beide
    # detectors en van daar aan classify_apnea_type().
    "PSGSCORING_PHASE_ANGLE_NEEDS_EFFORT",
    "PSGSCORING_SHAPE_EVIDENCE_GRADING",
    "PSGSCORING_SHAPE_EVIDENCE_SCALE",
    # Argument aan detect_arousals(); een harnas dat die functie direct
    # aanroept moet de vlag zelf meegeven.
    "PSGSCORING_SCORE_WAKE_AROUSALS",
    "PSGSCORING_AROUSAL_ALPHA_BAND_WIDE",
    "PSGSCORING_THERMISTOR_GATE",
    "PSGSCORING_PLM_EVENT_LIST_CAP",
    "PSGSCORING_PLM_OFFSET_AASM",
    "PSGSCORING_PLM_TIME_BASE",
    # Argumenten van score_hypopneas_breathwise, door de pipeline gezet.
    "PSGSCORING_BREATH_AROUSAL_LATENCY",
    "PSGSCORING_BREATH_CAND_MIN_DUR",
    "PSGSCORING_BREATH_TEMPLATE_DUR",
}


def test_alleen_via_pipeline_leesbaar_is_een_bewuste_lijst(kaart):
    pipe = kaart["pipeline.py"][0]
    elders = set().union(*(g for n, (g, _) in kaart.items() if n != "pipeline.py"))
    alleen_pipeline = pipe - elders

    nieuw = alleen_pipeline - PIPELINE_ONLY
    assert not nieuw, (
        "deze vlaggen worden ALLEEN in pipeline.py gelezen en staan niet in "
        "PIPELINE_ONLY. Elk meetharnas dat de detectoren rechtstreeks "
        "aanroept ziet ze niet — dat leverde op 29-08 vier keer een arm op "
        f"die niets deed. Beslis expliciet: {sorted(nieuw)}")

    verdwenen = PIPELINE_ONLY - alleen_pipeline
    assert not verdwenen, (
        "deze staan in PIPELINE_ONLY maar zijn inmiddels ook elders leesbaar; "
        f"haal ze uit de lijst: {sorted(verdwenen)}")


def test_de_vlaggen_van_29_08_zijn_buiten_de_pipeline_bereikbaar(kaart):
    """De drie die de val lieten springen, moeten leesbaar zijn zonder pipeline."""
    elders = set().union(*(g for n, (g, _) in kaart.items() if n != "pipeline.py"))
    for vlag in ("PSGSCORING_AROUSAL_MIN_INTERVAL_S",
                 "PSGSCORING_EVENT_GAP_TOLERANCE_BREATHS",
                 "PSGSCORING_EVENT_MIN_QUALIFYING_FRACTION"):
        assert vlag in elders, (
            f"{vlag} is alleen via de pipeline leesbaar; de meetharnassen "
            "roepen de detectoren rechtstreeks aan en zien hem dan niet")
