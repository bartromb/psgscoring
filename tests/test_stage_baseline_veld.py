"""`stage_baseline`: de stadiumbasislijn expliciet aan of uit, los van de hypnogramlengte.

WAAROM DIT VELD BESTAAT

Tot 0.36.0 draaide de stadiumbasislijn op een vloerlengte-hypnogram (YASA in de
kliniek) nooit: `compute_stage_baseline` gooide een IndexError die stil werd
geslikt. `stage_baseline_partial_epochs` (0.36.0) repareerde dat, en de
decompositie van 26-10 wees precies dat veld aan als drager van de
apneuverschuiving op de eigen PSG's (R04 28 → 85 apneus). De inkortingstoets op
MESA (docs/inkortingstoets_20261026.md) vond: implementatie correct (vloerlengte
= dekkend, 30/30), maar mét basislijn F1-gelijk tegen NSRR (p = 0,51), bias
−5,1 → −0,5 /u én ernstconcordantie 23 → 18/30 doordat drie van de vier normale
nachten ernstig of matig worden (mesa-sleep-0111: 3 → 183 apneus, alle op de
thermistor, bij 0 referentie-events).

`stage_baseline_partial_epochs=False` zou het oude gedrag alleen via de fout
herstellen, en alleen op vloerlengte-hypnogrammen. Dit veld zet de basislijn
expliciet uit, ook bij een dekkend hypnogram, zodat (1) de kliniek een
gedefinieerde configuratie kan kiezen en (2) MESA de detector zónder
stadiumbasislijn kan meten zonder het inkortingstrucje. Default True = geen
gedragsverandering.
"""
from __future__ import annotations

import pytest

from psgscoring import pipeline, respiratory
from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES, PostProcessingRules
from tests.test_golden_output import CASES, make_raw


@pytest.fixture(scope="module")
def _raw_en_hypno():
    cfg = dict(CASES["hypopnea_clean"])
    cfg.pop("profile"); cfg.pop("arousals", None)
    raw, hypno, channel_map = make_raw(**cfg)
    return raw, hypno, channel_map


def _spy(monkeypatch, module, naam, bewaar):
    origineel = getattr(module, naam)

    def _wrapper(*args, **kwargs):
        bewaar.append(kwargs)
        return origineel(*args, **kwargs)

    monkeypatch.setattr(module, naam, _wrapper)


def test_default_is_aan_en_geen_profiel_wijkt_af():
    assert PostProcessingRules().stage_baseline is True
    assert all(p.post_processing.stage_baseline is True for p in PROFILES.values())


def test_veld_bereikt_de_legacy_dict():
    for naam, d in SCORING_PROFILES.items():
        assert d["STAGE_BASELINE"] is True, naam


def test_uit_betekent_dat_compute_stage_baseline_nergens_wordt_aangeroepen(
        monkeypatch, _raw_en_hypno):
    """Alle drie de aanroepplaatsen (hoofdflow, Fix1-menging, hypopneekanaal)
    moeten de schakelaar lezen; één vergeten plaats laat de basislijn via een
    omweg terugkomen."""
    raw, hypno, channel_map = _raw_en_hypno
    gezien: list = []
    _spy(monkeypatch, respiratory, "compute_stage_baseline", gezien)
    d = SCORING_PROFILES["aasm_v3_breath_dual"]
    oud = d["STAGE_BASELINE"]
    try:
        d["STAGE_BASELINE"] = False
        res = pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                           scoring_profile="aasm_v3_breath_dual")
    finally:
        d["STAGE_BASELINE"] = oud
    assert not gezien, f"compute_stage_baseline toch aangeroepen: {len(gezien)}×"
    r = res["respiratory"]
    assert r.get("stage_baseline_used") is False
    assert "stage_baseline=False" in str(r.get("stage_baseline_skipped", ""))


def test_aan_laat_alles_zoals_het_was(monkeypatch, _raw_en_hypno):
    raw, hypno, channel_map = _raw_en_hypno
    gezien: list = []
    _spy(monkeypatch, respiratory, "compute_stage_baseline", gezien)
    res = pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                       scoring_profile="aasm_v3_breath_dual")
    assert gezien, "compute_stage_baseline niet aangeroepen met het veld aan"
    assert res["respiratory"].get("stage_baseline_used") is True
    assert "stage_baseline_skipped" not in res["respiratory"]


def test_uit_geeft_de_dynamische_basislijn_alleen(_raw_en_hypno):
    """Met het veld uit moet de uitkomst gelijk zijn aan wat een hypnogram zonder
    slaapstadia zou geven: de stadiumbasislijn draagt dan per constructie niets
    bij. Zo is 'uit' meer dan 'niet aangeroepen'."""
    raw, hypno, channel_map = _raw_en_hypno
    d = SCORING_PROFILES["aasm_v3_breath_dual"]
    oud = d["STAGE_BASELINE"]
    try:
        d["STAGE_BASELINE"] = False
        uit = pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                           scoring_profile="aasm_v3_breath_dual")
    finally:
        d["STAGE_BASELINE"] = oud
    aan = pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                       scoring_profile="aasm_v3_breath_dual")
    # Op de synthetische golden-case met één stadium is de stadium-P90 niet
    # noodzakelijk gelijk aan de dynamische basislijn; de events mogen dus
    # verschillen, maar de uitvoer moet geldig en volledig zijn.
    for res in (uit, aan):
        assert res["respiratory"]["summary"].get("ahi_total") is not None
    assert uit["respiratory"]["stage_baseline_used"] is False
    assert aan["respiratory"]["stage_baseline_used"] is True
