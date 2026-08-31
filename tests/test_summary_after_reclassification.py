"""Issue #18: de samenvatting beschrijft een andere eventlijst dan het rapport toont.

De laatste ``_compute_summary()`` staat op stap 9; stap 11 herklassificeert
daarna events (CSR-gedreven obstructief→centraal, mixed-decompositie) en
vervangt de eventlijst. ``n_obstructive``, ``n_central``, ``n_mixed``, de
type-indices en ``oahi`` beschrijven daardoor de toestand VOOR die
herklassificatie, terwijl de eventlijst, de confidence-tabel en elke
per-event-figuur de toestand ERNA beschrijven.

In het rapport zichtbaar als een verschil tussen de n-kolom en de
confidence-uitsplitsing dat exact gelijk is aan ``n_csr_reclassified``. Niet
cosmetisch: op één echte opname schoof ``oahi`` 32,2 -> 28,6, over de grens
ernstig/matig heen. ``ahi_total`` beweegt niet — er wordt alleen omgelabeld.

De correctie zit achter ``summary_after_reclassification``; default uit, zodat
``mesa_shhs`` en de papercijfers byte-identiek blijven.
"""
from __future__ import annotations

import os
import warnings
from collections import Counter

import numpy as np
import pytest

import psgscoring
from psgscoring.constants import SCORING_PROFILES

SF = 32.0
DUR_S = 900
BR = 0.25


def _make_mixed_csr_raw():
    """Periodieke apneus (CSR-positief) waarvan een deel ademarbeid houdt.

    De arbeidhoudende apneus worden obstructief geclassificeerd en liggen
    binnen het CSR-venster — precies de events die stap 11 naar centraal
    omzet. Zonder die twee soorten naast elkaar is de test leeg.
    """
    mne = pytest.importorskip("mne")
    mne.set_log_level("ERROR")
    n = int(SF * DUR_S)
    t = np.arange(n) / SF
    rng = np.random.default_rng(7)

    flow_amp = np.ones(n)
    eff_amp = np.ones(n)
    spo2 = np.full(n, 96.0)
    # 13 s apneu in een cyclus van 40 s. Beide getallen zijn kritisch:
    # _flag_csr_events vergelijkt het inter-event interval met de gemeten
    # periodiciteit binnen 12 s, dus bij een apneu van 20 s valt geen enkel
    # event binnen de tolerantie en markeert de stap niets; onder 13 s wordt de
    # apneu niet meer gedetecteerd. Verander dit niet zonder
    # test_the_fixture_actually_reclassifies opnieuw te draaien.
    start, i = 60.0, 0
    while start + 13.0 < DUR_S - 20:
        t0, t1 = start, start + 13.0
        a, b = int(t0 * SF), int(t1 * SF)
        flow_amp[a:b] *= 0.02
        # Elke derde apneu houdt ademarbeid -> obstructief geclassificeerd.
        eff_amp[a:b] *= 1.0 if i % 3 == 2 else 0.05
        d0, nadir, d1 = int((t1 - 2) * SF), int((t1 + 4) * SF), int((t1 + 10) * SF)
        spo2[d0:nadir] = np.linspace(96.0, 91.0, nadir - d0)
        spo2[nadir:d1] = np.linspace(91.0, 96.0, d1 - nadir)
        start += 40.0
        i += 1

    # Thorax en buik lopen in fase (geen paradox) en houden de lage ruis van de
    # bestaande CSR-fixture: verhoog je die, dan valt de effort-classificatie
    # terug op "obstructief" bij gebrek aan bruikbare ademarbeid en blijft er
    # niets over om te herklassificeren.
    base = np.sin(2 * np.pi * BR * t)
    flow = flow_amp * base + rng.normal(0, 0.005, n)
    thorax = eff_amp * base + rng.normal(0, 0.005, n)
    abdomen = 0.9 * eff_amp * base + rng.normal(0, 0.005, n)

    data = np.vstack([flow, thorax, abdomen, spo2])
    info = mne.create_info(["FLOW", "THORAX", "ABDOMEN", "SPO2"], SF, ch_types="misc")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    hypno = ["W"] + ["N2"] * (DUR_S // 30 - 1)
    cmap = {"flow": "FLOW", "thorax": "THORAX", "abdomen": "ABDOMEN", "spo2": "SPO2"}
    return raw, hypno, cmap


def _run(profile="aasm_v3_rec"):
    """Draait MET de CSR-herclassificatie aan, en dat moet sinds 31-08-2026
    expliciet.

    Deze module toetst `summary_after_reclassification`: of de samenvatting en
    de eventlijst hetzelfde beeld geven nadat events van type wisselen. Die
    vraag heeft alleen zin als er events van type WISSELEN, en de
    CSR-herclassificatie is wat ze laat wisselen.

    Sinds die stap default UIT staat (kappa 0,090 tegen 0,185 op de
    CSR-nachten van MESA) zou de fixture stilzwijgend niets meer
    herclassificeren, en zouden deze vier tests groen blijven zonder iets te
    meten. De env-override zet hem hier aan; `mesa_shhs` en `chicago_1999`
    houden hem sowieso aan voor hun reproductie.
    """
    raw, hypno, cmap = _make_mixed_csr_raw()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _oud = os.environ.get("PSGSCORING_CSR_RECLASSIFICATION")
        os.environ["PSGSCORING_CSR_RECLASSIFICATION"] = "1"
        try:
            return psgscoring.run_pneumo_analysis(
                raw, hypno, channel_map=cmap, scoring_profile=profile)
        finally:
            if _oud is None:
                os.environ.pop("PSGSCORING_CSR_RECLASSIFICATION", None)
            else:
                os.environ["PSGSCORING_CSR_RECLASSIFICATION"] = _oud


def _counts(out):
    """(samenvatting, eventlijst) — de twee beelden die uiteen mochten lopen."""
    s = out["respiratory"]["summary"]
    c = Counter(e.get("type") for e in out["respiratory"]["events"])
    return (
        {"obstructive": s.get("n_obstructive"), "central": s.get("n_central"),
         "mixed": s.get("n_mixed")},
        {"obstructive": c.get("obstructive", 0), "central": c.get("central", 0),
         "mixed": c.get("mixed", 0)},
    )


@pytest.fixture(scope="module")
def clinical_run():
    """Zoals de software hem uitlevert: `aasm_v3_rec` met de vlag aan."""
    return _run()


@pytest.fixture
def pinned_run(monkeypatch):
    """Het gedrag van de reproductieprofielen, waar de vlag uit blijft."""
    monkeypatch.setitem(
        SCORING_PROFILES, "aasm_v3_rec",
        {**SCORING_PROFILES["aasm_v3_rec"],
         "SUMMARY_AFTER_RECLASSIFICATION": False},
    )
    return _run()


def test_the_fixture_actually_reclassifies(clinical_run):
    """De wacht bij de wacht: zonder herklassificatie is alles hieronder leeg."""
    assert clinical_run["cheyne_stokes"].get("csr_detected") is True
    assert clinical_run["postprocess"]["n_csr_reclassified"] > 0


def test_a_pinned_profile_keeps_the_pre_reclassification_summary(pinned_run):
    """Karakterisering, geen goedkeuring: dit is het beeld dat `mesa_shhs` en
    `chicago_1999` blijven tonen, en precies wat het rapport tegensprak."""
    summary, events = _counts(pinned_run)
    n_recl = pinned_run["postprocess"]["n_csr_reclassified"]
    assert summary != events
    assert summary["obstructive"] - events["obstructive"] == n_recl
    assert events["central"] - summary["central"] == n_recl


def test_a_clinical_profile_shows_one_and_the_same_picture(clinical_run):
    summary, events = _counts(clinical_run)
    assert clinical_run["postprocess"]["n_csr_reclassified"] > 0
    assert summary == events


def test_the_flag_moves_the_labels_not_the_ahi(pinned_run, clinical_run):
    """ahi_total telt events; de herklassificatie voegt er geen toe en laat
    er geen weg. Alleen oahi en de type-indices mogen bewegen."""
    a = pinned_run["respiratory"]["summary"]
    b = clinical_run["respiratory"]["summary"]
    assert b["ahi_total"] == pytest.approx(a["ahi_total"], abs=1e-9)
    assert len(pinned_run["respiratory"]["events"]) == \
        len(clinical_run["respiratory"]["events"])
    assert b["oahi"] < a["oahi"], "obstructief -> centraal hoort de OAHI te verlagen"


def test_the_recompute_does_not_drop_the_keys_that_step_9b_added(clinical_run):
    """Dezelfde val als in v0.7.4: een kale vervanging gooide RERA/RDI weg.
    Daarom mergen we in plaats van te vervangen."""
    s = clinical_run["respiratory"]["summary"]
    for key in ("rdi", "rera_index", "n_rera"):
        assert s.get(key) is not None, f"{key} verdwenen na de herberekening"


def test_every_selectable_profile_reports_the_final_events():
    """Alles wat een clinicus kan kiezen, hoort één beeld te tonen."""
    from psgscoring.profiles import PROFILES
    off = [name for name, p in PROFILES.items()
           if p.family in ("clinical", "exploratory")
           and not p.post_processing.summary_after_reclassification]
    assert off == [], f"selecteerbare profielen zonder de correctie: {off}"


def test_the_reproduction_profiles_are_pinned():
    """Papercijfers en NSRR-reproductie mogen hier niet in meelopen."""
    from psgscoring.profiles import PROFILES
    on = [name for name, p in PROFILES.items()
          if p.family in ("dataset", "legacy")
          and p.post_processing.summary_after_reclassification]
    assert on == [], f"reproductieprofielen met de vlag aan: {on}"
    assert SCORING_PROFILES["mesa_shhs"].get(
        "SUMMARY_AFTER_RECLASSIFICATION") is False


def test_the_dataclass_default_stays_off():
    """De veilige faalstand: een nieuw datasetprofiel dat je vergeet vast te
    pinnen, breekt anders stilzwijgend de reproductie."""
    from psgscoring.profiles import PostProcessingRules
    assert PostProcessingRules().summary_after_reclassification is False
