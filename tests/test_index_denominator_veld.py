"""`index_denominator`: de noemer van de AHI-familie op de klinische profielen is de volle slaaptijd.

Aanleiding 02-11-2026 (`docs/laatste10_humaan_vs_machine_20261102.md`): op de laatste tien eigen
PSG's deelde de machine door de slaap zónder artefact-epochs (234–511 per nacht), een menselijke
scoorder deelt door de TST. Op L05 gaf dat 130,5 tegen 97,1 /u bij dezelfde telling. Barts
beslissing: "AHI op volle TST benadert menselijke scoring. Pas toe."

Het veld geldt voor alle indices die uit `_compute_summary` komen (AHI-familie, REM/NREM, per
type), voor de RERA-index/RDI en de PLM-index; ODI gaat via `spo2_index_denominator` mee en de
randindices via `ancillary_tst_excludes_artifacts`. De detectie zelf blijft artefact-epochs
overslaan (sleep mask); alleen de noemer verandert. `mesa_shhs` en `chicago_1999` blijven op de
oude noemer (reproductie).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from psgscoring import pipeline
from psgscoring.constants import SCORING_PROFILES
from psgscoring.indices import (INDEX_DENOMINATORS, denominator_artifact_epochs,
                                index_denominator_from_profile)
from psgscoring.profiles import PROFILES, PostProcessingRules
from psgscoring.respiratory import _compute_summary
from tests.test_golden_output import CASES, make_raw

REPO = Path(__file__).resolve().parents[1]
HYPNO = ["W"] * 4 + ["N2"] * 20 + ["R"] * 8 + ["N3"] * 8      # 40 epochs: 36 slaap = 18,0 min
ART = [5, 6, 7, 30]                                           # 3 NREM-, 1 REM-epoch als artefact


def _events(n: int) -> list[dict]:
    return [{"type": "obstructive", "onset_s": 300.0 + 40.0 * i, "duration_s": 15.0,
             "stage": "N2", "epoch": 10 + i, "confidence": 0.9, "desaturation_pct": 4.0,
             "min_spo2": 90.0, "corroboration": None} for i in range(n)]


# ── veld en registry ───────────────────────────────────────────────────────

def test_veld_default_en_klinische_profielen():
    assert PostProcessingRules().index_denominator == "artifact_free_sleep"
    klinisch = [p for p in PROFILES.values() if p.family == "clinical"]
    assert len(klinisch) >= 7
    for p in klinisch:
        pp = p.post_processing
        assert pp.index_denominator == "sleep", p.name
        assert pp.spo2_index_denominator == "sleep", p.name
        assert pp.ancillary_tst_excludes_artifacts is False, p.name
    for naam in ("mesa_shhs", "chicago_1999"):
        assert PROFILES[naam].post_processing.index_denominator == "artifact_free_sleep", naam


def test_alle_niet_bevroren_profielen_delen_door_de_volle_slaaptijd():
    """Eén noemer voor elk profiel dat in de kliniek gekozen kan worden; alleen de
    twee reproductieprofielen houden de oude noemer."""
    for naam, p in PROFILES.items():
        pp = p.post_processing
        if naam in ("mesa_shhs", "chicago_1999"):
            assert pp.index_denominator == "artifact_free_sleep", naam
        else:
            assert pp.index_denominator == "sleep", naam
            assert pp.spo2_index_denominator == "sleep", naam
            assert pp.ancillary_tst_excludes_artifacts is False, naam


def test_afgeleide_profielen_erven_de_noemer_van_breath():
    for naam in ("aasm_v3_breath_dual", "aasm_v3_breath_dual_v2"):
        assert PROFILES[naam].post_processing.index_denominator == "sleep", naam


def test_legacy_sleutel_in_de_profieldict():
    assert SCORING_PROFILES["aasm_v3_rec"]["INDEX_DENOMINATOR"] == "sleep"
    assert SCORING_PROFILES["aasm_v3_breath_dual"]["INDEX_DENOMINATOR"] == "sleep"
    assert SCORING_PROFILES["mesa_shhs"]["INDEX_DENOMINATOR"] == "artifact_free_sleep"


def test_lezer_valideert_en_vertaalt_naar_artefactlijst():
    assert INDEX_DENOMINATORS == ("artifact_free_sleep", "sleep")
    assert index_denominator_from_profile(None) == "artifact_free_sleep"
    assert index_denominator_from_profile({}) == "artifact_free_sleep"
    assert index_denominator_from_profile({"INDEX_DENOMINATOR": "sleep"}) == "sleep"
    with pytest.raises(ValueError):
        index_denominator_from_profile({"INDEX_DENOMINATOR": "tib"})
    assert denominator_artifact_epochs({"INDEX_DENOMINATOR": "artifact_free_sleep"}, [1, 2]) == [1, 2]
    assert denominator_artifact_epochs({"INDEX_DENOMINATOR": "sleep"}, [1, 2]) is None
    assert denominator_artifact_epochs({"INDEX_DENOMINATOR": "sleep"}, None) is None


# ── _compute_summary ───────────────────────────────────────────────────────

def test_sleep_deelt_door_de_volle_slaaptijd_en_houdt_de_artefacttelling():
    oud = _compute_summary(_events(6), HYPNO, ART)
    nieuw = _compute_summary(_events(6), HYPNO, ART, denominator="sleep")
    assert oud["tst_minutes"] == 16.0 and oud["ahi_total"] == pytest.approx(6 / (16 / 60), abs=0.05)
    assert nieuw["tst_minutes"] == 18.0 and nieuw["ahi_total"] == pytest.approx(6 / (18 / 60), abs=0.05)
    assert nieuw["index_denominator_h"] == pytest.approx(0.3, abs=1e-3)
    assert (oud["rem_min"], oud["nrem_min"]) == (3.5, 12.5)
    assert (nieuw["rem_min"], nieuw["nrem_min"]) == (4.0, 14.0)
    assert nieuw["obstructive_index"] == pytest.approx(20.0, abs=0.05)
    # herkomst: beide tijden zichtbaar, de artefacttelling blijft
    assert nieuw["index_denominator"] == "sleep"
    assert oud["index_denominator"] == "artifact_free_sleep"
    assert nieuw["n_artifact_epochs_excluded"] == 4 and oud["n_artifact_epochs_excluded"] == 4
    assert nieuw["sleep_minutes"] == 18.0 and nieuw["artifact_free_sleep_minutes"] == 16.0
    assert oud["sleep_minutes"] == 18.0 and oud["artifact_free_sleep_minutes"] == 16.0


def test_default_is_het_oude_gedrag():
    a = _compute_summary(_events(6), HYPNO, ART)
    b = _compute_summary(_events(6), HYPNO, ART, denominator="artifact_free_sleep")
    assert a == b


def test_onbekende_noemer_wordt_geweigerd():
    with pytest.raises(ValueError):
        _compute_summary(_events(1), HYPNO, ART, denominator="tib")


def test_alle_slaap_als_artefact_blijft_onder_sleep_berekenbaar():
    alles = [i for i, s in enumerate(HYPNO) if s != "W"]
    oud = _compute_summary(_events(3), HYPNO, alles)
    nieuw = _compute_summary(_events(3), HYPNO, alles, denominator="sleep")
    assert oud["indices_computable"] is False and oud["ahi_total"] is None
    assert "artefactuitsluiting" in oud["index_unavailable_reason"]
    assert nieuw["indices_computable"] is True and nieuw["ahi_total"] == pytest.approx(10.0, abs=0.05)
    assert nieuw["index_unavailable_reason"] is None


# ── bedrading in de pipeline ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def _raw_en_hypno():
    cfg = dict(CASES["apnea_clean"])
    cfg.pop("profile"); cfg.pop("arousals", None)
    raw, hypno, channel_map = make_raw(**cfg)
    return raw, hypno, channel_map


def _spy(monkeypatch, naam, bewaar):
    origineel = getattr(pipeline, naam)

    def _wrapper(*args, **kwargs):
        bewaar.append((args, kwargs))
        return origineel(*args, **kwargs)

    monkeypatch.setattr(pipeline, naam, _wrapper)


@pytest.mark.parametrize("profiel, modus", [("aasm_v3_rec", "sleep"), ("mesa_shhs", "artifact_free_sleep")])
def test_pipeline_geeft_de_noemer_aan_elke_samenvatting(monkeypatch, _raw_en_hypno, profiel, modus):
    raw, hypno, channel_map = _raw_en_hypno
    summ: list = []; rera: list = []
    _spy(monkeypatch, "_compute_summary", summ)
    _spy(monkeypatch, "_compute_rera_rdi", rera)
    out = pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                       artifact_epochs=[3, 4], scoring_profile=profiel)
    assert summ, "_compute_summary niet via de pipeline aangeroepen"
    assert all(k.get("denominator") == modus for _, k in summ), [k.get("denominator") for _, k in summ]
    assert rera
    args, kwargs = rera[0]
    art = kwargs.get("artifact_epochs", args[3] if len(args) > 3 else None)
    assert art == (None if modus == "sleep" else [3, 4])
    s = out["respiratory"]["summary"]
    assert s["index_denominator"] == modus
    volle = sum(1 for st in hypno if st != "W") * 0.5
    assert s["sleep_minutes"] == pytest.approx(volle, abs=0.01)
    assert s["n_artifact_epochs_excluded"] == 2
    if modus == "sleep":
        assert s["tst_minutes"] == pytest.approx(volle, abs=0.01)
    else:
        assert s["tst_minutes"] == pytest.approx(volle - 1.0, abs=0.01)


def test_plm_index_volgt_de_noemer(monkeypatch, _raw_en_hypno):
    raw, hypno, channel_map = _raw_en_hypno
    if not any(r in channel_map for r in ("leg_l", "leg_r", "leg")):
        pytest.skip("golden-case zonder beenkanalen")
    gezien: list = []
    _spy(monkeypatch, "analyze_plm", gezien)
    pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                 artifact_epochs=[3, 4], scoring_profile="aasm_v3_rec")
    assert gezien and gezien[0][1].get("artifact_epochs") is None
    gezien.clear()
    pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                 artifact_epochs=[3, 4], scoring_profile="mesa_shhs")
    assert gezien and gezien[0][1].get("artifact_epochs") == [3, 4]


def test_elke_aanroep_van_de_samenvatting_draagt_de_noemer():
    """Broncontrole: een vergeten aanroep valt stil terug op de oude noemer."""
    for bestand in ("psgscoring/pipeline.py", "psgscoring/respiratory.py"):
        src = (REPO / bestand).read_text(encoding="utf-8")
        aanroepen = [m.start() for m in re.finditer(r"(?<!def )_compute_summary\((?!\))", src)]
        aanroepen = [pos for pos in aanroepen
                     if not src[src.rfind("\n", 0, pos) + 1: pos].lstrip().startswith("#")]   # geen commentaar
        assert aanroepen, bestand
        for pos in aanroepen:
            stuk = src[pos: pos + 400]
            assert "denominator=" in stuk.split(")\n")[0] + ")", f"{bestand}: aanroep zonder denominator= bij offset {pos}"
