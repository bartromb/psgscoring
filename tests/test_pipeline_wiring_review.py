"""De profielvelden uit de review van 11-10-2026 bereiken hun lezer via de
pipeline.

Elk veld heeft een lezer in zijn eigen module (`spo2`, `ancillary`, `plm`);
dit bestand toetst dat `run_pneumo_analysis` die lezers ook werkelijk
aanroept met het profiel en de artefactlijst. Zonder deze bedrading geldt
de default (= het gerepareerde gedrag) ook op de bevroren profielen, en
dan pinnen hun snapshots niets.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring import pipeline
from tests.test_golden_output import CASES, make_raw


@pytest.fixture(scope="module")
def _raw_en_hypno():
    cfg = dict(CASES["apnea_clean"])
    cfg.pop("profile"); cfg.pop("arousals", None)
    raw, hypno, channel_map = make_raw(**cfg)
    return raw, hypno, channel_map, None


def _spy(monkeypatch, module, naam, bewaar):
    origineel = getattr(module, naam)

    def _wrapper(*args, **kwargs):
        bewaar.append(kwargs)
        return origineel(*args, **kwargs)

    monkeypatch.setattr(pipeline, naam, _wrapper)


def test_analyze_spo2_krijgt_artefactlijst_en_noemer_uit_het_profiel(
        monkeypatch, _raw_en_hypno):
    raw, hypno, channel_map, _ = _raw_en_hypno
    gezien: list = []
    _spy(monkeypatch, pipeline, "analyze_spo2", gezien)
    pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                 artifact_epochs=[3, 4],
                                 scoring_profile="mesa_shhs")
    assert gezien, "analyze_spo2 niet aangeroepen"
    assert gezien[0].get("artifact_epochs") == [3, 4]
    assert gezien[0].get("index_denominator") == "sleep"   # pin mesa_shhs


def test_detect_cheyne_stokes_krijgt_de_gelijkrichtingsvlag(
        monkeypatch, _raw_en_hypno):
    raw, hypno, channel_map, _ = _raw_en_hypno
    gezien: list = []
    _spy(monkeypatch, pipeline, "detect_cheyne_stokes", gezien)
    pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                 scoring_profile="mesa_shhs")
    assert gezien, "detect_cheyne_stokes niet aangeroepen"
    assert all(k.get("period_rectified") is True for k in gezien)   # pin


def test_analyze_plm_krijgt_het_uitsluitingsvenster(monkeypatch, _raw_en_hypno):
    raw, hypno, channel_map, _ = _raw_en_hypno
    if not any(r in channel_map for r in ("leg_l", "leg_r", "leg")):
        pytest.skip("golden-case zonder beenkanalen")
    gezien: list = []
    _spy(monkeypatch, pipeline, "analyze_plm", gezien)
    pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                 scoring_profile="mesa_shhs")
    assert gezien and gezien[0].get("resp_exclusion") == "end"      # pin


def test_de_klinische_default_is_het_gerepareerde_gedrag(monkeypatch, _raw_en_hypno):
    raw, hypno, channel_map, _ = _raw_en_hypno
    gezien_spo2: list = []
    gezien_csr: list = []
    _spy(monkeypatch, pipeline, "analyze_spo2", gezien_spo2)
    _spy(monkeypatch, pipeline, "detect_cheyne_stokes", gezien_csr)
    pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                 scoring_profile="aasm_v3_rec")
    assert gezien_spo2[0].get("index_denominator") == "artifact_free_sleep"
    assert all(k.get("period_rectified") is False for k in gezien_csr)


def test_de_sweep_slaat_een_gecrashte_arm_over(monkeypatch, _raw_en_hypno):
    """Een arm met success=False mag niet als AHI 0 in het interval komen."""
    raw, hypno, channel_map, _ = _raw_en_hypno
    origineel = pipeline.detect_respiratory_events

    def _kapot(*args, **kwargs):
        from psgscoring.constants import SCORING_PROFILES
        if kwargs.get("scoring_profile") is SCORING_PROFILES["strict"] or kwargs.get("scoring_profile") is SCORING_PROFILES.get("aasm_v3_strict"):
            return {"success": False, "error": "kapot", "events": [],
                    "summary": {"ahi_total": 0.0}}
        return origineel(*args, **kwargs)

    monkeypatch.setattr(pipeline, "detect_respiratory_events", _kapot)
    out = pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                       scoring_profile="aasm_v3_rec")
    iv = out.get("ahi_interval") or {}
    # De gecrashte arm mag niet als AHI 0,0 in het interval terechtkomen.
    assert iv.get("strict", {}).get("ahi") is None
    assert iv.get("strict", {}).get("success") is False
    assert iv["interval"][0] > 0.0
