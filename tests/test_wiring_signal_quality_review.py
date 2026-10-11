"""Bedrading van de signaal-/kwaliteitsvelden uit de review van 11-10-2026:
`stage_baseline_partial_epochs`, `linearise_after_dc_removal`,
`thermistor_band_power_antialias` en de genegeerde kanaaloverrides.

De lezers zitten in `signal`, `signal_quality` en `utils`; zonder deze
bedrading geldt de keyword-default (= het gerepareerde gedrag) ook op de
bevroren profielen en pinnen hun snapshots niets.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring import pipeline, respiratory, signal_quality
from psgscoring.constants import SCORING_PROFILES
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


@pytest.mark.parametrize("profiel, verwacht", [("mesa_shhs", False), ("aasm_v3_rec", True)])
def test_compute_stage_baseline_krijgt_partial_epochs_van_het_profiel(
        monkeypatch, _raw_en_hypno, profiel, verwacht):
    raw, hypno, channel_map = _raw_en_hypno
    gezien: list = []
    # respiratory importeert de naam zelf (`from .signal import compute_stage_baseline`).
    _spy(monkeypatch, respiratory, "compute_stage_baseline", gezien)
    pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                 scoring_profile=profiel)
    assert gezien, "compute_stage_baseline niet aangeroepen"
    # De primaire pas staat voorop; de intervalsweep draait zijn eigen profielen.
    assert gezien[0].get("partial_epochs") is verwacht


def _synthetisch(sf=32.0, dur_s=600.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sf * dur_s)) / sf
    flow = np.sin(2 * np.pi * 0.25 * t) + 0.01 * rng.standard_normal(t.size)
    pres = flow ** 2 * np.sign(flow) + 0.5          # drukachtig, met offset
    hypno = ["N2"] * int(dur_s // 30)
    return flow, pres, hypno, sf


@pytest.mark.parametrize("vlag", [True, False])
def test_preprocess_flow_krijgt_de_dc_verwijderingsvlag_voor_het_drukkanaal(
        monkeypatch, vlag):
    flow, pres, hypno, sf = _synthetisch()
    sp = dict(SCORING_PROFILES["aasm_v3_dual"])
    sp["LINEARISE_AFTER_DC_REMOVAL"] = vlag
    gezien: list = []
    _spy(monkeypatch, respiratory, "preprocess_flow", gezien)
    respiratory.detect_respiratory_events(
        flow_data=flow, sf_flow=sf, hypop_flow=pres, sf_hypop=sf,
        thorax_data=None, abdomen_data=None,
        spo2_data=None, sf_spo2=1.0, hypno=hypno, scoring_profile=sp)
    druk = [k for k in gezien if k.get("is_nasal_pressure")]
    assert druk, "geen preprocess_flow-aanroep voor het drukkanaal gezien"
    assert all(k.get("linearise_after_dc_removal") is vlag for k in druk)


@pytest.mark.parametrize("vlag", [True, False])
def test_resolve_flow_channels_geeft_antialias_door_aan_de_bandvermogenpoort(
        monkeypatch, vlag):
    flow, pres, hypno, sf = _synthetisch()
    gezien: list = []
    _spy(monkeypatch, signal_quality, "assess_thermistor_band_power", gezien)
    output = {"meta": {}}
    ch = {"flow": "Therm", "flow_pressure": "Pres"}
    pipeline._resolve_flow_channels(
        flow, sf, pres, sf, flow, sf, ch, output,
        additive_thermistor=True, thermistor_gate="respiratory_band",
        thermistor_band_power_antialias=vlag)
    assert gezien and gezien[0].get("antialias") is vlag


def test_de_pipeline_leest_antialias_uit_het_profiel(monkeypatch, _raw_en_hypno):
    """De aanroep in run_pneumo_analysis geeft de profielwaarde door."""
    raw, hypno, channel_map = _raw_en_hypno
    gezien: list = []
    _spy(monkeypatch, pipeline, "_resolve_flow_channels", gezien)
    pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                 scoring_profile="mesa_shhs")
    assert gezien and gezien[0].get("thermistor_band_power_antialias") is False


def test_een_override_naar_een_onbestaand_kanaal_staat_in_meta(_raw_en_hypno):
    raw, hypno, channel_map = _raw_en_hypno
    cm = dict(channel_map); cm["flow"] = "Bestaat Niet"
    out = pipeline.run_pneumo_analysis(raw, hypno, channel_map=cm,
                                       scoring_profile="aasm_v3_rec")
    genegeerd = out["meta"].get("channel_overrides_ignored")
    assert genegeerd and any(g.get("channel") == "Bestaat Niet" for g in genegeerd)
