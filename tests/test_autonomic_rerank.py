"""De autonome re-ranker: vlag, gedragsneutraliteit en het bevroren recept.

Pleth fase 1 (2026-09-08) repliceerde: een logistische re-ranker
[LGBM-kans, PWA-min-ratio, PWA-vlag, HR-stijging, duur, REM] bovenop de
kandidatenlijst geeft ΔF1 +0,0097 (30/40, p=0,0001) zonder tellingsschade.
Deze tests borgen dat de bibliotheekimplementatie exact dát recept draagt:

  1. default UIT, overal — geen profiel verandert gedrag;
  2. vlag aan zonder bruikbare pleth/HR = het ongewijzigde pad, met reden
     in de provenance (het gevalideerde domein had beide getuigen);
  3. vlag aan mét signalen: top-K-selectie (K = drempelkeuze), zelfde
     samenvoegregel, provenance zichtbaar op het leveringsoppervlak;
  4. het bevroren model in de package is byte-voor-byte het
     replicatiemodel (coëfficiëntenwacht).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from psgscoring.arousal import (
    _AUTONOMIC_MODEL,
    autonomic_rerank_selection,
    detect_arousals_multi,
)
from psgscoring.profiles import PROFILES

SF = 100.0


def _synth_eeg(n_s=1200, bursts=()):
    rng = np.random.default_rng(7)
    x = rng.normal(0, 8e-6, int(n_s * SF)) * 1e6  # µV-schaal na *1e6? nee: al µV
    x = rng.normal(0, 8.0, int(n_s * SF))
    t = np.arange(int(n_s * SF)) / SF
    for b0 in bursts:
        m = (t >= b0) & (t < b0 + 5)
        x[m] += 60.0 * np.sin(2 * np.pi * 11.0 * t[m])
    return x


def _hypno(n_s=1200):
    return ["N2"] * int(n_s // 30)


def _pleth_vlak(n_s=1200, sf=SF):
    t = np.arange(int(n_s * sf)) / sf
    return np.sin(2 * np.pi * 1.1 * t)  # stabiele pols, geen dalingen


def _hr_vlak(n_s=1200, sf=SF, stijg_bij=None):
    r = np.full(int(n_s * sf), 60.0)
    if stijg_bij is not None:
        i0 = int(stijg_bij * sf)
        r[i0:i0 + int(8 * sf)] = 75.0
    return r


# ── 1. vlagstand per profiel ─────────────────────────────────────────────

def test_vlag_alleen_aan_op_rec_en_zijn_meetarmen():
    """Gebruikersbeslissing 2026-09-08: AAN op aasm_v3_rec (het klinische
    standaardprofiel) én op de meetarmen die per contract "rec + precies
    één knop" zijn — die moeten het anker volgen, anders is elke gepaarde
    meting voortaan een twee-knopsvergelijking. UIT op al het andere; de
    bevroren profielen (mesa_shhs, chicago_1999, ...) blijven gepind
    False voor byte-identiteit."""
    AAN = {"aasm_v3_rec", "aasm_v3_pair_scalefree", "aasm_v3_amplitude",
           "aasm_v3_env_chunked", "aasm_v3_env_rectify",
           "aasm_v3_env_breath", "aasm_v3_env_decimated"}
    for naam, p in PROFILES.items():
        verwacht = naam in AAN
        assert p.post_processing.arousal_autonomic_rerank is verwacht, (
            f"{naam}: verwacht {verwacht}")


# ── 4. bevroren model ────────────────────────────────────────────────────

def test_bevroren_model_is_het_replicatiemodel():
    """Coëfficiëntenwacht: het gevalideerde model mag alleen bewust
    veranderen (nieuwe afleiding + replicatie), nooit door drift."""
    assert _AUTONOMIC_MODEL["features"] == [
        "proba", "pwa_min_ratio", "pwa_flag", "hr_rise", "d", "is_rem"]
    np.testing.assert_allclose(
        _AUTONOMIC_MODEL["coef"][0], 2.152465310304989, rtol=1e-12)
    np.testing.assert_allclose(
        _AUTONOMIC_MODEL["intercept"], -5.381246712500639, rtol=1e-12)
    assert _AUTONOMIC_MODEL["n_kandidaten"] == 59831


# ── 2/3. gedrag ─────────────────────────────────────────────────────────

def _kandidaten():
    return [
        {"onset_s": 100.0, "duration_s": 6.0, "proba": 0.90},
        {"onset_s": 200.0, "duration_s": 5.0, "proba": 0.85},
        {"onset_s": 300.0, "duration_s": 4.0, "proba": 0.60},
        {"onset_s": 400.0, "duration_s": 5.0, "proba": 0.55},
    ]


def test_selectie_houdt_k_van_de_drempel():
    """K = het aantal kandidaten dat de drempel haalt (hier 2); de
    re-ranker mag herordenen, niet bijtellen."""
    ev, prov = autonomic_rerank_selection(
        _kandidaten(), threshold=0.80,
        pleth=_pleth_vlak(), sf_pleth=SF,
        hr=_hr_vlak(stijg_bij=400.0), sf_hr=SF,
        hypno=_hypno(), min_interval_s=10.0)
    assert prov["active"] is True
    assert prov["k"] == 2
    assert len(ev) <= 2


def test_hr_stijging_kan_herordenen():
    """Kandidaat 400 s (proba 0,55) krijgt een HR-stijging van 15 bpm;
    kandidaat 300 s (proba 0,60) niets. Met een kansverschil van maar
    0,05 hoort de autonome getuige de rangorde te kunnen draaien —
    dat is precies wat fase 1 gevalideerd heeft."""
    ev, prov = autonomic_rerank_selection(
        _kandidaten(), threshold=0.58,   # K = 3
        pleth=_pleth_vlak(), sf_pleth=SF,
        hr=_hr_vlak(stijg_bij=400.0), sf_hr=SF,
        hypno=_hypno(), min_interval_s=10.0)
    onsets = [e["onset_s"] for e in ev]
    assert 400.0 in onsets, (
        "de HR-gesteunde kandidaat hoort de derde plek te winnen")
    assert 300.0 not in onsets


def test_zonder_pleth_ongewijzigd_pad_met_reden():
    ev, prov = autonomic_rerank_selection(
        _kandidaten(), threshold=0.80,
        pleth=None, sf_pleth=None,
        hr=_hr_vlak(), sf_hr=SF,
        hypno=_hypno(), min_interval_s=10.0)
    assert ev is None
    assert prov["active"] is False
    assert "pleth" in prov["reason"]


def test_multi_pad_vlag_uit_is_onaangeroerd():
    """Zonder vlag verandert er NIETS aan het multi-pad — geen
    autonomic-provenance, zelfde events als altijd."""
    der = [("EEG3", _synth_eeg(bursts=(100, 200))),
           ("EEG1", _synth_eeg(bursts=(100,)))]
    uit = detect_arousals_multi(der, SF, _hypno(), min_interval_s=10.0)
    assert "autonomic_rerank" not in (uit.get("summary") or {})


def test_multi_pad_vlag_aan_zonder_signalen_meldt_reden():
    der = [("EEG3", _synth_eeg(bursts=(100, 200))),
           ("EEG1", _synth_eeg(bursts=(100,)))]
    uit = detect_arousals_multi(der, SF, _hypno(), min_interval_s=10.0,
                                autonomic_rerank=True)
    prov = (uit.get("summary") or {}).get("autonomic_rerank")
    assert prov is not None, "de vlagstatus hoort op het leveringsoppervlak"
    assert prov["active"] is False


def test_score_is_de_bevroren_logistiek():
    """De score van één kandidaat is exact sigmoid(coef·z + intercept) met
    de bevroren mu/sd — geen hertraining, geen andere normalisatie."""
    from psgscoring.arousal import _autonomic_score
    feats = np.array([[0.90, 1.0, 0.0, 15.0, 6.0, 0.0]])
    m = _AUTONOMIC_MODEL
    z = (feats - np.array(m["mu"])) / np.array(m["sd"])
    verwacht = 1.0 / (1.0 + np.exp(-(z @ np.array(m["coef"]) + m["intercept"])))
    np.testing.assert_allclose(_autonomic_score(feats), verwacht, rtol=1e-12)


def test_pipeline_levert_de_provenance(monkeypatch):
    """Leveringsoppervlak: via run_pneumo_analysis met env-vlag aan draagt de
    arousal-summary de rerank-provenance; zonder vlag staat er niets.

    Zelfde les als de drempelkoppeling: een vlag die alleen in de registry
    bestaat is decoratie."""
    import mne
    import psgscoring

    sf, n_s = 128.0, 1200.0
    n = int(sf * n_s)
    rng = np.random.default_rng(7)
    namen = ["EEG1", "EEG2", "EEG3", "EMG", "Pres", "Pleth", "HR"]
    info = mne.create_info(namen, sf, ch_types="misc", verbose=False)
    data = rng.normal(0, 2e-5, (len(namen), n))
    t = np.arange(n) / sf
    data[namen.index("Pleth")] = np.sin(2 * np.pi * 1.1 * t)
    data[namen.index("HR")] = 60.0
    hypno = ["N2"] * int(n_s // 30)

    monkeypatch.setenv("PSGSCORING_AROUSAL_AUTONOMIC_RERANK", "1")
    uit = psgscoring.run_pneumo_analysis(
        mne.io.RawArray(data, info, verbose=False), hypno=hypno,
        scoring_profile="aasm_v3_rec")
    s = (uit.get("arousal") or {}).get("summary") or {}
    if "lgbm_threshold" not in s:
        pytest.skip("LGBM-model niet beschikbaar in deze omgeving")
    assert "autonomic_rerank" in s, (
        "de vlagstatus hoort op het leveringsoppervlak")

    # Sinds 08-09 staat de vlag AAN op aasm_v3_rec: zonder env hoort de
    # provenance er dus ook te staan, en env=0 dwingt hem uit.
    monkeypatch.delenv("PSGSCORING_AROUSAL_AUTONOMIC_RERANK")
    uit2 = psgscoring.run_pneumo_analysis(
        mne.io.RawArray(data, info, verbose=False), hypno=hypno,
        scoring_profile="aasm_v3_rec")
    s2 = (uit2.get("arousal") or {}).get("summary") or {}
    assert "autonomic_rerank" in s2

    monkeypatch.setenv("PSGSCORING_AROUSAL_AUTONOMIC_RERANK", "0")
    uit3 = psgscoring.run_pneumo_analysis(
        mne.io.RawArray(data, info, verbose=False), hypno=hypno,
        scoring_profile="aasm_v3_rec")
    s3 = (uit3.get("arousal") or {}).get("summary") or {}
    assert "autonomic_rerank" not in s3
