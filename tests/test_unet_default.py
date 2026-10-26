"""Standaardkeuze van de arousaldetector.

Beslissing van de gebruiker op 2026-10-25, ná de pre-geregistreerde replicatie van
27-09 (`docs/arousal_unet_replicatie_20260927.md`: SHHS1 150 verse nachten F1
0,543 → 0,676 op 134/147, MESA-76 0,556 → 0,687, PSG-IPA 0,745 op 5/5) en de
doorwerkingsmeting van 09-10 (`docs/unet_doorwerking_20261009.md`): `unet_v1` is
de standaard op de klinische en exploratieve profielen; `lgbm` blijft GEPIND op
de vijf profielen die een externe regelset of een gepubliceerde dataset-analyse
reproduceren (zelfde groep die `arousal_lgbm=False` pint). Het werkpunt τ blijft
0,35 (tellingsneutraal op PSG-IPA); 0,25 (NSRR-neutraal) is een open beslissing.

Deze tests faalden eerst op de oude code (default "lgbm" op alle 22 profielen).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from psgscoring.profiles import PROFILES, PostProcessingRules

GEPIND_LGBM = {"aasm_v2_rec", "aasm_v1_rec", "cms_medicare", "mesa_shhs", "chicago_1999"}


def test_field_default_is_unet_v1():
    pp = PostProcessingRules()
    assert pp.arousal_detector == "unet_v1"
    # Het werkpunt is bewust NIET veranderd: 0,35 is het PSG-IPA-tellingsneutrale
    # punt uit de replicatie; de keuze 0,35/0,25 staat open.
    assert pp.arousal_unet_threshold == 0.35


def test_clinical_and_exploratory_profiles_default_to_unet():
    for naam, p in PROFILES.items():
        verwacht = "lgbm" if naam in GEPIND_LGBM else "unet_v1"
        assert p.post_processing.arousal_detector == verwacht, naam


def test_reproduction_profiles_stay_on_lgbm():
    # Dezelfde vijf als de `arousal_lgbm=False`-groep: een U-Net maakt geen deel
    # uit van AASM v1/v2 of de CMS-regels, en mesa_shhs/chicago_1999 moeten
    # paper v31/v37 reproduceren.
    assert GEPIND_LGBM <= set(PROFILES)
    for naam in GEPIND_LGBM:
        pp = PROFILES[naam].post_processing
        assert pp.arousal_detector == "lgbm", naam
        assert pp.arousal_lgbm is False, naam


def test_legacy_dict_carries_the_default():
    from psgscoring.constants import _profile_to_legacy_dict
    assert _profile_to_legacy_dict(PROFILES["aasm_v3_breath_dual"])["AROUSAL_DETECTOR"] == "unet_v1"
    assert _profile_to_legacy_dict(PROFILES["mesa_shhs"])["AROUSAL_DETECTOR"] == "lgbm"


def _raw_with_eeg_eog_emg(sf=128.0, minutes=10, seed=7):
    import mne
    rng = np.random.default_rng(seed)
    n = int(sf * 60 * minutes)
    t = np.arange(n) / sf
    resp = np.sin(2 * np.pi * 0.25 * t)
    chans = [resp, 0.8 * resp, 0.7 * resp, np.full(n, 96.0),
             rng.normal(0, 20e-6, n), rng.normal(0, 30e-6, n), rng.normal(0, 5e-6, n)]
    names = ["FLOW", "THORAX", "ABDOMEN", "SPO2", "EEG C4-M1", "EOG E1-M2", "EMG chin"]
    types = ["misc", "misc", "misc", "misc", "eeg", "eog", "emg"]
    info = mne.create_info(names, sf, ch_types=types)
    raw = mne.io.RawArray(np.vstack(chans), info, verbose=False)
    hypno = ["N2"] * (2 * minutes)
    return raw, hypno


@pytest.mark.skipif(os.environ.get("PSGSCORING_AROUSAL_DETECTOR") is not None,
                    reason="env-override actief; de profielstandaard is dan niet zichtbaar")
def test_pipeline_runs_unet_under_the_production_profile_without_env():
    pytest.importorskip("onnxruntime")
    import psgscoring
    raw, hypno = _raw_with_eeg_eog_emg()
    res = psgscoring.run_pneumo_analysis(raw, hypno=hypno, scoring_profile="aasm_v3_breath_dual")
    s = (res.get("arousal") or {}).get("summary") or {}
    assert s.get("detector") == "unet_v1", s
    assert "unet_fallback_reason" not in s


@pytest.mark.skipif(os.environ.get("PSGSCORING_AROUSAL_DETECTOR") is not None,
                    reason="env-override actief")
def test_pipeline_keeps_lgbm_chain_on_mesa_shhs_without_env():
    import psgscoring
    raw, hypno = _raw_with_eeg_eog_emg()
    res = psgscoring.run_pneumo_analysis(raw, hypno=hypno, scoring_profile="mesa_shhs")
    s = (res.get("arousal") or {}).get("summary") or {}
    # "afwezig = gewone LGBM-keten" (0.35.0); een expliciete "lgbm" mag ook.
    assert s.get("detector") in (None, "lgbm"), s
