"""Tweede U-Net-gewicht ``unet_v2`` (opt-in, schaalmeting 26-10-2026).

Arm MS uit `docs/arousal_unet_schaal_replicatie_20261026.md` (1 262 MESA + 1 235 SHHS1 nachten,
zelfde architectuur en ingangen als v1): SHHS1-150b mediane F1 0,670 → 0,739, MESA-76 +0,011,
PSG-IPA 4/5. Eigen werkpunt 0,30 (validatie-optimum; 0,35 ligt voor dit gewicht buiten het
vlakke stuk). ``unet_v1`` blijft de standaard; v2 is nergens profieldefault.
Het profielveld ``arousal_unet_threshold`` is nu ``None`` = werkpunt van het gewicht.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import numpy as np
import pytest

from psgscoring import arousal_unet as au
from psgscoring.profiles import PROFILES, PostProcessingRules


def test_registry_has_v2_with_matching_checksums():
    assert set(au.WEIGHTS) >= {"unet_v1", "unet_v2"}
    onnx, meta_p = au.paths("unet_v2")
    assert onnx.exists() and meta_p.exists()
    meta = json.loads(meta_p.read_text())
    assert hashlib.sha256(onnx.read_bytes()).hexdigest() == meta["onnx_sha256"]
    assert meta["source_sha256"].startswith("e75bee7a")
    assert meta["config"]["in_ch"] == 3
    assert au.default_threshold("unet_v2") == 0.30 == meta["tau_default"]
    assert au.default_threshold("unet_v1") == 0.35
    # v1 blijft wat het was
    assert au.paths("unet_v1") == (au.ONNX_PATH, au.META_PATH)
    with pytest.raises(KeyError):
        au.paths("unet_v9")


def test_models_manifest_lists_v2():
    reg = json.loads((au.DATA_DIR / "models.json").read_text())
    assert reg["arousal_unet_v2"]["file"] == "arousal_unet_v2.onnx"
    assert reg["arousal_unet_v2"]["sha256"] == json.loads(au.paths("unet_v2")[1].read_text())["onnx_sha256"]


def test_available_per_weight():
    pytest.importorskip("onnxruntime")
    assert au.available("unet_v2") == (True, None)
    assert au.available() == (True, None)


def _synth(sf=128.0, minutes=8, seed=11):
    rng = np.random.default_rng(seed)
    n = int(sf * 60 * minutes)
    eeg = rng.normal(0, 20e-6, n); eog = rng.normal(0, 30e-6, n); emg = rng.normal(0, 5e-6, n)
    t = np.arange(n) / sf
    for a in (60, 150, 300, 400):
        m = (t >= a) & (t < a + 6)
        eeg[m] += 60e-6 * np.sin(2 * np.pi * 10 * t[m]); emg[m] += 40e-6 * rng.normal(0, 1, m.sum())
    return eeg, eog, emg, sf, ["N2"] * (2 * minutes)


def test_detect_with_v2_reports_its_own_weight_and_threshold():
    pytest.importorskip("onnxruntime")
    eeg, eog, emg, sf, hypno = _synth()
    r2 = au.detect_arousals_unet(eeg, sf, eog, sf, emg, sf, hypno, weight="unet_v2")
    assert r2["success"], r2.get("error")
    s = r2["summary"]
    assert s["detector"] == "unet_v2" and s["unet_threshold"] == 0.30
    assert "arousal_unet_v2" in s["models"]
    assert s["unet_onnx_sha256"] == json.loads(au.paths("unet_v2")[1].read_text())["onnx_sha256"]
    for e in r2["events"]:
        assert e["derivation"] == "unet_v2"
    r1 = au.detect_arousals_unet(eeg, sf, eog, sf, emg, sf, hypno)
    assert r1["summary"]["detector"] == "unet_v1" and r1["summary"]["unet_threshold"] == 0.35
    # expliciete τ wint van het gewichtsdefault
    r2b = au.detect_arousals_unet(eeg, sf, eog, sf, emg, sf, hypno, weight="unet_v2", threshold=0.2)
    assert r2b["summary"]["unet_threshold"] == 0.2


def test_pipeline_helpers_resolve_threshold_per_weight(monkeypatch):
    from psgscoring.pipeline import _arousal_detector, _arousal_unet_threshold
    monkeypatch.delenv("PSGSCORING_AROUSAL_DETECTOR", raising=False)
    monkeypatch.delenv("PSGSCORING_AROUSAL_UNET_THRESHOLD", raising=False)
    assert _arousal_detector({"AROUSAL_DETECTOR": "unet_v2"}) == "unet_v2"
    monkeypatch.setenv("PSGSCORING_AROUSAL_DETECTOR", "unet_v2")
    assert _arousal_detector({"AROUSAL_DETECTOR": "lgbm"}) == "unet_v2"
    monkeypatch.delenv("PSGSCORING_AROUSAL_DETECTOR", raising=False)
    assert _arousal_unet_threshold({"AROUSAL_UNET_THRESHOLD": None}, "unet_v2") == 0.30
    assert _arousal_unet_threshold({"AROUSAL_UNET_THRESHOLD": None}, "unet_v1") == 0.35
    assert _arousal_unet_threshold({"AROUSAL_UNET_THRESHOLD": None}) == 0.35
    assert _arousal_unet_threshold({"AROUSAL_UNET_THRESHOLD": 0.2}, "unet_v2") == 0.2
    monkeypatch.setenv("PSGSCORING_AROUSAL_UNET_THRESHOLD", "0.25")
    assert _arousal_unet_threshold({"AROUSAL_UNET_THRESHOLD": None}, "unet_v2") == 0.25


def test_profile_field_none_means_weight_default_and_v2_is_nowhere_default():
    from psgscoring.constants import _profile_to_legacy_dict
    assert PostProcessingRules().arousal_unet_threshold is None
    assert _profile_to_legacy_dict(PROFILES["aasm_v3_breath_dual"])["AROUSAL_UNET_THRESHOLD"] is None
    for naam, p in PROFILES.items():
        assert p.post_processing.arousal_detector != "unet_v2", naam


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
    return raw, ["N2"] * (2 * minutes)


def test_pipeline_runs_v2_via_env_override(monkeypatch):
    pytest.importorskip("onnxruntime")
    import psgscoring
    monkeypatch.setenv("PSGSCORING_AROUSAL_DETECTOR", "unet_v2")
    monkeypatch.delenv("PSGSCORING_AROUSAL_UNET_THRESHOLD", raising=False)
    raw, hypno = _raw_with_eeg_eog_emg()
    res = psgscoring.run_pneumo_analysis(raw, hypno=hypno, scoring_profile="aasm_v3_breath_dual")
    s = (res.get("arousal") or {}).get("summary") or {}
    assert s.get("detector") == "unet_v2", s
    assert s.get("unet_threshold") == 0.30
    assert "unet_fallback_reason" not in s
    assert s.get("autonomic_rerank", {}).get("active") in (None, False)


def test_pipeline_v2_falls_back_to_lgbm_without_eog(monkeypatch):
    import mne, psgscoring
    monkeypatch.setenv("PSGSCORING_AROUSAL_DETECTOR", "unet_v2")
    raw, hypno = _raw_with_eeg_eog_emg()
    raw = raw.copy().drop_channels(["EOG E1-M2"])
    res = psgscoring.run_pneumo_analysis(raw, hypno=hypno, scoring_profile="aasm_v3_breath_dual")
    s = (res.get("arousal") or {}).get("summary") or {}
    assert s.get("detector") in (None, "lgbm")
    assert "unet_v2" in str(s.get("unet_fallback_reason", "")) or "EOG" in str(s.get("unet_fallback_reason", ""))
