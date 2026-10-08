"""Bevroren U-Net-arousaldetector ``unet_v1`` (opt-in, default lgbm).

De replicatie van 27-09-2026 won op SHHS1/MESA/PSG-IPA met een bevroren
gewicht (sha256 cc91bb86…); de bibliotheek mag alleen dát gewicht draaien
(checksum-wacht), moet zonder EOG/kin-EMG hoorbaar terugvallen op de
LGBM-keten, en mag op geen enkel profiel standaard aan staan.
"""
import hashlib
import json
import logging
import os

import numpy as np
import pytest

from psgscoring import arousal_unet as au

ort = pytest.importorskip("onnxruntime")


def test_metadata_matches_the_shipped_weight():
    meta = json.loads(au.META_PATH.read_text())
    assert au.ONNX_PATH.exists()
    assert hashlib.sha256(au.ONNX_PATH.read_bytes()).hexdigest() == meta["onnx_sha256"]
    assert meta["source_sha256"].startswith("cc91bb86")
    assert meta["fs"] == 50 and meta["pad_multiple"] == 128
    ok, why = au.available()
    assert ok, why


def test_every_registered_profile_defaults_to_lgbm():
    from psgscoring.profiles import PROFILES
    assert all(p.post_processing.arousal_detector == "lgbm" for p in PROFILES.values())
    for naam in ("mesa_shhs", "chicago_1999"):
        assert PROFILES[naam].post_processing.arousal_detector == "lgbm"


def test_normalize_removes_offset_unit_rms_and_clips():
    """De bench normaliseerde in volt (mne) met epsilon 1e-7 in de RMS; de
    pipeline levert dezelfde volt (`_pick_eeg` → raw.get_data), dus de
    bibliotheek houdt die epsilon letterlijk. Op EEG-schaal (10 µV = 1e-5 V)
    weegt hij ~1 %; een factor 1e6 (µV) verschuift de uitkomst dus ≤ 1 %."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=50 * 60 * 60) * 1e-5 + 3.0e-5     # 60 min, 18 min-venster past
    a = au.normalize(x)
    b = au.normalize(x * 1e6)
    assert a.dtype == np.float32 and np.isfinite(a).all()
    assert np.abs(a).max() <= 20.0
    mid = slice(50 * 60 * 20, 50 * 60 * 40)               # buiten de randen
    assert abs(float(a[mid].mean())) < 0.05
    assert 0.95 < float(a[mid].std()) < 1.05
    assert np.abs(a - b).max() <= 0.15 and float(b[mid].std()) > float(a[mid].std())


def test_to_50hz_length_and_bandlimit():
    sf = 256.0
    t = np.arange(0, 120, 1 / sf)
    x = np.sin(2 * np.pi * 10 * t)
    y = au.to_50hz(x, sf)
    assert abs(y.size - 120 * 50) <= 1
    assert np.corrcoef(y[500:-500], np.sin(2 * np.pi * 10 * np.arange(y.size)[500:-500] / 50))[0, 1] > 0.99


def test_prob_to_events_threshold_gap_and_minimum_duration():
    fs = 50
    p = np.zeros(fs * 100)
    p[fs * 10: fs * 14] = 0.9            # 4 s → event
    p[fs * 20: fs * 22] = 0.9            # 2 s → te kort
    p[fs * 30: fs * 33] = 0.9            # 3 s en
    p[fs * 33 + fs // 2: fs * 36] = 0.9  # na gat van 0,5 s → samengevoegd
    ev = au.prob_to_events(p, fs, 0.5, smooth_s=0.0)
    assert [round(a) for a, _ in ev] == [10, 30]
    assert round(ev[1][1] - ev[1][0]) == 6


def test_gate_sleep_drops_events_that_start_in_wake():
    hypno = ["W", "N2", "N2", "R"]
    ev = [(10.0, 15.0), (40.0, 45.0), (100.0, 104.0)]
    assert au.gate_sleep(ev, hypno) == [(40.0, 45.0), (100.0, 104.0)]


def _synthetic_night(sf=256.0, minutes=20, seed=1):
    rng = np.random.default_rng(seed)
    n = int(sf * 60 * minutes)
    eeg = rng.normal(size=n) * 10.0
    eog = rng.normal(size=n) * 20.0
    emg = rng.normal(size=n) * 5.0
    hypno = ["N2"] * (2 * minutes)
    return eeg, eog, emg, hypno, sf


def test_unet_runs_end_to_end_and_reports_provenance():
    eeg, eog, emg, hypno, sf = _synthetic_night()
    r = au.detect_arousals_unet(eeg, sf, eog, sf, emg, sf, hypno, threshold=0.35, min_interval_s=10.0)
    assert r["success"], r.get("error")
    s = r["summary"]
    assert s["detector"] == "unet_v1" and s["unet_threshold"] == 0.35
    assert s["unet_onnx_sha256"] == json.loads(au.META_PATH.read_text())["onnx_sha256"]
    assert s["unet_n_samples"] == 20 * 60 * 50
    for e in r["events"]:
        assert e["duration_s"] >= 3.0 and e["derivation"] == "unet_v1"
        assert 0.0 <= e["confidence"] <= 1.0


def test_predict_pads_to_a_multiple_of_128_and_crops_back():
    X = np.zeros((3, 50 * 61), dtype=np.float32)
    p = au.predict(X)
    assert p.shape == (50 * 61,)
    assert np.all((p >= 0) & (p <= 1))


def test_run_arousal_analysis_falls_back_without_eog(caplog):
    from psgscoring.arousal import run_arousal_respiratory_analysis
    eeg, eog, emg, hypno, sf = _synthetic_night(minutes=10)
    with caplog.at_level(logging.WARNING, logger="psgscoring.arousal"):
        out = run_arousal_respiratory_analysis(
            eeg_data=eeg, sf_eeg=sf, flow_data=None, flow_norm=None, sf_flow=None,
            resp_events=[], hypno=hypno, emg_data=emg, eog_data=None,
            arousal_detector="unet_v1", lgbm=False)
    assert out["success"]
    assert out["summary"]["detector"] == "lgbm"
    assert "EOG" in out["summary"]["unet_fallback_reason"]
    assert "terugval op lgbm" in caplog.text


def test_run_arousal_analysis_uses_unet_when_everything_is_there():
    from psgscoring.arousal import run_arousal_respiratory_analysis
    eeg, eog, emg, hypno, sf = _synthetic_night(minutes=10)
    out = run_arousal_respiratory_analysis(
        eeg_data=eeg, sf_eeg=sf, flow_data=None, flow_norm=None, sf_flow=None,
        resp_events=[], hypno=hypno, emg_data=emg, eog_data=eog,
        arousal_detector="unet_v1", autonomic_rerank=True)
    assert out["success"]
    assert out["summary"]["detector"] == "unet_v1"
    assert "unet_fallback_reason" not in out["summary"]
    assert out["arousals"]["summary"]["autonomic_rerank"]["active"] is False


def test_env_override_selects_detector(monkeypatch, caplog):
    from psgscoring.pipeline import _arousal_detector, _arousal_unet_threshold
    monkeypatch.delenv("PSGSCORING_AROUSAL_DETECTOR", raising=False)
    assert _arousal_detector({"AROUSAL_DETECTOR": "lgbm"}) == "lgbm"
    monkeypatch.setenv("PSGSCORING_AROUSAL_DETECTOR", "unet_v1")
    assert _arousal_detector({"AROUSAL_DETECTOR": "lgbm"}) == "unet_v1"
    monkeypatch.setenv("PSGSCORING_AROUSAL_DETECTOR", "foo")
    with caplog.at_level(logging.WARNING, logger="psgscoring.pipeline"):
        assert _arousal_detector({"AROUSAL_DETECTOR": "lgbm"}) == "lgbm"
    assert "PSGSCORING_AROUSAL_DETECTOR" in caplog.text
    monkeypatch.setenv("PSGSCORING_AROUSAL_UNET_THRESHOLD", "0.25")
    assert _arousal_unet_threshold({"AROUSAL_UNET_THRESHOLD": 0.35}) == 0.25


def test_wrong_checksum_is_refused(monkeypatch):
    monkeypatch.setattr(au, "sha256_of", lambda p: "0" * 64)
    monkeypatch.delenv("PSGSCORING_UNET_SKIP_SHA", raising=False)
    ok, why = au.available()
    assert not ok and "checksum" in why
