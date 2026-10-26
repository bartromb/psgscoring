"""Modelidentiteit: één manifest, een checksumwacht, en de identiteit op het
leveringsoppervlak.

Het U-Net had al een checksumwacht (`arousal_unet.available`). De twee
LightGBM-boosters niet: een ander bestand met dezelfde naam laadde stil, en
de arousal-summary zei nergens WELK model er gedraaid had. `psgscoring/data/
models.json` legt per model bestand, sha256, feature-namen en herkomst vast;
de laders weigeren een afwijkende checksum voor de boosters en toetsen de
feature-geometrie; de summary draagt `models` met bestand en sha.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from psgscoring import arousal as ar
from psgscoring import arousal_unet as au
from psgscoring import ml_classifier as ml
from psgscoring import utils
from psgscoring.arousal import _AROUSAL_LGBM_FEATURE_ORDER
from psgscoring.ml_classifier import FEATURE_COLUMNS
from psgscoring.utils import MODELS_MANIFEST_PATH, check_model_file, models_manifest

DATA = Path(ar.__file__).parent / "data"


def test_het_manifest_bestaat_en_benoemt_de_vijf_modellen():
    assert MODELS_MANIFEST_PATH.exists()
    m = models_manifest()
    assert set(m) == {"arousal_classifier_v3", "lightgbm_v06_q7holdout",
                      "autonomic_rerank_v1", "arousal_unet_v1", "arousal_unet_v2"}


@pytest.mark.parametrize("sleutel", ["arousal_classifier_v3", "lightgbm_v06_q7holdout",
                                     "autonomic_rerank_v1", "arousal_unet_v1", "arousal_unet_v2"])
def test_elk_manifestitem_klopt_met_het_bestand(sleutel):
    item = models_manifest()[sleutel]
    for veld in ("file", "sha256", "n_features", "feature_names", "trained_on",
                 "frozen_date", "doc"):
        assert veld in item, veld
    pad = DATA / item["file"]
    assert pad.exists(), pad
    assert hashlib.sha256(pad.read_bytes()).hexdigest() == item["sha256"]
    assert len(item["feature_names"]) == item["n_features"]
    ident = check_model_file(sleutel, pad)
    assert ident == {"file": item["file"], "sha256": item["sha256"], "manifest": True}


def test_de_arousal_booster_draagt_de_featurevolgorde_van_de_code():
    item = models_manifest()["arousal_classifier_v3"]
    assert item["feature_names"] == list(_AROUSAL_LGBM_FEATURE_ORDER)
    assert item["n_features"] == 50


def test_de_kandidaat_booster_draagt_de_featurekolommen_van_de_code():
    item = models_manifest()["lightgbm_v06_q7holdout"]
    assert item["feature_names"] == list(FEATURE_COLUMNS)


def test_het_unet_item_hergebruikt_de_bestaande_checksum():
    assert models_manifest()["arousal_unet_v1"]["sha256"] == au.meta()["onnx_sha256"]


def test_een_vreemd_pad_wordt_niet_geweigerd_maar_wel_benoemd(tmp_path):
    p = tmp_path / "eigen_model.txt"
    p.write_text("x")
    ident = check_model_file("arousal_classifier_v3", p)
    assert ident["manifest"] is False and ident["file"] == "eigen_model.txt"
    assert ident["sha256"] == hashlib.sha256(b"x").hexdigest()


def test_een_afwijkende_checksum_wordt_geweigerd(tmp_path):
    p = tmp_path / "arousal_classifier_v3.txt"
    p.write_text("niet het model")
    with pytest.raises(ValueError, match="sha256"):
        check_model_file("arousal_classifier_v3", p)


# ── de laders ───────────────────────────────────────────────────────────

def test_de_arousal_lader_weigert_een_afwijkende_checksum(monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.setattr(ar, "_AROUSAL_LGBM_BOOSTER", None)
    monkeypatch.setattr(ar, "_AROUSAL_LGBM_IDENTITY", None)
    monkeypatch.setattr(utils, "file_sha256", lambda p: "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        ar._load_arousal_lgbm_booster()


def test_de_arousal_lader_toetst_het_aantal_features(monkeypatch):
    lgb = pytest.importorskip("lightgbm")
    monkeypatch.setattr(ar, "_AROUSAL_LGBM_BOOSTER", None)
    monkeypatch.setattr(ar, "_AROUSAL_LGBM_IDENTITY", None)
    echte = lgb.Booster

    class _Nep:
        def num_feature(self):
            return 49

    monkeypatch.setattr(lgb, "Booster", lambda model_file=None: _Nep())
    try:
        with pytest.raises(ValueError, match="49"):
            ar._load_arousal_lgbm_booster()
    finally:
        monkeypatch.setattr(lgb, "Booster", echte)


def test_de_kandidaat_lader_weigert_een_afwijkende_checksum(monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.setattr(ml, "_BOOSTER_CACHE", {})
    monkeypatch.setattr(ml, "_BOOSTER_IDENTITY", {})
    monkeypatch.setattr(utils, "file_sha256", lambda p: "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        ml.load_booster("data/lightgbm_v06_q7holdout.txt")


def test_de_kandidaat_lader_toetst_de_featurenamen(monkeypatch):
    lgb = pytest.importorskip("lightgbm")
    monkeypatch.setattr(ml, "_BOOSTER_CACHE", {})
    monkeypatch.setattr(ml, "_BOOSTER_IDENTITY", {})

    class _Nep:
        def feature_name(self):
            return ["a", "b"]

    monkeypatch.setattr(lgb, "Booster", lambda model_file=None: _Nep())
    with pytest.raises(ValueError, match="feature"):
        ml.load_booster("data/lightgbm_v06_q7holdout.txt")


def test_de_echte_boosters_laden_nog(monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.setattr(ar, "_AROUSAL_LGBM_BOOSTER", None)
    monkeypatch.setattr(ar, "_AROUSAL_LGBM_IDENTITY", None)
    monkeypatch.setattr(ml, "_BOOSTER_CACHE", {})
    monkeypatch.setattr(ml, "_BOOSTER_IDENTITY", {})
    assert ar._load_arousal_lgbm_booster().num_feature() == 50
    assert ml.load_booster("data/lightgbm_v06_q7holdout.txt").feature_name() == list(FEATURE_COLUMNS)
    assert ar._AROUSAL_LGBM_IDENTITY["sha256"] == models_manifest()["arousal_classifier_v3"]["sha256"]


# ── het leveringsoppervlak ──────────────────────────────────────────────

def test_de_arousal_summary_benoemt_het_model(monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.delenv("PSGSCORING_AROUSAL_LGBM", raising=False)
    sf, n_s = 128.0, 900
    rng = np.random.default_rng(3)
    t = np.arange(int(n_s * sf)) / sf
    eeg = rng.normal(0, 8.0, t.size) + 30.0 * np.sin(2 * np.pi * 1.5 * t)
    for b0 in (120, 300, 480, 660):
        m = (t >= b0) & (t < b0 + 6)
        eeg[m] += 60.0 * np.sin(2 * np.pi * 11.0 * t[m])
    emg = rng.normal(0, 5.0, t.size)
    uit = ar.detect_arousals(eeg, sf, ["N2"] * (n_s // 30), emg_data=emg, lgbm=True,
                             min_interval_s=10.0)
    s = uit["summary"]
    if "lgbm_n_pre" not in s:
        pytest.skip(f"classifier niet gedraaid: {s.get('lgbm_skipped_reason')}")
    verwacht = models_manifest()["arousal_classifier_v3"]
    assert s["models"]["arousal_classifier_v3"]["sha256"] == verwacht["sha256"]
    assert s["models"]["arousal_classifier_v3"]["file"] == verwacht["file"]


def test_de_ml_classifier_meta_benoemt_het_model():
    pytest.importorskip("lightgbm")
    kand = [{"onset_s": 100.0, "duration_s": 15.0, "type": "hypopnea", "stage": "N2"},
            {"onset_s": 400.0, "duration_s": 12.0, "type": "obstructive", "stage": "N2"}]
    _, _, meta = ml.apply_ml_reclassification(
        kand, [], [], ["N2"] * 40, 1200.0, 0.33, 5, 95.0, 0,
        booster_path="data/lightgbm_v06_q7holdout.txt")
    assert meta["status"] == "ok", meta
    verwacht = models_manifest()["lightgbm_v06_q7holdout"]
    assert meta["models"]["lightgbm_v06_q7holdout"]["sha256"] == verwacht["sha256"]


def test_de_unet_summary_meldt_de_echte_checksum_als_de_wacht_uit_staat(monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("PSGSCORING_UNET_SKIP_SHA", "1")
    monkeypatch.setattr(au, "meta", lambda: {"onnx_sha256": "deadbeef"})
    ok, why = au.available()
    assert ok, why
    sf, n_s = 128.0, 600
    rng = np.random.default_rng(1)
    eeg, eog, emg = (rng.normal(size=int(n_s * sf)) * a for a in (10.0, 20.0, 5.0))
    u = au.detect_arousals_unet(eeg, sf, eog, sf, emg, sf, ["N2"] * (n_s // 30))
    assert u["success"], u.get("error")
    echte = hashlib.sha256(au.ONNX_PATH.read_bytes()).hexdigest()
    assert u["summary"]["unet_onnx_sha256"] == echte
    assert u["summary"]["unet_skip_sha"] is True
    assert u["summary"]["models"]["arousal_unet_v1"] == {
        "file": "arousal_unet_v1.onnx", "sha256": echte, "skip_sha": True}
