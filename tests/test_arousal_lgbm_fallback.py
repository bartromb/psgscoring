"""Arousal: wat er gebeurt als de hybride niet kan draaien.

In hybride modus (`PSGSCORING_AROUSAL_LGBM=1`) zet `detect_arousals` de
drempels op de RUIME kandidaatwaarden (ratio 1,2 / abrupt 1,0) en laat een
LightGBM-model daarna wegfilteren. Faalt dat model -- bestand ontbreekt,
lightgbm niet geinstalleerd, corrupte booster -- dan logt de code
"falling back to rule-based output", maar `result["events"]` bevat op dat
moment de KANDIDATEN, niet de regelgebaseerde uitkomst.

Dat is het gevaarlijke geval: een installatie zonder model levert dan stil een
veel te hoge arousal-index, met een logregel die het tegendeel beweert. Zolang
de vlag opt-in is, is het een voetangel; wordt hij ooit default, dan is het
een productiedefect.
"""
import numpy as np
import pytest

from psgscoring.arousal import AROUSAL_RATIO_THRESH, detect_arousals

SF = 100.0
DUR_S = 900


def _recording(seed=11):
    t = np.arange(int(DUR_S * SF)) / SF
    rng = np.random.default_rng(seed)
    eeg = (60.0 * np.sin(2 * np.pi * 1.5 * t)
           + 6.0 * np.sin(2 * np.pi * 6.0 * t)
           + 4.0 * np.sin(2 * np.pi * 10.0 * t)
           + rng.normal(0.0, 1.0, t.size))
    # een handvol echte verschuivingen
    for at in (120.0, 300.0, 480.0, 660.0):
        s, e = int(at * SF), int((at + 6.0) * SF)
        eeg[s:e] = (28.0 * np.sin(2 * np.pi * 10.0 * t[s:e])
                    + 28.0 * np.sin(2 * np.pi * 20.0 * t[s:e])
                    + rng.normal(0.0, 1.0, e - s))
    return eeg * 1e-6, ["N2"] * int(DUR_S / 30)


def test_a_missing_model_falls_back_to_the_rule_based_result(monkeypatch):
    """Met een onvindbaar model hoort de uitkomst gelijk te zijn aan het
    regelgebaseerde pad -- niet aan de ruime kandidatenlijst."""
    eeg, hypno = _recording()
    regels = detect_arousals(eeg, SF, hypno)

    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    monkeypatch.setattr("psgscoring.arousal.AROUSAL_LGBM_MODEL_PATH",
                        "/nonexistent/arousal_classifier_v3.txt")
    monkeypatch.setattr("psgscoring.arousal._AROUSAL_LGBM_BOOSTER", None)
    kapot = detect_arousals(eeg, SF, hypno)

    assert kapot["success"], kapot.get("error")
    assert len(kapot["events"]) == len(regels["events"]), (
        f"model ontbreekt maar er komen {len(kapot['events'])} events uit tegen "
        f"{len(regels['events'])} regelgebaseerd -- dit is de kandidatenlijst "
        f"op ratio 1,2 in plaats van {AROUSAL_RATIO_THRESH}"
    )
    assert kapot["events"] == regels["events"]


def test_the_summary_says_the_model_did_not_run(monkeypatch):
    """Een consument moet kunnen zien dat de hybride niet gedraaid heeft."""
    eeg, hypno = _recording()
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    monkeypatch.setattr("psgscoring.arousal.AROUSAL_LGBM_MODEL_PATH",
                        "/nonexistent/arousal_classifier_v3.txt")
    monkeypatch.setattr("psgscoring.arousal._AROUSAL_LGBM_BOOSTER", None)
    out = detect_arousals(eeg, SF, hypno)
    s = out["summary"]
    assert s.get("lgbm_available") is False, (
        "de samenvatting zegt niet dat de classifier niet beschikbaar was"
    )
    assert "lgbm_n_post" not in s, (
        "lgbm_n_post suggereert dat er gefilterd is terwijl dat niet gebeurd is"
    )
