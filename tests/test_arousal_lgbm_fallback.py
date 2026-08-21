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


# ══════════════════════════════════════════════════════════════
# Het profielveld — `arousal_lgbm`
# ══════════════════════════════════════════════════════════════

def test_the_profile_field_reaches_the_detector(monkeypatch):
    """`lgbm=True` schakelt de hybride in zonder env-variabele.

    Tot v0.23.0 was het hybride pad ALLEEN via PSGSCORING_AROUSAL_LGBM te
    bereiken. Dat maakte de keuze installatiebreed: `mesa_shhs` kon niet
    gepind blijven terwijl de klinische profielen hem gebruikten. Zonder een
    profielveld is de beslissing alles-of-niets, en dat is geen beslissing die
    per profiel genomen kan worden.
    """
    monkeypatch.delenv("PSGSCORING_AROUSAL_LGBM", raising=False)
    monkeypatch.delenv("YASAFLASKIFIED_AROUSAL_LGBM", raising=False)
    eeg, hypno = _recording()

    uit = detect_arousals(eeg, SF, hypno)
    aan = detect_arousals(eeg, SF, hypno, lgbm=True)

    assert "lgbm_available" not in uit["summary"], (
        "zonder de vlag hoort er geen lgbm-sleutel in de samenvatting te staan")
    assert aan["summary"].get("lgbm_available") is True, (
        "profielveld bereikt de detector niet")


def test_the_env_variable_still_wins(monkeypatch):
    """De env blijft werken en overschrijft het profiel — in beide richtingen.

    Een installatie moet hem kunnen forceren of uitzetten, en een meting moet
    kunnen aantonen dat hij niet actief was.
    """
    eeg, hypno = _recording()
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "0")
    assert "lgbm_available" not in detect_arousals(
        eeg, SF, hypno, lgbm=True)["summary"]
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    assert detect_arousals(eeg, SF, hypno, lgbm=False)["summary"].get(
        "lgbm_available") is True


def test_no_profile_enables_it_by_default():
    from psgscoring.constants import SCORING_PROFILES
    for name, d in SCORING_PROFILES.items():
        assert "AROUSAL_LGBM" in d, name
    aan = [n for n, d in SCORING_PROFILES.items() if d["AROUSAL_LGBM"]]
    assert not aan, f"vlag hoort nergens default aan te staan: {aan}"
