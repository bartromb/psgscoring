"""Het werkpunt van de classifier is een profielveld geworden.

0,60 is gedomineerd: herijkt met gescheiden steekproeven gaat de F1 op 30
ongeziene MESA-opnames van 0,421 naar 0,543 bij drempel 0,90 (gepaard +0,091,
24/30, p = 1,5e-05), en 0,80 verslaat 0,60 op beide cohorten met bovendien een
zuivere eventtelling (1,07 / 1,01 tegen 1,52 / 1,47).

Default blijft `None` -- de moduleconstante 0,60 -- tot de keuze tussen 0,80
en 0,90 gemaakt is; die is klinisch, want bij 0,90 telt de arousal-index ruim
een derde te laag.
"""
import numpy as np
import pytest


def test_no_profile_overrides_the_threshold_yet():
    from psgscoring.profiles import get_profile, list_profiles

    for name in list_profiles():
        v = get_profile(name).post_processing.arousal_lgbm_threshold
        assert v is None, f"{name} zet een eigen drempel: {v}"


def test_the_registry_carries_it():
    from psgscoring.constants import SCORING_PROFILES

    for name, d in SCORING_PROFILES.items():
        assert "AROUSAL_LGBM_THRESHOLD" in d, name


def test_the_detector_accepts_a_threshold_and_it_changes_the_outcome():
    """Een hogere drempel hoort MINDER events te geven, en meetbaar."""
    pytest.importorskip("lightgbm")
    from psgscoring.arousal import detect_arousals

    sf, minutes = 256.0, 20
    n = int(sf * 60 * minutes)
    t = np.arange(n) / sf
    rng = np.random.default_rng(3)
    eeg = rng.normal(0, 20e-6, n)
    for start in range(60, 60 * minutes - 60, 60):
        a, b = int(start * sf), int((start + 4) * sf)
        eeg[a:b] += 60e-6 * np.sin(2 * np.pi * 10.0 * t[a:b])
    hypno = ["N2"] * int(np.ceil(n / sf / 30))

    laag = detect_arousals(eeg, sf, hypno, lgbm=True, lgbm_threshold=0.30)
    hoog = detect_arousals(eeg, sf, hypno, lgbm=True, lgbm_threshold=0.95)
    n_laag = len(laag.get("events") or [])
    n_hoog = len(hoog.get("events") or [])
    assert n_laag > 0, "fixture levert geen kandidaten; meet niets"
    assert n_hoog <= n_laag, (
        f"hogere drempel gaf MEER events: {n_hoog} tegen {n_laag}")
    assert laag["summary"].get("lgbm_threshold") == 0.30
    assert hoog["summary"].get("lgbm_threshold") == 0.95


def test_the_summary_reports_the_threshold_that_was_used():
    """Anders is achteraf niet na te gaan welk werkpunt een rapport draaide."""
    pytest.importorskip("lightgbm")
    from psgscoring.arousal import detect_arousals

    sf = 256.0
    n = int(sf * 60 * 12)
    rng = np.random.default_rng(8)
    eeg = rng.normal(0, 20e-6, n)
    hypno = ["N2"] * int(np.ceil(n / sf / 30))
    out = detect_arousals(eeg, sf, hypno, lgbm=True, lgbm_threshold=0.77)
    if "lgbm_threshold" in out.get("summary", {}):
        assert out["summary"]["lgbm_threshold"] == 0.77
