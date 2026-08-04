"""Een gerapporteerd minimum van 20 bpm is de filtergrens, niet de patiënt.

Drie opnames toonden een minimale hartfrequentie van 20,2 · 32,6 · 20,0 bpm.
Een echte sinusbradycardie van 20/min is een klinisch alarm; twee van die drie
liggen exact op de ondergrens van het plausibiliteitsfilter in
``analyze_heart_rate``, en dat is de signatuur van een oximeter die even
loslaat. Het extremum is zo betrouwbaar als het slechtste sample.

Er komt geen stille correctie: de getallen blijven staan, er komen robuuste
percentielen naast, en het oordeel "betrouwbaar of niet" wordt expliciet zodat
het rapport kan kiezen wat het toont.
"""
from __future__ import annotations

import numpy as np

from psgscoring.ancillary import analyze_heart_rate

SF = 1.0
N = 3600  # één uur op 1 Hz
HYPNO = ["N2"] * (N // 30)


def _steady(bpm=62.0, jitter=2.0, seed=3):
    rng = np.random.default_rng(seed)
    return bpm + jitter * rng.standard_normal(N)


def test_a_clean_pulse_channel_is_reliable():
    s = analyze_heart_rate(_steady(), SF, HYPNO)["summary"]
    assert s["hr_reliable"] is True
    assert s["hr_unreliable_reason"] == ""
    assert s["hr_implausible_pct"] == 0.0
    assert 55 < s["avg_hr"] < 70


def test_a_dropout_that_lands_on_the_filter_edge_is_flagged():
    """Het waargenomen geval: sensoruitval duwt het minimum naar de grens."""
    hr = _steady()
    hr[1000:1200] = 0.0          # sonde los -> 0, valt buiten het filter
    hr[1200:1210] = 20.1         # de rand van het uitvalvenster overleeft wel
    s = analyze_heart_rate(hr, SF, HYPNO)["summary"]

    assert s["min_hr"] < 21, "het oude getal blijft zichtbaar"
    assert s["hr_reliable"] is False
    assert "sensoruitval" in s["hr_unreliable_reason"]
    assert s["hr_implausible_pct"] > 1.0


def test_the_robust_percentiles_ignore_the_dropout():
    """p1 is waar het rapport op terug kan vallen als het extremum niets zegt."""
    hr = _steady()
    hr[1000:1200] = 0.0
    hr[1200:1210] = 20.1
    s = analyze_heart_rate(hr, SF, HYPNO)["summary"]
    assert s["min_hr"] < 25
    assert 50 < s["hr_p1"] < 62, "percentiel hoort bij de echte hartslag te liggen"
    assert s["hr_p1"] - s["min_hr"] > 25


def test_a_single_bad_sample_does_not_condemn_the_recording():
    """Eén uitschieter op een uur is geen sensoruitval."""
    hr = _steady()
    hr[500] = 21.0
    s = analyze_heart_rate(hr, SF, HYPNO)["summary"]
    assert s["hr_reliable"] is True
    assert s["min_hr"] == 21.0


def test_a_raw_ecg_waveform_is_never_reported_as_a_heart_rate():
    """Zonder pulse-kanaal viel de pijplijn terug op het ruwe ECG. In ADC-
    eenheden overleeft een deel van de golfvorm het filter en rolt eruit als
    'hartfrequentie'."""
    rng = np.random.default_rng(5)
    ecg_adc = 120.0 * np.sin(2 * np.pi * 1.2 * np.arange(N) / SF) \
        + 30 * rng.standard_normal(N)
    s = analyze_heart_rate(ecg_adc, SF, HYPNO, source="ecg")["summary"]
    assert s["hr_reliable"] is False
    assert s["hr_source"] == "ecg"
    assert "R-piekdetectie" in s["hr_unreliable_reason"]


def test_the_existing_keys_are_untouched():
    """Additief: wat er stond blijft staan, met dezelfde betekenis."""
    s = analyze_heart_rate(_steady(), SF, HYPNO)["summary"]
    for key in ("avg_hr", "min_hr", "max_hr", "std_hr",
                "n_tachycardia", "n_bradycardia"):
        assert key in s
    assert s["min_hr"] == min(s["min_hr"], s["hr_p1"])
