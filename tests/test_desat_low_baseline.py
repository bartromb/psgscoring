"""Blok 2E: 2 %-desaturatie onder een lage baseline.

Dit is een AFWIJKING van AASM Regel 1A, geen reparatie. De suite pint vooral
dat hij default uit staat, dat geen enkel profiel hem aanzet, en dat elk event
dat er gebruik van maakt gemarkeerd wordt.
"""
from __future__ import annotations

import numpy as np

from psgscoring.breath_scoring import _pre_event_below_local_baseline
from psgscoring.profiles import PROFILES, PostProcessingRules

SF = 1.0


def test_default_is_uit():
    assert PostProcessingRules().desat_low_baseline_relaxation is False


def test_geen_enkel_geleverd_profiel_zet_het_aan():
    aan = [n for n, p in PROFILES.items()
           if p.post_processing.desat_low_baseline_relaxation]
    assert aan == [], (
        f"dit is een regelafwijking en mag nooit stilzwijgend aanstaan: {aan}")


def test_veld_bereikt_de_legacy_dict():
    import psgscoring.constants as C
    for naam, d in C.SCORING_PROFILES.items():
        assert d["DESAT_LOW_BASELINE_RELAXATION"] is False, naam


def test_dalende_saturatie_wordt_herkend():
    """Laatste 30 s duidelijk onder het niveau van de 2 minuten ervoor."""
    s = np.concatenate([np.full(90, 96.0), np.full(30, 90.0)])
    assert _pre_event_below_local_baseline(s, SF, 120.0) is True


def test_vlakke_saturatie_op_het_90e_percentiel_vuurt_niet():
    s = np.full(120, 96.0)
    assert _pre_event_below_local_baseline(s, SF, 120.0) is False


def test_zonder_bruikbare_data_vuurt_hij_niet():
    assert _pre_event_below_local_baseline(None, SF, 120.0) is False
    assert _pre_event_below_local_baseline(np.full(120, np.nan), SF, 120.0) is False
    assert _pre_event_below_local_baseline(np.full(5, 96.0), SF, 120.0) is False


def test_implausibele_waarden_tellen_niet_mee():
    """Sensoruitval als 0 mag de baseline niet omlaag trekken."""
    s = np.concatenate([np.full(60, 0.0), np.full(30, 96.0), np.full(30, 96.0)])
    assert _pre_event_below_local_baseline(s, SF, 120.0) is False
