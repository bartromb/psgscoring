"""Spectrale effortclassificatie: de ademband is méér dan twee bins.

`spectral_effort_classifier` nam `nperseg = 4 * sf`, dus bins van 0,25 Hz.
De ademband 0,1-0,5 Hz is dan {0,25, 0,5}: een ademritme op 0,15 of 0,18 Hz
(9-11 ademteugen per minuut, gewoon tijdens slaap) lekt grotendeels naar de
DC-bin die buiten de band valt, terwijl de hartband van 0,8-2,5 Hz zeven bins
krijgt. Het resultaat was systematisch "indeterminate" of zelfs cardiaal
dominant op een band die duidelijk ademde.

Met `resolution="event"` is de vensterlengte het event zelf (10-60 s, dus
0,017-0,1 Hz per bin) en wordt per band de dichtheid geïntegreerd. `"4s"` is
het oude gedrag, dat de bevroren profielen houden.
"""
import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.ecg_effort import (
    ecg_effort_assessment,
    spectral_effort_classifier,
    spectral_resolution_from_profile,
)
from psgscoring.profiles import PROFILES

SF = 200.0


def _band(dur_s=30.0, f_eff=0.15, a_eff=1.0, f_card=1.2, a_card=0.8, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * SF)) / SF
    return (a_eff * np.sin(2 * np.pi * f_eff * t)
            + a_card * np.sin(2 * np.pi * f_card * t)
            + 0.01 * rng.normal(size=t.size))


def test_een_ademende_band_heet_effort_present():
    x = _band()
    uit = spectral_effort_classifier(x, SF, 0, len(x))
    assert uit["classification_hint"] == "effort_present", uit
    assert uit["respiratory_fraction"] > 0.5
    assert uit["resolution"] == "event"


def test_het_oude_pad_blijft_bereikbaar_en_toont_het_defect():
    x = _band()
    uit = spectral_effort_classifier(x, SF, 0, len(x), resolution="4s")
    assert uit["classification_hint"] != "effort_present", uit
    assert uit["resolution"] == "4s"


def test_cardiogene_pulsatie_blijft_cardiaal_dominant():
    x = _band(a_eff=0.05, a_card=1.0)
    uit = spectral_effort_classifier(x, SF, 0, len(x))
    assert uit["cardiac_dominant"] and uit["classification_hint"] == "probable_central"


def test_de_assessment_geeft_de_resolutie_door():
    # De assessment eist signaal VOORBIJ end_idx; het event is 5-35 s van 40 s.
    x = _band(dur_s=40.0)
    a, b = int(5 * SF), int(35 * SF)
    nieuw = ecg_effort_assessment(None, x, None, SF, a, b)
    oud = ecg_effort_assessment(None, x, None, SF, a, b, spectral_resolution="4s")
    assert nieuw["spectral_detail"]["classification_hint"] == "effort_present"
    assert oud["spectral_detail"]["classification_hint"] != "effort_present"


def test_onbekende_resolutie_wordt_geweigerd():
    x = _band()
    with pytest.raises(ValueError):
        spectral_effort_classifier(x, SF, 0, len(x), resolution="8s")


# ── profielveld en lezer ────────────────────────────────────────────────

def test_veld_event_op_klinisch_4s_op_de_bevroren_profielen():
    assert PROFILES["aasm_v3_rec"].post_processing.spectral_effort_resolution == "event"
    for naam in ("mesa_shhs", "chicago_1999"):
        assert PROFILES[naam].post_processing.spectral_effort_resolution == "4s", naam
        assert SCORING_PROFILES[naam]["SPECTRAL_EFFORT_RESOLUTION"] == "4s"


def test_de_lezer_volgt_de_legacy_sleutel():
    assert spectral_resolution_from_profile(SCORING_PROFILES["aasm_v3_rec"]) == "event"
    assert spectral_resolution_from_profile(SCORING_PROFILES["mesa_shhs"]) == "4s"
    assert spectral_resolution_from_profile(None) == "event"
