"""Cheyne-Stokes: de periodiciteit mag niet de helft of anderhalf keer zijn.

`detect_cheyne_stokes` bandfiltert de flow-envelope op 0,005-0,05 Hz, een
band die de gelijkstroom weghaalt. Daarna stond er `np.abs(...)`: een
nulgemiddelde sinus gelijkrichten verdubbelt zijn frequentie, dus de
autocorrelatie piekte op T/2 (88 s -> 44 s) of, als T/2 onder de 40 s-grens
viel, op 3T/2 (60 s -> 90 s). `csr_detected` bleef vaak waar; alleen het
getal dat in het rapport staat -- en dat `_flag_csr_events` gebruikt om
events aan de cyclus te toetsen -- was fout.

De gelijkrichting is weg achter `csr_period_rectified` (default uit; de
bevroren profielen houden het oude gedrag). Daarnaast telt `sleep_min` een
geheel hypnogram nu zoals de rest van het pakket: 0 is wake.
"""
import numpy as np
import pytest

from psgscoring.ancillary import csr_period_rectified_from_profile, detect_cheyne_stokes
from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES

SF = 32.0
N_S = 20 * 60


def _env(period_s):
    t = np.arange(int(N_S * SF)) / SF
    return 1.0 + 0.8 * np.sin(2 * np.pi * t / period_s)


def _hypno():
    return ["N2"] * (N_S // 30)


@pytest.mark.parametrize("period_s", [88.0, 60.0])
def test_de_periodiciteit_is_de_cyclusduur(period_s):
    uit = detect_cheyne_stokes(_env(period_s), SF, _hypno())
    assert uit["success"] and uit["csr_detected"]
    assert abs(uit["periodicity_s"] - period_s) <= 2.0, uit["periodicity_s"]


@pytest.mark.parametrize("period_s,oud", [(88.0, 44.0), (60.0, 90.0)])
def test_het_gelijkgerichte_pad_blijft_bereikbaar_en_toont_het_defect(period_s, oud):
    uit = detect_cheyne_stokes(_env(period_s), SF, _hypno(), period_rectified=True)
    assert uit["csr_detected"]
    assert abs(uit["periodicity_s"] - oud) <= 2.0, uit["periodicity_s"]


def test_een_geheel_hypnogram_telt_wake_niet_als_slaap():
    x = _env(88.0)
    tekst = detect_cheyne_stokes(x, SF, ["N2"] * (N_S // 30))
    getal = detect_cheyne_stokes(x, SF, [2] * (N_S // 30))
    assert getal["csr_pct_sleep"] == tekst["csr_pct_sleep"]
    wake_t = detect_cheyne_stokes(x, SF, ["W"] * (N_S // 30))
    wake_g = detect_cheyne_stokes(x, SF, [0] * (N_S // 30))
    assert wake_g["csr_pct_sleep"] == wake_t["csr_pct_sleep"]
    assert wake_g["csr_pct_sleep"] != getal["csr_pct_sleep"]


# ── het profielveld ─────────────────────────────────────────────────────

def test_veld_uit_op_klinisch_aan_op_de_bevroren_profielen():
    assert PROFILES["aasm_v3_rec"].post_processing.csr_period_rectified is False
    for naam in ("mesa_shhs", "chicago_1999"):
        assert PROFILES[naam].post_processing.csr_period_rectified is True, naam
        assert SCORING_PROFILES[naam]["CSR_PERIOD_RECTIFIED"] is True


def test_de_lezer_volgt_de_legacy_sleutel():
    assert csr_period_rectified_from_profile(SCORING_PROFILES["aasm_v3_rec"]) is False
    assert csr_period_rectified_from_profile(SCORING_PROFILES["chicago_1999"]) is True
    assert csr_period_rectified_from_profile(None) is False
