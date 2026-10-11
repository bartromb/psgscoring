"""Eén eenheidsheuristiek voor EEG en EMG, met een mV-trap en detrending.

`arousal.py` had op drie plaatsen `if max|x| < 0.01: x *= 1e6` -- geen
mV-trap, geen melding. `plm.py` had de volledige versie (V / mV / uV met een
waarschuwing). Beide maten de amplitude op het RUWE signaal: een
gelijkspanning-gekoppelde opname met 145 mV offset (zie de BDF-casus) las
als "mV" en kreeg x1e3 waar x1e6 hoorde -- het EEG van 50 uV werd 0,05 uV.

`utils.to_microvolts` trekt eerst de gelijkspanning af en meet dan. Op uV- en
V-invoer zonder offset is dat byte-identiek aan de oude regel.
"""
import logging

import numpy as np
import pytest

from psgscoring.plm import _detect_lm_channel
from psgscoring.utils import to_microvolts

SF = 100.0


def _sinus(amp, offset=0.0, n_s=60):
    # 12,5 Hz op 100 Hz: de fasestap is 45 graden, dus de top (90 graden)
    # wordt exact bemonsterd en max|x| is precies `amp`.
    t = np.arange(int(n_s * SF)) / SF
    return offset + amp * np.sin(2 * np.pi * 12.5 * t)


def _amplitude_uv(x):
    return float(np.max(np.abs(x - np.mean(x))))


def test_volt():
    assert _amplitude_uv(to_microvolts(_sinus(5e-5))) == pytest.approx(50.0, rel=1e-6)


def test_millivolt(caplog):
    with caplog.at_level(logging.WARNING, logger="psgscoring.utils"):
        uit = to_microvolts(_sinus(0.05))
    assert _amplitude_uv(uit) == pytest.approx(50.0, rel=1e-6)
    assert "mV" in caplog.text


def test_microvolt_blijft_ongemoeid():
    x = _sinus(50.0)
    np.testing.assert_array_equal(to_microvolts(x), x)


def test_volt_met_145_mv_gelijkspanning():
    """Het offset mag de trap niet kiezen: het EEG eronder is 50 uV in volt."""
    uit = to_microvolts(_sinus(5e-5, offset=0.145))
    assert _amplitude_uv(uit) == pytest.approx(50.0, rel=1e-6)
    assert float(np.mean(uit)) == pytest.approx(145000.0, rel=1e-6)


def test_expliciete_eenheid_wint_van_de_heuristiek():
    x = _sinus(50.0)
    np.testing.assert_allclose(to_microvolts(x, unit="V"), x * 1e6)
    np.testing.assert_allclose(to_microvolts(x, unit="mV"), x * 1e3)
    np.testing.assert_array_equal(to_microvolts(x, unit="uV"), x)
    with pytest.raises(ValueError):
        to_microvolts(x, unit="nV")


def test_een_dood_kanaal_blijft_nul():
    np.testing.assert_array_equal(to_microvolts(np.zeros(100)), np.zeros(100))


def test_de_plm_detector_ziet_volt_en_microvolt_als_hetzelfde_signaal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, int(SF * 60))
    a, b = int(SF * 20), int(SF * 21.5)
    x[a:b] += rng.normal(0, 40.0, b - a)
    uv = _detect_lm_channel(x, SF)
    v = _detect_lm_channel(x * 1e-6, SF)
    assert len(uv) == 1 and uv == v
