"""De effortband wordt gesneden op zijn EIGEN bemonstering, niet op die van het ECG.

`respiratory._get_ecg_assessment` rekende de eventgrenzen om naar de
ECG-frequentie en gaf die indices ook mee voor de thorax- en abdomenband, die
op de flowfrequentie staan. Bij gelijke frequenties (het geval in de pijplijn,
waar MNE alles op één frequentie zet) klopt dat toevallig; bij een ECG op
256 Hz naast banden op 32 Hz lag het venster acht keer te ver, meestal voorbij
het einde van de band, en verviel de spectrale toets zonder melding. Gevonden
bij de codelezing van 10-11-2026.
"""
import numpy as np
import pytest

from psgscoring.respiratory import _get_ecg_assessment

SF_FLOW, SF_ECG = 32.0, 256.0
DUUR_S = 900
EVENT = (600.0, 630.0)


def _signalen():
    t_f = np.arange(0, DUUR_S, 1 / SF_FLOW)
    t_e = np.arange(0, DUUR_S, 1 / SF_ECG)
    # Band: ademhaling op 0,25 Hz, behalve tijdens het event; daar blijft
    # alleen de cardiogene pulsatie op 1,2 Hz over (een centrale apneu).
    band = np.sin(2 * np.pi * 0.25 * t_f) + 0.05 * np.sin(2 * np.pi * 1.2 * t_f)
    in_event = (t_f >= EVENT[0]) & (t_f < EVENT[1])
    band[in_event] = 0.05 * np.sin(2 * np.pi * 1.2 * t_f[in_event])
    # ECG: een smalle piek per slag, 72/min.
    ecg = np.zeros_like(t_e)
    for slag in np.arange(0.5, DUUR_S, 1 / 1.2):
        i = int(slag * SF_ECG)
        ecg[i:i + 4] = 1.0
    return band, ecg


def _oordeel(sf_ecg, ecg):
    band, _ = _signalen()
    return _get_ecg_assessment(
        ecg, None, None, band, band, SF_FLOW, sf_ecg,
        int(EVENT[0] * SF_FLOW), int(EVENT[1] * SF_FLOW))


def test_bij_ongelijke_frequenties_wordt_de_band_op_het_event_gesneden():
    _, ecg = _signalen()
    uit = _oordeel(SF_ECG, ecg)
    assert uit is not None
    assert uit["spectral_detail"], "de spectrale toets is stil vervallen"
    assert uit["spectral_cardiac_dominant"] is True


def test_bij_gelijke_frequenties_verandert_er_niets():
    """Het pijplijngeval: zelfde oordeel als met de omrekening van factor 1."""
    band, _ = _signalen()
    t = np.arange(0, DUUR_S, 1 / SF_FLOW)
    ecg = np.zeros_like(t)
    for slag in np.arange(0.5, DUUR_S, 1 / 1.2):
        ecg[int(slag * SF_FLOW)] = 1.0
    uit = _get_ecg_assessment(
        ecg, None, None, band, band, SF_FLOW, SF_FLOW,
        int(EVENT[0] * SF_FLOW), int(EVENT[1] * SF_FLOW))
    assert uit["spectral_detail"]
    assert uit["spectral_cardiac_dominant"] is True


def test_ademhaling_op_de_band_leest_niet_als_hartslag():
    """Een event vroeg in de opname met doorlopende ademhaling op de band (een
    obstructieve apneu). Het foute venster viel dan BINNEN de band en werd
    bovendien op de ECG-frequentie gelezen: 0,25 Hz ademhaling leek 2 Hz en
    dus cardiaal -- het omgekeerde van wat er op de band staat."""
    t_f = np.arange(0, DUUR_S, 1 / SF_FLOW)
    band = np.sin(2 * np.pi * 0.25 * t_f)
    a, b = 60.0, 90.0
    _, ecg = _signalen()
    uit = _get_ecg_assessment(
        ecg, None, None, band, band, SF_FLOW, SF_ECG,
        int(a * SF_FLOW), int(b * SF_FLOW))
    assert uit["spectral_detail"]
    assert uit["spectral_cardiac_dominant"] is False


@pytest.mark.parametrize("sf_ecg", [SF_FLOW, SF_ECG])
def test_zonder_ecg_blijft_het_oordeel_bestaan(sf_ecg):
    uit = _oordeel(sf_ecg, None)
    assert uit is not None and uit["spectral_detail"]
