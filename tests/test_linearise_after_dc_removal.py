"""De wortellinearisatie draaide vóór elke DC-verwijdering.

Neusdruk is evenredig met flow² (Bernoulli); de wortel maakt daar weer flow
van. Maar `preprocess_flow` nam de wortel van het RUWE signaal, offset
inbegrepen. Een drukopnemer met een offset van 0,5x de drukamplitude -- een
DC-gekoppelde versterker, een niet-genulde canule -- geeft dan:

    normaal   p = f² + 0,5   piek 1,5 / dal -0,5  ->  wortel 1,22 / -0,71
    halvering p = f²/4 + 0,5 piek 0,75 / dal +0,25 -> wortel 0,87 / 0,50

De halvering komt zo als ~80 % reductie uit de envelope: het dal keert niet
eens meer van teken. De lezing hangt dus af van een versterkerinstelling.

Veld `linearise_after_dc_removal`: eerst de trage component eraf (0,05 Hz
hoogdoorlaat, dezelfde onderkant als de bandpass), dan de wortel, dan de
bandpass. **Default `False` op ALLE profielen**: dit verschuift elke
hypopneemeting op drukkanalen met offset en hoort eerst gemeten te worden.
"""
import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES, PostProcessingRules
from psgscoring.signal import preprocess_flow

SF = 32.0
DUUR_S = 600.0
DAL = (240.0, 330.0)          # 90 s halvering, ruim voor een stabiele envelope
VOOR = (120.0, 210.0)         # referentievenster ervoor


def _druk(offset_fractie=0.5, drop=0.5):
    """Druk ∝ sign(f)·f² met een DC-offset als fractie van de drukamplitude."""
    t = np.arange(int(DUUR_S * SF)) / SF
    flow = np.sin(2 * np.pi * 0.25 * t)
    a, b = int(DAL[0] * SF), int(DAL[1] * SF)
    flow[a:b] *= drop
    return np.sign(flow) * flow ** 2 + offset_fractie * 1.0 ** 2


def _reductie(env):
    """1 - (envelope in het dal / envelope ervoor), beide als mediaan en met
    10 s marge voor de flanken."""
    m = int(10 * SF)
    dal = env[int(DAL[0] * SF) + m:int(DAL[1] * SF) - m]
    voor = env[int(VOOR[0] * SF) + m:int(VOOR[1] * SF) - m]
    return 1.0 - float(np.median(dal) / np.median(voor))


def test_met_dc_verwijdering_meet_een_halvering_als_halvering():
    env = preprocess_flow(_druk(), SF, is_nasal_pressure=True,
                          linearise_after_dc_removal=True)
    assert _reductie(env) == pytest.approx(0.50, abs=0.07)


def test_zonder_dc_verwijdering_meet_dezelfde_halvering_als_80_procent():
    """Het oude gedrag, met het getal erbij. Dit is de default."""
    env = preprocess_flow(_druk(), SF, is_nasal_pressure=True)
    r = _reductie(env)
    assert 0.70 < r < 0.90, r


def test_de_vlag_doet_expliciet_uit_hetzelfde_als_weggelaten():
    a = preprocess_flow(_druk(), SF, is_nasal_pressure=True)
    b = preprocess_flow(_druk(), SF, is_nasal_pressure=True,
                        linearise_after_dc_removal=False)
    np.testing.assert_array_equal(a, b)


def test_zonder_offset_maakt_de_vlag_vrijwel_niets_uit():
    """Zonder DC is er niets te verwijderen; de hoogdoorlaat raakt 0,25 Hz
    nauwelijks."""
    p = _druk(offset_fractie=0.0)
    oud = _reductie(preprocess_flow(p, SF, is_nasal_pressure=True))
    nieuw = _reductie(preprocess_flow(p, SF, is_nasal_pressure=True,
                                      linearise_after_dc_removal=True))
    assert oud == pytest.approx(0.50, abs=0.05)
    assert abs(oud - nieuw) < 0.03, (oud, nieuw)


def test_zonder_linearisatie_is_de_vlag_dood():
    """Hij hoort alleen bij `is_nasal_pressure=True`; een thermistor krijgt
    geen wortel en dus ook geen DC-stap ervoor."""
    p = _druk()
    a = preprocess_flow(p, SF, is_nasal_pressure=False)
    b = preprocess_flow(p, SF, is_nasal_pressure=False,
                        linearise_after_dc_removal=True)
    np.testing.assert_array_equal(a, b)


# ── het veld ─────────────────────────────────────────────────────────

def test_default_is_uit_op_elk_profiel():
    """Eerst meten. Geen enkel profiel, ook geen exploratief, zet dit aan."""
    assert PostProcessingRules().linearise_after_dc_removal is False
    aan = {n for n, p in PROFILES.items()
           if p.post_processing.linearise_after_dc_removal}
    assert aan == set(), aan


def test_veld_bereikt_de_legacy_dict():
    for naam, d in SCORING_PROFILES.items():
        assert d["LINEARISE_AFTER_DC_REMOVAL"] is False, naam
