"""`get_desaturation`: de "nadir binnen 3 s"-toets moet TIJD meten.

Het segment werd eerst gefilterd op 50-100 % en pas daarna geïndexeerd. Een
sensoruitval van 10 s aan het begin van een event schoof de index van de
nadir dus 10 posities naar voren: een echte 4 %-desaturatie met nadir op
11 s na de onset las als "nadir op 1 s" en werd op de 3 s-regel afgewezen.
De waarde zelf (92 %) klopte; alleen de desaturatie werd None.

De tijd-as blijft nu staan (geldig-masker, argmin binnen het masker). Op een
segment zonder ongeldige samples is dat byte-identiek aan het oude pad.
"""
import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES
from psgscoring.spo2 import get_desaturation, nadir_time_axis_from_profile

SF = 1.0


def _spo2(dropout=True):
    s = np.full(600, 96.0)
    # event 100-120 s, 4 % desaturatie met nadir op 111 s
    s[100:125] = 93.0
    s[111] = 92.0
    if dropout:
        s[100:110] = 0.0            # 10 s sensoruitval aan het begin
    return s


def test_een_uitval_aan_de_onset_kost_de_desaturatie_niet_meer():
    desat, nadir = get_desaturation(_spo2(), 100.0, 20.0, SF)
    assert (desat, nadir) == (4.0, 92.0)


def test_het_oude_pad_blijft_bereikbaar_en_toont_het_defect():
    desat, nadir = get_desaturation(_spo2(), 100.0, 20.0, SF, nadir_time_axis=False)
    assert (desat, nadir) == (None, 92.0)


def test_zonder_ongeldige_samples_zijn_beide_paden_gelijk():
    s = _spo2(dropout=False)
    assert get_desaturation(s, 100.0, 20.0, SF) == \
        get_desaturation(s, 100.0, 20.0, SF, nadir_time_axis=False) == (4.0, 92.0)


def test_een_echte_vroege_nadir_wordt_nog_steeds_afgewezen():
    """De regel zelf verandert niet: nadir op 1 s na de onset met < 5 % daling
    blijft een circulatoir onmogelijke koppeling."""
    s = np.full(600, 96.0)
    s[101] = 92.0
    assert get_desaturation(s, 100.0, 20.0, SF)[0] is None


# ── het profielveld ─────────────────────────────────────────────────────

def test_veld_aan_op_klinisch_uit_op_de_bevroren_profielen():
    assert PROFILES["aasm_v3_rec"].post_processing.spo2_nadir_time_axis is True
    for naam in ("mesa_shhs", "chicago_1999"):
        assert PROFILES[naam].post_processing.spo2_nadir_time_axis is False, naam
        assert SCORING_PROFILES[naam]["SPO2_NADIR_TIME_AXIS"] is False


def test_de_lezer_volgt_de_legacy_sleutel():
    assert nadir_time_axis_from_profile(SCORING_PROFILES["aasm_v3_rec"]) is True
    assert nadir_time_axis_from_profile(SCORING_PROFILES["mesa_shhs"]) is False
    assert nadir_time_axis_from_profile(None) is True
