"""Blok 2B: `max_events_per_desaturation`.

De tests pinnen drie dingen: het veld is default uit en bereikt de detector,
de begrenzing DEGRADEERT (verwijdert niet), en ze raakt uitsluitend de
desaturatietak. De fixture is bewust zo gebouwd dat een limiter die niets doet
de tests laat falen — een fixture waarin toch al niets te begrenzen valt, meet
niets en blijft groen.
"""
from __future__ import annotations


import numpy as np
import pytest

from psgscoring.profiles import PROFILES
from psgscoring.respiratory import limit_events_per_desaturation


# ── het veld zelf ────────────────────────────────────────────────────

def test_default_is_uit():
    from psgscoring.profiles import PostProcessingRules
    assert PostProcessingRules().max_events_per_desaturation is None


def test_geen_enkel_geleverd_profiel_zet_het_aan():
    aan = [n for n, p in PROFILES.items()
           if p.post_processing.max_events_per_desaturation is not None]
    assert aan == [], f"profielen zouden byte-identiek blijven: {aan}"


def test_veld_bereikt_de_legacy_dict():
    # GEEN importlib.reload hier: reload vervangt het moduleobject en laat
    # pipeline/respiratory met een verouderde SCORING_PROFILES achter, wat
    # later draaiende tests laat omvallen. De dict is bij import al afgeleid.
    import psgscoring.constants as C
    for naam, d in C.SCORING_PROFILES.items():
        assert "MAX_EVENTS_PER_DESATURATION" in d, naam
        assert d["MAX_EVENTS_PER_DESATURATION"] is None, naam


# ── fixture: één diepe desaturatie, drie hypopneus eromheen ──────────

SF = 1.0
DUUR_S = 1200


def _spo2_met_een_desaturatie(nadir_t=300.0, diepte=6.0, breedte_s=40.0):
    """Vlak op 96 %, met één daling van `diepte` % rond `nadir_t`."""
    s = np.full(int(DUUR_S * SF), 96.0)
    a, b = int((nadir_t - breedte_s / 2) * SF), int((nadir_t + breedte_s / 2) * SF)
    s[a:b] = 96.0 - diepte
    return s


def _hypno(n_epochs=DUUR_S // 30):
    return ["N2"] * n_epochs


def _ev(onset, p, desat=6.0, type_="hypopnea"):
    return {"type": type_, "onset_s": onset, "duration_s": 15.0,
            "desaturation_pct": desat, "p_scored": p, "stage": "N2"}


def test_fixture_levert_werkelijk_een_desaturatie():
    """Zonder deze controle kan de hele suite groen staan op een vlak signaal."""
    from psgscoring.spo2 import detect_desaturations
    d = detect_desaturations(_spo2_met_een_desaturatie(), SF,
                             np.ones(int(DUUR_S * SF), dtype=bool), drop_pct=3.0)
    assert len(d) == 1, f"fixture moet precies 1 desaturatie geven, gaf {len(d)}"


def test_drie_events_op_een_desaturatie_worden_begrensd_tot_twee():
    events = [_ev(270.0, 0.9), _ev(290.0, 0.7), _ev(310.0, 0.5)]
    acc, rej, st = limit_events_per_desaturation(
        events, [], _spo2_met_een_desaturatie(), SF, _hypno(), max_events=2)
    assert st["n_groups_over_limit"] == 1
    assert st["n_degraded"] == 1
    assert len(acc) == 2
    # de laagste p_scored valt af, niet de eerste of de laatste in tijd
    assert {e["p_scored"] for e in acc} == {0.9, 0.7}


def test_gedegradeerd_niet_verwijderd():
    events = [_ev(270.0, 0.9), _ev(290.0, 0.7), _ev(310.0, 0.5)]
    acc, rej, st = limit_events_per_desaturation(
        events, [], _spo2_met_een_desaturatie(), SF, _hypno(), max_events=2)
    assert len(rej) == 1
    assert rej[0]["reject_reason"] == "desat_reuse_limit"
    # alle oorspronkelijke velden blijven staan — de ML-promotie leest ze
    assert rej[0]["p_scored"] == 0.5
    assert rej[0]["onset_s"] == 310.0
    assert rej[0]["desaturation_pct"] == 6.0
    # niets raakt zoek
    assert len(acc) + len(rej) == len(events)


def test_none_is_een_no_op():
    events = [_ev(270.0, 0.9), _ev(290.0, 0.7), _ev(310.0, 0.5)]
    acc, rej, st = limit_events_per_desaturation(
        events, [], _spo2_met_een_desaturatie(), SF, _hypno(), max_events=None)
    assert acc == events and rej == [] and st["n_degraded"] == 0


def test_limiet_boven_het_aantal_verandert_niets():
    events = [_ev(270.0, 0.9), _ev(290.0, 0.7), _ev(310.0, 0.5)]
    acc, rej, st = limit_events_per_desaturation(
        events, [], _spo2_met_een_desaturatie(), SF, _hypno(), max_events=5)
    assert len(acc) == 3 and st["n_degraded"] == 0


def test_arousal_bevestigde_events_blijven_ongemoeid():
    """Zonder desaturatie is er niets hergebruikt, dus niets te begrenzen."""
    events = [_ev(270.0, 0.9, desat=None), _ev(290.0, 0.7, desat=None),
              _ev(310.0, 0.5, desat=None)]
    acc, rej, st = limit_events_per_desaturation(
        events, [], _spo2_met_een_desaturatie(), SF, _hypno(), max_events=1)
    assert len(acc) == 3 and st["n_degraded"] == 0


def test_ondiepe_desaturatie_telt_niet_als_bevestiging():
    events = [_ev(270.0, 0.9, desat=1.2), _ev(290.0, 0.7, desat=1.5),
              _ev(310.0, 0.5, desat=0.8)]
    acc, rej, st = limit_events_per_desaturation(
        events, [], _spo2_met_een_desaturatie(), SF, _hypno(), max_events=1)
    assert len(acc) == 3 and st["n_degraded"] == 0


def test_apneus_worden_niet_begrensd():
    events = [_ev(270.0, 0.9, type_="obstructive"),
              _ev(290.0, 0.7, type_="obstructive"),
              _ev(310.0, 0.5, type_="central")]
    acc, rej, st = limit_events_per_desaturation(
        events, [], _spo2_met_een_desaturatie(), SF, _hypno(), max_events=1)
    assert len(acc) == 3 and st["n_degraded"] == 0


def test_twee_desaturaties_worden_apart_geteld():
    """Twee groepen van twee bij limiet 1 = twee degradaties, niet drie."""
    s = _spo2_met_een_desaturatie(nadir_t=300.0)
    a, b = int(700 * SF), int(740 * SF)
    s[a:b] = 90.0
    events = [_ev(270.0, 0.9), _ev(295.0, 0.6),
              _ev(680.0, 0.8), _ev(705.0, 0.4)]
    acc, rej, st = limit_events_per_desaturation(
        events, [], s, SF, _hypno(), max_events=1)
    assert st["n_desaturations"] == 2
    assert st["n_groups_over_limit"] == 2
    assert st["n_degraded"] == 2
    assert {e["p_scored"] for e in acc} == {0.9, 0.8}


def test_wakker_signaal_geeft_geen_desaturaties():
    """De slaapmasker-tak: buiten slaap wordt niets gedetecteerd of begrensd."""
    events = [_ev(270.0, 0.9), _ev(290.0, 0.7), _ev(310.0, 0.5)]
    acc, rej, st = limit_events_per_desaturation(
        events, [], _spo2_met_een_desaturatie(), SF,
        ["W"] * (DUUR_S // 30), max_events=1)
    assert len(acc) == 3 and st["n_degraded"] == 0


def test_gelijkspel_valt_op_de_vroegste_terug():
    """Anders hangt de uitkomst van de sorteerstabiliteit af."""
    events = [_ev(310.0, 0.7), _ev(270.0, 0.7), _ev(290.0, 0.7)]
    acc, _rej, st = limit_events_per_desaturation(
        events, [], _spo2_met_een_desaturatie(), SF, _hypno(), max_events=1)
    assert st["n_degraded"] == 2
    assert acc[0]["onset_s"] == 270.0


def test_bestaande_rejected_blijven_staan():
    events = [_ev(270.0, 0.9), _ev(290.0, 0.7), _ev(310.0, 0.5)]
    bestaand = [{"type": "hypopnea", "onset_s": 10.0, "reject_reason": "iets"}]
    _acc, rej, _st = limit_events_per_desaturation(
        events, bestaand, _spo2_met_een_desaturatie(), SF, _hypno(),
        max_events=2)
    assert len(rej) == 2
    assert rej[0]["reject_reason"] == "iets"


def test_zonder_spo2_gebeurt_er_niets():
    events = [_ev(270.0, 0.9), _ev(290.0, 0.7)]
    acc, rej, st = limit_events_per_desaturation(
        events, [], None, SF, _hypno(), max_events=1)
    assert acc == events and st["n_degraded"] == 0


@pytest.mark.parametrize("limiet", [1, 2, 3])
def test_limiet_wordt_werkelijk_gerespecteerd(limiet):
    events = [_ev(265.0 + 12 * i, 0.9 - 0.05 * i) for i in range(5)]
    acc, _rej, _st = limit_events_per_desaturation(
        events, [], _spo2_met_een_desaturatie(), SF, _hypno(),
        max_events=limiet)
    assert len(acc) == limiet
