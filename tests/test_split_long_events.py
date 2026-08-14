"""Blok 2D: lange events splitsen op fysiologische ankers.

Herkomst van het idee: CAISR-resp (Nasiri et al., Sleep 2025), dat events
> 60 s splitst. Implementatie is eigen werk vanaf de specificatie; zie
`docs/third_party_comparison.md`.

De suite moet aantoonbaar falen als de splitser niets doet — anders meet ze
niets.
"""
from __future__ import annotations

import pytest

from psgscoring.profiles import PROFILES, PostProcessingRules
from psgscoring.respiratory import split_long_events


def _ev(onset, duur, type_="hypopnea"):
    return {"type": type_, "onset_s": onset, "duration_s": duur,
            "stage": "N2", "epoch": int(onset // 30)}


# ── het veld ─────────────────────────────────────────────────────────

def test_default_is_uit():
    assert PostProcessingRules().split_events_longer_than_s is None


def test_geen_enkel_geleverd_profiel_splitst():
    aan = [n for n, p in PROFILES.items()
           if p.post_processing.split_events_longer_than_s is not None]
    assert aan == [], f"profielen zouden byte-identiek blijven: {aan}"


def test_veld_bereikt_de_legacy_dict():
    import psgscoring.constants as C
    for naam, d in C.SCORING_PROFILES.items():
        assert d["SPLIT_EVENTS_LONGER_THAN_S"] is None, naam


# ── het splitsen ─────────────────────────────────────────────────────

def test_lang_event_splitst_op_desaturaties():
    """90 s met twee desaturaties erin wordt drie delen."""
    ev = [_ev(100.0, 90.0)]
    acc, rej, st = split_long_events(ev, [], threshold_s=60.0,
                                     desat_onsets=[130.0, 160.0])
    assert st["n_split"] == 1 and st["n_parts"] == 3
    assert st["anchor_desat"] == 1 and st["anchor_arousal"] == 0
    assert [(e["onset_s"], e["duration_s"]) for e in acc] == [
        (100.0, 30.0), (130.0, 30.0), (160.0, 30.0)]


def test_zonder_desaturatie_valt_hij_terug_op_arousals():
    acc, _rej, st = split_long_events([_ev(100.0, 90.0)], [], threshold_s=60.0,
                                      desat_onsets=[], arousal_onsets=[145.0])
    assert st["anchor_arousal"] == 1 and st["n_parts"] == 2
    assert len(acc) == 2


def test_desaturatie_gaat_voor_op_arousal():
    _acc, _rej, st = split_long_events([_ev(100.0, 90.0)], [], threshold_s=60.0,
                                       desat_onsets=[140.0],
                                       arousal_onsets=[120.0, 160.0])
    assert st["anchor_desat"] == 1 and st["anchor_arousal"] == 0


def test_zonder_anker_blijft_het_event_heel():
    ev = [_ev(100.0, 90.0)]
    acc, _rej, st = split_long_events(ev, [], threshold_s=60.0,
                                      desat_onsets=[], arousal_onsets=[])
    assert st["n_split"] == 0 and acc == ev


def test_kort_event_wordt_niet_aangeraakt():
    ev = [_ev(100.0, 40.0)]
    acc, _rej, st = split_long_events(ev, [], threshold_s=60.0,
                                      desat_onsets=[120.0])
    assert st["n_candidates"] == 0 and acc == ev


def test_anker_te_dicht_bij_de_rand_telt_niet():
    """Anders ontstaat een fragment dat per constructie te kort is."""
    ev = [_ev(100.0, 90.0)]
    for anker in (105.0, 185.0):
        acc, _rej, st = split_long_events(ev, [], threshold_s=60.0,
                                          desat_onsets=[anker])
        assert st["n_split"] == 0, f"anker op {anker} had genegeerd moeten worden"
        assert acc == ev


def test_te_kort_deel_gaat_naar_rejected_niet_weg():
    """Ankers op 118 en 122 geven een middendeel van 4 s."""
    acc, rej, st = split_long_events([_ev(100.0, 90.0)], [], threshold_s=60.0,
                                     desat_onsets=[118.0, 122.0])
    assert st["n_fragments_rejected"] == 1
    assert any(r.get("reject_reason") == "split_fragment" for r in rej)
    assert len(acc) + len(rej) == 3, "geen enkel deel mag verdwijnen"


def test_delen_dragen_hun_herkomst():
    acc, _rej, _st = split_long_events([_ev(100.0, 90.0)], [], threshold_s=60.0,
                                       desat_onsets=[130.0, 160.0])
    for deel in acc:
        det = deel["classify_detail"]
        assert det["split_from"] == {"onset_s": 100.0, "duration_s": 90.0}
        assert det["split_anchor"] == "desaturation"


def test_subtype_wordt_geerfd_niet_herbepaald():
    acc, _rej, _st = split_long_events([_ev(100.0, 90.0, "central")], [],
                                       threshold_s=60.0, desat_onsets=[140.0])
    assert all(e["type"] == "central" for e in acc)


def test_none_is_een_no_op():
    ev = [_ev(100.0, 90.0)]
    acc, rej, st = split_long_events(ev, [], threshold_s=None,
                                     desat_onsets=[130.0])
    assert acc == ev and rej == [] and st["n_split"] == 0


def test_originele_events_worden_niet_gemuteerd():
    ev = [_ev(100.0, 90.0)]
    origineel = dict(ev[0])
    split_long_events(ev, [], threshold_s=60.0, desat_onsets=[130.0])
    assert ev[0] == origineel


def test_totale_duur_blijft_behouden():
    acc, rej, _st = split_long_events([_ev(100.0, 90.0)], [], threshold_s=60.0,
                                      desat_onsets=[130.0, 160.0])
    assert round(sum(e["duration_s"] for e in acc + rej), 2) == 90.0


@pytest.mark.parametrize("drempel", [30.0, 60.0, 120.0])
def test_drempel_bepaalt_wat_kandidaat_is(drempel):
    _acc, _rej, st = split_long_events([_ev(100.0, 90.0)], [],
                                       threshold_s=drempel,
                                       desat_onsets=[140.0])
    assert st["n_candidates"] == (1 if 90.0 > drempel else 0)
