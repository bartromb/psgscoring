"""Grensoffset-meting: de helpers die de getekende delta's afleiden.

Geen scoringspad. Deze toetsen pinnen de tekenconventie (algoritme − scorer)
en de lost-to-iou-telling — de twee plekken waar een stille verwisseling de
hele meting onbruikbaar zou maken zonder zichtbaar te falen.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from measure_boundary_offsets import lost_to_iou, signed_deltas  # noqa: E402
from validate_psgipa import LEGACY_MATCHER  # noqa: E402

M = dict(LEGACY_MATCHER)


def test_sign_convention_algo_minus_scorer():
    """Algoritme begint 2 s later en eindigt 3 s eerder dan de scorer:
    d_onset = +2, d_offset = −3. Wie de tekens omdraait, leest elke offset
    in de verkeerde richting en 'corrigeert' straks de verkeerde kant op."""
    algo = [(102.0, 117.0, "hypopnea")]
    ref = [(100.0, 120.0, "hypopnea")]
    d = signed_deltas(algo, ref, M)
    assert len(d) == 1
    assert d[0]["d_onset"] == 2.0
    assert d[0]["d_offset"] == -3.0
    assert d[0]["d_duration"] == -5.0


def test_perfect_match_gives_zero_deltas():
    ev = [(50.0, 65.0, "obstructive")]
    d = signed_deltas(ev, list(ev), M)
    assert d[0]["d_onset"] == 0.0 and d[0]["d_offset"] == 0.0
    assert d[0]["family"] == "apnea"


def test_unmatched_events_produce_no_deltas():
    """Alleen gematchte paren tellen: de offsetmeting gaat over grenzen van
    gevonden events, niet over gemiste events — die horen in lost_to_iou of
    in de F1, niet hier."""
    algo = [(0.0, 12.0, "hypopnea")]
    ref = [(500.0, 512.0, "hypopnea")]
    assert signed_deltas(algo, ref, M) == []


def test_lost_to_iou_counts_found_but_misaligned():
    """Referentie-event 100–120; algoritme 116–140: overlap 4 s, IoU
    4/40 = 0,10 < 0,20 → gevonden maar anders afgebakend → telt."""
    algo = [(116.0, 140.0, "hypopnea")]
    ref = [(100.0, 120.0, "hypopnea")]
    assert lost_to_iou(algo, ref, 0.20) == 1


def test_lost_to_iou_ignores_true_misses_and_true_matches():
    algo = [(0.0, 12.0, "hypopnea"), (200.0, 215.0, "hypopnea")]
    ref = [
        (1.0, 13.0, "hypopnea"),     # echte match (hoge IoU) → telt niet
        (500.0, 512.0, "hypopnea"),  # geen enkele overlap → telt niet
    ]
    assert lost_to_iou(algo, ref, 0.20) == 0


def test_human_deltas_are_symmetric_in_distribution():
    """Mens-tegen-mens is de referentieverdeling. Draai je a en b om, dan
    klappen de tekens — maar |delta| en de spreiding blijven gelijk. De
    samenvatting gebruikt daarom de verdeling, niet de richting."""
    a = [(100.0, 118.0, "hypopnea")]
    b = [(103.0, 120.0, "hypopnea")]
    d_ab = signed_deltas(a, b, M)[0]
    d_ba = signed_deltas(b, a, M)[0]
    assert d_ab["d_onset"] == -d_ba["d_onset"]
    assert d_ab["d_offset"] == -d_ba["d_offset"]
