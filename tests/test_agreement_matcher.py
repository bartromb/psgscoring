"""tests/test_agreement_matcher.py — koppelregel, 1 s-κ en grensverschuiving.

Waarom dit bestaat (literatuurcheck 25-10-2026, docs/literatuur_update_20261025.md):
de arousalliteratuur rapporteert event-F1 met *elke overlap* als treffer
(Pitkänen e.a. 2026, Ehrlich e.a. 2024) en daarnaast een seconde-per-seconde κ
en de verschuiving van begin en einde. Onze cijfers staan op IoU 0,20. Zonder
dezelfde koppelregel is "0,745 tegen 0,61" geen vergelijking. De matcher
krijgt daarom een tweede koppelregel en drie maten erbij; het bestaande pad
blijft byte-identiek.

Elke test hieronder faalde op de code van vóór de uitbreiding (de kwarg
`matcher` bestond niet; de nieuwe sleutels ontbraken).
"""
from __future__ import annotations

import pytest

from psgscoring.agreement import IOU_THRESH, compare_event_sets


def ev(onset, dur, typ="arousal"):
    return {"onset_s": float(onset), "duration_s": float(dur), "type": typ}


# ── koppelregel "any" ───────────────────────────────────────────────────

def test_any_overlap_pairs_what_iou_rejects():
    """A = [0,10], B = [8,18]: overlap 2 s, IoU 0,111 — IoU 0,20 weigert, elke overlap paart."""
    a, b = [ev(0, 10)], [ev(8, 10)]
    strict = compare_event_sets(a, b)
    assert strict["n_shared"] == 0, "IoU 0,111 hoort onder de drempel te blijven"
    loose = compare_event_sets(a, b, matcher="any")
    assert loose["n_shared"] == 1
    assert loose["matcher"] == "any"
    assert strict["matcher"] == "iou"
    # De IoU van het gepaarde paar blijft de IoU, ook onder "any".
    assert loose["median_iou"] == pytest.approx(2 / 18, abs=1e-4)


def test_any_overlap_requires_positive_overlap():
    """Aangrenzend ([0,10] en [10,20]) is geen overlap."""
    r = compare_event_sets([ev(0, 10)], [ev(10, 10)], matcher="any")
    assert r["n_shared"] == 0
    assert r["n_only_a"] == 1 and r["n_only_b"] == 1


def test_any_overlap_takes_the_largest_overlap_first_and_is_symmetric():
    """A = [0,10]; B1 = [8,30] (overlap 2 s), B2 = [5,9] (overlap 4 s): A paart met B2."""
    a = [ev(0, 10)]
    b = [ev(8, 22), ev(5, 4)]
    r = compare_event_sets(a, b, matcher="any")
    assert r["n_shared"] == 1
    assert r["median_iou"] == pytest.approx(4 / 10, abs=1e-4), "B2 (IoU 0,4), niet B1"
    rb = compare_event_sets(b, a, matcher="any")
    assert rb["n_shared"] == r["n_shared"] and rb["median_iou"] == r["median_iou"]


def test_unknown_matcher_is_refused():
    with pytest.raises(ValueError):
        compare_event_sets([ev(0, 10)], [ev(0, 10)], matcher="fraction")


# ── 1 s-κ ───────────────────────────────────────────────────────────────

def test_kappa_1s_hand_computed():
    """A dekt seconden 0–9, B seconden 2–11, duur 20 s.

    n11 = 8, n10 = 2, n01 = 2, n00 = 8 → p_o = 16/20 = 0,8;
    p_e = (10·10 + 10·10)/400 = 0,5; κ = (0,8 − 0,5)/(1 − 0,5) = 0,6.
    """
    r = compare_event_sets([ev(0, 10)], [ev(2, 10)], duration_s=20.0)
    assert r["kappa_1s"] == pytest.approx(0.6, abs=1e-4)


def test_kappa_1s_default_duration_is_the_last_offset():
    """Zonder duur: tot het einde van het laatste event van beide lijsten (12 s).

    n11 = 8, n10 = 2, n01 = 2, n00 = 0 → p_o = 8/12; p_e = (10·10 + 2·2)/144;
    κ = (0,6667 − 0,7222)/(1 − 0,7222) = −0,2.
    """
    r = compare_event_sets([ev(0, 10)], [ev(2, 10)])
    assert r["kappa_1s"] == pytest.approx(-0.2, abs=1e-4)


def test_kappa_1s_identical_disjoint_and_empty():
    a = [ev(0, 10), ev(50, 5)]
    assert compare_event_sets(a, list(a), duration_s=100)["kappa_1s"] == pytest.approx(1.0)
    r = compare_event_sets([ev(0, 10)], [ev(50, 10)], duration_s=100)
    assert r["kappa_1s"] < 0, "twee disjuncte events: slechter dan toeval"
    assert compare_event_sets([], [], duration_s=100)["kappa_1s"] is None, \
        "zonder events is κ niet gedefinieerd, en 1,0 zou een leugen zijn"


def test_kappa_1s_bins_count_any_overlap_with_the_second():
    """Een event [0,5; 1,2] raakt seconde 0 én seconde 1."""
    r = compare_event_sets([ev(0.5, 0.7)], [ev(0, 2)], duration_s=4.0)
    assert r["kappa_1s"] == pytest.approx(1.0)


# ── grensverschuiving en duur ───────────────────────────────────────────

def test_onset_and_offset_shift_are_b_minus_a_over_pairs():
    """A = [100,110]; B = [101,5; 112]: begin +1,5 s, einde +2,0 s."""
    r = compare_event_sets([ev(100, 10)], [ev(101.5, 10.5)])
    assert r["n_shared"] == 1
    assert r["onset_shift_median_s"] == pytest.approx(1.5)
    assert r["offset_shift_median_s"] == pytest.approx(2.0)
    assert r["median_duration_a"] == pytest.approx(10.0)
    assert r["median_duration_b"] == pytest.approx(10.5)
    rb = compare_event_sets([ev(101.5, 10.5)], [ev(100, 10)])
    assert rb["onset_shift_median_s"] == pytest.approx(-1.5)
    assert rb["offset_shift_median_s"] == pytest.approx(-2.0)


def test_shift_is_none_without_pairs_and_duration_none_without_events():
    r = compare_event_sets([ev(0, 10)], [ev(500, 10)])
    assert r["onset_shift_median_s"] is None and r["offset_shift_median_s"] is None
    assert r["median_duration_a"] == 10.0 and r["median_duration_b"] == 10.0
    e = compare_event_sets([], [ev(500, 10)])
    assert e["median_duration_a"] is None and e["median_duration_b"] == 10.0


# ── het bestaande pad verandert niet ────────────────────────────────────

#: Vastgelegd op 25-10-2026 met de code van vóór de uitbreiding (0.36.2):
#: `compare_event_sets(A, B, label_a="x", label_b="y")` op de lijsten hieronder.
_FROZEN_0362 = {
    "excl_bare_uncertain": {
        "jaccard": 0.6, "median_iou": 0.7647, "min_iou": 0.25, "n_a": 4, "n_b": 4,
        "n_only_a": 1, "n_only_b": 1, "n_shared": 3, "n_type_changed": 2,
        "per_category": {"apnea": {"only_a": 1, "only_b": 0, "shared": 2},
                         "hypopnea": {"only_a": 0, "only_b": 1, "shared": 1}},
        "type_changes": {"hypopnea -> hypopnea_uncertain": 1, "obstructive -> central": 1}},
    "iou_thresh": 0.2, "jaccard": 0.5, "labels": {"a": "x", "b": "y"},
    "median_iou": 0.7647, "min_iou": 0.25, "n_a": 5, "n_b": 4,
    "n_bare_uncertain": {"a": 1, "b": 0}, "n_only_a": 2, "n_only_b": 1,
    "n_shared": 3, "n_type_changed": 2,
    "per_category": {"apnea": {"only_a": 2, "only_b": 0, "shared": 2},
                     "hypopnea": {"only_a": 0, "only_b": 1, "shared": 1}},
    "type_changes": {"hypopnea -> hypopnea_uncertain": 1, "obstructive -> central": 1},
}

_NEW_KEYS = {"matcher", "kappa_1s", "onset_shift_median_s", "offset_shift_median_s",
             "median_duration_a", "median_duration_b"}


def _frozen_input():
    a = [ev(10, 15, "obstructive"), ev(100, 20, "hypopnea"), ev(300, 12, "central"),
         ev(500, 10, "uncertain"), ev(96, 10, "obstructive")]
    b = [ev(12, 15, "central"), ev(100, 21, "hypopnea_uncertain"),
         ev(700, 20, "hypopnea"), ev(90, 10, "obstructive")]
    return a, b


def test_default_path_is_byte_identical_to_0362_on_every_old_key():
    a, b = _frozen_input()
    r = compare_event_sets(a, b, label_a="x", label_b="y")
    old_keys = set(_FROZEN_0362)
    assert {k: r[k] for k in old_keys if k != "excl_bare_uncertain"} == \
        {k: v for k, v in _FROZEN_0362.items() if k != "excl_bare_uncertain"}
    x = r["excl_bare_uncertain"]
    assert {k: x[k] for k in _FROZEN_0362["excl_bare_uncertain"]} == \
        _FROZEN_0362["excl_bare_uncertain"]
    assert r["matcher"] == "iou" and r["iou_thresh"] == IOU_THRESH


def test_new_keys_are_present_on_both_variants():
    a, b = _frozen_input()
    r = compare_event_sets(a, b)
    assert _NEW_KEYS - {"matcher"} <= set(r["excl_bare_uncertain"])
    assert _NEW_KEYS <= set(r)
    # Drie paren in de vastgelegde invoer: (10,25)↔(12,27), (100,120)↔(100,121),
    # (96,106)↔(90,100): onset-verschuivingen +2, 0, −6 → mediaan 0; einde +2, +1, −6 → +1.
    assert r["onset_shift_median_s"] == pytest.approx(0.0)
    assert r["offset_shift_median_s"] == pytest.approx(1.0)
