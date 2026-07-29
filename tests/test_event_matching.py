"""
tests/test_event_matching.py — event-matching in de PSG-IPA-harness.

`match_events()` is uit `analyse_one()` getrokken. De legacy-modus
(type_aware=False, optimal=False) moet exact doen wat de inline-lus deed,
want paper v31 rapporteert die getallen. Deze module pint dat vast met een
kopie van de oude logica en een randomised equivalentietest.

De harness staat in de repo-root, niet in het package — vandaar de
sys.path-toevoeging. `psgscoring/` zelf wordt hier niet geraakt.

Run:
    pytest tests/test_event_matching.py -v
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validate_psgipa import (  # noqa: E402
    iou,
    match_events,
    match_events_symmetric,
    pairwise_baseline,
    percentile_of,
)


# ══════════════════════════════════════════════════════════════
#  Referentie-implementatie: letterlijke kopie van de inline-lus
#  zoals die vóór de refactor in analyse_one() stond.
# ══════════════════════════════════════════════════════════════

def _legacy_match(algo_events, ref_events):
    matched_a, matched_r, onset_diffs = set(), set(), []
    for i, (a0, a1, _) in enumerate(algo_events):
        best_j, best_v = -1, 0.0
        for j, (r0, r1, _) in enumerate(ref_events):
            if j in matched_r:
                continue
            v = iou(a0, a1, r0, r1)
            if v >= 0.20 and v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            matched_a.add(i)
            matched_r.add(best_j)
            onset_diffs.append(abs(algo_events[i][0] - ref_events[best_j][0]))
    tp = len(matched_a)
    fp = len(algo_events) - tp
    fn = len(ref_events) - len(matched_r)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec_ = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0
    return tp, fp, fn, f1, onset_diffs


def _random_events(rng, n, span=30000.0):
    """Willekeurige eventlijst met realistische duren (10-90 s)."""
    out = []
    for _ in range(n):
        onset = rng.uniform(0.0, span)
        dur = rng.uniform(10.0, 90.0)
        etype = rng.choice(["obstructive", "central", "mixed", "hypopnea"])
        out.append((onset, onset + dur, etype))
    return sorted(out)


def _clustered_events(rng, n, span=3000.0):
    """Dicht op elkaar: dwingt concurrentie om dezelfde partner af."""
    out = []
    for _ in range(n):
        onset = rng.uniform(0.0, span)
        dur = rng.uniform(10.0, 40.0)
        out.append((onset, onset + dur, rng.choice(["hypopnea", "central"])))
    return sorted(out)


# ══════════════════════════════════════════════════════════════
#  1. Byte-identieke invariant: legacy == match_events()
# ══════════════════════════════════════════════════════════════

def test_legacy_equivalence_over_random_cases():
    """>=200 willekeurige gevallen, vaste seed: identieke tp/fp/fn/f1."""
    rng = random.Random(20260729)
    n_cases = 250
    for case in range(n_cases):
        a = _random_events(rng, rng.randint(0, 40))
        b = _random_events(rng, rng.randint(0, 40))

        tp, fp, fn, f1, diffs = _legacy_match(a, b)
        m = match_events(a, b, iou_thresh=0.20)

        assert m["tp"] == tp, f"case {case}: tp {m['tp']} != {tp}"
        assert m["fp"] == fp, f"case {case}: fp {m['fp']} != {fp}"
        assert m["fn"] == fn, f"case {case}: fn {m['fn']} != {fn}"
        assert m["f1"] == pytest.approx(f1), f"case {case}: f1 {m['f1']} != {f1}"

        expected_dt = (sum(diffs) / len(diffs)) if diffs else None
        if expected_dt is None:
            assert m["mean_dt"] is None
        else:
            assert m["mean_dt"] == pytest.approx(expected_dt)


def test_legacy_equivalence_on_clustered_events():
    """Clusters zijn waar greedy het meest afwijkt — apart afgedekt."""
    rng = random.Random(31415)
    for case in range(150):
        a = _clustered_events(rng, rng.randint(1, 25))
        b = _clustered_events(rng, rng.randint(1, 25))

        tp, fp, fn, f1, _ = _legacy_match(a, b)
        m = match_events(a, b, iou_thresh=0.20)

        assert (m["tp"], m["fp"], m["fn"]) == (tp, fp, fn), f"case {case}"
        assert m["f1"] == pytest.approx(f1)


def test_legacy_equivalence_edge_cases():
    """Lege lijsten en identieke lijsten."""
    ev = [(0.0, 20.0, "hypopnea"), (100.0, 130.0, "central")]
    for a, b in [([], []), (ev, []), ([], ev), (ev, ev)]:
        tp, fp, fn, f1, _ = _legacy_match(a, b)
        m = match_events(a, b, iou_thresh=0.20)
        assert (m["tp"], m["fp"], m["fn"]) == (tp, fp, fn)
        assert m["f1"] == pytest.approx(f1)


# ══════════════════════════════════════════════════════════════
#  2. Handmatig narekenbare gevallen
# ══════════════════════════════════════════════════════════════

def test_identical_lists_give_perfect_score():
    ev = [(0.0, 20.0, "hypopnea"), (60.0, 90.0, "central"), (200.0, 240.0, "obstructive")]
    m = match_events(ev, ev)
    assert m["tp"] == 3
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["f1"] == pytest.approx(1.0)
    assert m["mean_dt"] == pytest.approx(0.0)


def test_disjoint_lists_give_zero():
    a = [(0.0, 20.0, "hypopnea")]
    b = [(1000.0, 1020.0, "hypopnea")]
    m = match_events(a, b)
    assert m["tp"] == 0
    assert m["f1"] == pytest.approx(0.0)
    assert m["mean_dt"] is None


def test_iou_threshold_boundary():
    """
    Twee events van 12 s met 8 s verschuiving: inter = 12-8 = 4,
    span = 12+8 = 20 -> IoU = 0.20, matcht net. Bij 9 s: 3/21 < 0.20.
    """
    a = [(0.0, 12.0, "hypopnea")]
    assert match_events(a, [(8.0, 20.0, "hypopnea")])["tp"] == 1
    assert match_events(a, [(9.0, 21.0, "hypopnea")])["tp"] == 0


def test_duration_mismatch_boundary():
    """Algo 60 s bevat ref 12 s -> IoU = 12/60 = 0.20, net een match; 70 s niet."""
    ref = [(10.0, 22.0, "hypopnea")]
    assert match_events([(0.0, 60.0, "hypopnea")], ref)["tp"] == 1
    assert match_events([(0.0, 70.0, "hypopnea")], ref)["tp"] == 0


def test_matched_pairs_reports_indices_and_iou():
    a = [(0.0, 20.0, "hypopnea")]
    b = [(0.0, 20.0, "hypopnea")]
    m = match_events(a, b)
    assert len(m["matched_pairs"]) == 1
    i, j, v = m["matched_pairs"][0]
    assert (i, j) == (0, 0)
    assert v == pytest.approx(1.0)


def test_precision_recall_are_consistent_with_counts():
    a = [(0.0, 20.0, "hypopnea"), (500.0, 520.0, "central")]
    b = [(0.0, 20.0, "hypopnea")]
    m = match_events(a, b)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_legacy_mode_ignores_event_type():
    """Zonder type_aware matcht een hypopnee met een apneu — bewust vastgelegd."""
    a = [(0.0, 20.0, "hypopnea")]
    b = [(0.0, 20.0, "obstructive")]
    assert match_events(a, b)["tp"] == 1


# ══════════════════════════════════════════════════════════════
#  3. Richtingsafhankelijkheid van greedy matching
# ══════════════════════════════════════════════════════════════

# Op de echte PSG-IPA-data is de asymmetrie exact 0 voor alle 5 opnames:
# scorer-events liggen ver genoeg uit elkaar dat greedy in beide richtingen
# dezelfde partners kiest. Dat is een meting, geen aanname — en dit geval
# bewijst dat de detectie werkt, zodat die 0 niet stilletjes een bug is.
ASYMMETRIC_A = [
    (6.5, 61.5, "hypopnea"), (74.1, 117.9, "hypopnea"),
    (82.9, 118.6, "hypopnea"), (93.6, 147.3, "hypopnea"),
]
ASYMMETRIC_B = [
    (47.9, 63.1, "hypopnea"), (76.1, 89.2, "hypopnea"), (95.7, 125.4, "hypopnea"),
]


def test_greedy_matching_can_be_direction_dependent():
    """Zwaar overlappende events: richting verandert de gevonden TP."""
    ab = match_events(ASYMMETRIC_A, ASYMMETRIC_B)
    ba = match_events(ASYMMETRIC_B, ASYMMETRIC_A)
    assert ab["tp"] != ba["tp"], "dit geval hoort juist asymmetrisch te zijn"


def test_symmetric_wrapper_detects_and_averages_asymmetry():
    m = match_events_symmetric(ASYMMETRIC_A, ASYMMETRIC_B)
    assert m["asymmetry"] > 0.0
    assert m["f1"] == pytest.approx((m["f1_ab"] + m["f1_ba"]) / 2.0)


def test_symmetric_wrapper_is_order_independent():
    """f1 mag niet afhangen van de volgorde van de argumenten."""
    ab = match_events_symmetric(ASYMMETRIC_A, ASYMMETRIC_B)
    ba = match_events_symmetric(ASYMMETRIC_B, ASYMMETRIC_A)
    assert ab["f1"] == pytest.approx(ba["f1"])
    assert ab["asymmetry"] == pytest.approx(ba["asymmetry"])


# ══════════════════════════════════════════════════════════════
#  4. Mens-tegen-mens baseline
# ══════════════════════════════════════════════════════════════

def _sets(n, events):
    return [(f"scorer{i}", list(events)) for i in range(n)]


@pytest.mark.parametrize("n,expected", [(2, 1), (3, 3), (5, 10), (12, 66)])
def test_pair_count_is_n_choose_2(n, expected):
    ev = [(0.0, 20.0, "hypopnea"), (100.0, 130.0, "central")]
    r = pairwise_baseline(_sets(n, ev), iou_thresh=0.20)
    assert r["summary"]["n_pairs"] == expected
    assert len(r["pairs"]) == expected


def test_identical_scorers_give_baseline_of_one():
    """Sanity: een scorer tegen zichzelf geeft F1 = 1.0."""
    ev = [(0.0, 20.0, "hypopnea"), (100.0, 130.0, "central")]
    r = pairwise_baseline(_sets(12, ev), iou_thresh=0.20)
    s = r["summary"]
    assert s["n_pairs"] == 66
    assert s["median"] == pytest.approx(1.0)
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(1.0)
    assert s["max_asymmetry"] == pytest.approx(0.0)


def test_baseline_spread_reflects_disagreement():
    """Eén afwijkende scorer verlaagt het minimum, niet de mediaan."""
    agree = [(0.0, 20.0, "hypopnea"), (100.0, 130.0, "central")]
    odd = [(500.0, 520.0, "hypopnea")]
    sets = _sets(4, agree) + [("outlier", odd)]
    s = pairwise_baseline(sets, iou_thresh=0.20)["summary"]
    assert s["n_pairs"] == 10
    assert s["min"] == pytest.approx(0.0)
    assert s["median"] == pytest.approx(1.0)


def test_empty_scorer_set_is_handled():
    assert pairwise_baseline([], iou_thresh=0.20)["summary"]["n_pairs"] == 0


# ══════════════════════════════════════════════════════════════
#  5. Percentiel
# ══════════════════════════════════════════════════════════════

def test_percentile_of_basic():
    pop = [0.0, 0.25, 0.75, 1.0]
    assert percentile_of(0.5, pop) == pytest.approx(50.0)
    assert percentile_of(-1.0, pop) == pytest.approx(0.0)
    assert percentile_of(2.0, pop) == pytest.approx(100.0)


def test_percentile_of_counts_ties_as_half():
    assert percentile_of(0.5, [0.5, 0.5]) == pytest.approx(50.0)
    assert percentile_of(0.5, [0.0, 0.5]) == pytest.approx(75.0)


def test_percentile_of_empty_population_is_none():
    assert percentile_of(0.5, []) is None
    assert percentile_of(None, [0.1, 0.2]) is None
