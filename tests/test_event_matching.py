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

from validate_psgipa import iou, match_events  # noqa: E402


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
