"""tests/test_agreement.py — de paarsgewijze eventovereenkomst.

De matcher beantwoordt de vraag die een indextabel oproept en niet kan
beantwoorden: zijn twee even grote eventverzamelingen dezelfde verzameling?
Een test die dat niet scherp pint, meet niets -- een matcher die alles paart
geeft ook Jaccard 1,0.
"""
from __future__ import annotations

import pytest

from psgscoring.agreement import IOU_THRESH, compare_event_sets, event_category


def ev(onset, dur, typ="obstructive"):
    return {"onset_s": float(onset), "duration_s": float(dur), "type": typ}


# ── de drie gevallen uit de specificatie ────────────────────────────────

def test_identical_lists_give_jaccard_one():
    a = [ev(10, 15), ev(100, 20, "hypopnea"), ev(300, 12, "central")]
    r = compare_event_sets(a, list(a))
    assert r["n_shared"] == 3
    assert r["n_only_a"] == 0 and r["n_only_b"] == 0
    assert r["jaccard"] == 1.0
    assert r["median_iou"] == 1.0


def test_disjoint_lists_give_jaccard_zero():
    a = [ev(10, 15), ev(100, 20)]
    b = [ev(500, 15), ev(900, 20)]
    r = compare_event_sets(a, b)
    assert r["n_shared"] == 0
    assert r["n_only_a"] == 2 and r["n_only_b"] == 2
    assert r["jaccard"] == 0.0
    assert r["median_iou"] is None


def test_iou_threshold_is_inclusive_at_the_boundary():
    """Precies op 0,20 hoort te paren; net eronder niet.

    A = [0,10], B = [8,10]: overlap 2, unie 10, IoU exact 0,20.
    A = [0,10], B = [8,18]: overlap 2, unie 18, IoU 0,111.
    """
    on_boundary = compare_event_sets([ev(0, 10)], [ev(8, 2)])
    assert on_boundary["n_shared"] == 1, "IoU == drempel moet paren (>=)"
    assert on_boundary["median_iou"] == pytest.approx(0.20)

    below = compare_event_sets([ev(0, 10)], [ev(8, 10)])
    assert below["n_shared"] == 0


# ── de eigenschappen die een naïeve implementatie mist ──────────────────

def test_matching_is_globally_optimal_not_first_come_first_served():
    """Beste paren eerst, niet in volgorde van A.

    A1 overlapt B1 (0,25) en B2 (0,43); A2 overlapt alleen B2, maar met 0,90.
    Een matcher die greedy over A loopt geeft A1 zijn eigen beste keuze (B2) en
    laat A2 met lege handen achter: 1 paar. Beste-eerst vindt er 2.

    Deze test faalt op de vorm die `corroborate_apnea_events` gebruikt -- daar
    is de asymmetrie een keuze, want de AASM-sensor bepaalt de grenzen. Hier is
    er geen bevoorrechte lijst.
    """
    a = [ev(96, 10), ev(100, 9)]
    b = [ev(90, 10), ev(100, 10)]
    r = compare_event_sets(a, b)
    assert r["n_shared"] == 2, "beste-eerst hoort beide paren te vinden"
    assert r["n_only_a"] == 0 and r["n_only_b"] == 0


def test_result_is_symmetric_in_its_arguments():
    """A tegen B moet hetzelfde zeggen als B tegen A.

    Anders hangt een rapportcijfer af van de kolomvolgorde, en dat is geen
    cijfer maar een artefact.
    """
    a = [ev(96, 10), ev(100, 9), ev(400, 20, "hypopnea")]
    b = [ev(90, 10), ev(100, 10)]
    ab = compare_event_sets(a, b)
    ba = compare_event_sets(b, a)
    assert ab["n_shared"] == ba["n_shared"]
    assert ab["jaccard"] == ba["jaccard"]
    assert ab["n_only_a"] == ba["n_only_b"]
    assert ab["n_only_b"] == ba["n_only_a"]


def test_matched_events_with_different_labels_are_counted():
    """Hetzelfde event, ander label -- de categorie die een index verbergt."""
    a = [ev(10, 15, "obstructive"), ev(100, 20, "hypopnea")]
    b = [ev(10, 15, "central"), ev(100, 20, "hypopnea")]
    r = compare_event_sets(a, b)
    assert r["n_shared"] == 2
    assert r["n_type_changed"] == 1
    assert r["type_changes"] == {"obstructive -> central": 1}


# ── de twee uncertain-klassen ───────────────────────────────────────────

def test_bare_uncertain_is_excluded_but_hypopnea_uncertain_is_not():
    """`uncertain` valt buiten ahi_total; `hypopnea_uncertain` telt gewoon mee.

    Ze door elkaar halen laat profielen verschillen die alleen in hun
    uncertain-boekhouding uiteenlopen.
    """
    a = [ev(10, 15, "uncertain"), ev(100, 20, "hypopnea_uncertain"),
         ev(200, 12, "obstructive")]
    r = compare_event_sets(a, list(a))

    assert r["n_a"] == 3 and r["n_shared"] == 3
    assert r["n_bare_uncertain"] == {"a": 1, "b": 1}

    x = r["excl_bare_uncertain"]
    assert x["n_a"] == 2, "alleen het kale type verdwijnt"
    assert x["n_shared"] == 2
    assert x["jaccard"] == 1.0


def test_the_two_variants_actually_differ_when_uncertain_is_asymmetric():
    """Guard op de guard.

    Als beide varianten altijd hetzelfde getal geven, is de splitsing
    decoratie. Hier ziet A een kaal `uncertain` waar B niets ziet: inclusief
    telt dat als verschil, exclusief niet.
    """
    a = [ev(10, 15, "obstructive"), ev(500, 12, "uncertain")]
    b = [ev(10, 15, "obstructive")]
    r = compare_event_sets(a, b)
    assert r["n_only_a"] == 1 and r["jaccard"] == 0.5
    x = r["excl_bare_uncertain"]
    assert x["n_only_a"] == 0 and x["jaccard"] == 1.0
    assert r["jaccard"] != x["jaccard"]


# ── categorie-indeling ──────────────────────────────────────────────────

@pytest.mark.parametrize("typ,cat", [
    ("obstructive", "apnea"), ("central", "apnea"), ("mixed", "apnea"),
    ("uncertain", "apnea"),
    ("hypopnea", "hypopnea"), ("hypopnea_central", "hypopnea"),
    ("hypopnea_uncertain", "hypopnea"), ("hypopnea_obstr", "hypopnea"),
])
def test_event_category_follows_the_substring_rule_the_index_uses(typ, cat):
    assert event_category({"type": typ}) == cat


def test_per_category_counts_add_up_to_the_totals():
    a = [ev(10, 15), ev(100, 20, "hypopnea"), ev(300, 12, "central")]
    b = [ev(10, 15), ev(700, 20, "hypopnea")]
    r = compare_event_sets(a, b)
    pc = r["per_category"]
    assert sum(v["shared"] for v in pc.values()) == r["n_shared"]
    assert sum(v["only_a"] for v in pc.values()) == r["n_only_a"]
    assert sum(v["only_b"] for v in pc.values()) == r["n_only_b"]


def test_empty_input_does_not_pretend_to_agree():
    r = compare_event_sets([], [])
    assert r["n_shared"] == 0
    assert r["jaccard"] is None, "0/0 is geen 1,0 en geen 0,0"
    r2 = compare_event_sets([ev(10, 15)], [])
    assert r2["jaccard"] == 0.0 and r2["n_only_a"] == 1


def test_threshold_is_the_harness_threshold():
    assert IOU_THRESH == 0.20
