"""Paarsgewijze overeenkomst tussen twee scoringen van dezelfde opname.

WAAROM DIT BESTAAT

Een profielvergelijking die alleen indices toont, beantwoordt de vraag niet die
ze oproept. Op mesa-sleep-0001 leverden ``aasm_v3_rec``, ``aasm_v3_pressure`` en
``aasm_v2_rec`` alle drie AHI 19,6 uit alle drie 126 events. Zijn dat dezelfde
126 events, of drie verschillende verzamelingen die toevallig even groot zijn?
Uit een indextabel is dat principieel niet af te lezen, en het is precies het
verschil tussen "deze regels doen hetzelfde" en "deze regels doen iets anders
en komen toevallig uit op hetzelfde getal".

Deze module beantwoordt die vraag met dezelfde IoU-logica die de
validatieharness gebruikt.

TWEE DINGEN DIE ANDERS ZIJN DAN JE ZOU VERWACHTEN

1. **De matching is symmetrisch.** ``corroborate_apnea_events`` loopt greedy
   over de thermistorlijst omdat daar één sensor de AASM-grenzen bepaalt; die
   asymmetrie is daar een keuze. Hier is er geen bevoorrechte lijst: A tegen B
   moet hetzelfde antwoord geven als B tegen A, anders hangt een rapportcijfer
   af van de kolomvolgorde. Daarom worden alle paren boven de drempel op IoU
   gesorteerd en van hoog naar laag toegewezen.

2. **Kale ``uncertain`` telt niet mee in de AHI, ``hypopnea_uncertain`` wel.**
   Dat is geen detail van deze module maar van de index: ``ahi_total`` sluit
   het kale type uit, terwijl ``hypopnea_uncertain`` via substring-matching
   gewoon meetelt. De eventlijst bevat dus events die de index niet telt. Een
   overeenkomsttelling die dat negeert toont spookverschillen tussen profielen
   die alleen in hun uncertain-boekhouding uiteenlopen. Daarom rapporteert
   ``compare_event_sets`` beide varianten en laat ze de lezer kiezen in plaats
   van er stilzwijgend één op te leggen.
"""
from __future__ import annotations

from collections.abc import Iterable
from statistics import median
from typing import Any

from .postprocess import _iou

__all__ = ["IOU_THRESH", "compare_event_sets", "event_category"]

#: Dezelfde drempel als de validatieharness en ``corroborate_apnea_events``.
IOU_THRESH = 0.20

#: Het kale type dat BUITEN ``ahi_total`` valt. ``hypopnea_uncertain`` is een
#: ander type en telt wél mee; die twee door elkaar halen is de valkuil.
BARE_UNCERTAIN = "uncertain"


def _span(e: dict) -> tuple[float, float]:
    a = float(e.get("onset_s") or 0.0)
    return a, a + float(e.get("duration_s") or 0.0)


def event_category(e: dict) -> str:
    """``"hypopnea"`` of ``"apnea"``.

    Er is geen ``subtype``-veld: ``type`` draagt het subtype zelf, als
    ``hypopnea_central``, ``hypopnea_uncertain``, ``obstructive``, ``central``,
    ``mixed`` of ``uncertain``. De substring-regel hieronder is dezelfde die de
    index gebruikt om hypopneus te herkennen.
    """
    return "hypopnea" if str(e.get("type", "")).startswith("hypopnea") else "apnea"


def _match(a: list[dict], b: list[dict], iou_thresh: float
           ) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Symmetrische greedy toewijzing: beste paren eerst.

    Retourneert ``(paren, ongepaard_a, ongepaard_b)``. Ties worden op index
    gebroken zodat dezelfde invoer altijd dezelfde uitvoer geeft -- een
    rapportcijfer dat per run wisselt is geen cijfer.
    """
    cand: list[tuple[float, int, int]] = []
    for i, ea in enumerate(a):
        a0, a1 = _span(ea)
        for j, eb in enumerate(b):
            b0, b1 = _span(eb)
            v = _iou(a0, a1, b0, b1)
            if v >= iou_thresh:
                cand.append((v, i, j))
    # Hoogste IoU eerst; bij gelijkspel de laagste indices, voor determinisme.
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for v, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, v))
    return (pairs,
            {i for i in range(len(a)) if i not in used_a},
            {j for j in range(len(b)) if j not in used_b})


def _tally(a: list[dict], b: list[dict], iou_thresh: float) -> dict[str, Any]:
    pairs, only_a, only_b = _match(a, b, iou_thresh)

    per_cat: dict[str, dict[str, int]] = {
        c: {"shared": 0, "only_a": 0, "only_b": 0} for c in ("apnea", "hypopnea")
    }
    for i, j, _v in pairs:
        # Een gepaard event kan in beide lijsten een andere categorie hebben.
        # Tel het bij die van A; het verschil zelf staat in `type_changes`.
        per_cat[event_category(a[i])]["shared"] += 1
    for i in only_a:
        per_cat[event_category(a[i])]["only_a"] += 1
    for j in only_b:
        per_cat[event_category(b[j])]["only_b"] += 1

    type_changes: dict[str, int] = {}
    for i, j, _v in pairs:
        ta, tb = str(a[i].get("type")), str(b[j].get("type"))
        if ta != tb:
            type_changes[f"{ta} -> {tb}"] = type_changes.get(f"{ta} -> {tb}", 0) + 1

    n_shared = len(pairs)
    union = n_shared + len(only_a) + len(only_b)
    ious = [v for _i, _j, v in pairs]
    return {
        "n_a": len(a),
        "n_b": len(b),
        "n_shared": n_shared,
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        # Jaccard over de vereniging: 1,0 betekent exact dezelfde events, 0,0
        # geen enkele overlap boven de drempel.
        "jaccard": round(n_shared / union, 4) if union else None,
        "median_iou": round(median(ious), 4) if ious else None,
        "min_iou": round(min(ious), 4) if ious else None,
        # Gepaard maar anders geclassificeerd: hetzelfde event, ander label.
        # Dit is de categorie die een indextabel volledig onzichtbaar maakt.
        "n_type_changed": sum(type_changes.values()),
        "type_changes": dict(sorted(type_changes.items(),
                                    key=lambda kv: -kv[1])),
        "per_category": per_cat,
    }


def compare_event_sets(events_a: Iterable[dict],
                       events_b: Iterable[dict],
                       *,
                       iou_thresh: float = IOU_THRESH,
                       label_a: str = "a",
                       label_b: str = "b") -> dict[str, Any]:
    """Vergelijk twee eventlijsten van dezelfde opname.

    ``events_a`` / ``events_b``  lijsten met ``onset_s``, ``duration_s``,
                                 ``type`` -- zoals ``respiratory["events"]``
                                 ze teruggeeft.

    Retourneert een dict met de telling **inclusief** kale ``uncertain``, en
    onder ``excl_bare_uncertain`` dezelfde telling met die events uit beide
    lijsten verwijderd. Dat tweede getal is het getal dat bij ``ahi_total``
    hoort; het eerste beschrijft de volledige eventlijst. Welke van de twee een
    rapport toont is een keuze, en die keuze hoort zichtbaar te zijn.
    """
    a = list(events_a or [])
    b = list(events_b or [])

    out = _tally(a, b, iou_thresh)
    out["labels"] = {"a": label_a, "b": label_b}
    out["iou_thresh"] = iou_thresh

    a_x = [e for e in a if str(e.get("type")) != BARE_UNCERTAIN]
    b_x = [e for e in b if str(e.get("type")) != BARE_UNCERTAIN]
    out["excl_bare_uncertain"] = _tally(a_x, b_x, iou_thresh)
    out["n_bare_uncertain"] = {"a": len(a) - len(a_x), "b": len(b) - len(b_x)}
    return out
