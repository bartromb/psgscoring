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

3. **De koppelregel is een parameter, en hij staat in de uitvoer.** De
   arousalliteratuur telt *elke overlap* als treffer (Pitkänen e.a., Sleep
   2026; Ehrlich e.a., Sci Rep 2024); dit project koppelt op IoU 0,20, en dat
   is strenger. "0,745 tegen 0,61" is dus geen vergelijking zolang de regel
   niet dezelfde is. Daarom kent ``compare_event_sets`` ``matcher="iou"``
   (default, byte-identiek aan vroeger) en ``matcher="any"``, en geeft het
   naast de eventtelling ook de seconde-per-seconde Cohen's κ (``kappa_1s``)
   en de mediane verschuiving van begin en einde over de gekoppelde paren —
   de drie maten waarmee de literatuur arousaldetectoren naast scoorders zet
   (docs/literatuur_update_20261025.md §1).
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import median
from typing import Any

from .postprocess import _iou

__all__ = ["IOU_THRESH", "MATCHERS", "compare_event_sets", "event_category"]

#: Dezelfde drempel als de validatieharness en ``corroborate_apnea_events``.
IOU_THRESH = 0.20

#: Koppelregels. ``"iou"``: IoU ≥ ``iou_thresh`` (projectstandaard).
#: ``"any"``: elke overlap > 0 s telt (de regel van de arousalliteratuur).
MATCHERS = ("iou", "any")

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


def _overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _match(a: list[dict], b: list[dict], iou_thresh: float, matcher: str = "iou"
           ) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Symmetrische greedy toewijzing: beste paren eerst.

    ``matcher="iou"``: kandidaat als IoU ≥ ``iou_thresh``, hoogste IoU eerst.
    ``matcher="any"``: kandidaat als de overlap > 0 s, grootste overlap (in
    seconden) eerst, daarna hoogste IoU. De derde waarde van elk paar is in
    beide gevallen de IoU, zodat ``median_iou`` onder beide regels hetzelfde
    betekent.

    Retourneert ``(paren, ongepaard_a, ongepaard_b)``. Ties worden op index
    gebroken zodat dezelfde invoer altijd dezelfde uitvoer geeft -- een
    rapportcijfer dat per run wisselt is geen cijfer.
    """
    if matcher not in MATCHERS:
        raise ValueError(f"matcher {matcher!r}: kies uit {MATCHERS}")
    cand: list[tuple[float, float, int, int]] = []
    for i, ea in enumerate(a):
        a0, a1 = _span(ea)
        for j, eb in enumerate(b):
            b0, b1 = _span(eb)
            v = _iou(a0, a1, b0, b1)
            if matcher == "iou":
                if v >= iou_thresh:
                    cand.append((v, v, i, j))
            else:
                ov = _overlap_s(a0, a1, b0, b1)
                if ov > 0.0:
                    cand.append((ov, v, i, j))
    # Hoogste score eerst; bij gelijkspel de laagste indices, voor determinisme.
    cand.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))

    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for _s, v, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, v))
    return (pairs,
            {i for i in range(len(a)) if i not in used_a},
            {j for j in range(len(b)) if j not in used_b})


def _mask_1s(events: list[dict], n: int) -> list[bool]:
    """Binair masker per seconde: bin ``t`` is waar zodra een event het
    interval ``[t, t+1)`` raakt (overlap > 0). Een event van nul seconden
    raakt niets, net zoals het onder beide koppelregels niets paart."""
    m = [False] * n
    for e in events:
        e0, e1 = _span(e)
        if e1 <= e0:
            continue
        t0 = max(0, int(math.floor(e0)))
        t1 = min(n, int(math.ceil(e1)))
        for t in range(t0, t1):
            m[t] = True
    return m


def _kappa_1s(a: list[dict], b: list[dict], duration_s: float | None) -> float | None:
    """Cohen's κ tussen de 1 s-maskers van A en B over ``[0, duration_s)``.

    Zonder duur: tot het einde van het laatste event van beide lijsten. Geen
    events, geen duur of een verwachte overeenstemming van 1 (beide maskers
    constant) geeft ``None`` -- niet 1,0, want dan "stemt" lege lucht overeen.
    """
    if duration_s is None:
        ends = [_span(e)[1] for e in a] + [_span(e)[1] for e in b]
        duration_s = max(ends) if ends else 0.0
    n = int(math.ceil(float(duration_s)))
    if n <= 0:
        return None
    ma, mb = _mask_1s(a, n), _mask_1s(b, n)
    n11 = sum(1 for x, y in zip(ma, mb) if x and y)
    n10 = sum(1 for x, y in zip(ma, mb) if x and not y)
    n01 = sum(1 for x, y in zip(ma, mb) if y and not x)
    n00 = n - n11 - n10 - n01
    po = (n11 + n00) / n
    pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (n * n)
    if 1.0 - pe <= 0.0:
        return None
    return (po - pe) / (1.0 - pe)


def _tally(a: list[dict], b: list[dict], iou_thresh: float,
           matcher: str = "iou", duration_s: float | None = None) -> dict[str, Any]:
    pairs, only_a, only_b = _match(a, b, iou_thresh, matcher)

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
    # Verschuiving b − a per gekoppeld paar: positief = B begint/eindigt later.
    d_on = [_span(b[j])[0] - _span(a[i])[0] for i, j, _v in pairs]
    d_off = [_span(b[j])[1] - _span(a[i])[1] for i, j, _v in pairs]
    dur_a = [_span(e)[1] - _span(e)[0] for e in a]
    dur_b = [_span(e)[1] - _span(e)[0] for e in b]
    kappa = _kappa_1s(a, b, duration_s)
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
        # Seconde-per-seconde Cohen's κ (Pitkänen e.a. 2026 rapporteren deze
        # naast de event-F1); None zonder events of zonder duur.
        "kappa_1s": round(kappa, 4) if kappa is not None else None,
        # Mediane verschuiving b − a over de gekoppelde paren, in seconden.
        "onset_shift_median_s": round(median(d_on), 4) if d_on else None,
        "offset_shift_median_s": round(median(d_off), 4) if d_off else None,
        "median_duration_a": round(median(dur_a), 4) if dur_a else None,
        "median_duration_b": round(median(dur_b), 4) if dur_b else None,
    }


def compare_event_sets(events_a: Iterable[dict],
                       events_b: Iterable[dict],
                       *,
                       iou_thresh: float = IOU_THRESH,
                       matcher: str = "iou",
                       duration_s: float | None = None,
                       label_a: str = "a",
                       label_b: str = "b") -> dict[str, Any]:
    """Vergelijk twee eventlijsten van dezelfde opname.

    ``events_a`` / ``events_b``  lijsten met ``onset_s``, ``duration_s``,
                                 ``type`` -- zoals ``respiratory["events"]``
                                 ze teruggeeft.
    ``matcher``      ``"iou"`` (IoU ≥ ``iou_thresh``, projectstandaard) of
                     ``"any"`` (elke overlap > 0 s, de regel van de
                     arousalliteratuur). Staat als ``matcher`` in de uitvoer:
                     een F1 zonder zijn koppelregel is geen F1.
    ``duration_s``   opnameduur voor ``kappa_1s``; zonder opgave tot het einde
                     van het laatste event van beide lijsten.

    Retourneert een dict met de telling **inclusief** kale ``uncertain``, en
    onder ``excl_bare_uncertain`` dezelfde telling met die events uit beide
    lijsten verwijderd. Dat tweede getal is het getal dat bij ``ahi_total``
    hoort; het eerste beschrijft de volledige eventlijst. Welke van de twee een
    rapport toont is een keuze, en die keuze hoort zichtbaar te zijn.

    Naast de telling: ``kappa_1s`` (seconde-per-seconde Cohen's κ),
    ``onset_shift_median_s`` / ``offset_shift_median_s`` (mediaan van b − a
    over de gekoppelde paren) en ``median_duration_a`` / ``_b``.
    """
    a = list(events_a or [])
    b = list(events_b or [])

    out = _tally(a, b, iou_thresh, matcher, duration_s)
    out["labels"] = {"a": label_a, "b": label_b}
    out["iou_thresh"] = iou_thresh
    out["matcher"] = matcher

    a_x = [e for e in a if str(e.get("type")) != BARE_UNCERTAIN]
    b_x = [e for e in b if str(e.get("type")) != BARE_UNCERTAIN]
    out["excl_bare_uncertain"] = _tally(a_x, b_x, iou_thresh, matcher, duration_s)
    out["n_bare_uncertain"] = {"a": len(a) - len(a_x), "b": len(b) - len(b_x)}
    return out
