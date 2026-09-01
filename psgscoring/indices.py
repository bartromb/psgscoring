"""Eén plek voor "events per uur", en voor de vraag wanneer dat niet kan.

Aanleiding is een echt rapport dat **REI 81000,0/u** meldde bij 81 hypopnees, en
daarop "Ernstig SAS — therapie CPAP". De oorzaak was overal dezelfde regel:

    total_sleep_h = max(total_sleep_s / 3600, 0.001)

bedoeld tegen deling door nul. Maar 0,001 uur is 3,6 seconden, dus de index werd
het AANTAL MAAL DUIZEND — en dat getal ging ongehinderd door de
ernstclassificatie. Een noemer die er niet is, is geen kleine noemer.

Die regel stond op **twaalf** plaatsen: arousal.py (7×), pipeline.py (3×),
plm.py, spo2.py — naast respiratory.py, waar hij het eerst opviel. De eerste
reparatie raakte alleen respiratory.py, waardoor de AHI klopte terwijl de
arousal-index, PLM-index, ODI, RERA-index en RDI nog steeds het aantal maal
duizend waren. Vandaar deze module: één regel, één plaats.

**Nul is geen antwoord.** "AHI 0,0" naast 81 gescoorde events leest als *geen
events* — geruststellend fout, en dat is klinisch erger dan zichtbaar fout. Geen
REM-slaap geeft geen REM-index van 0; er is domweg geen REM om events per uur
van uit te drukken. De aanroeper hoort dat verschil te tonen.
"""

from __future__ import annotations

__all__ = ["hours_of", "per_hour"]


def per_hour(n: float | None, hours: float | None,
             ndigits: int = 1) -> float | None:
    """``n`` per uur, of ``None`` als er geen bruikbare noemer is.

    Geen ondergrens op ``hours``: is die nul, dan bestaat de index niet.
    """
    if n is None or hours is None:
        return None
    try:
        h = float(hours)
    except (TypeError, ValueError):
        return None
    if h <= 0:
        return None
    return round(float(n) / h, ndigits)


def hours_of(hypno: list, predicate, epoch_len_s: float,
             artifact_set: set | None = None) -> float:
    """Uren slaap die aan ``predicate`` voldoen, artefact-epochs uitgesloten.

    Retourneert een echte 0.0 wanneer er niets overblijft — het is aan
    ``per_hour`` om daar ``None`` van te maken, zodat de reden zichtbaar blijft
    op de plek waar hij thuishoort.
    """
    art = artifact_set or set()
    n = sum(1 for i, s in enumerate(hypno or [])
            if predicate(s) and i not in art)
    return n * epoch_len_s / 3600.0


# ═══════════════════════════════════════════════════════════════════════════
# Hoe zeker is een index bij deze ziektelast?
# ═══════════════════════════════════════════════════════════════════════════

#: Gemeten scoorder-tegen-scoorder overeenstemming op PSG-IPA, per opname, over
#: alle 66 paren (12 scoorders). Zie docs/nacht_20260901_bevindingen.md.
#:
#:     opname   events per scoorder   F1 mens-mens
#:     SN3            273-339             0,948
#:     SN1             27-38              0,826
#:     SN5             25-101             0,556
#:     SN2              8-33              0,549
#:     SN4              1-38              0,553
#:
#: De ankerpunten hieronder zijn de mediane eventtelling per opname tegen de
#: gemeten F1. Er wordt lineair tussen geïnterpoleerd -- met vijf opnames is
#: elke rijkere vorm een aanname die de data niet draagt.
_AGREEMENT_ANCHORS = ((20.0, 0.55), (33.0, 0.55), (70.0, 0.56),
                      (34.0, 0.83), (300.0, 0.95))

#: Onder deze eventtelling zijn menselijke scoorders het onderling zo oneens
#: dat een enkel getal de spreiding niet draagt. Volgt uit de meting: de drie
#: opnames onder ~100 events liggen alle drie rond F1 0,55, de twee erboven op
#: 0,83 en 0,95.
AGREEMENT_LOW_BURDEN_EVENTS = 100


def expected_scorer_agreement(n_events: int | None) -> dict:
    """Hoe goed zijn MENSEN het bij deze ziektelast onderling eens?

    Dit is geen uitspraak over onze detector. Het is een eigenschap van de
    OPNAME: bij weinig events lopen menselijke scoringen zo uiteen dat er geen
    scherpe referentie bestaat om tegen te vergelijken -- niet voor ons, en niet
    voor een lezer die twee rapporten naast elkaar legt.

    Op de lichtste PSG-IPA-opname scoorde de ene expert één event en de andere
    achtendertig, met kappa 0,000 op het subtype.

    Waarom dit in een rapport hoort: een AHI van 8 en een AHI van 40 dragen niet
    dezelfde zekerheid. Wie ze als gelijkwaardig behandelt, doet precies wat de
    meting verbiedt.

    Returns
    -------
    dict met ``f1_human`` (verwachte overeenstemming tussen twee scoorders),
    ``band`` (``"hoog"``/``"laag"``/``"onbekend"``), ``what`` (wat het getal
    beschrijft), ``source`` en ``n_scorer_pairs``.
    """
    basis = {
        "what": ("verwachte overeenstemming tussen twee menselijke scoorders "
                 "bij deze ziektelast; beschrijft de OPNAME, niet de detector"),
        "source": ("PSG-IPA, 5 opnames, 12 scoorders, event-F1 met IoU 0,20 "
                   "(gemeten 2026-09-01)"),
        "n_scorer_pairs": 330,
    }
    if not n_events or n_events <= 0:
        return {**basis, "f1_human": None, "band": "onbekend"}

    punten = sorted(_AGREEMENT_ANCHORS)
    xs = [p[0] for p in punten]
    ys = [p[1] for p in punten]
    # Monotone interpolatie: de gemeten punten liggen niet monotoon (SN1 met 34
    # events haalt 0,83 waar SN5 met 70 events 0,56 haalt), en een kromme die
    # daar doorheen slingert zou suggereren dat 50 events slechter is dan 34.
    # Cumulatief maximum houdt de uitspraak conservatief: nooit een hogere
    # verwachting dan bij een lagere ziektelast al gemeten is.
    ys_mono = []
    hoogste = 0.0
    for y in ys:
        hoogste = max(hoogste, y)
        ys_mono.append(hoogste)

    if n_events <= xs[0]:
        f1 = ys_mono[0]
    elif n_events >= xs[-1]:
        f1 = ys_mono[-1]
    else:
        f1 = ys_mono[-1]
        for (x0, y0), (x1, y1) in zip(zip(xs, ys_mono), list(zip(xs, ys_mono))[1:]):
            if x0 <= n_events <= x1:
                t = (n_events - x0) / (x1 - x0) if x1 > x0 else 0.0
                f1 = y0 + t * (y1 - y0)
                break
    return {**basis, "f1_human": round(float(f1), 3),
            "band": "laag" if n_events < AGREEMENT_LOW_BURDEN_EVENTS else "hoog"}
