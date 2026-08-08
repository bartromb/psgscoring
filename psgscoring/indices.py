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
