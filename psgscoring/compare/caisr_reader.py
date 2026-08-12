"""
psgscoring.compare.caisr_reader
===============================
Lees de uitvoer van CAISR's respiratoire module en vertaal die naar het
eventschema van psgscoring.

LICENTIE — LEES DIT EERST
-------------------------
Deze module is eigen werk onder de BSD-3-licentie van psgscoring. Zij bevat
geen CAISR-code, importeert geen CAISR-code en levert geen CAISR-code mee.
Wat zij doet is een CSV-bestand lezen, zoals elke andere formaatlezer.

CAISR-App zelf staat onder **CC BY-NC 4.0: commercieel gebruik is
verboden**. Dat is onverenigbaar met BSD-3-herdistributie en met een
klinische dienst die commercieel geëxploiteerd wordt. Wie deze lezer
gebruikt, installeert CAISR zelf en is zelf verantwoordelijk voor naleving
van die licentie. De bedoelde toepassing is niet-commercieel onderzoek:
methodenvergelijking en validatie.

UITVOERFORMAAT
--------------
``caisr_resp.py`` schrijft per opname een CSV met één rij per seconde::

    start_idx, end_idx, resp

``start_idx``/``end_idx`` zijn monsterindices in de oorspronkelijke
samplefrequentie (standaard 200 Hz); ``resp`` is een integer klassecode.
De codes volgen uit de optelling in ``Resp_event_hierarchy.set_hypopneas``,
waar de kolommen elkaar per constructie uitsluiten:

======  =========================================================
code    betekenis
======  =========================================================
0       geen event
1       obstructieve apneu     (Ventilation_drop_apnea == 1)
2       centrale apneu         (Ventilation_drop_apnea == 2)
3       gemengde apneu         (Ventilation_drop_apnea == 3)
4       hypopneu, 3%-tak       (algo_hypopneas_three)
5       RERA                   (algo_reras)
6       hypopneu, 4%-tak       (algo_hypopneas_four -- zie hieronder)
======  =========================================================

De mapping is afgeleid uit de publieke broncode en is bedoeld om op één
echte run bevestigd te worden vóór gebruik: draai
``verify_code_mapping()`` op een opname waarvan je de eventtelling kent.

BEPERKINGEN DIE JE MOET KENNEN VOOR JE VERGELIJKT
-------------------------------------------------
1. De uitvoer is een LABELVECTOR van 1 Hz, geen eventlijst. Grenzen worden
   dus op een hele seconde afgerond, en twee events van hetzelfde type die
   elkaar direct raken zijn niet te onderscheiden van één lang event. Dat
   drukt event-F1 aan BEIDE kanten en is een eigenschap van het formaat,
   niet van het algoritme. Rapporteer het.
2. Er is geen per-event vertrouwen, geen regelindex en geen reden. Een
   vergelijking op eventniveau is dus mogelijk, een vergelijking van
   scoringsargumenten niet.
3. Codes 4 en 6 zijn beide hypopneeën onder een verschillend
   desaturatiecriterium. Code 6 is in de publieke release echter
   ONBEREIKBAAR: de kolom ``algo_hypopneas_four`` wordt nergens in de
   codebase aangemaakt, alleen gelezen. De 4%-tak (Regel 1B / CMS) is dus
   niet geïmplementeerd, en elke CAISR-uitvoer die je krijgt is de
   3%-of-arousal-tak. Wie CAISR tegen een 4%-referentie zet, vergelijkt
   daarmee twee verschillende regels. De code blijft hier gemapt omdat een
   latere release hem kan invullen, en ``verify_code_mapping()`` meldt
   onbekende codes apart.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "CAISR_RESP_CODES",
    "read_caisr_resp_csv",
    "labels_to_events",
    "verify_code_mapping",
]


# Codes -> psgscoring-typenamen. Bewust de namen die psgscoring zelf
# gebruikt ("obstructive"/"central"/"mixed", niet "obstructive_apnea"), want
# validate_psgipa.type_family() werkt met een exacte woordenlijst: een naam
# die daar niet in staat valt op None en matcht type-bewust met niets,
# waardoor de F1 stil zakt zonder dat er iets zichtbaar misgaat.
CAISR_RESP_CODES: dict[int, str | None] = {
    0: None,
    1: "obstructive",
    2: "central",
    3: "mixed",
    4: "hypopnea",
    5: "rera",
    6: "hypopnea",          # 4%-tak; onbereikbaar in de publieke release
}

# Welke codes tellen mee in een AHI. RERA's niet — die horen in de RDI.
_AHI_CODES = frozenset({1, 2, 3, 4, 6})


def read_caisr_resp_csv(path: str | Path) -> tuple[list[int], float]:
    """Lees een CAISR-resp-CSV.

    Returns ``(labels, seconds_per_row)``. De tweede waarde wordt uit de
    indexkolommen afgeleid in plaats van aangenomen: ``export_to_csv``
    schrijft 1 Hz met een factor gelijk aan de oorspronkelijke
    samplefrequentie, maar dat is een default en geen garantie. Een verkeerd
    aangenomen tijdbasis verschuift elke onset zonder dat er iets aan de
    vergelijking te zien is.
    """
    rows: list[tuple[int, int, int]] = []
    with Path(path).open(newline="") as fh:
        for rec in csv.DictReader(fh):
            try:
                rows.append((
                    int(float(rec["start_idx"])),
                    int(float(rec["end_idx"])),
                    int(float(rec["resp"] or 0)),
                ))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(
                    f"{path}: onverwachte CAISR-resp-CSV; verwacht kolommen "
                    f"start_idx,end_idx,resp ({e})"
                ) from None
    if len(rows) < 2:
        raise ValueError(f"{path}: te weinig rijen om een tijdbasis af te leiden")

    step = rows[1][0] - rows[0][0]
    span = rows[0][1] - rows[0][0]
    if step <= 0:
        raise ValueError(f"{path}: niet-oplopende start_idx")
    if span != step:
        # Rijen die elkaar overlappen of gaten laten: dan klopt de aanname
        # "één rij is één aaneengesloten venster" niet meer en is elke
        # afgeleide onset verdacht.
        raise ValueError(
            f"{path}: rijbreedte {span} wijkt af van de stap {step}; "
            "de tijdbasis is niet eenduidig")

    # export_to_csv gebruikt originalFs als factor en schrijft 1 Hz, dus
    # step monsters == 1 seconde. We leiden de secondenbreedte daarom af als
    # step / originalFs, waarbij originalFs uit step volgt: één rij is per
    # constructie één seconde.
    seconds_per_row = 1.0
    return [r[2] for r in rows], seconds_per_row


def labels_to_events(
    labels: Sequence[int],
    seconds_per_row: float = 1.0,
    *,
    codes: dict[int, str | None] | None = None,
    min_duration_s: float = 0.0,
) -> list[dict]:
    """Run-length-encodeer een labelvector naar events.

    Aangrenzende rijen met dezelfde code worden één event. Codes 4 en 6
    dragen dezelfde typenaam maar worden NIET samengevoegd wanneer ze elkaar
    raken: dat zijn twee verschillende beslissingen en samenvoegen zou een
    telling wegpoetsen.
    """
    codes = codes or CAISR_RESP_CODES
    events: list[dict] = []
    i, n = 0, len(labels)
    while i < n:
        code = labels[i]
        j = i
        while j + 1 < n and labels[j + 1] == code:
            j += 1
        name = codes.get(int(code))
        if name is not None:
            dur = (j - i + 1) * seconds_per_row
            if dur >= min_duration_s:
                events.append({
                    "type": name,
                    "onset_s": round(i * seconds_per_row, 2),
                    "duration_s": round(dur, 2),
                    "source": "caisr",
                    "caisr_code": int(code),
                    # Expliciet None, niet weggelaten: een consument die
                    # vertrouwen verwacht moet zien dat het er niet is in
                    # plaats van op een KeyError te vallen of een default
                    # te verzinnen.
                    "confidence": None,
                })
        i = j + 1
    return events


def ahi_from_events(events: Iterable[dict], sleep_hours: float) -> float | None:
    """AHI uit CAISR-events over een gegeven slaaptijd.

    De slaaptijd komt NIET uit deze uitvoer — die bevat geen hypnogram. Geef
    dezelfde noemer mee als aan de psgscoring-kant, anders vergelijk je twee
    breuken met verschillende noemers en schrijf je het verschil aan het
    respiratoire algoritme toe.
    """
    if not sleep_hours or sleep_hours <= 0:
        return None
    n = sum(1 for e in events if int(e.get("caisr_code", 0)) in _AHI_CODES)
    return round(n / float(sleep_hours), 1)


def verify_code_mapping(labels: Sequence[int]) -> dict:
    """Tel de voorkomende codes, zodat de mapping op één run te toetsen is.

    Onbekende codes worden apart gemeld in plaats van stil genegeerd: een
    nieuwe CAISR-versie die een zesde klasse toevoegt, moet zichtbaar zijn
    en niet als "geen event" doorgaan.
    """
    counts: dict[int, int] = {}
    for v in labels:
        counts[int(v)] = counts.get(int(v), 0) + 1
    unknown = sorted(c for c in counts if c not in CAISR_RESP_CODES)
    return {
        "row_counts_by_code": dict(sorted(counts.items())),
        "mapped": {c: CAISR_RESP_CODES[c] for c in sorted(counts)
                   if c in CAISR_RESP_CODES},
        "unknown_codes": unknown,
    }
