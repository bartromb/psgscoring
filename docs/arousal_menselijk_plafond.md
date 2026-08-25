# Het menselijk plafond voor arousal-F1: 0,679

**Datum:** 2026-08-25
**Cohort:** PSG-IPA, 5 opnames, 12 scoorders, **330 scoorderparen**
**Maat:** event-level F1, greedy IoU-matching op 0,20 (`psgscoring.agreement._match`)

---

## Waarom dit getal ontbrak, en waarom dat een probleem was

Elke arousal-F1 in dit project is tot nu toe gerapporteerd zonder plafond
ernaast. 0,546 klinkt matig tegen 1,0 en goed tegen 0,3, en zonder te weten
waar mensen zelf uitkomen is niet te zeggen welke van de twee het is.

Scoorder tegen scoorder, alle paren:

| opname | mediaan | bereik |
|---|---:|---|
| SN1 | 0,642 | 0,504–0,761 |
| SN2 | **0,492** | 0,192–0,742 |
| SN3 | 0,692 | 0,533–0,794 |
| SN4 | 0,767 | 0,562–0,869 |
| SN5 | 0,766 | 0,454–0,866 |
| **alle 330 paren** | **0,679** | |

**Perfecte overeenstemming bestaat hier niet.** Twee menselijke scoorders halen
mediaan 0,679, en op SN2 zijn ze het onderling zó oneens (0,492, met paren tot
0,192) dat een algoritme daar nauwelijks kan falen.

## Waar wij staan

Met de vlaggen zoals ze in productie draaien (werkpunt 0,80, event-locked uit):

| | F1 |
|---|---:|
| menselijk plafond | 0,679 |
| **wij, PSG-IPA** | **0,546** |
| gat | 0,133 |

**80 % van het menselijke plafond.** Per opname is het beeld ongelijk: op SN2
halen wij 0,51 tegen een plafond van 0,49 — binnen de menselijke onenigheid.
Op SN4 halen wij 0,40 tegen een plafond van 0,767, en dát is een echte
achterstand.

## Het traject

Zelfde cohort, zelfde maat:

| | F1 |
|---|---:|
| 21-08-2026, regelgebaseerd | 0,182 |
| na de classifier (v0.25) | 0,505 |
| **nu (0.27.5)** | **0,546** |
| menselijk plafond | 0,679 |

Van het oorspronkelijke gat van 0,497 is 0,364 gedicht; er rest 0,133.

## Waarom dit de event-locked vlag verklaart

Die vlag levert +0,018 F1 op PSG-IPA en +0,011 op MESA (zie
`arousal_event_locked_bevinding.md`). Dat is 13 % respectievelijk 8 % van het
resterende gat — en dat gat bestaat voor een deel uit onenigheid die tussen
mensen onderling ook bestaat.

**Let op de valstrik die dit blootlegt:** de koppelingsfractie steeg met 7,2
procentpunt terwijl F1 met 0,018 bewoog. Een maat die alleen vraagt "volgt er
een arousal op een respiratoir event" beloont arousals die bij het event-einde
liggen, ook wanneer geen scoorder daar iets zag. Elke toekomstige vlag die
arousals rond events toevoegt hoort op F1 beoordeeld te worden, niet op
koppeling.

## Beperkingen

- **n = 5 opnames.** De 330 paren komen uit dezelfde vijf nachten.
- Het plafond komt uit de `EEG_arousals`-subtree (12 unieke scoringen). De
  `Resp_events`-subtree draagt er **3 unieke** — daar is de arousal-annotatie
  gedeeld. De twee subtrees zijn bovendien andere exports met andere duur
  (SN3: 6,57 u tegen 8,13 u), dus getallen uit het ene horen niet naast
  spreidingen uit het andere.
- Onze eigen 0,546 is gemeten op `Resp_events`; het plafond op
  `EEG_arousals`. Beide zijn arousal-F1 op dezelfde vijf nachten, maar niet op
  exact hetzelfde signaal. Het getal is een richtpunt, geen precisiemeting.
