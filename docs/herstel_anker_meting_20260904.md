# Herstel-anker: afleiding geslaagd, replicatie loopt

*2026-09-04 (avond). Vlag `local_baseline_recovery_anchor`, gemeten op de 15
vooraf bekende zwaarste hoge-AHI-nachten + 10 laag-AHI-controles, twee armen
gepaard, regel vooraf vastgelegd.*

| groep | bias | F1 | oordeel |
|---|---|---|---|
| HOOG (15) | −29,6 → −23,6, p=0,0001 | 0,175 → 0,212, p=0,0001 | **beide criteria gehaald** |
| LAAG (10) | +4,0 → +6,2, p=0,23 | vlak | niet aantoonbaar slechter |

De F1 stijgt mee waar de regel alleen "niet zakken" eiste — telling en
localisatie dezelfde kant op.

Kanttekeningen: (1) dit dicht ~6/u van −29,6 — de validator droeg 63 % van de
terughaalbare 29 %; de 55 % nooit-kandidaten zijn onaangeroerd en zijn de
volgende vraag; (2) LAAG beweegt nominaal omhoog bij n=10 — geen vrijbrief.

**Replicatie (gestart):** 100 verse MESA-opnames over het hele spectrum, twee
armen, per menselijk-AHI-tertiel; slaagt als het hoogste tertiel de
biasverbetering herhaalt (p<0,05) en de onderste twee tertielen niet
aantoonbaar verslechteren. Default blijft uit tot die er ligt.

---

# Replicatie (110 onaangeraakte opnames, per tertiel): mislukt op de vastgelegde maat

| tertiel | bias | \|bias\| | F1 |
|---|---|---|---|
| laag (36) | −1,9 → +0,7 | 4,6 → 5,1 (p=0,38) | 0,231 → 0,263 (p=0,09) |
| midden (36) | −5,3 → +2,9 | 9,2 → 9,1 (p=0,23) | 0,291 → **0,345** (p=0,006) |
| HOOG (38) | −13,2 → **−1,1** | 15,0 → 13,6 (**p=0,41**) | 0,401 → **0,433** (p=0,0001) |

Ontleed (HOOG): de verschuiving is uniform **+8,6/u (p=7e-12)** — de
systematische ondertelling verdwijnt — maar de spreiding groeit (sd 18,0→19,6),
overshoot-nachten verdubbelen (6→12) en binnen ±5/u gaat maar 21→24 %.
RMS-bias 21,0→19,6. De F1 repliceert in alle tertielen.

**Oordeel: de vooraf vastgelegde |bias|-toets is niet gehaald; default blijft
uit.** Wat wél staat: betere localisatie overal, en een gecentreerde
verdeling in plaats van een systematisch te lage. De vijfde ruil-in-plaats-
van-dominantie van deze week.

## De voor de hand liggende verfijning — en waarom hij niet vannacht komt

Het anker corrigeert nu élk event; de schade zit op nachten waar het
pre-venster niet event-dicht was. De conditionele vorm — anker alleen wanneer
kale en verankerde referentie sterk uiteenlopen (dat is per event meetbaar
zonder labels) — is de logische volgende stap, maar de drempel daarvoor
afleiden op deze set en dan pas repliceren is een nieuwe cyclus. Geen
haastwerk om 03:00; dit is een beslispunt voor de ochtend, mét deze tabel.
