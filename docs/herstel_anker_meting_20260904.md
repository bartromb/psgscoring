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
