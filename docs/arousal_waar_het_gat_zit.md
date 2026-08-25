# Waar het arousal-gat zit: de selectie, en op SN3 ook de pool

**Datum:** 2026-08-26 (nacht van 25 op 26 augustus)
**Cohort:** PSG-IPA, 5 opnames, 12 scoorders, event-F1 bij IoU 0,20
**Referentie:** `EEG_arousals` (12 unieke scoorderssets), NOOIT `Resp_events`

---

## Samenvatting

| | mediane F1 |
|---|---:|
| wij, productie (v3 @ 0,80) | **0,514** |
| beste **vaste** drempel (0,70) | 0,519 |
| **orakel op de drempel**, per opname | **0,519** |
| menselijk plafond (330 scoorderparen) | 0,679 |
| **plafond van de KANDIDATENPOOL** | **0,896** |

Twee families zijn hiermee uitgesloten en één plek is aangewezen.

## Uitgesloten: elke drempelstrategie

Een orakel dat de drempel MET kennis van het antwoord per opname kiest, haalt
**+0,005** boven vast-0,80. Adaptief, lastafhankelijk, per opname geleerd —
allemaal begrensd door dit getal.

Op SN3 kiest het orakel **0,80**, de huidige waarde. Verlagen naar 0,60
repareert de TELLING (133 tegen 142 menselijk) en verslechtert de F1
(0,386 tegen 0,419): de extra events liggen op verkeerde plekken.

## Uitgesloten: de twee featurevarianten

| | opzet | uitkomst |
|---|---|---|
| v4 | begrensde fracties **erbij** | alle criteria gehaald, doelwit onbewogen (SN3 0,52 → 0,54); fracties droegen 2,5 % gain |
| v5 | z-scores binnen de opname, **in plaats van** | poort gefaald (OOF AP 0,721 → 0,676); SN3 bewoog wél (→ 0,61) maar SN1/SN2 liepen weg naar 1,31 en 1,51 |

v5 leert wel iets: **absolute featurewaarden dragen echte informatie** — ze
weggooien kost 0,045 AP. Het model kijkt niet "te absoluut".

## Aangewezen: de selectie

Het orakel op de SELECTIE — houd exact de kandidaten die met een menselijke
arousal matchen — haalt **0,896**, ruim boven het menselijke plafond van 0,679.

| opname | pool | nu | pool-plafond | mens |
|---|---:|---:|---:|---:|
| SN1 | 1163 | 0,514 | **0,928** | 0,642 |
| SN2 | 1188 | 0,396 | **0,896** | 0,492 |
| SN3 | 1453 | 0,419 | **0,669** | 0,692 |
| SN4 | 1410 | 0,601 | **0,851** | 0,767 |
| SN5 | 1601 | 0,708 | **0,948** | 0,766 |

**De arousals ZITTEN in de pool.** Van 0,514 naar 0,896 is 0,382 die volledig
in de selectie ligt — en die selectie is het LightGBM-model, niet de drempel.

Dit weerlegt mijn eigen redenering van een uur eerder. Ik concludeerde uit
"324 kandidaten op drempel 0,30 met F1 0,269" dat de kandidaten verkeerd
lagen. Fout: op 0,30 selecteer je nog steeds, alleen slecht. De pool bevat
1453 kandidaten waarvan er ~142 juist zijn; ruimhartig houden levert vooral
verkeerde op.

## En op SN3 óók de pool

Bij perfecte precisie geldt F1 = 2R/(1+R), dus het plafond zegt hoeveel van de
menselijke arousals de pool überhaupt bevat:

| opname | pooldekking |
|---|---:|
| SN5 | 90 % |
| SN1 | 87 % |
| SN2 | 81 % |
| SN4 | 74 % |
| **SN3** | **50 %** |

SN3 is de enige opname waar het pool-plafond (0,669) ONDER het menselijke
plafond (0,692) ligt. Daar is dus twee dingen aan de hand: de kandidaatstap
mist de helft van de arousals, én de selectie haalt uit wat er wél is maar
0,419 van de 0,669.

## Wat dit betekent voor het v4-issue

Het issue `issues/v4_fractie_features.md` gaat over modelhertraining, en die
richting blijft juist — maar de features die ik koos waren het niet. De
opdracht is scherper geworden: er ligt **0,382 mediane F1 in de selectie**, en
het huidige model haalt daar 0,514 van 0,896.

Dat is een veel groter doel dan alles wat vandaag geprobeerd is, en het
verklaart waarom drempels en featuretweaks niets opleverden: ze schuiven aan
de marge van een beslissing die fundamenteel beter kan.

## Openstaand, in volgorde

1. **Waarom haalt het model 0,514 van 0,896?** Foutanalyse op de kandidaten
   die het model verwerpt terwijl ze matchen — welke features onderscheiden
   die van de kandidaten die het terecht verwerpt?
2. **Waarom mist de kandidaatstap de helft van SN3?** Dat is de
   regelgebaseerde detector met zijn 2-seconden-vensters en ratio-drempels,
   niet het model.
3. n = 5. Alles hierboven vraagt replicatie op MESA per opname.
