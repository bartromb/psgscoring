# Preregistratie — spectrale-verschuivingscriterium voor arousaldetectie

Datum: 2026-08-21. **Geschreven vóór enige meting van de nieuwe methode.**
Vastgelegd volgens dezelfde regel als de enveloppe-as (`docs/`): het
acceptatiecriterium staat op papier voordat de cohortcijfers bestaan.

## Het defect

`detect_arousals` (psgscoring/arousal.py:526) beslist met:

    arousal_pow = alpha_narrow + theta + beta          # regel 592
    arousal_mask = arousal_pow > ratio_thresh * local_bl   # regel 751

`local_bl` is de mediaan van de laagste 50 % van de voorafgaande 120 s
stabiele slaap (`_rolling_baseline`, regel 620). Het criterium is dus
**"vermogen in de snelle banden > 2x de eigen rustige vloer"**.

De AASM-definitie is een andere grootheid: *"an abrupt shift of EEG
frequency"* — een verschuiving van de **frequentie**, niet een stijging
van het **vermogen**. Vermogen in alpha/theta/beta stijgt ook bij zaken
die geen frequentieverschuiving zijn: bewegingsartefact, zweetartefact,
elektrode-instabiliteit, en een algemene amplitudetoename.

`delta_pow` wordt op regel 596 wél berekend en komt in de beslissing
**niet voor** — alleen als rapportageveld `seg_delta` op regel 841, dat
niet eens in het event-dict terechtkomt. De noemer die van een
vermogensmaat een frequentiemaat maakt, ligt klaar en wordt weggegooid.

Dit is dezelfde klasse defect als de twee RIP-poorten in het manuscript:
een criterium dat iets anders meet dan wat het beweert te meten.

## Wat de meting op PSG-IPA liet zien (bestaand gedrag, 0.22.0)

Vijf opnames, twaalf scoorders elk, event-F1 mediaan **0,266** tegen
0,692 scoorder-tegen-scoorder. De fout is niet een verschuiving maar
**inconsistentie**: index 4,4x te hoog op SN2, 2,3x te laag op SN3.

## De ingreep

Nieuw, scale-free criterium — de fractie van het spectrale vermogen in
de snelle banden:

    r(t) = (alpha + theta + beta) / (delta + alpha + theta + beta + sigma)

`r` is dimensieloos, begrensd op [0,1] en **invariant onder een
amplitudeschaling van het EEG**. Het stijgt alleen als het
zwaartepunt van het spectrum omhoog schuift — precies wat de AASM
beschrijft. Detectie op een absolute toename van `r` boven de lokale
basislijn, niet op een verhouding: op een begrensde grootheid is een
absoluut increment tussen opnames vergelijkbaar, op een onbegrensde
vermogensmaat is het dat nooit.

**De drempels worden hier vastgelegd, vóór de meting**, gekozen uit de
grootheid zelf en niet uit de data:

| constante | waarde | betekenis |
|---|---|---|
| `AROUSAL_SHIFT_DELTA` | **0,15** | `r` moet 0,15 absoluut boven de lokale basislijn liggen — een substantiële verplaatsing van het spectrale zwaartepunt |
| `AROUSAL_SHIFT_ABRUPT` | **0,10** | `r` in de eerste seconde ligt 0,10 absoluut boven de 3 s ervoor (abruptheid, event-niveau) |

In REM houdt de regel dezelfde vorm als nu: daar is theta het
achtergrondritme, dus telt alleen de alpha-fractie `alpha / totaal`.

Een gevoeligheidsveeg over `AROUSAL_SHIFT_DELTA` wordt gerapporteerd om
te tonen of het resultaat op een mesrand staat. **Beoordeeld wordt
uitsluitend de vooraf vastgelegde 0,15**; de veeg is beschrijving, geen
keuze.

Achter profielvlag `arousal_spectral_shift`, **default False**.
Bestaand gedrag blijft byte-identiek als de vlag uit staat.

## Acceptatiecriterium (vastgelegd vóór de meting)

Het gemeten defect is inconsistentie, dus daar wordt op afgerekend.
Laat `q_i = index_algoritme / index_scoordermediaan` per opname.

**Primair.** De spreiding `max(q)/min(q)` over de vijf PSG-IPA
opnames daalt van de huidige **10,3** naar **< 3,0**.

**Secundair.** De mediane event-F1 over de vijf opnames daalt niet
onder de huidige **0,266**.

**Randvoorwaarden voor promotie naar default:**
1. Beide bovenstaande gehaald.
2. Geen enkele respiratoire index verandert met de vlag uit
   (arousals voeden RERA-detectie) — golden 9/9 byte-identiek.
3. `mesa_shhs` en `chicago_1999` blijven gepind op het oude gedrag.

Wordt het primaire criterium niet gehaald, dan blijft de vlag
experimenteel en gaat hij niet default — ongeacht hoe gunstig de
F1 uitvalt. Een enkele gunstige F1 zonder consistentiewinst is
precies het patroon dat `rectify_lowpass` op PSG-IPA liet zien en
dat op MESA niet repliceerde.

---

# Uitkomst — 21 augustus 2026

**Het criterium is weerlegd. Het gaat niet default en het lost het gemeten
defect niet op.**

Gemeten op de vijf PSG-IPA arousal-opnames, twaalf scoorders per opname,
event-F1 met greedy IoU-matching op 0,20. `q = index_algoritme /
index_scoordermediaan`.

| | spreiding max(q)/min(q) | mediane F1 | q per opname |
|---|---:|---:|---|
| huidig gedrag | **10,13** | 0,182 | 0,87 · 4,36 · 0,43 · 0,90 · 1,00 |
| delta = 0,05 | 3,64 | 0,139 | 3,54 · 10,94 · 3,65 · 4,81 · 3,01 |
| delta = 0,10 | 4,09 | 0,167 | 2,15 · 8,79 · 2,90 · 2,91 · 2,19 |
| **delta = 0,15 (preregistratie)** | **4,82** | **0,146** | 1,12 · 5,42 · 1,72 · 1,50 · 1,35 |
| delta = 0,20 | 6,05 | 0,120 | 0,48 · 2,90 · 0,86 · 0,77 · 0,83 |
| delta = 0,25 | 8,75 | 0,070 | 0,18 · 1,59 · 0,32 · 0,36 · 0,47 |
| delta = 0,30 | 8,08 | 0,049 | 0,08 · 0,67 · 0,16 · 0,18 · 0,29 |

Scoorder-tegen-scoorder mediane F1: **0,692**.

- **Primair (spreiding < 3,0): NIET GEHAALD** — 4,82 op de vooraf vastgelegde
  waarde, en geen enkele waarde in de veeg haalt 3,0.
- **Secundair (F1 ≥ huidig): NIET GEHAALD** — 0,146 tegen 0,182.

De veeg laat bovendien geen mesrand zien maar een *monotone* wissel: lage
delta detecteert veel te veel, hoge delta veel te weinig, en de spreiding is
op géén punt aanvaardbaar. Er is dus geen waarde die het criterium redt, en
de post-hoc verleiding om er een te kiezen bestaat hier eenvoudig niet.

Let op één afwijking van de preregistratie: die noemde 0,266 als huidige
mediane F1, uit een eerdere meting met de multi-derivatiemodus. Dit harnas
draait single-derivatie op één centraal kanaal, waar het huidige gedrag 0,182
haalt. Op beide referenties is het nieuwe criterium slechter.

## Wat de meting wél opleverde

**De duur klopt niet.** Door mensen gescoorde arousals duren mediaan **8,6 s**
(PSG-IPA, 7528 events over 12 scoorders) en **11,0 s** (MESA/NSRR). De
detector geeft mediaan **3,6 s** — hij plakt tegen `AROUSAL_MIN_DUR_S = 3.0`.
Dat is een tweede, onafhankelijk defect, en gezien de IoU-matching
waarschijnlijk een zwaardere oorzaak van de lage F1 dan het detectiecriterium.

**MESA is bruikbaar als replicatiecohort.** De NSRR-annotaties bevatten
`Arousal|Arousal ()` (en PLM-events) over 2056 opnames, van een ander lab met
een andere montage. Harnas: `docs/mesa_arousal_harness.py`. Daarmee is de
opzet die bij de enveloppe-as werkte — PSG-IPA kiest, MESA repliceert — voor
arousals nu ook mogelijk. Dat was ze tot nu toe niet.

## Wat blijft staan

De vlag `arousal_spectral_shift` blijft in de code, default uit, met de tests
in `tests/test_arousal_spectral_shift.py`. Twee redenen: het is de enige plek
waar vastligt dat het huidige criterium een **pure amplitudestap zonder
frequentieverschuiving** als AASM-arousal scoort (`test_pure_amplitude_step_*`
legt beide kanten vast), en de doorvoer van profiel naar detector is nu
gebouwd en getest, zodat een volgend criterium er niet opnieuw doorheen
geleid hoeft te worden.
