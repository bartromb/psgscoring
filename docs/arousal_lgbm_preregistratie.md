# Preregistratie — mag het hybride LGBM-pad default aan?

Datum: 2026-08-21. Geschreven **voordat de laatste PSG-IPA-opname (SN5)
gemeten was** en voordat er iets aan de defaults veranderd is.

## Waar dit over gaat

`PSGSCORING_AROUSAL_LGBM=1` schakelt een bestaand pad in: het kandidaatstadium
draait op ruime drempels (ratio 1,2 / abrupt 1,0) en een LightGBM-model filtert
wat overblijft. Het model (`psgscoring/data/arousal_classifier_v3.txt`) is
getraind op **MESA q∈{5,6}**, 653 proefpersonen, 562k kandidaten. Het staat
sinds YASAFlaskified v0.9.8 in de repo en **default uit**.

Wat er tot vandaag van gevalideerd was: de arousal-INDEX (Pearson r 0,84 op
PSG-IPA, r 0,66 op de MESA q=7-holdout tegen 0,08 regelgebaseerd; paper v37
§5.5). Event-niveau nooit.

**PSG-IPA is voor dit model een extern cohort** — ander lab, andere montage,
twaalf scoorders per opname, geen enkele opname in de training. De meting van
vandaag is dus externe validatie, geen ontdekking; daarom staat er hierboven
geen drempelkeuze te preregistreren. De veeg over
`AROUSAL_LGBM_THRESHOLD` is beschrijvend, en de default 0,60 is de waarde die
beoordeeld wordt.

## Wat er nog moet gebeuren voordat de vlag default aan mag

Een betere arousaldetectie is niet vanzelf een betere PSG-analyse. Arousals
voeden twee dingen die verderop meetellen:

- **Rule 1B-hypopneus** — een debietdaling van ≥30 % telt mee bij desaturatie
  *of* arousal. Meer of andere arousals verplaatsen hypopneus in en uit de AHI.
- **RERA-detectie** (`detect_reras`), en daarmee de RDI.

Daarom, vastgelegd vóór die meting:

**Voorwaarde 1 — respiratoir niet slechter.** Op **MESA n = 150** (zelfde
steekproef en referentie als `docs/mesa_n150_hermeting_20260814.md`, profiel
`aasm_v3_rec` en `aasm_v3_breath`): de AHI-bias mag niet meer dan **1,0/u**
verslechteren en de event-F1 niet meer dan **0,01** dalen ten opzichte van
dezelfde run met de vlag uit. Verbetert er iets, prima; dit criterium bewaakt
alleen dat de arousalwinst niet respiratoir wordt afbetaald.

**Voorwaarde 2 — looptijd.** De hybride berekent 50 kenmerken per kandidaat
op een veel ruimer kandidaatstadium. De looptijd per opname mag niet meer dan
**verdubbelen**; zie `docs/performance_policy.md`. Wordt dat overschreden,
dan gaat de vlag alleen aan op de klinische profielen en niet op de
dataset-profielen, of hij wacht op optimalisatie.

**Voorwaarde 3 — reproduceerbaarheid.** `mesa_shhs` en `chicago_1999` blijven
op het regelgebaseerde pad, zodat paper v31/v37 reproduceerbaar blijft. Golden
9/9 byte-identiek met de vlag uit.

**Voorwaarde 4 — de beslissing ligt bij de gebruiker.** Ook als 1 tot 3
gehaald worden, gaat de default pas om na expliciete toestemming. Dit
verandert klinische output op elke nieuwe opname.

## Waarom dit hier staat en niet als conclusie achteraf

De verleiding is groot: het regelgebaseerde pad is vandaag tweemaal
tevergeefs bijgesteld, en dan is een bestaande vlag die het wél doet een
opluchting. Precies daarom staan de voorwaarden op papier voordat de laatste
cijfers binnen zijn.
