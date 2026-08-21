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

---

# Meting — 21 augustus 2026, PSG-IPA (extern voor dit model)

Vijf opnames, twaalf scoorders elk, event-F1 met greedy IoU-matching op 0,20,
single-derivatie op één centraal kanaal. `q = index_algoritme /
index_scoordermediaan`.

| | mediane F1 | mediane duur | spreiding max(q)/min(q) |
|---|---:|---:|---:|
| regels alleen | 0,182 | 4,0 s | 10,13 |
| hybride, drempel 0,30 | 0,409 | 5,5 s | 1,53 |
| hybride, drempel 0,45 | 0,429 | 5,8 s | 1,79 |
| **hybride, drempel 0,60 (meegeleverde default)** | **0,463** | **6,2 s** | **2,10** |
| hybride, drempel 0,75 | 0,442 | 6,5 s | 2,21 |
| hybride, drempel 0,85 | 0,409 | 7,4 s | 2,67 |

Mens tegen mens: **0,692**, mediane menselijke eventduur 8,3 s.

Per opname bij drempel 0,60:

| | scoorder-index | regels | hybride | mens |
|---|---:|---:|---:|---:|
| SN1 | 24,2 | 0,326 (idx 21,0) | **0,463** (idx 26,6) | 0,642 |
| SN2 | 8,5 | 0,160 (idx 37,1) | **0,347** (idx 11,0) | 0,492 |
| SN3 | 18,0 | 0,154 (idx 7,8) | **0,340** (idx 11,1) | 0,692 |
| SN4 | 14,3 | 0,182 (idx 12,8) | **0,505** (idx 13,5) | 0,767 |
| SN5 | 27,7 | 0,499 (idx 27,6) | **0,666** (idx 31,6) | 0,766 |

De hybride haalt 67 % van het menselijke plafond waar de regels op 26 %
zitten, en `q` ligt op elke opname tussen 0,61 en 1,29 tegen 0,43 tot 4,36.
De overdetectie die geen enkele drempelingreep wegkreeg (SN2: 37,1 tegen een
scoordermediaan van 8,5) verdwijnt: 11,0. Op SN5 komt de eventduur uit op
9,2 s tegen menselijk 9,1 s.

De meegeleverde drempel 0,60 is óók het optimum van de veeg. Dat is niet door
mij afgesteld — hij stond al zo in de code — maar het is het vermelden waard,
want het betekent dat er hier geen keuze te maken viel.

## Een voetangel die eerst weg moest

Gevonden bij het nalopen van dit pad, vóór enige defaultwissel.

In hybride modus zet `detect_arousals` de drempels op de RUIME
kandidaatwaarden (ratio 1,2 / abrupt 1,0) en laat de classifier daarna
wegfilteren. Faalde die -- model ontbreekt, lightgbm niet geïnstalleerd,
corrupte booster -- dan logde de code *"falling back to rule-based output"*
terwijl `result["events"]` op dat moment de **kandidatenlijst** bevatte. De
samenvatting zei bovendien nergens dat de classifier niet gedraaid had.

Gemeten op PSG-IPA, single derivatie:

| | regels | hybride MET model | hybride ZONDER model | scoordermediaan |
|---|---:|---:|---:|---:|
| SN2 | 203 ev (37,1/u) | 60 (11,0) | **777 (142,1)** | 8,5 |
| SN4 | 94 ev (12,8/u) | 99 (13,5) | **979 (133,8)** | 14,3 |

Een installatie met de vlag aan maar zonder model rapporteert dus een
arousal-index van 134 tot 142 per uur waar 11 tot 14 hoort — een factor tien,
met een logregel die het tegendeel beweert. Zolang de vlag opt-in is, is dat
een voetangel; wordt hij default, dan is het een productiedefect.

Gerepareerd: het model wordt geladen **voordat** de kandidaatdrempels
verruimd worden, en lukt dat niet, dan blijven de regelgebaseerde drempels
staan. Mislukt het filteren pas ná een geslaagde laadpoging, dan wordt er
opnieuw gedetecteerd op de regeldrempels (met een privé-parameter die de
recursie stopt). `summary["lgbm_available"]` zegt of de classifier gedraaid
heeft, en verschijnt alleen als de hybride ook gevraagd was. Tests in
`tests/test_arousal_lgbm_fallback.py`.

## Stand van de voorwaarden

| | |
|---|---|
| 1. respiratoir niet slechter (MESA n=150) | **nog niet gemeten** |
| 2. looptijd ≤ 2× | **nog niet gemeten** |
| 3. `mesa_shhs`/`chicago_1999` gepind, golden 9/9 | te doen bij de wissel |
| 4. toestemming gebruiker | openstaand |

Voorwaarde 1 is de zwaarste en de enige die de uitkomst nog kan keren. De
hybride vindt op vier van vijf opnames MEER arousals dan de regels; via Rule
1B betekent dat meer arousal-gekoppelde hypopneus in de AHI.
