# Preregistratie — de classifierdrempel opnieuw ijken, met gescheiden steekproeven

*23 augustus 2026, vóór enige kalibratie.*

## Waarom

`AROUSAL_LGBM_THRESHOLD = 0,60` is nooit onafhankelijk gevalideerd. De veeg die
hem bevestigde (`arousal_lgbm_preregistratie.md`) draaide op **dezelfde vijf
PSG-IPA-opnames** waarop het resultaat gerapporteerd werd — een fit, geen
validatie — en bovendien in een configuratie die we niet meer draaien:
**single** derivatie. Productie draait multi.

De diagnose van 22-08-2026 wijst bovendien precies naar deze knop. Het gat naar
de menselijke F1 is **precisie**, geen localisatie: de detector vindt de
arousals waar de scoorders het over eens zijn (86 % bij 11–12 van de 12), maar
**16 % van onze events wordt door geen enkele scoorder gedekt** en bijna de
helft niet door een meerderheid. De drempel verhogen snijdt juist de
laagst-scorende kandidaten weg.

## Configuratie

Multi-derivatie, hybride aan, EOG-reject uit, **artefactlijst genegeerd**.

Dat laatste is een keuze en ze is voorwaardelijk. De artefactmeting van
23-08-2026 laat zien dat negeren op 30 van 30 MESA-opnames wint en op PSG-IPA
repliceert; ijken in de configuratie die aantoonbaar slechter is, levert een
drempel op die daarna niet overdraagt. **Besluit de gebruiker de lijst tóch te
blijven gebruiken, dan moet deze ijking over.** Dat staat hier vooraf.

## Steekproeven — GESCHEIDEN

| | zaad | n | rol |
|---|---|---|---|
| kalibratie | 20260825 | 15 | drempel kiezen; sluit de validatieset uit |
| validatie | 20260824 | 30 | afrekenen; dezelfde 30 als de artefactmeting |
| replicatie | PSG-IPA | 5 | tekencontrole, twaalf scoorders |

De validatieset is al gemeten bij drempel 0,60 (arm A van de artefactmeting,
F1 0,421), dus daar hoeft alleen de gekozen drempel bij.

## Keuzeregel — vastgelegd vóór het kijken

Geveegd wordt **0,50 · 0,60 · 0,70 · 0,80 · 0,90**.

Gekozen wordt de drempel met de **hoogste mediane F1 op de kalibratieset**.
Bij een verschil kleiner dan 0,005 tussen de beste twee wint de **hogere**
drempel — het gediagnosticeerde probleem is precisie, en bij gelijke F1 is
minder verzonnen events het betere product.

## Beslisregel — vooraf

De gekozen drempel vervangt 0,60 **alleen als beide**:

1. mediane **gepaarde** ΔF1 op de validatieset ≥ **+0,010**, en
2. het teken repliceert op PSG-IPA.

**Weerlegd** bij ΔF1 ≤ 0. Daartussen: onbeslist, 0,60 blijft.

Komt de veeg zelf uit op 0,60, dan is dat óók een uitkomst: dan is de waarde
voor het eerst onafhankelijk bevestigd in plaats van op de evaluatieset
gekozen, en verandert er niets.

## Wat dit niet uitwijst

Of het model zelf beter kan. Een drempel verschuift alleen het werkpunt op een
bestaande ROC; hij maakt geen nieuwe scheiding. Blijft de precisie ook bij een
hoge drempel steken, dan zit het probleem in de kenmerken of in de
trainingsreferentie (MESA, één scoorder per opname) en niet in het werkpunt.

Evenmin: of de twaalf-scoorder-fractie als trainingsdoel beter zou werken. Dat
is een apart spoor en het vereist hertrainen, niet herijken.

---

## Kalibratie-uitkomst en een amendement, vóór de validatie

Kalibratie op 15 MESA-opnames (zaad 20260825, overlap met validatie 0):

| drempel | F1 | precisie | recall | events (mediaan) |
|---:|---:|---:|---:|---:|
| 0,50 | 0,459 | 0,351 | 0,679 | 314 |
| 0,60 | 0,493 | 0,402 | 0,667 | 267 |
| 0,70 | 0,527 | 0,486 | 0,654 | 229 |
| 0,80 | 0,562 | 0,571 | 0,629 | 194 |
| **0,90** | **0,607** | **0,710** | 0,513 | 133 |

Monotoon stijgend over de hele veeg. De precisie verdubbelt bijna (0,351 →
0,710) terwijl de recall maar van 0,679 naar 0,513 zakt — precies het profiel
dat de diagnose voorspelde: er zaten veel events in die niemand markeerde.

**Het probleem: het optimum ligt op de RAND van mijn eigen veeg.** De
vooraf vastgelegde keuzeregel wijst 0,90 aan, maar een grid dat op zijn
grenspunt eindigt heeft het optimum niet gevonden — het is er alleen tegenaan
gelopen.

**Amendement, vastgelegd vóór enige validatiemeting:** de veeg wordt op
**dezelfde kalibratieopnames** uitgebreid met **0,93 · 0,95 · 0,97 · 0,99**.

Waarom dit geen p-hacking is: de kalibratieset bestaat om het werkpunt te
kiezen, en de validatieset is er niet bij betrokken en blijft ongezien. Wat wél
fout zou zijn, is het grid uitbreiden nádat de validatie is gedraaid, of het
uitbreiden op de validatieset. Geen van beide gebeurt.

De keuzeregel blijft ongewijzigd: hoogste mediane F1 op de kalibratieset,
en bij een verschil onder 0,005 wint de hogere drempel.

**Wat ik verwacht en waarom het ertoe doet dat ik het opschrijf:** ergens
boven 0,90 moet de F1 omslaan, want bij een drempel van 1,0 blijven er nul
events over. Ligt het optimum ook na uitbreiding op de rand (0,99), dan is er
iets anders aan de hand dan een werkpunt — dan produceert het model kansen die
niet kalibreren, en is de conclusie een andere.

### Uitkomst van de uitgebreide veeg

Zelfde 15 kalibratieopnames:

| drempel | F1 | precisie | recall | events |
|---:|---:|---:|---:|---:|
| 0,50 | 0,459 | 0,351 | 0,679 | 314 |
| 0,60 | 0,493 | 0,402 | 0,667 | 267 |
| 0,70 | 0,527 | 0,486 | 0,654 | 229 |
| 0,80 | 0,562 | 0,571 | 0,629 | 194 |
| **0,90** | **0,607** | 0,710 | 0,513 | 133 |
| 0,93 | 0,543 | 0,760 | 0,449 | 106 |
| 0,95 | 0,507 | 0,832 | 0,365 | 79 |
| 0,97 | 0,386 | 0,851 | 0,250 | 55 |
| 0,99 | 0,059 | 0,846 | 0,030 | 4 |

**Eén duidelijk maximum, binnen het grid.** De curve slaat om na 0,90, zoals
vooraf opgeschreven dat hij moest. Het optimum lag dus niet op de rand omdat de
kansen niet kalibreren, maar omdat mijn eerste grid te kort was.

**GEKOZEN: 0,90.** Het verschil met de tweede (0,93 bij 0,543) is 0,064, ruim
boven de 0,005 waar de tie-breakregel voor bestond.

Ter oriëntatie op de omvang: bij 0,60 produceert de detector mediaan **267**
events tegen een referentie rond de 130; bij 0,90 zijn dat er **133**. De
huidige default overdetecteert met ongeveer een factor twee, en dat is precies
wat de diagnose op PSG-IPA liet zien — 16 % van onze events werd door geen
enkele van de twaalf scoorders gedekt.

Het script viel na de tabel om op een `KeyError` in de slotregel (die verwees
naar 0,60, dat in de tweede veeg niet meer in het grid zat). De tabel is
volledig; alleen de JSON van de tweede ronde is niet weggeschreven.

**Nog niets besloten.** 0,90 is gekozen op de kalibratieset. De validatie op
de 30 ongeziene opnames moet nog, en die beslist — mediane gepaarde ΔF1 ≥
+0,010 én tekencontrole op PSG-IPA.

---

# Uitkomst — 23 augustus 2026

## Validatie, MESA n=30 (ongezien)

| drempel | F1 | precisie | recall | events | referentie |
|---:|---:|---:|---:|---:|---:|
| 0,60 (huidig) | 0,421 | 0,364 | 0,597 | 228 | 128 |
| **0,90** | **0,543** | 0,673 | 0,443 | 85 | |

Gepaarde ΔF1 **+0,0910**, beter op **24/30**, Wilcoxon **p = 1,5·10⁻⁵**.
Criterium 1 ruim gehaald.

## Tekencontrole, PSG-IPA n=5 (twaalf scoorders)

| drempel | F1 | precisie | recall | ratio |
|---:|---:|---:|---:|---:|
| 0,60 | 0,505 | 0,425 | 0,649 | 1,47 |
| **0,80** | **0,514** | 0,599 | 0,520 | **1,01** |
| 0,90 | 0,478 | 0,653 | 0,398 | 0,60 |

| tegen 0,60 | gepaarde mediaan | beter op | teken |
|---|---:|---:|---|
| 0,80 | **+0,0300** | **5/5** | repliceert |
| 0,90 | +0,0070 | 3/5 | repliceert |

**Formeel haalt 0,90 beide criteria** — ΔF1 +0,091 op MESA, teken repliceert op
PSG-IPA. Volgens de vooraf vastgelegde regel is 0,90 dus de uitkomst, en dat
laat ik staan.

## Maar het volledige beeld stelt die uitkomst ter discussie

Drie dingen die de regel niet ving:

1. **Het optimum verschilt per cohort.** MESA wijst 0,90 aan, PSG-IPA 0,80.
   Op PSG-IPA is 0,90 op de marginale F1 zelfs *slechter* dan 0,60 (0,478
   tegen 0,505), en de gepaarde winst leunt op 3 van 5 opnames.
2. **0,80 wint overtuigender waar het meeste bewijs per opname zit.** Vijf van
   vijf tegen drie van vijf, +0,030 tegen +0,007.
3. **De eventtelling.** Bij 0,80 is de verhouding gedetecteerd/referentie 1,01
   op PSG-IPA en 1,07 op de MESA-kalibratie — zuiver op beide. Bij 0,90 is dat
   0,60 respectievelijk 0,64–0,81: een arousal-index die een derde te laag
   uitkomt.

**Wat hoe dan ook vaststaat: 0,60 is gedomineerd.** 0,80 verbetert de F1 op
beide cohorten (MESA-kalibratie 0,493 → 0,562; PSG-IPA 0,505 → 0,514, 5/5) én
maakt de telling zuiver. Welk gewicht je ook aan F1 tegen index-zuiverheid
geeft, het huidige werkpunt is niet de juiste keuze.

## Wat ik hiervan wel en niet maak

Ik verschuif de uitkomstmaat **niet** achteraf van F1 naar de eventtelling. De
regel wees 0,90 aan en dat staat er.

Maar de keuze tussen 0,80 en 0,90 is klinisch, niet statistisch: de
arousal-index belandt in het rapport, en 36 % te laag is een ander soort fout
dan 47 % te hoog. Die weging is niet van mij.

**Beperking van deze meting:** PSG-IPA is n=5, dus de cohortverschillen kunnen
ruis zijn. Wie het wil beslechten, meet 0,80 tegen 0,90 gepaard op een tweede
MESA-steekproef, disjunct van beide gebruikte sets, met de eventtelling als
vooraf gekozen uitkomstmaat.

**Niets geïmplementeerd.** De drempel is nog steeds 0,60; er bestaat al een
env-override `PSGSCORING_AROUSAL_LGBM_THRESHOLD`.
