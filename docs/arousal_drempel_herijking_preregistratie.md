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
