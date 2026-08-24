# Preregistratie — wat doen de twee openstaande wijzigingen SAMEN?

*24 augustus 2026, 02:45. Vóór de meting.*

## Waarom apart meten niet genoeg is

Er liggen twee gevalideerde wijzigingen klaar:

- `hypopnea_strictness` 0,50 → 0,30 (respiratoir);
- `plm_offset_aasm` aan (beenbewegingen).

Beide zijn los gemeten. Maar wie ze aanzet, zet ze samen aan, en ze zijn niet
onafhankelijk: de strictness verandert het aantal hypopneus en dus de AHI, en
via de RERA-koppeling ook de RDI; de PLM-regel verandert de beenbewegingen en
via `plm_arousal_index` een afgeleide arousalmaat.

Een gebruiker die beslist, verdient het gecombineerde cijfer — niet de som van
twee losse.

## Opzet

MESA n=30, zaad 20260824 (dezelfde validatieset), gepaard, volle pipeline,
artefact-epochs zoals productie, huidige uitgerolde stand als basis.

| arm | strictness | PLM-einde |
|---|---|---|
| A | 0,50 (nu) | huidig (nu) |
| D | 0,30 | AASM |

Arm A is al gemeten; alleen D moet erbij.

## Wat gerapporteerd wordt

AHI, RDI, arousal-index, PLM-index en event-F1, met de gepaarde verschuivingen
en het aandeel opnames waarop de AHI- of RDI-ernstklasse verandert.

## Karakterisering, geen slaag/zak

De twee wijzigingen zijn elk al aangenomen op hun eigen vooraf vastgelegde
regel. Deze meting beslist niets; ze maakt zichtbaar wat de gebruiker aanzet.

**Meldgrens:** verschuift de AHI- of RDI-ernstklasse op meer dan een kwart van
de opnames, dan leg ik dat apart voor.

## Wat dit niet uitwijst

Of de combinatie beter is dan elk apart. Daarvoor zouden alle vier de hoeken
gemeten moeten worden, en de vraag die voorligt is niet "welke combinatie" maar
"wat gebeurt er als ik ja zeg".
