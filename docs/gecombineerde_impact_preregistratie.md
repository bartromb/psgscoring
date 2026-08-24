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

---

# Uitkomst — 24 augustus 2026, 03:55

## Wat de meting wél laat zien

Gepaard over 30 opnames, A = huidige uitgerolde stand, D = beide wijzigingen
aan:

| | A (nu) | D | verschil |
|---|---:|---:|---:|
| event-F1 mediaan | 0,48 | 0,54 | +0,06 |
| AHI mediaan | 19,05 | 22,85 | +3,80 |
| RDI mediaan | 28,70 | 31,70 | +3,00 |
| arousal-index mediaan | 19,50 | 19,50 | 0,00 |
| AHI-bias mediaan | −3,28 | −0,05 | +3,23 |

Gepaarde ΔF1 **+0,0346**, beter op 26/30, p = 2,6·10⁻⁷. AHI-ernstklasse
verschuift op 7/30 = **23 %**, RDI op 3/30 = 10 %. Beide onder de meldgrens.

Dat is **exact** de strictness-uitkomst, tot op het cijfer. De arousal-index
beweegt niet.

## Wat de meting NIET laat zien, en dat is mijn fout

Ik kan hieruit **niet** concluderen dat de twee wijzigingen onafhankelijk zijn.
`validate_mesa.py` legt geen enkel PLM-veld vast — geen `n_lm_sleep`, geen
`plm_index`, niets. De env-vlag stond aan, maar of hij effect had is uit de
uitvoer niet af te lezen.

Dezelfde soort blindheid als de apneuteller die nul gaf: het harnas kon een
uitspraak niet dragen die ik er wel uit wilde halen.

Wat er dus feitelijk staat: **de strictness-validatie reproduceert exact** op
een onafhankelijke run met dezelfde seed — op zichzelf een geslaagde
reproductiecontrole — en over de PLM-kant zegt deze meting niets.

**Gerepareerd**: het harnas schrijft nu `n_lm_total`, `n_lm_sleep`,
`n_plm_eligible`, `n_plm`, `plm_index` en `n_events_truncated` mee, met een
test die faalt als die velden verdwijnen. De gecombineerde vraag is daarmee
beantwoordbaar geworden, maar nog niet beantwoord.
