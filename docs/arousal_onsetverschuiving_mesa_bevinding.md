# MESA-replicatie van de +2 s onsetverschuiving — gehaald

Gemeten 2026-08-26 tegen `arousal_onsetverschuiving_mesa_preregistratie.md`.
n = 30, productieconfiguratie, werkpunt 0,80, referentie de `Arousal`-events uit
de NSRR-XML (onset én duur), event-F1 bij IoU 0,20.

## Uitkomst

| verschuiving | mediane F1 | gem. paarsgewijze Δ | beter |
|---|---:|---:|---:|
| −2 s | 0,3537 | −0,0597 | 0/30 |
| −1 s | 0,3875 | −0,0255 | 0/30 |
| **0 s** *(nu)* | 0,4100 | — | — |
| +1 s | **0,4378** | +0,0136 | 22/30 |
| **+2 s** | 0,4305 | **+0,0140** | **22/30** |
| +3 s | 0,4288 | +0,0077 | 17/30 |
| +4 s | 0,4071 | −0,0127 | 10/30 |
| +6 s | 0,3371 | −0,0944 | 0/30 |

**Criterium 1 gehaald:** Δ = **+0,0140**, beter op **22 van 30** (26 verschillen
niet nul), tweezijdige tekentoets **p = 0,00053**.
**Criterium 2 gehaald:** het maximum van de reeks ligt op **+2 s**.

## Waarom dit een sterke replicatie is

De twee cohorten verschillen op vrijwel alles wat ertoe kan doen:

| | PSG-IPA | MESA |
|---|---|---|
| scoorders per opname | 12 | 1 |
| afleidingen | 3 (F+C+O) | **1** |
| duur menselijke arousals | 8,6 s | 11,0 s |
| n | 5 | 30 |
| **gem. Δ bij +2 s** | **+0,0123** | **+0,0140** |
| maximum van de reeks | +2 s | +2 s |

Ander apparatuurpark, andere scoorconventie, andere afleidingsopzet, zes keer
zoveel opnames — en hetzelfde optimum met vrijwel dezelfde grootte. Dat maakt
het een eigenschap van **onze detector**, niet van de PSG-IPA-scoorders. Dat was
precies de vraag die de replicatie moest beantwoorden.

## Twee eerlijke kanttekeningen

1. **+1 en +2 liggen dicht bij elkaar**, op beide cohorten (MESA +0,0136 tegen
   +0,0140; PSG-IPA +0,0097 tegen +0,0123). Op MESA piekt de **mediane** F1
   zelfs op +1 (0,4378 tegen 0,4305) terwijl de **gepaarde** maat op +2 piekt.
   Dat is dezelfde val als bij v6: de mediaan volgt één opname. De vooraf
   vastgelegde maat is de gepaarde, en die wijst +2 aan — maar het verschil
   tussen +1 en +2 is klein genoeg om niet als vastgesteld te lezen.
2. **De helft van het mechanisme is nog onverklaard.** Het gecentreerde
   2 s-Hanningvenster verklaart ~1 s (zie
   `arousal_onsetverschuiving_bevinding.md`); waar de tweede seconde vandaan
   komt weet ik niet. Dat de verschuiving op twee onafhankelijke cohorten
   hetzelfde optimum geeft, maakt een toevalstreffer wel onwaarschijnlijk.

## Wat er nog gemeten moet worden vóór een default-flip

**Arousals zijn niet alleen een uitkomst, ze zijn ook een invoer.** Ze
bevestigen Rule 1B-hypopneus en dragen de RERA-detectie. Een verschuiving van
+2 s verplaatst elke arousal ten opzichte van de respiratoire events, en kan
dus de **AHI en de RDI** veranderen — ook als de arousaltelling identiek blijft.
Die meting is niet gedaan.

Zolang die ontbreekt is dit een gemeten verbetering van de arousal-**localisatie**
en niets meer. Een vlag hoort default uit, en een default-flip vereist eerst een
gepaarde AHI/RDI-meting op MESA, net als bij elke eerdere arousalwijziging.

Zie [[feedback_verify_the_delivery_surface]]: een verbetering in de bibliotheek
is pas geleverd als de maat die de kliniek leest hem ook laat zien.

## Reproductie

`scripts/sweep_arousal_onset_offset_mesa.py 30`; ruwe cijfers per opname in
`arousal_onsetverschuiving_mesa.json`.
