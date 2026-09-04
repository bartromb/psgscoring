# Verliesrekening op de zwaarst onderdetecterende hoge-AHI-nachten

*2026-09-04, 15 nachten met de grootste onderdetectie uit de verse
149-opnameset (rec-bias tot −40/u), `aasm_v3_rec` op 0.32.0. Dezelfde
verliesrekening die bij de arousals de EEG1-vondst opleverde.*

## De rekening

| bak | n | aandeel |
|---|---:|---:|
| menselijke events | 3706 | 100 % |
| gedekt door een **eind**event | 591 | **15,9 %** |
| gedekt door een **afgewezen** hypopneu | 1073 | **29,0 %** ← met filters/werkpunt terug te halen |
| **nooit kandidaat** geweest | 2042 | **55,1 %** ← het echte plafond |

Van de nooit-kandidaten is 90 % een menselijke hypopneu (1840), 9 % obstructieve
apneu, <1 % centraal.

## Waaróm de 1073 werden afgewezen (geaggregeerd)

| reden | n | aandeel |
|---|---:|---:|
| `local_reduction … < 20 %` | 394 | 36,7 % |
| `local_reduction −x % < 20 %` (flow BOVEN de lokale basislijn) | 280 | 26,1 % |
| `no_desaturation` (Rule 1A: geen desat én geen arousal) | 201 | 18,7 % |
| `stable_breathing_cv < 0,45` | 198 | 18,5 % |

## De diagnose: de lokale basislijn bezwijkt onder hoge last

**62,8 % van de terughaalbare verliezen** sneuvelt op de lokale
basislijnvalidatie — en ruim een kwart daarvan heeft zelfs een NEGATIEVE
reductie: de flow tijdens het menselijke event ligt bóven de "pre-event
basislijn". Dat kan alleen wanneer die basislijn zelf pathologisch laag is:
op een nacht met AHI 60+ bestaat het pre-event-venster grotendeels uit
vórige events, en tegen een ingezakte referentie verdwijnt elke reductie.

Dit is exact de fout die deze week drie keer langs kwam in andere gedaanten
(nachtpercentiel-snurk, arousal-basislijn-hypothese, prevalentie-afhankelijke
gradering): **een lokale referentie die aanneemt dat de omgeving gezond is,
breekt precies waar de ziekte het ergst is.** En de moeilijkheid-schaalt-met-
last-regel van het project krijgt hiermee zijn mechanisme: het is geen
diffuse moeilijkheid, het is de referentie.

De `stable_breathing`-bak (18,5 %) heeft vermoedelijk hetzelfde mechanisme:
periodieke ademhaling maakt de amplitude-CV laag ("stabiel") terwijl er
niets stabiels aan is; alle gevallen zaten net onder de 0,45.

De `no_desaturation`-bak (18,7 %) is het doelwit van de al geplande
gegradeerde Rule 1B (arousalkans in plaats van gebinariseerd event).

## Wat dit betekent voor de 55 % nooit-kandidaten

De kandidaatgeneratie gebruikt dezelfde envelope en (rollende) basislijn. Als
de referentie inzakt, komt een kandidaat nooit onder de drempel uit — het
plafond en de afwijzingen delen vermoedelijk één mechanisme. Dat is toetsbaar:
hertest de dekking met een basislijn die alleen uit HERSTEL-ademhaling put
(zoals het breath-pad in passage 1 al doet met zijn sjabloon), of met een
globale-percentielvloer onder de lokale referentie.

## Reparatiekandidaten, in volgorde

1. **Basislijn uit hersteladem** (of vloer onder de lokale referentie) —
   raakt kandidaatstap én validator tegelijk; potentieel het grootste deel
   van 55 % + 63 %-van-29 %. Achter vlag, meten op de hoge-tertielnachten
   eerst, dan vol cohort.
2. **Gegradeerde Rule 1B** (gepland punt 6) — gericht op de 18,7 %.
3. **Stabiliteitsfilter lastbewust maken** — pas na 1, want mogelijk lost
   1 dit mee op.
