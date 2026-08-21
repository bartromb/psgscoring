# Preregistratie — cap én lokale basislijn samen

Datum: 2026-08-21. **Geschreven vóór de meting.**

## Stand

Drie varianten zijn nu gemeten tegen de specificatie (`azarbarzin`), op acht
MESA-opnames met de menselijke NSRR-events:

| | mediaan ratio | bereik | Spearman | risicoband |
|---|---:|---|---:|---:|
| default (`max(l,g)`, ongekapt) | 1,32 | 0,29–2,34 (×8,0) | 0,69 | 7/8 |
| + cap | — | 0,28–1,42 (×5,1) | 0,76 | 7/8 |
| `local_baseline_only`, ongekapt | 0,44 | 0,22–0,95 (×4,3) | **0,93** | 4/8 |

Dat derde getal dwingt een correctie op mijn eerdere formulering. Ik schreef
dat de spreiding "van het venster komt, niet van de basislijn". Het venster is
inderdaad het mechanisme — de vensterlengte is een sprongfunctie en de daling
door de cap volgt de eventdichtheid met r = 0,89. Maar het nachtbrede plafond
is één van de twee dingen die dat venster aansturen, via de hersteldrempel
`basislijn − 1 %`. `local_baseline_only` is dus géén zuivere
basislijnwijziging: het verkort óók de vensters. Daarom haalt het in zijn
eentje al ρ 0,93.

## De vraag

Voegt de cap iets toe bovenop het weghalen van het plafond, of is de
eenvoudiger ingreep genoeg?

## Acceptatiecriterium (vastgelegd vóór de meting)

Gemeten: `cap_at_next_event=True` én `local_baseline_only=True`, tegen
dezelfde specificatie en dezelfde acht opnames.

**Primair.** Spearman ρ t.o.v. de specificatie **> 0,93** — dus strikt beter
dan `local_baseline_only` alleen. Haalt de combinatie dat niet, dan voegt de
cap niets toe aan de rangorde en is de eenvoudiger ingreep de juiste.

**Secundair.** Het ratiobereik krimpt onder **×3,0** (nu ×4,3 voor lokaal
alleen).

**Bewaking.** Het aantal opnames in dezelfde risicoband als de specificatie
wordt gerapporteerd. Let op: lokaal alleen scoort daar 4/8 tegen 7/8 voor de
default, omdat het op ~0,44× de schaal zit. Een betere rangorde bij een
slechtere bandtoewijzing is geen winst zolang er afkapwaarden op die schaal
gebruikt worden.

**Wat er NIET uit volgt.** Ook als beide criteria gehaald worden, gaat er geen
default om. Er is geen ijkpunt op dit cohort, n = 8, en de vergelijking is
tegen onze eigen spec-implementatie — niet tegen Azarbarzins code. Wat deze
meting oplevert is welke van de twee ingrepen nodig is, niet of het getal
klopt.

## Verwachting

Met de cap zou het basislijnverschil zich weer als schaalfactor moeten
gedragen, dus ρ hoort ten minste gelijk te blijven aan de 0,93 van lokaal
alleen. Komt hij daar duidelijk onder, dan doet de cap iets dat de rangorde
verstoort en hoort hij niet in de combinatie.
