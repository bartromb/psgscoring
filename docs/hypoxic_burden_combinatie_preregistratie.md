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

---

# Uitkomst — 21 augustus 2026

**Beide criteria gehaald. De bewaking niet — en die beslist.**

| variant | mediane ratio | bereik | factor | Spearman | Pearson | zelfde band |
|---|---:|---|---:|---:|---:|---:|
| default | 1,32 | 0,29–2,34 | ×8,0 | 0,69 | 0,79 | 7/8 |
| + cap | 0,77 | 0,28–1,42 | ×5,1 | 0,76 | 0,84 | 7/8 |
| lokaal alleen | 0,44 | 0,22–0,95 | ×4,4 | 0,93 | 0,87 | 4/8 |
| **beide** | **0,32** | **0,21–0,54** | **×2,6** | **0,95** | **0,91** | **1/8** |

- **Primair (ρ > 0,93): GEHAALD** — 0,95. De cap voegt dus wél iets toe
  bovenop het weghalen van het plafond, zij het weinig.
- **Secundair (bereik < ×3,0): GEHAALD** — ×2,6.
- **Bewaking: 1 van 8.** Tegen 7/8 voor de default.

## Wat er werkelijk gebeurd is

De twee ingrepen samen zetten een **onvoorspelbare** afwijking om in een
**systematische**. Dat is de gunstigste uitkomst die een meetprobleem kan
hebben: een consistente onderschatting van ongeveer een derde laat zich
corrigeren, een spreiding van 0,29 tot 2,34 niet.

Maar precies daardoor wordt zichtbaar wat eronder ligt: het percentiel-pad
zit structureel op een andere schaal dan de specificatie, en dat is geen
gevolg van het venster of het plafond. Het komt van de basislijn zelf — een
90e percentiel over de 120 s vóór de ONSET tegen het maximum over de 100 s
vóór het EINDE. Dat verschil laat zich niet wegrepareren; het is de definitie.

## Conclusie

**Het percentiel-pad valt in de juiste volgorde te krijgen, niet op de juiste
schaal.** Wie een getal wil dat naast de literatuur te leggen is, moet niet
dit pad verder repareren maar `baseline_method="azarbarzin"` gebruiken.

Daarmee splitst de beslissing in twee, en dat is een gebruikersbeslissing:

1. **Blijven op `percentile`.** Dan horen cap en `local_baseline_only` erbij —
   ze maken het getal intern consistent — en horen de gepubliceerde
   afkapwaarden uit het rapport, want ze gelden niet op die schaal. Dat laatste
   is per 21-08-2026 al gebeurd: de referentie "< 20" verschijnt alleen nog
   naast `azarbarzin`.
2. **Over naar `azarbarzin`.** Dan zijn cap en `local_baseline_only` niet meer
   nodig: die repareren een pad dat je dan niet meer gebruikt.

Optie 2 is de enige die het getal vergelijkbaar maakt. Optie 1 maakt het
alleen consistent met zichzelf.

**Geen enkele vlag gaat hierop default.** n = 8, geen ijkpunt op dit cohort,
en de vergelijking is tegen onze eigen implementatie van de specificatie —
niet tegen Azarbarzins code. Wat deze meting oplevert is welke route zin heeft,
niet welk getal klopt.
