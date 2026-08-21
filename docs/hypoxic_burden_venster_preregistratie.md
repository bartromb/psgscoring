# Preregistratie — het integratievenster van de hypoxic burden

Datum: 2026-08-21. **Geschreven vóór de meting van de ingreep.**

## Het defect

`compute_hypoxic_burden` integreert in het `percentile`-pad van eventonset tot
het eerste sample waarop SpO2 ≥ `basislijn − 1 %`, of tot 120 s ná het
eventeinde als dat niet gebeurt. Dat is een binaire toets per event: herstellen
of vollopen. Gemeten over acht MESA-opnames verklaart het aandeel events dat de
cap haalt de afwijking t.o.v. de gepubliceerde definitie met **r = 0,89**, en
de verdeling van vensterlengtes is tweetoppig in plaats van geleidelijk
(`mesa-sleep-1020`: mediaan 21,3 s, maar 35 % aan de cap).

Gevolg: op `mesa-sleep-1374` overlapt **73 van de 114** vensters het vorige, en
telt dezelfde hypoxemie meermaals mee.

## De ingreep

`hypoxic_burden_cap_at_next_event`: kap de integratie van elk event af bij de
**onset van het volgende event**. Elke seconde hypoxemie hoort dan bij hoogstens
één event.

Geen nieuwe constante, en geen keuze over attributie: wat ná de onset van het
volgende event gebeurt, hoort per definitie bij dát event. De vensterlengte
wordt daarmee een functie van de eventafstand — continu — in plaats van van een
drempelovergang.

Achter een profielvlag, **default uit**.

## Verwachting, vooraf

1. **Overlap verdwijnt** — per constructie 0 %.
2. **De spreiding van de ratio t.o.v. de specificatie krimpt.** Nu 0,29–2,34
   met ρ 0,69. Verwachting: het bereik halveert ruwweg en ρ stijgt boven 0,80.
3. **De daling is het grootst waar de eventdichtheid het hoogst is.** Op
   opnames met weinig overlap (3135: 3 %, 3823: 5 %) verandert er vrijwel
   niets; op 1374 (64 %) en 6157 (55 %) het meest.

Komt 3 niet uit, dan werkt er iets anders dan de veronderstelde oorzaak en
telt de winst niet — dezelfde regel als bij de arousal-hysterese, waar het
mechanisme-criterium de ingreep terecht afkeurde.

## Wat dit NIET is

Geen implementatie van Azarbarzin. De publicatie zegt niets over overlappende
vensters; dedupliceren is dus een eigen keuze, geen reparatie naar de
specificatie toe. Wat het wél is: het opheffen van een dubbeltelling die onder
geen enkele lezing van "totale oppervlakte" bedoeld kan zijn.

De volgorde blijft zoals vastgesteld: eerst het venster, dan pas de basislijn.

---

# Uitkomst — 21 augustus 2026

**Mechanisme bevestigd, uitkomstcriterium niet gehaald. De vlag gaat niet
default.**

| opname | overlap | default | +cap | spec | daling |
|---|---:|---:|---:|---:|---:|
| 1374 | 64 % | 266,14 | 132,19 | 113,97 | 50 % |
| 6157 | 55 % | 168,26 | 81,50 | 112,99 | 52 % |
| 3743 | 37 % | 64,73 | 43,76 | 49,35 | 32 % |
| 2747 | 31 % | 81,03 | 66,62 | 105,61 | 18 % |
| 1020 | 27 % | 100,27 | 64,29 | 45,13 | 36 % |
| 2149 | 24 % | 66,98 | 41,16 | 50,52 | 39 % |
| 3823 | 5 % | 25,16 | 22,92 | 53,77 | 9 % |
| 3135 | 3 % | 2,82 | 2,70 | 9,71 | 4 % |

**Verwachting 1 — overlap verdwijnt.** Gehaald, per constructie.

**Verwachting 3 — de daling loopt mee met de eventdichtheid.** Gehaald,
**r = 0,89**. `3135` (3 % overlap) en `3823` (5 %) bewegen nauwelijks (4 % en
9 %), `1374` en `6157` halveren. Het is dus werkelijk de dubbeltelling en niet
iets anders.

**Verwachting 2 — spreiding halveert, ρ boven 0,80. NIET GEHAALD.**

| | default | +cap | vastgelegd |
|---|---:|---:|---|
| ratiobereik | factor 8,0 | factor 5,1 | ruwweg halveren |
| variatiecoëfficiënt | 0,59 | 0,47 | — |
| Spearman t.o.v. spec | 0,69 | **0,76** | > 0,80 |
| Pearson t.o.v. spec | 0,79 | 0,84 | — |
| zelfde risicoband | 7/8 | 7/8 | — |

Alles beweegt de goede kant op, maar niet ver genoeg. Volgens de vooraf
vastgelegde regel blijft de vlag daarmee experimenteel.

## Wat dit zegt over de volgende stap

Dit is precies wat te verwachten was uit de vaststelling dat de twee
afwijkingen elkaar VERMENIGVULDIGEN. De cap haalt de overlap weg, maar de
hersteldrempel is nog altijd `basislijn − 1 %`, en het nachtbrede plafond duwt
nog steeds events over de 120 s-grens waar geen volgend event in de buurt
ligt. Op `mesa-sleep-1020` — 27 % overlap maar 35 % van de events aan de cap —
blijft `cap/spec` op 1,42 steken.

De volgorde uit `docs/hypoxic_burden_bevinding.md` blijft dus staan, met één
toevoeging: het venster is een noodzakelijke maar niet voldoende stap. Wat
resteert is de hersteldrempel zelf, en die is niet los te zien van de
basislijn.

**Voorstel voor de volgende preregistratie:** cap én `local_baseline_only`
samen, tegen dezelfde specificatie. Verwachting: mét de cap gedraagt het
basislijnverschil zich weer als schaalfactor, dus ρ hoort dan wél boven 0,80
te komen. Valt dat tegen, dan zit de rest in de 120 s-grens zelf en is een
ensemble-afgeleid venster de enige overgebleven route.
