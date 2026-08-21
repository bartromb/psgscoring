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
