# De hypoxic burden staat op een andere schaal dan de literatuur

*21 augustus 2026. Uitgevoerd op verzoek, naar aanleiding van W31b in
`docs/PAPER_REVISIE_v40.md`: is de driftgevoeligheid van onze hypoxic burden
een eigenschap van ONZE implementatie of van de gepubliceerde definitie?*

## Het antwoord op de gestelde vraag

**Van onze implementatie.** De constructie `baseline = max(lokaal,
nachtbreed)` zit uitsluitend in ons `percentile`-pad. Azarbarzin leidt per
opname een zoekvenster af uit het ensemble-gemiddelde van alle op het
eventeinde uitgelijnde SpO2-curves en neemt de basislijn aan de linkerflank
van dát venster — zuiver lokaal, zonder nachtbreed plafond. Die methode is bij
ons ook geïmplementeerd, als `baseline_method="ensemble"`, en staat default
uit.

Per W31b's eigen beslisregel betekent dat: het blijft staan waar het staat, in
S13, als beperking van psgscoring.

## Maar het is groter dan drift

Gemeten op acht MESA-opnames, met de **menselijk gescoorde NSRR-events** als
invoer zodat onze eventdetector geen rol speelt. Drift = 95e percentiel SpO2
in het eerste uur slaap minus dat in het laatste uur.

| opname | drift | n_ev | `max(l,g)` — onze default | lokaal alleen | ensemble | max/lokaal |
|---|---:|---:|---:|---:|---:|---:|
| 1374 | −0,8 | 114 | 266,14 | 108,83 | **18,33** | 2,45 |
| 2149 | +0,4 | 93 | 66,98 | 24,03 | **23,61** | 2,79 |
| 2747 | 0,0 | 234 | 81,03 | 46,51 | **23,39** | 1,74 |
| 3135 | −0,1 | 66 | 2,82 | 2,13 | **3,48** | 1,32 |
| 3743 | −0,9 | 186 | 64,73 | 21,43 | **22,31** | 3,02 |
| 3823 | −1,1 | 175 | 25,16 | 17,06 | **35,87** | 1,47 |
| 6157 | +2,2 | 95 | 168,26 | 62,46 | **41,95** | 2,69 |
| 1020 | −1,9 | 97 | 100,27 | 12,32 | **15,53** | 8,14 |

**Drift verklaart het niet.** `mesa-sleep-2747` heeft drift exact 0,0 en toch
een factor 1,74; `mesa-sleep-1020` werd 's nachts juist béter (−1,9) en heeft
met 8,14 de grootste factor. Het nachtbrede plafond ligt structureel boven de
lokale basislijn, ongeacht of de saturatie wegzakt.

## Het mechanisme, gemeten op mesa-sleep-1374

Er zijn twee afwijkingen, en ze werken allebei omhoog.

**1. Het integratievenster is ruim vier keer zo breed.**

| | |
|---|---|
| ensemble-zoekvenster (Azarbarzin) | 30,6 s |
| ons percentiel-venster | **mediaan 126,5 s** (90e 142,4 · max 170,7) |

Het percentiel-pad integreert van eventONSET tot herstel, of tot 120 s ná het
einde als er geen herstel komt. Op deze opname:

- **60 van 114 events** lopen de volle 120 s uit — de saturatie keert nooit
  terug tot basislijn − 1 %.
- **73 van 114 events** overlappen het venster van het vorige event. Dezelfde
  desaturatie telt dus meermaals mee.
- Totaal geïntegreerd: **193 min op een nacht van 540 min**, oftewel 36 %
  dekking, voor events die samen nog geen uur beslaan.

**2. De basislijn ligt hoger.** Lokaal 92,6 % tegen een nachtbreed plafond van
94,2 %: 1,6 procentpunt extra deficit op élk sample binnen die 193 minuten.

## Waarom de bestaande verantwoording niet dekt wat er gebeurt

De docstring citeert He et al. voor de stelling dat de percentiel-basislijn
vergelijkbaar is met de ensemble-methode. Dat kan kloppen voor de BASISLIJN.
Maar onze implementatie combineert die basislijn met een heel ander
INTEGRATIEVENSTER, en die combinatie is niet wat daar vergeleken is. De
citatie verantwoordt één van de twee afwijkingen, niet het paar.

## Wat dit klinisch betekent

`generate_pdf_report.py:2633` drukt de hypoxic burden af mét referentiewaarde
**"< 20"**. Die grens komt uit literatuur waarin de burden volgens Azarbarzin
berekend is. Op deze acht opnames legt onze default er **7 van 8** boven, de
gepubliceerde methode **4 van 8**. De gerapporteerde waarde wordt dus
vergeleken met een afkapwaarde waarmee hij niet commensurabel is.

## Wat hier NIET geclaimd wordt

De vergelijking is tussen onze twee implementaties, waarvan er één in onze
eigen docstring als "Azarbarzin original" staat. Ik heb de publicatie zelf
niet naast de code gelegd. Voor de conclusie "onze default wijkt af van de
gepubliceerde definitie" is dat voldoende — de afwijking zit in het plafond en
het venster, en beide zijn in de code expliciet als onze eigen keuze
gemarkeerd. Voor de sterkere claim "onze `ensemble` is een getrouwe
implementatie van Azarbarzin" is het dat niet; die verdient een directe
vergelijking met Figuur 1 van Azarbarzin et al. (Eur Heart J 2019) voordat er
een default op omgaat.

## Voorstel

1. Verifieer `_ensemble_search_window` tegen de publicatie zelf.
2. Meet daarna op een groter cohort of `ensemble` default kan worden. Dat is
   een gedragswijziging op een gerapporteerde grootheid, dus met
   preregistratie en toestemming.
3. Tot die tijd: laat de referentiewaarde "< 20" niet naast een getal staan
   dat op een andere schaal ligt. Dat is de goedkoopste ingreep en de enige
   die vandaag al klinisch scheelt.

Meetscript: `docs/meet_hypoxic_burden_varianten.py`.
