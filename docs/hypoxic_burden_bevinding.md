# De hypoxic burden wijkt af van de gepubliceerde definitie — maar niet als schaalverschuiving

> **Herzien, later op 21 augustus 2026.** De eerste versie van dit document
> concludeerde dat onze burden "op een andere schaal staat dan de literatuur"
> en dat onze default 4 van 8 opnames in de hoogrisicoband > 73 legt tegen 0
> van 8 voor de gepubliceerde methode. **Dat klopte niet.** Het berustte op
> `ensemble`-waarden die te laag waren door een defect in de
> vensterafleiding, en op de aanname dat ons `ensemble`-pad de publicatie
> volgde. Dat doet het niet. Met de specificatie opgezocht en correct
> geïmplementeerd valt 7 van de 8 opnames in dezelfde risicoband. De
> onderstaande secties zijn bijgewerkt; de conclusie is veranderd, de
> vastgestelde afwijkingen niet.

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

## De verantwoording rust op een fixture die het defect niet kan tonen

De wiki (`psgscoring.wiki/Hypoxic-Burden.md`) vergelijkt de twee methodes en
concludeert dat ze dicht bij elkaar liggen:

> On synthetic data (30 events, 8 h): Percentile 9,76 · Ensemble 10,54.
> Difference: ~8%.

Dertig events over acht uur is één per zestien minuten. Dan overlappen de
integratievensters nooit en herstelt de saturatie ruimschoots tussen twee
events — precies de twee omstandigheden waaronder het verschil ontstaat. Op
de acht MESA-opnames hierboven, met 66 tot 234 events, is het verschil geen
8 % maar een factor 2 tot 14.

Dezelfde valkuil als bij de tests van vandaag: een fixture die te braaf is om
te meten wat ze zou moeten meten. Een hermeting hoort op echte opnames met
realistische eventdichtheid, niet op synthetische.

## Wat dit klinisch betekent, preciezer

De wiki noemt drie banden uit Azarbarzin 2019: < 20 laag risico, 20–73 matig,
**> 73 hoog** — en in die bovenste band liet CPAP een beschermend effect zien
(HR 0,57; Pinilla et al., ERJ 2023) waar het in het ongestratificeerde cohort
geen effect had.

Op deze acht opnames:

| | < 20 | 20–73 | **> 73** |
|---|---:|---:|---:|
| onze default | 1 | 3 | **4** |
| ensemble | 3 | 5 | **0** |

Het gaat dus niet om een randgeval rond een afkapwaarde, maar om de band die
een behandelargument draagt.

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


---

# Herziening — de specificatie opgezocht, en wat er dan overblijft

## De definitie, uit de literatuur

> *"For each individually identified apnea or hypopnea, the maximum SpO2
> during the 100 seconds before the end of the event is considered as the
> pre-event baseline oxygen saturation. For each event, the area under this
> baseline value was calculated over a subject-specific search window that was
> determined from the ensemble average of time-aligned SpO2 curves."*

Het venster is *"the interval between the pre-event and post-event maximum
oxygen saturation values"*; de individuele oppervlaktes worden gesommeerd en
gedeeld door de totale slaaptijd.

De review in Archivos de Bronconeumología merkt daarbij op dat de publicatie
implementatiedetails mist die nodig zijn voor onafhankelijke replicatie — over
het omgaan met overlappende vensters staat er niets. Dedupliceren zou dus een
eigen keuze zijn, geen reparatie.

## Vier afwijkingen, alle vier bevestigd

| | onze `percentile` (default) | onze `ensemble` | specificatie |
|---|---|---|---|
| basislijn | max(90e pct van 120 s vóór **onset**, 95e pct nachtbreed) | max van de eerste 3 s ván het zoekvenster | **max over `[einde − 100 s, einde]`** |
| venster | onset → herstel of +120 s (mediaan 126 s) | ensemble-afgeleid | ensemble-afgeleid |
| linkerflank | n.v.t. | kon ná het eventeinde landen | pre-event maximum |

De vierde afwijking is gerepareerd (`_ensemble_search_window` dwingt de
linkerflank nu vóór het eventeinde) en dat was materieel: `mesa-sleep-1374`
ging van 18,33 naar 44,31 en `mesa-sleep-2747` van 23,39 naar 66,52.

## De meting, met de specificatie erbij

`baseline_method="azarbarzin"` implementeert de definitie zoals geciteerd.
Toegevoegd, default uit; de twee bestaande paden zijn ongewijzigd.

| opname | `max(l,g)` default | lokaal | ensemble | **azarbarzin** | default/spec |
|---|---:|---:|---:|---:|---:|
| 1374 | 266,14 | 108,83 | 44,31 | **113,97** | 2,34 |
| 2149 | 66,98 | 24,03 | 23,64 | **50,52** | 1,33 |
| 2747 | 81,03 | 46,51 | 66,52 | **105,61** | 0,77 |
| 3135 | 2,82 | 2,13 | 3,48 | **9,71** | 0,29 |
| 3743 | 64,73 | 21,43 | 22,31 | **49,35** | 1,31 |
| 3823 | 25,16 | 17,06 | 36,46 | **53,77** | 0,47 |
| 6157 | 168,26 | 62,46 | 41,95 | **112,99** | 1,49 |
| 1020 | 100,27 | 12,32 | 15,53 | **45,13** | 2,22 |

## De herziene conclusie

**Geen schaalverschuiving maar spreiding.** De verhouding default/specificatie
loopt van 0,29 tot 2,34, op zes opnames te hoog en op twee te laag, zonder
patroon. Er bestaat dus geen correctiefactor die het rechttrekt — per patiënt
is de afwijking onvoorspelbaar. Dat is lastiger dan een systematische
afwijking, niet makkelijker.

**De risicoband komt meestal wél overeen.** Met de banden < 20 / 20–73 / > 73
vallen 7 van de 8 opnames in dezelfde klasse; alleen `mesa-sleep-1020`
verschuift (hoog → matig). De klinische stelling uit de eerste versie van dit
document — dat de banden systematisch verschillen — wordt door deze meting
NIET gedragen.

**Wat er niet mee bewezen is.** Er is geen ijkpunt op dit cohort: MESA
publiceert geen hypoxic-burdenvariabele, en van SHHS staat hier alleen 36 KB
scripts. `azarbarzin` volgt de geciteerde definitie, maar of het getal
overeenkomt met wat Azarbarzins eigen code oplevert is hiermee niet getoetst.

## Wat hier de volgende stap is

1. Meet `azarbarzin` tegen `percentile` op MESA n = 150, gepaard, met een
   vooraf vastgelegd criterium. n = 8 is te klein voor een defaultwissel.
2. Geef `baseline_method` door aan de samenvatting en het rapport. Een getal
   dat op vier manieren berekend kan worden, hoort te zeggen welke het was.
3. Laat de referentiewaarde "< 20" pas staan wanneer het getal ernaast op de
   gepubliceerde definitie berust.
4. Een regressietest met realistische eventdichtheid, zodat de val van de
   synthetische fixture niet terugkomt.
