# `plm_offset_aasm`: de cohorttegenspraak opgelost — de referentie, niet de montage

**Datum:** 2026-08-24 (avond)
**Status:** de vlag blijft **default uit**. Dit document draait de weerlegging
van eerder vandaag terug; het is geen beslissing om hem aan te zetten.

---

## De tegenspraak

Vanmiddag (`plm_aasm_offset_index_weerlegd.md`) mat ik op MESA n=6 dat de vlag
de PLM-telling verder van de referentie brengt, op 6/6, p = 0,031. Op PSG-IPA
n=5 met twaalf scoorders meet ik het omgekeerde.

Twee dingen verschilden tegelijk, en dat is precies waarom die meting niets kon
beslissen:

| | MESA | PSG-IPA |
|---|---|---|
| scoorders | 1 | 12 |
| geannoteerd | alleen linkerbeen | beide benen, mét zijde |
| ons kanaal | één kaal `Leg`, zijde onbekend | `EMG LAT` + `EMG RAT`, bilateraal samengevoegd |

## De toets die ze uit elkaar haalt

PSG-IPA opnieuw gedraaid **in MESA-modus**: alleen `EMG LAT`, geen bilaterale
samenvoeging, tegen de linkerbeen-annotaties van dezelfde twaalf scoorders. De
montage is dan gelijk aan MESA; alleen de referentie verschilt.

| | mediane afwijking (uit → aan) | dichterbij op | binnen scoordersspreiding (uit → aan) |
|---|---|---:|---|
| **bilateraal** | 45 → **24** | 4/5 | **0/5 → 4/5** |
| **één been (MESA-modus)** | 38 → 38 | 4/5 | **0/5 → 3/5** |

De vlag verbetert de overeenstemming **ook in de MESA-montage**. Het
cohortverschil komt dus niet van de montage.

## Wat dat betekent voor de MESA-uitkomst

Dan blijft de referentie over. MESA's beenscoring heeft één scoorder, annoteert
alleen het linkerbeen, en telde op **twee volledig gescoorde nachten nul
beenbewegingen** (965 en 956 events, inclusief arousals en respiratoire
events). PSG-IPA laat zien dat twaalf scoorders op elke nacht 68–723 bewegingen
vinden. Een nacht met werkelijk nul is daarmee onwaarschijnlijk.

De MESA-meting mat vermoedelijk hún onderscoring, niet onze overdetectie. Ze
telt niet mee als weerlegging.

## Wat er wél staat

Zonder de vlag vinden wij ongeveer de **helft** van de PLM's die de mens telt
(ratio 0,51 bilateraal, 0,52 enkelzijdig) en liggen we op **0 van de 5** binnen
de spreiding van de twaalf scoorders. Op SN3 en SN5 vinden we er **nul** tegen
respectievelijk 13 en 221. Dat is een echte onderdetectie op het sterkste
cohort dat dit project heeft.

De bewegingsdetectie zelf is redelijk (LM-ratio 0,86); het verlies zit in de
SERIEdetectie. Omdat de menselijke bewegingen door exact dezelfde serieregel
gaan, ligt dat niet aan de regel maar aan de tijdsverdeling van onze
bewegingen — met de te korte duur uit de 8 µV-offset als de aannemelijkste
oorzaak, en dat is precies wat deze vlag repareert.

## Waarom hij tóch niet omgaat

- **n = 5.** Wilcoxon p = 0,19 (bilateraal) en 0,31 (enkelzijdig); bij n=5 kan
  die maat niet onder 0,06 komen. De 0/5 → 4/5 is overtuigender dan de
  p-waarde, maar het blijft vijf opnames.
- **SN4 gaat de verkeerde kant op:** 417 → 687 tegen een scoordersmediaan van
  548, buiten de spreiding 504–616. De vlag overschiet daar.
- **De PLM-index verandert klinisch fors** en dat is een gebruikersbeslissing,
  geen meetuitkomst.
- **MESA is niet weerlegd, alleen gediskwalificeerd als referentie voor
  beenbewegingen.** Een tweede onafhankelijk cohort met betrouwbare
  beenannotatie zou de zaak sluiten; dat is er nu niet.

## Wat dit harnas nu kan

`scripts/validate_plm_psgipa.py` — twaalf scoorders, beide benen, met
`--single-leg` om een enkelzijdige montage na te bootsen. De menselijke
bewegingen gaan door **exact dezelfde keten** als de onze (bilaterale
samenvoeging, slaapfilter, respiratoire uitsluiting, seriedetectie), zodat een
verschil aan de LM-detectie toe te schrijven is en niet aan de serielogica.

Twee grendels erin, na een eigen fout: een onbekende stadiumtekst is een harde
fout (de eerste versie las "Sleep stage N1" niet en verklaarde 568 van de 846
epochs tot wake, in beide armen), en een hypnogram met minder dan een half uur
slaap stopt het harnas.
