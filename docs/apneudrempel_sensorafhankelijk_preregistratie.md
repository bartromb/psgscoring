# Preregistratie — een sensorafhankelijke apneudrempel

*22 augustus 2026, vóór enige kalibratie. De kalibratiesteekproef is nog niet
aangeraakt.*

## Aanleiding

`aasm_v3_breath` hanteert één `flow_reduction_threshold = 0,9`, ongeacht welke
sensor de apneus draagt, terwijl het profiel `sensor = thermistor` declareert.
Gemeten op door mensen gescoorde apneus (zie
`apneudrempel_sensorafhankelijk_bevinding.md`):

- de neusdruk zakt mediaan 89,6 %, de thermistor 80,3 %;
- **50 %** van de menselijke apneus haalt 90 % op de druk, **13 %** op de
  thermistor;
- de AUC apneu-vs-hypopneu is gelijk (gepaard +0,009, binnen de vooraf
  vastgelegde band van 0,03).

Dezelfde informatie, andere schaal. Eén drempel kan niet beide schalen lezen.

## Wat er precies verandert

Een tweede drempel voor de thermistor. **Default = huidig gedrag**: zolang de
nieuwe waarde gelijk is aan `flow_reduction_threshold` verandert er niets, en
de gepinde reproductieprofielen (`mesa_shhs`, `chicago_1999`, `cms_medicare`,
`aasm_v1_rec`, `aasm_v2_rec`) blijven daar hoe dan ook op staan.

## Steekproeven — GESCHEIDEN

| | zaad | n | rol |
|---|---|---|---|
| kalibratie | 20260823 | 30 | drempel afleiden, `--exclude-seed 20260801 --exclude-n 50` |
| validatie | 20260801 | 50 | afrekenen; dit is dezelfde set als de poortmeting |

Disjunct per constructie: de kalibratieset trekt uit de pool ná verwijdering
van de validatieset, met het mechanisme dat het harnas daar al voor heeft.

## De afleidingsregel — vastgelegd vóór het kijken

De thermistordrempel is de waarde waarbij het **fout-positieve percentage
gelijk is aan dat van de neusdruk bij 0,90** op de kalibratiesteekproef, waar
"fout-positief" betekent: een geannoteerde **hypopneu** die de drempel haalt en
dus als apneu zou tellen.

Waarom deze regel en geen optimalisatie: hij ijkt de twee sensoren op elkaar
zonder iets te maximaliseren. Een regel die F1 optimaliseert op de
kalibratieset zou een tweede vrijheidsgraad introduceren en de validatie
verzwakken. Deze regel heeft één uitkomst en die is uit te rekenen.

Afronden op twee decimalen. Valt de waarde boven 0,90, dan is er niets te
ijken en vervalt het hele voorstel.

## De vergelijking

| arm | poort | drempel |
|---|---|---|
| A — huidige default | `envelope_agreement` | 0,90 beide sensoren |
| B — voorstel | `respiratory_band` | 0,90 druk, afgeleide waarde thermistor |

Arm A is al gedraaid (poortmeting van vandaag, n=50, seed 20260801). Arm B
draait op exact dezelfde 50 opnames.

Waarom de poort meeverandert: bij `envelope_agreement` wordt de thermistor op
⅔ van de opnames afgekeurd, dus een thermistordrempel zou daar nauwelijks iets
doen. De vraag die ertoe doet is of de **AASM-conforme** configuratie —
apneus op de thermistor, correct geijkt — beter scoort dan de huidige
terugval. Dat is één samengestelde vergelijking en dat staat hier zo.

## Beslisregel — vooraf

Arm B wordt default op de niet-gepinde v3-profielen **alleen als beide**:

1. mediane **gepaarde** ΔF1 ≥ **+0,010**, en
2. de mediane gepaarde |AHI-bias| verslechtert niet met meer dan **1,0/u**.

**Weerlegd** bij mediane gepaarde ΔF1 ≤ 0. Daartussen: onbeslist, blijft
opt-in, en dat rapporteer ik als zodanig.

Ik noteer expliciet **gepaard**, en niet het verschil van de twee medianen.
Bij de poortmeting van vanochtend scheelde dat precies tussen aannemen en
verwerpen: +0,011 tegen 0,0000.

## Meldgrens

Verschuift de AHI-ernstklasse op meer dan een kwart van de opnames, dan leg ik
dat als apart punt voor.

## Wat dit niet kan uitwijzen

Welke van de twee wijzigingen het doet. Arm B verandert poort én drempel
tegelijk; wint hij, dan is niet te zeggen of de winst uit de sensorkeuze of uit
de ijking komt. Dat is bewust — de twee zijn niet los van elkaar zinvol — maar
het betekent dat een winst niet als bewijs voor de poort afzonderlijk mag
gelden. De poortmeting van vandaag blijft staan: bij een drempel van 0,90 is
`respiratory_band` weerlegd.
