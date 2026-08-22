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

---

# Uitkomst — 22 augustus 2026

**WEERLEGD.** Arm B wordt geen default. De vlag blijft bestaan met
`None` als default, dus er verandert niets aan uitgeleverd gedrag.

Gepaard over 50 MESA-opnames, `aasm_v3_breath`, seed 20260801, psgscoring
0.25.0. Kalibratie: 26 opnames, disjunct (overlap 0).

| | arm A (nu) | arm B | verschil |
|---|---:|---:|---:|
| F1 mediaan | 0,499 | 0,492 | −0,007 |
| **gepaarde ΔF1, mediaan** | | | **−0,0010** |
| AHI-bias mediaan | −4,19 | −1,86 | **+2,33** |
| \|AHI-bias\| mediaan | 6,67 | 6,30 | −0,37 |
| apneus totaal | 2223 | 2393 | +170 |

Gepaard: beter op 12, slechter op 26, gelijk op 12. Wilcoxon p = 0,009.
AHI-ernstklasse verschuift op 10/50 (20 %), onder de meldgrens.

Beslisregel: mediane gepaarde ΔF1 ≤ 0 → **weerlegd**.

## De ijking deed precies wat ze moest doen

Drie configuraties naast elkaar, dezelfde 50 opnames:

| | apneus | F1 med | bias med | \|bias\| med |
|---|---:|---:|---:|---:|
| envelope + 0,90 (huidige default) | 2223 | 0,499 | −4,19 | 6,67 |
| band + 0,90 (poortmeting vanochtend) | **1223** | 0,510 | −5,13 | 6,81 |
| band + 0,72 (arm B) | **2393** | 0,492 | **−1,86** | **6,30** |

Het zuivere drempeleffect (band 0,90 → band 0,72) herstelt de apneus die de
poort had verloren: **1223 → 2393**, met een gepaarde ΔF1 van precies 0,0000
(beter op 15, slechter op 17) en een |bias| die van 6,81 naar 6,30 zakt. De
voorspelling uit de kalibratie klopt dus: de thermistor mist bij 0,90 het
merendeel van de apneus, en 0,72 haalt ze terug.

## Wat dit oplevert en wat niet

**De AHI-bias verbetert fors** — mediaan −4,19 → −1,86, meer dan een halvering
van de systematische onderschatting. Voor een klinische index is dat niet
niks.

**De event-F1 verslechtert licht maar consistent** — gepaard −0,0010, slechter
op 26 van 50, p = 0,009. De extra apneus vallen dus niet allemaal op de
plaatsen waar de referentie er ook een heeft.

De vooraf vastgelegde regel is F1-primair, en die regel gold vóór ik deze
cijfers zag. **Ik verschuif hem niet achteraf naar de bias.** Dat is precies
waar preregistratie voor bestaat: een uitkomst die op één as wint en op de
gekozen as verliest, is een verwerping, geen aanleiding om de as te wisselen.

## Wat ik hieruit wél meeneem

De richting is niet dood, de **maat** is de vraag. Dat de bias meer dan
halveert terwijl de F1 nauwelijks beweegt, wijst erop dat de teruggewonnen
apneus grotendeels echt zijn maar net anders liggen dan de referentie ze legt
— plausibel bij een IoU-drempel van 0,20 op events die op een andere sensor
zijn afgebakend.

Wie dit wil vervolgen, hoort **vooraf** te kiezen of de AHI-bias of de
event-F1 de uitkomstmaat is, en die keuze te verdedigen vóór de meting. Beide
zijn verdedigbaar; ze samen achteraf wegen is dat niet.

Een tweede spoor dat deze meting openlaat: de 0,72 komt uit een benadering van
de detector (mediane daling over een Hilbert-omhullende), niet uit de keten
zelf. Een ijking op de werkelijke `flow_norm` van de detector kan een andere
waarde geven.
