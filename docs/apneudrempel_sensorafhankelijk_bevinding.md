# Waarom de thermistor als apneusensor verliest: de drempel is voor de neusdruk geijkt

*22 augustus 2026. Diagnostisch, geen preregistratie — er wordt hier niets
afgesteld en niets aangenomen.*

## Waar de vraag vandaan komt

De poortmeting van vandaag liet `respiratory_band` verliezen: meer thermistors
door de poort, en de apneutelling **halveerde** (2223 → 1223 over 50
MESA-opnames). Ik schreef daarbij dat de neusdruk "op dit cohort de betere
apneusensor" is. **Dat was te kort door de bocht**, en deze meting laat zien
waarom.

## De meting

Zeven MESA-opnames, **597 door mensen gescoorde apneus** uit de
NSRR-annotatie — niet uit onze detector, dus geen circulariteit. Per event de
mediane ademamplitude (bandfilter 0,10–0,70 Hz, Hilbert-omhullende) tijdens
het event, gedeeld door de mediane amplitude in de 60 s ervoor.

| | neusdruk | thermistor |
|---|---:|---:|
| mediane amplitudedaling | **89,6 %** | **80,3 %** |
| aandeel events dat ≥ 90 % haalt | **50,0 %** | **13,3 %** |

De neusdruk zakt dieper op **7 van de 7** opnames. Per opname:

| opname | n | druk | therm | ≥90 % druk | ≥90 % therm |
|---|---:|---:|---:|---:|---:|
| mesa-sleep-5295 | 6 | 89,6 | 52,3 | 50 % | 0 % |
| mesa-sleep-5208 | 148 | 90,7 | 83,4 | 55 % | 26 % |
| mesa-sleep-0033 | 181 | 85,9 | 68,1 | 34 % | 6 % |
| mesa-sleep-1356 | 60 | 88,0 | 80,3 | 40 % | 13 % |
| mesa-sleep-6804 | 55 | 93,0 | 70,6 | 65 % | 0 % |
| mesa-sleep-1693 | 21 | 86,0 | 81,2 | 43 % | 24 % |
| mesa-sleep-3913 | 126 | 92,1 | 81,9 | 60 % | 31 % |

## Wat het verklaart

`aasm_v3_breath` gebruikt `flow_reduction_threshold = 0,9` — **één drempel,
ongeacht de sensor**. Dezelfde menselijke apneu haalt die drempel op de
neusdruk in de helft van de gevallen en op de thermistor in **één op de acht**.

Daarmee is de halvering van de apneutelling geen eigenschap van de thermistor
maar van de drempel. De keten:

1. Het profiel zegt `sensor = thermistor` — AASM-conform, want de regel
   schrijft een thermische opnemer voor.
2. De poort keurt de thermistor op ⅔ van de MESA-opnames af, dus in de
   praktijk wordt er op de **neusdruk** gescoord.
3. De drempel van 0,9 werkt daar, omdat de neusdruk de daling **uitvergroot**:
   het signaal loopt ongeveer met het kwadraat van de flow, dus een daling van
   70 % in flow oogt als ~90 % in amplitude.
4. Laat de poort de thermistor wél door, dan raakt diezelfde drempel nog maar
   13 % van de echte apneus.

**De AASM schrijft de thermische opnemer juist voor omdat de neusdruk
overdrijft.** Onze apneudetectie leunt op precies die overdrijving om haar
criterium te halen. Dat is geen detail: het betekent dat de goede uitkomst op
de neusdruk deels uit twee fouten komt die elkaar opheffen.

## Wat hier NIET uit volgt

**Niet** dat de drempel op ~0,80 moet. Dat getal komt uit deze zeven opnames,
en het op dit cohort afstellen en er vervolgens op meten is een fit die zich
als validatie voordoet — dezelfde fout die `hypopnea_strictness` op PSG-IPA
maakte.

**Niet** dat de thermistor een slechtere sensor is. Hij zakt ondieper; of hij
apneus bétér onderscheidt van niet-apneus is hiermee niet gemeten. Daarvoor
zou je ook de amplitudedaling op momenten ZONDER apneu moeten kennen — een
sensor die overal ondieper zakt verliest geen onderscheidend vermogen.

**Niet** dat dit buiten MESA geldt. Eén cohort, één type thermistor, n = 7
opnames.

## Wat de volgende stap zou zijn

Een **sensorafhankelijke apneudrempel**, met een eigen preregistratie en een
kalibratie- en validatiesteekproef die van elkaar gescheiden zijn. De
diagnostiek hierboven hoort daar de motivering te zijn, niet de ijking.

De ontbrekende meting die er eerst bij hoort: de amplitudedaling op
**niet-apneu**-momenten, op beide sensoren. Zonder die is niet te zeggen of
een lagere drempel op de thermistor echte apneus terugwint of alleen ruis
binnenlaat.
