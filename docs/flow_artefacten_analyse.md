# Artefacten in de ademhalingsdetectie: wat er staat, en wat er ontbreekt

*23 augustus 2026. Analyse met metingen waar ik ze kon doen; expliciet
gemarkeerd waar het redenering is.*

## Wat er nu draait

Drie mechanismen, en **twee ervan kijken naar het EEG**:

| mechanisme | waarop berekend | wat het doet |
|---|---|---|
| `run_artifact_detection` (YF) | **EEG-kanalen**, piek > 500 µV of vlak | epochs uit `build_sleep_mask` → poort apneu- **en** hypopneudetectie |
| clipping-terugkoppeling (`tasks.py:306`) | **EEG-kanaal** | voegt nog meer epochs aan diezelfde lijst toe |
| `_detect_signal_gaps` (`respiratory.py:52`) | **flow zelf** | sluit vlakke/bevroren stukken ≥ 10 s uit, plus 15 s herstelramp |

Alleen het derde kijkt naar het signaal waarop gescoord wordt. En dat derde
werkt in de praktijk niet.

## Het flow-mechanisme is effectief dood — gemeten

`_detect_signal_gaps` markeert een stuk als uitval bij `|x| < 1e-5`
(**absolute** drempel op het RUWE signaal) of `diff == 0`. Gemeten op
`mesa-sleep-0001`:

| kanaal | bereik | \|x\| < 1e-5 | diff == 0 | 1e-5 als deel van p1–p99 |
|---|---|---:|---:|---:|
| `Pres` | −19,07 … 1,96 | **0,00 %** | 0,00 % | 1,3·10⁻⁵ |
| `Flow` | −2,50 … 6,39 | 0,06 % | 0,00 % | 1,4·10⁻⁵ |
| `Therm` | −0,005 … 0,003 | **6,44 %** | 0,00 % | 7,2·10⁻³ |

Dezelfde drempel, dezelfde opname: op de neusdruk vuurt hij **nooit**, op de
thermistor op **6 % van de samples**. Het verschil is puur amplitudeschaal.

`diff == 0` vuurt nergens, en dat is te verwachten: echte ADC-data heeft geen
exact gelijke opeenvolgende samples. Een **losgeraakte canule ruist**, hij
staat niet op precies nul.

Dit is de derde keer dat dit patroon opduikt: de RIP-poort mat met een
absolute MAD-drempel in feite de EDF-eenheid (gerepareerd), de artefactregel
gooit met een absolute 500 µV mediaan 19,9 % van de MESA-nacht weg
(gemeten vandaag), en hier weer. **Een absolute drempel op een signaal
waarvan de eenheid per montage verschilt, meet de eenheid.**

## Waarom flow-artefacten een ánder probleem zijn dan EEG-artefacten

De foutrichting is omgekeerd, en dat bepaalt het ontwerp.

- **EEG-artefact** kost je events: je onderdrukt epochs en verliest arousals
  die er wél waren. Gemeten: F1 0,421 → 0,338 op MESA, 30/30 slechter.
- **Flow-artefact maakt events**: een vlak of dood flowsignaal ís per definitie
  "geen ademhaling", en dat is precies de definitie van een apneu.

Daarom kan één artefactmasker voor beide niet kloppen. Voor het EEG is
onderdrukken schadelijk; voor de flow is niet-onderdrukken schadelijk. De
huidige opzet doet het omgekeerde van allebei: hij onderdrukt de
ademhalingsdetectie op grond van het EEG, en laat echte flow-uitval
grotendeels door.

## De flow-artefacten die er klinisch toe doen

**1. Losgeraakte of verschoven canule.** Geeft ruisend bijna-nul: geen
ademhaling zichtbaar, dus aaneengesloten apneus. De maximumduur van 90 s
begrenst het losse event, niet het aantal. Onderscheidend in het signaal: de
**effortbanden blijven bewegen** en de **SpO₂ reageert niet**. Onze
apneudefinitie eist geen desaturatie, dus niets houdt dit tegen.

**2. Mondademhaling.** Neusdruk zakt minutenlang naar bijna nul terwijl de
patiënt prima door de mond ademt; de thermistor ziet het wel. Dit is dé
klassieke bron van vals-positieve apneus en het is exact de reden dat de AASM
een thermische opnemer vóórschrijft voor apneus.

Bij ons komt dat ongelukkig samen: de thermistorpoort keurt de thermistor op
**⅔** van de MESA-opnames af, dus we scoren apneus op de neusdruk — precies de
sensor die bij mondademhaling blind is. En de apneudrempel is, zoals vandaag
gemeten, feitelijk voor die neusdruk geijkt.

**3. Clipping van de druktransducer.** Railt het signaal, dan is een
amplitudereductie niet meer meetbaar. Er is wél een clipping-detectie in
`signal_quality`, maar de terugkoppeling naar het artefactmasker draait op het
**EEG-kanaal**, niet op de flow.

**4. Basislijndrift.** Grotendeels opgevangen doordat de basislijn rollend is.
Van de vier het minst dringend.

## Wat ik zou bouwen, in deze volgorde

**a. De vlakke-detectie schaalvrij maken.** Zelfde reparatie als bij de
RIP-poort: oordeel op een verhouding tot de eigen spreiding van het kanaal in
plaats van een absoluut getal. Kleinste ingreep, en hij maakt een mechanisme
dat er al staat voor het eerst werkzaam. Achter een profielvlag, want hij gaat
events wegnemen.

**b. Een artefactvlag PER SIGNAALGROEP.** EEG-artefact raakt de
EEG-afhankelijke stappen (stadiëring, arousals), flow-artefact raakt de
ademhalingsdetectie, SpO₂-artefact de desaturaties. Eén vlag voor de hele
opname is de categoriefout die vandaag aan het licht kwam. Dit is de
eigenlijke reparatie en de grootste ingreep.

**c. Kruiscontrole tegen de tweede sensor en de effortbanden.** Flow vlak
terwijl thermistor óf beide banden ademhaling tonen = geen apneu maar een
sensorprobleem of mondademhaling. Dit vraagt geen nieuwe detector, alleen een
poort die bestaande kanalen naast elkaar legt — en het pakt punt 1 en 2
tegelijk.

**Wat ik NIET zou doen zonder referentie:** een drempel afstellen op "hoeveel
apneus voelen te veel". Er is geen artefactreferentie in MESA of PSG-IPA (net
zomin als een RERA-referentie), dus elke ijking hier moet indirect: het effect
op de AHI-overeenstemming met de menselijke scoring, met gescheiden
kalibratie- en validatiesteekproeven.

## Status

Analyse, geen wijziging. Punt (a) en (c) zijn beide meetbaar met het bestaande
harnas zodra de lopende meting klaar is.
