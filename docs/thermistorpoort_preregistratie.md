# Preregistratie — welke poort beslist of apneus op de thermistor gescoord worden

*Opgesteld 22-08-2026, vóór enige uitkomst op MESA. De verkenning van de
doorlaatpercentages liep nog toen dit werd vastgelegd.*

## Wat er nu gebeurt

`assess_flow_sensor_agreement` bepaalt op **elke** opname of apneus op de
thermistor of op de neusdruk gescoord worden. Negentien van de eenentwintig
profielen gebruiken hem, waaronder `aasm_v3_rec`, `aasm_v3_breath` en
`aasm_v3_prob`. Alleen `aasm_v3_breath_dual` en `aasm_v3_prob_dual` staan op
`respiratory_band`.

De maat is op 06-08-2026 weerlegd als maat voor de vraag die hij beantwoordt.
Zes synthetische signalen die **allemaal op 0,25 Hz ademen** geven agreement
1,000 tot −0,985, afhankelijk van hun trage amplitudemodulatie:

| geval | agreement |
|---|---:|
| identieke amplitudemodulatie | 1,000 door |
| thermistor 1 s vertraagd | 1,000 door |
| modulatie 90° verschoven | 0,002 af |
| geen modulatie op de thermistor | 0,094 af |
| modulatie in tegenfase | −0,985 af |

De maat reageert uitsluitend op overeenstemming in trage amplitudemodulatie
en is blind voor de vraag of beide sensoren dezelfde ademhaling zien. Op negen
lokale montages haalde **één** de drempel, terwijl vier spectraal gemeten
opnames exact dezelfde ademfrequentie op beide sensoren lieten zien en de
thermistor het schónere ademspectrum had.

Dat maakt de poort verdacht, **niet aantoonbaar schadelijk**: een poort kan
op de verkeerde grond oordelen en tóch de betere sensorkeuze maken. Dat is wat
hier gemeten wordt.

## De vraag

Scoort `respiratory_band` — een toets op één kanaal, adembandvermogen ≥ 0,70 —
dichter bij menselijke scoring dan `envelope_agreement`?

## Opzet

- **Cohort:** MESA, gepaarde run over dezelfde opnames, seed vastgelegd in het
  runscript. MESA heeft `Pres` én `Therm`, dus de poort draait echt.
- **Armen:** `PSGSCORING_THERMISTOR_GATE=envelope_agreement` tegen
  `respiratory_band`. Env-override, zodat de registry niet muteert en beide
  armen aantoonbaar op hetzelfde profiel draaien.
- **Profiel:** `aasm_v3_breath`.
- **Referentie:** de MESA-annotaties (apneus en hypopneus).
- **Maten:** event-F1 met greedy IoU-koppeling op 0,20, en AHI-bias.

## Beslisregel — vastgelegd vóór de meting

`respiratory_band` wordt default op de niet-gepinde v3-profielen **alleen
als beide** waar zijn:

1. mediane gepaarde ΔF1 ≥ **+0,010**, en
2. de mediane gepaarde AHI-bias verslechtert met niet meer dan **1,0/u** in
   absolute waarde.

**Weerlegd** als de mediane gepaarde ΔF1 ≤ 0. Daartussen: blijft opt-in, en
ik rapporteer het als onbeslist in plaats van het alsnog aan te zetten.

Gepind op `envelope_agreement`, ongeacht de uitkomst: `mesa_shhs`,
`chicago_1999`, `cms_medicare`, `aasm_v1_rec`, `aasm_v2_rec`. Die reproduceren
een externe regelset of paper v31/v37.

## Meldgrens

Verschuift de **AHI-ernstklasse** op meer dan een kwart van de opnames, dan
leg ik dat als apart punt voor in plaats van het in een verslag te vermelden.
Dezelfde grens als bij de RDI-karakterisering, en om dezelfde reden: een
sensorwissel die een kwart van de patiënten herklasseert is een beslissing,
geen detail.

## Wat deze meting NIET kan uitwijzen

Of de poort op de júiste grond oordeelt. Beide poorten worden hier op hun
uitkomst afgerekend, niet op hun redenering. Een poort die om de verkeerde
reden de goede sensor kiest, wint hier — en dat is voor een klinische keuze
verdedigbaar, maar het betekent dat een winst hier de synthetische weerlegging
van 06-08-2026 niet ongedaan maakt.

Evenmin uit te wijzen: hoe dit op de lokale AZORG-montages uitpakt. Daar is
geen menselijke referentie. De negen montages tonen alleen dat de poorten
verschillend oordelen, niet wie gelijk heeft.

## Derde poort

`breath_coherence` bestaat ook en wordt door geen enkel profiel gebruikt. Die
blijft hier buiten beschouwing: twee armen tegelijk vergelijken met één
beslisregel maakt de regel dubbelzinnig. Het doorlaatpercentage wordt wel
gerapporteerd.

---

## Addendum, vastgelegd vóór de uitkomst: de drempel van `respiratory_band`
## reproduceert zijn eigen onderbouwing niet op MESA

`THERMISTOR_BAND_POWER_MIN = 0,70` is niet gekozen maar afgeleid, en de code
zegt er netjes bij hoe: op negen lokale Somnomedics-montages lagen de waarden
in twee klassen met een leeg gat van 0,53 ertussen —

    0,982  0,981  0,977  0,970   |   0,441  0,396  0,318  0,036  0,000

en 0,70 ligt op het midden daarvan. Dat is precies de methode die deze
codebase elders ook gebruikt: leg de drempel in het gat, niet op een rond
getal.

**Op MESA is dat gat er niet.** Twaalf willekeurige opnames (seed 20260822):

    0,506  0,636  0,687 | 0,703  0,717  0,774  0,802  0,816  0,874  0,913
    0,966  0,985

Het grootste gat is 0,130 (tussen 0,506 en 0,636). De drempel valt tussen
**0,687 en 0,703** — een opening van 0,016 — en **3 van de 12** opnames liggen
binnen 0,05 van de drempel.

**Wat dat betekent voor deze meting.** Voor ongeveer een kwart van de MESA-
opnames is de beslissing van `respiratory_band` effectief willekeurig: een
verwaarloosbare verschuiving van de drempel draait ze om. De poort kan dus
winnen of verliezen op opnames waar hij eigenlijk niets beslist. Dat maakt de
meting niet ongeldig — de vraag is welke poort tot betere scoring leidt, en dat
blijft meetbaar — maar het bepaalt hoe hard de uitkomst mag worden gelezen.

**Wat ik hier bewust NIET doe:** de drempel op MESA herijken. Het grootste
MESA-gat ligt bij ~0,57 en zou 11 van de 12 doorlaten, wat er aantrekkelijk
uitziet. Maar dan zou ik de drempel afstellen op hetzelfde cohort waarop ik
hem vervolgens meet, en dat is een fit die zich als validatie voordoet —
dezelfde fout die `hypopnea_strictness` op PSG-IPA maakte en waarvoor deze
MESA-validatie juist bestaat. Als de drempel herijkt moet worden, hoort dat op
een steekproef die van de meetopnames gescheiden is, met een eigen
preregistratie.

---

# Uitkomst — 22 augustus 2026

**WEERLEGD.** `respiratory_band` blijft opt-in; `envelope_agreement` blijft de
default op de negentien profielen die hem gebruiken.

Gepaard over 50 MESA-opnames, `aasm_v3_breath`, seed 20260801, psgscoring
0.25.0.

| | envelope | band | verschil |
|---|---:|---:|---:|
| F1 mediaan (van de verdelingen) | 0,499 | 0,510 | +0,011 |
| **gepaarde ΔF1, mediaan** | | | **0,0000** |
| AHI-bias mediaan | −4,19 | −5,13 | −0,95 |
| \|AHI-bias\| mediaan | 6,67 | 6,81 | +0,13 |

Gepaard: beter op 4, slechter op 19, gelijk op 27. Wilcoxon over de 23
niet-nul verschillen: **p = 0,011** — significant slechter, niet gelijk.

**Let op het verschil tussen de twee eerste regels.** Het verschil van de
medianen is +0,011 en haalt criterium 1; de mediaan van de gepaarde
verschillen is 0,0000 en haalt het niet. De preregistratie zegt *gepaarde*
ΔF1, en dat is ook de juiste maat: de eerste regel vergelijkt twee verdelingen
alsof het losse steekproeven zijn, terwijl elke opname in beide armen zit. Wie
alleen naar de eerste regel keek, zou deze poort hebben aangenomen.

Beslisregel: mediane gepaarde ΔF1 ≤ 0 → **weerlegd**.

## Het mechanisme, en waarom het klopt

`respiratory_band` laat meer thermistors door, en apneus worden dan op de
thermistor gescoord in plaats van op de neusdruk. Op de 24 opnames waar er
iets verandert:

| | envelope | band |
|---|---:|---:|
| apneus, totaal over 50 | **2223** | **1223** |
| hypopneus, totaal over 50 | 3248 | 3564 |

De apneudetectie **halveert**. Een deel schuift door naar hypopneu (+316),
maar lang niet alles: netto verdwijnen er events, wat de AHI-bias negatiever
maakt (−4,19 → −5,13).

De thermistor is dus als apneusensor slechter dan de neusdruk op dit cohort —
precies omgekeerd aan wat de AASM-voorkeur voor een thermische opnemer
suggereert. Dat kan aan het MESA-`Therm`-kanaal liggen of aan een
apneudetector die op druksignalen is afgesteld; dat is met deze meting niet
te scheiden.

## Wat dit NIET zegt

Dat `envelope_agreement` een goede maat is. Die blijft aantoonbaar de
verkeerde eigenschap meten — zes signalen die allemaal op 0,25 Hz ademen
scoren van −0,985 tot +1,000 — en hij keurt op MESA aantoonbaar ademende
kanalen af (0,131 bij een bandvermogen van 0,985). Hij wint hier omdat hij
vaker de neusdruk kiest, en de neusdruk blijkt de betere apneusensor. **Een
poort die om de verkeerde reden de goede sensor kiest.**

Dat is voor een klinische default verdedigbaar en het is de reden dat er niets
verandert. Het is géén reden om de maat als juist te boeken.

## Een defect dat deze meting blootlegde

Het harnas rapporteerde **nul apneus op alle 50 opnames**, ook bij een AHI van
47. `n_apnea` zocht op de substring `"apnea"` in het eventtype, en de
bibliotheek schrijft dat woord nergens: apneus heten `obstructive`, `central`,
`mixed` en `uncertain`. Werkelijk gedetecteerd: 2223.

Alleen boekhouding — `match`, `ahi` en `n_events` lopen niet langs dat veld,
dus eerdere F1- en bias-cijfers uit dit harnas zijn ongemoeid. Maar elke
uitspraak over apneu-AANTALLEN uit dit harnas was fout, en mijn eerste
mechanisme-analyse hierboven was er ook op gebaseerd: ik concludeerde eerst
dat de poort de apneutelling niet raakte. Het omgekeerde is waar.

Gerepareerd, met een test die de typewoordenschat van het harnas tegen die van
`respiratory.py` houdt.

## Wat hierna zou moeten

Niet: deze poort nog eens proberen. Wel, als de apneu-as verder moet:
waaróm de thermistor als apneusensor verliest. Dat is een vraag over de
detector, niet over de poort, en hij raakt de AASM-conformiteit — wij scoren
apneus liever op druk terwijl de regel een thermische opnemer voorschrijft.
