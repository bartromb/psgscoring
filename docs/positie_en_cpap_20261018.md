# Twee productiedefecten uit de laatste tien uploads (18-10-2026)

Aanleiding: Bart zag dat de houding niet altijd klopt ("STA wordt soms als rugligging
gescoord") en dat de laatste upload, een CPAP-nacht, een OAHI van 21/u kreeg waar de
manuele scoring < 4/u geeft. Controle in situ op de server (losse container uit
`yasaflaskified:0.38.11`, uploads read-only, alleen getallen naar buiten; scripts in
`/data/slaapkliniek/metingen/positie_20261018/`), tien uploads van 18-10 tussen 10:50 en 15:09,
alle `aasm_v3_breath_dual`, psgscoring 0.36.1.

## 1. Houding: op alle tien nachten de percentielmethode

Het Somnomedics-kanaal `Pos.` is een **code op 4 Hz** (EDF-kop: fysiek 0–10 mV, digitaal
0–10, dus gehele getallen). In de tien nachten komen de codes 1, 2, 3, 4 en 6 voor; 5 vrijwel
nooit; code 6 is het vaakst (29–71 % van de nacht), code 1 zeldzaam (0–3 %).

| | |
|---|---|
| methode in `results.json` (`position_mapping_method`) | **`percentile` op 10/10** |
| `Pos.` apart geladen (4 Hz) | precies 6 waarden |
| `Pos.` in de volledige lading (zoals de app laadt) | **606 verschillende waarden**, 99,3 % binnen 0,05 mV van een geheel getal |
| `_position_levels` op die lading | één plateau over het hele bereik → `None` → percentiel |

Oorzaak: MNE brengt het 4 Hz-kanaal bij de volledige lading met een FFT naar 256 Hz; de
ringing op elke overgang legt een dunne brug tussen de niveaus en de gap-gebaseerde
niveaudetectie zag daardoor één breed plateau. De percentielmethode verdeelt de nacht dan op
duurkwantielen — precies de duurafhankelijkheid die de niveaudetectie moest wegnemen. Voorbeelden
uit de tien: 38,6 % "Upright" en 0 % "Right"; 71 % in één code en toch 36 % "Upright".

Tweede laag: ook met goede niveaus kent de code oplopende waarden toe als Prone, Left, Supine,
Right, Upright. Dat is een aanname, geen Somnomedics-tabel. **De legende van `Pos.`
(code → STA/rug/links/rechts/buik) is aan Bart gevraagd.**

Reparatie (psgscoring 0.36.2): niveaudetectie op histogrampieken (ongevoelig voor de
ringing), een optionele fabrikantentabel `position_code_map` (methode wordt dan `coded`; de
pipeline schaalt naar de EDF-eenheid), en in YASAFlaskified de tabel uit de jobconfig of de
installatie-env `YASAFLASKIFIED_POSITION_CODE_MAP`. Zonder tabel blijft het een rangorde-gok
met een eerlijk label en het bestaande voorbehoud in het PDF.

## 2. CPAP-nacht: de dode thermistor werd de apneusensor

Job `cff16f03` (upload `ANON_C26028BC.edf`, 40 kanalen, met een drukkanaal `Pressur` in
mbar dat de andere nachten niet hebben):

| | |
|---|---|
| CPAP-druk | ~0 mbar in uur 0–1, **7–9 mbar vanaf uur 2** |
| `Flow Th.` | MAD **0,000** in elk uur (vlak onder het masker) |
| thermistorpoort (`respiratory_band`, informatief) | `band_power 0,00`, `usable: false`, "draagt overwegend geen ademhaling" |
| wat er toch gebeurde | additieve modus: thermistor **wél** gebruikt, `apnea_sensor = Flow Th.` |
| apneus | 166, waarvan **165 `thermistor_only`**; ~20/u in uur 2–9; 55 zonder desaturatie ≥ 3 % |
| split-night-detector | niet aangeslagen (kijkt naar flowamplitude + SpO2, niet naar druk) |

Herscoring met dezelfde staging, artefact-epochs en kanaalkeuze:

| profiel | apneusensor | AHI | OAHI | events |
|---|---|---|---|---|
| `aasm_v3_breath_dual` (zoals productie) | Flow Th. (dood) | 21,5 | 21,5 | 166 apneus + 7 hypopneeën |
| `aasm_v3_breath` (blokkerende poort → thermistor afgewezen) | Pressure Flow | **1,0** | **1,0** | 1 apneu + 7 hypopneeën |
| `aasm_v3_rec` (blokkerende poort) | Pressure Flow | 0,7 | 0,7 | 1 apneu + 5 hypopneeën |

De manuele scoring (< 4/u) klopt; het verschil is volledig de dode sensor. Reparatie: een
sensor onder de dode vloer (`THERMISTOR_BAND_POWER_DEAD = 0,20`: vlak 0,00, witte ruis
analytisch 0,10, drift 0,17) wordt ook in additieve modus uitgesloten, met reden in
`flow_channels.thermistor_check`. Daarnaast meldt `meta.cpap_pressure` voortaan een
drukkanaal dat ≥ 30 min boven 4 mbar staat (informatief; de nacht blijft gescoord zoals
gevraagd, maar het verslag kan zeggen dat het een titratie was).

Nog open: de split-night-logica op het drukkanaal laten aanslaan (therapiesegment apart
indexeren), en de legende van `Pos.`.
