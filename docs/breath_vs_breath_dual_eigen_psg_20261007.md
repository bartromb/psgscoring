# `aasm_v3_breath` tegen `aasm_v3_breath_dual` op 20 eigen diagnostische PSG's

Datum: 2026-10-07. Preregistratie `docs/breath_vs_breath_dual_eigen_psg_preregistratie_20261007.md` (c59652f, geschreven vóór de meting). In situ op de Hetzner-server in een losse container uit het productie-image YASAFlaskified 0.38.9 (psgscoring **0.34.0**), uploads read-only, 16 CPU's, 4 analyses parallel, 39 min; de app bleef bereikbaar. Per opname identieke invoer (opgeslagen YASA-hypnogram, opgeslagen artefact-epochs, dezelfde kanaalkeuze en inleesroute als de worker), alleen het profiel verschilt. Steekproef 20 van 159 kandidaten (seed 20261007, 7/7/6 per tertiel van de opgeslagen AHI, grens laag/midden 7,1 /u, grens midden/hoog tussen 23,1 en 23,2 /u — de preregistratie schrijft 23,1, de toewijzing is dezelfde: R09 op 23,1 valt in midden, R02 op 23,2 in hoog). Koppeling naar de jobs blijft op de server; hier alleen R01–R20. Ruwe uitvoer: `/srv/CODE/docs/breath_dual_20261007/R*.json` (geen patiëntgegevens). **Verschilmeting zonder referentie**: er is geen menselijke scoring van deze nachten.

## Resultaat per opname

| R | tertiel | AHI breath | AHI dual | ΔAHI | ernst breath → dual | apneus breath (O/C/M) → dual | hypopneeën breath → dual |
|---|---|---|---|---|---|---|---|
| R01 | hoog | 55,9 | 55,9 | +0,0 | ernstig → ernstig | 201/0/0 → 202/0/1 | 17 → 15 |
| R02 | hoog | 17,3 | 23,2 | +5,9 | matig → matig | 23/0/0 → 105/2/0 | 102 → 61 |
| R03 | midden | 14,5 | 14,5 | +0,0 | licht → licht | 3/0/0 → 3/0/0 | 29 → 29 |
| R04 | midden | 6,6 | 7,8 | +1,2 | licht → licht | 21/0/0 → 28/0/0 | 12 → 11 |
| R05 | laag | 1,4 | 1,4 | +0,0 | normaal → normaal | 0/0/0 → 0/0/0 | 11 → 11 |
| R06 | laag | 4,6 | 4,6 | +0,0 | normaal → normaal | 2/0/0 → 2/0/0 | 6 → 6 |
| R07 | laag | 2,5 | 2,5 | +0,0 | normaal → normaal | 0/0/0 → 0/0/0 | 4 → 4 |
| R08 | midden | 8,3 | 8,3 | +0,0 | licht → licht | 22/0/0 → 22/0/0 | 29 → 29 |
| R09 | midden | 29,9 | 29,9 | +0,0 | matig → matig | 36/0/0 → 36/0/0 | 26 → 26 |
| R10 | midden | 17,9 | 17,9 | +0,0 | matig → matig | 101/0/0 → 101/0/0 | 44 → 44 |
| R11 | midden | 13,4 | 22,1 | +8,7 | licht → matig | 0/0/0 → 65/3/0 | 65 → 39 |
| R12 | hoog | 47,5 | 47,5 | +0,0 | ernstig → ernstig | 27/7/0 → 27/7/0 | 53 → 53 |
| R13 | midden | 9,3 | 10,0 | +0,7 | licht → licht | 20/1/0 → 23/0/0 | 5 → 5 |
| R14 | hoog | 90,7 | 90,7 | +0,0 | ernstig → ernstig | 158/0/0 → 158/0/0 | 145 → 145 |
| R15 | laag | 2,6 | 2,6 | +0,0 | normaal → normaal | 6/0/0 → 6/0/0 | 7 → 7 |
| R16 | hoog | 24,5 | 24,5 | +0,0 | matig → matig | 71/2/0 → 71/2/0 | 48 → 48 |
| R17 | laag | 4,9 | 5,1 | +0,2 | normaal → licht | 3/0/0 → 4/0/0 | 27 → 27 |
| R18 | hoog | 66,0 | 103,3 | +37,3 | ernstig → ernstig | 30/2/0 → 439/2/0 | 296 → 72 |
| R19 | laag | 1,3 | 1,3 | +0,0 | normaal → normaal | 1/0/0 → 1/0/0 | 11 → 11 |
| R20 | laag | 4,8 | 6,0 | +1,2 | normaal → licht | 5/0/0 → 12/0/0 | 23 → 23 |

## Lezing volgens de preregistratie

- **ΔAHI (dual − breath): mediaan 0,0, gemiddeld +2,8, bereik 0,0 … +37,3 /u; Wilcoxon p = 0,018 (normale benadering; exact 2/2⁷ = 0,0156 — er zijn maar 7 niet-nul verschillen en ze zijn alle positief, de toets zegt dus alleen dat het teken consistent is, niets over de grootte).** Op 13 van 20 opnames is de AHI gelijk, op 10 daarvan ook de apneu- en hypopneetelling; op de 7 andere geeft `breath_dual` een hogere AHI, nooit een lagere. **Op elk veld identiek is géén enkele opname**: RDI, RERA en FRI verschillen op 18/20 en de ventilatoire last op 5/20 (zie de sectie na het mechanisme).
- **Ernstklasse wisselt op 3/20:** R11 (licht → matig), R17 (normaal → licht), R20 (normaal → licht).
- **|ΔAHI| ≥ 5 /u op 3/20:** R02, R11, R18 (+5,9, +8,7 en +37,3).
- **Per tertiel** (opgeslagen AHI): laag n=7 gemiddeld +0,2 (max +1,2); midden n=7 gemiddeld +1,5 (max +8,7); hoog n=6 gemiddeld +7,2 (max +37,3). Het verschil is op elk tertiel mediaan nul en komt uit enkele opnames.
- De vooraf opgeschreven lezing "mediaan |ΔAHI| < 1 én geen ernstklasse-wissel → klinisch onverschillig" gaat **niet** op: de mediaan is nul, maar drie opnames wisselen van klasse en één opname verschilt 37 /u. De keuze is dus niet onverschillig; ze raakt een minderheid van de nachten hard.

## Mechanisme: de duale regel is een vereniging, geen bevestiging

`breath_dual` erft van `breath` en zet in `profiles.py` `_with_dual_apneas` **vier** dingen om (niet één): `dual_sensor_apnea=True`, `dual_sensor_corroboration=False`, `flow_reference="hypopnea"` en `thermistor_gate="respiratory_band"`, familie `exploratory`. Voor de AHI telt de eerste: apneus worden op **beide** flowsensoren gedetecteerd en samengevoegd (`corroborate_apnea_events`, `corroboration_licensed = False`, `keep_thermistor_only = keep_pressure_only = True`: n_kept = beide + alleen-thermistor + alleen-druk). De tweede sensor is dus **additief** — hij voegt apneus toe en verwijdert er nooit een (`profiles.py` `_with_dual_apneas`). Welke sensor `breath` zelf voor de apneus gebruikt, beslist de thermistorpoort (`envelope_agreement`; de drempel 0,40 is de constante `THERMISTOR_AGREEMENT_MIN` in `signal_quality.py`, het profiel kiest alleen de poortsoort, env-override `PSGSCORING_THERMISTOR_GATE`; de herhaling toont in productie `envelope_agreement` / 0,4 / `blocking`): poort **aan** → apneus op de thermistor (AASM-voorkeurssensor), poort **af** → apneus op de druk. Onder `breath_dual` staat de poort op `informational` en wordt de thermistor nooit verworpen (`pipeline.py`, `additive_thermistor`): de primaire detectiepas draait op alle 20 op de thermistor, de tweede pas op de druk, en de hypopneeën komen op beide profielen van de druk. De herhaling van de breath-arm en de mechanisme-tabel hieronder staan **niet in de preregistratie** (post-hoc, beschrijvend; de preregistratie veronderstelde ten onrechte dat de duale regel apneus degradeert of laat vallen — ze voegt alleen toe). Uit de herhaling van de breath-arm met vastgelegde `meta.flow_channels` (AHI op alle 20 gelijk aan de eerste run):

| R | thermistorpoort | overeenstemming | apneusensor `breath` | apneus thermistor / druk (dual-arm) | alleen-druk toegevoegd | alleen-thermistor toegevoegd | ΔAHI |
|---|---|---|---|---|---|---|---|
| R01 | af | 0,34 | druk | 83 / 201 | +120 | +2 | +0,0 |
| R02 | aan | 0,50 | thermistor | 23 / 108 | +85 | +0 | +5,9 |
| R03 | af | -0,01 | druk | 0 / 3 | +3 | +0 | +0,0 |
| R04 | aan | 0,60 | thermistor | 21 / 16 | +7 | +12 | +1,2 |
| R05 | af | 0,07 | druk | 0 / 0 | +0 | +0 | +0,0 |
| R06 | af | 0,09 | druk | 0 / 2 | +2 | +0 | +0,0 |
| R07 | aan | 0,42 | thermistor | 0 / 0 | +0 | +0 | +0,0 |
| R08 | af | -0,03 | druk | 0 / 23 | +23 | +0 | +0,0 |
| R09 | af | 0,32 | druk | 0 / 36 | +36 | +0 | +0,0 |
| R10 | af | 0,13 | druk | 0 / 103 | +103 | +0 | +0,0 |
| R11 | aan | 0,53 | thermistor | 0 / 68 | +68 | +0 | +8,7 |
| R12 | af | -0,15 | druk | 0 / 34 | +34 | +0 | +0,0 |
| R13 | af | 0,20 | druk | 6 / 21 | +17 | +2 | +0,7 |
| R14 | af | 0,10 | druk | 0 / 158 | +158 | +0 | +0,0 |
| R15 | af | 0,38 | druk | 0 / 6 | +6 | +0 | +0,0 |
| R16 | af | 0,07 | druk | 0 / 73 | +73 | +0 | +0,0 |
| R17 | af | -0,45 | druk | 1 / 3 | +3 | +1 | +0,2 |
| R18 | aan | 0,45 | thermistor | 32 / 458 | +426 | +0 | +37,3 |
| R19 | af | 0,03 | druk | 0 / 1 | +1 | +0 | +0,0 |
| R20 | af | 0,33 | druk | 8 / 5 | +4 | +7 | +1,2 |

- **Poort aan (5/20: R02, R04, R07, R11, R18):** `breath` scoort de apneus op de thermistor en vindt er precies zoveel als de thermistor-detector (23, 21, 0, 0, 32). De druk ziet er veel meer (108, 16, 0, 68, 458); `breath_dual` voegt die alleen-druk-apneus toe. Dat zijn de drie grote verschillen (R02 +5,9, R11 +8,7, R18 +37,3) en R04 (+1,2); R07 heeft op geen van beide sensoren apneus. Op R11 en R18 vindt de thermistor 0 van 68 en 32 van 458 — dezelfde sensorblindheid van de apneudrempel die op MESA gemeten is (thermistor haalt een fractie van wat de druk haalt bij gelijke AUC; `project_apnea_threshold_sensor`). De poort keurt de thermistor hier goed op overeenstemming 0,4–0,6 en levert hem dan uit aan een drempel die er weinig uithaalt.
- **Poort af, thermistor levert toch iets (4/20: R01, R13, R17, R20):** `breath` scoort op de druk; `breath_dual` voegt 1–7 alleen-thermistor-apneus van de afgekeurde sensor toe (ΔAHI 0,0–1,2; R17 en R20 wisselen daardoor van normaal naar licht rond de grens van 5 /u).
- **Poort af, thermistor leeg (11/20):** beide armen identiek.
- De eerste run en de herhaling van de breath-arm geven op alle 20 dezelfde AHI: de meting is reproduceerbaar binnen het image. De 12 productiejobs die op 0.34.0 draaiden, hebben een opgeslagen AHI die exact gelijk is aan de arm van hun profiel (12/12); de 4 afwijkers (R07, R09, R17, R19) zijn jobs van oudere of onbekende versie.
- De O/C/M-kolommen laten de `uncertain`-apneus weg (die tellen niet in `ahi_total`); mét die apneus sluiten alle 20 tellingen exact: breath = sensortelling, dual = n_kept. Op R18 zijn de 426 toegevoegde druk-apneus 409 obstructief + 17 `uncertain`; `ahi_incl_uncertain` stijgt daar +40,7 (66,0 → 106,7) in plaats van +37,3.

## Buiten de AHI: RDI, RERA, FRI en ventilatoire last verschuiven wél, en niet altijd omhoog

Post-hoc, buiten de vooraf vastgelegde lezing (die ging alleen over AHI, events en ernstklasse); gevonden door de onafhankelijke verificatie. De andere drie schakelaars van `_with_dual_apneas` werken hier door: omdat de thermistor onder `breath_dual` nooit wordt verworpen, draait de primaire pas op de thermistor en komt de lijst afgewezen hypopneeën — de bron van FRI en RERA en dus van de RDI — van een andere sensor dan onder `breath` (waar de poort op 15/20 de druk kiest). En `flow_reference="hypopnea"` legt het referentiesignaal voor sweep, anker, arousal-analyse, CSR en ventilatoire last op de druk, terwijl `breath` bij poort-aan de thermistor als referentie neemt — vandaar de VB-sprong op precies de 5 poort-aan-nachten.

| R | RDI breath → dual | ΔRDI | RERA | FRI | ventilatoire last (%) |
|---|---|---|---|---|---|
| R01 | 56,9 → 57,4 | +0,5 | 4 → 6 | 68 → 59 | 62,6 → 62,6 |
| R02 | 22,1 → 28,3 | +6,2 | 35 → 37 | 160 → 156 | 3,8 → 32,0 |
| R03 | 36,2 → 31,3 | -4,9 | 48 → 37 | 95 → 29 | 23,3 → 23,3 |
| R04 | 8,0 → 9,2 | +1,2 | 7 → 7 | 79 → 78 | 15,4 → 17,8 |
| R05 | 8,5 → 3,4 | -5,1 | 54 → 15 | 229 → 57 | 8,8 → 8,8 |
| R06 | 13,7 → 8,6 | -5,1 | 16 → 7 | 85 → 38 | 21,3 → 21,3 |
| R07 | 8,1 → 8,1 | +0,0 | 9 → 9 | 15 → 15 | 0,0 → 14,8 |
| R08 | 17,1 → 12,1 | -5,0 | 54 → 23 | 218 → 87 | 19,4 → 19,4 |
| R09 | 32,8 → 34,2 | +1,4 | 6 → 9 | 61 → 21 | 33,8 → 33,8 |
| R10 | 26,6 → 22,0 | -4,6 | 71 → 33 | 285 → 137 | 36,8 → 36,8 |
| R11 | 19,2 → 28,3 | +9,1 | 28 → 30 | 156 → 152 | 6,9 → 53,5 |
| R12 | 55,1 → 58,4 | +3,3 | 14 → 20 | 46 → 46 | 70,2 → 70,2 |
| R13 | 17,9 → 17,1 | -0,8 | 24 → 20 | 121 → 60 | 39,0 → 39,0 |
| R14 | 98,8 → 98,5 | -0,3 | 27 → 26 | 11 → 5 | 74,3 → 74,3 |
| R15 | 9,3 → 5,5 | -3,8 | 34 → 15 | 290 → 71 | 14,2 → 14,2 |
| R16 | 31,8 → 30,0 | -1,8 | 36 → 27 | 180 → 103 | 41,1 → 41,1 |
| R17 | 5,6 → 5,6 | +0,0 | 4 → 3 | 160 → 65 | 15,5 → 15,5 |
| R18 | 71,0 → 109,9 | +38,9 | 25 → 33 | 55 → 75 | 27,3 → 68,2 |
| R19 | 7,2 → 3,4 | -3,8 | 53 → 19 | 298 → 37 | 7,4 → 7,4 |
| R20 | 11,8 → 7,7 | -4,1 | 41 → 10 | 182 → 50 | 12,7 → 12,7 |

- **ΔRDI (dual − breath): mediaan -0,55, bereik -5,1 … +38,9 /u; lager op 11/20, hoger op 7/20, gelijk op 2/20; Wilcoxon p = 0,47 (n = 18),** De AHI-stelling "dual neemt nooit iets weg" geldt dus voor apneus en AHI, niet voor de RDI: op 11 nachten daalt de RDI onder `breath_dual`, tot −5,1 /u (R05 8,5 → 3,4, R06 13,7 → 8,6, R08 17,1 → 12,1), doordat de FRI-lijst van de thermistorpas kleiner is,
- **Ventilatoire last** verschilt op de 5 poort-aan-nachten: R02 3,8 → 32,0, R04 15,4 → 17,8, R07 0,0 → 14,8, R11 6,9 → 53,5, R18 27,3 → 68,2; op de 15 andere identiek.
- Hypopneu-subtypes verschuiven op R12 (centraal 7 → 4) en R16 (centraal 1 → 0, gemengd 2 → 0) bij gelijke totalen.
- RDI is een gerapporteerd klinisch getal. Wie `breath_dual` kiest om de apneu-onderdetectie van de thermistor te ondervangen, krijgt er een andere FRI/RERA-bron en een andere ventilatoire-lastreferentie bij; of dat gewenst is, is een aparte vraag die deze meting niet beantwoordt.

## Wat dit betekent

- `breath_dual` kan op deze montages alleen apneus **toevoegen**, nooit wegnemen; hypopneeën nemen af doordat events naar apneu verschuiven, en de AHI stijgt waar de tweede sensor apneus ziet die de eerste niet zag. Buiten de AHI is het profiel géén zuivere toevoeging: RDI daalt op 11/20, de ventilatoire last springt op de poort-aan-nachten. Of die extra apneus juist zijn, kan deze meting niet zeggen — daarvoor is een menselijke scoring van dezelfde nachten nodig (de EC-studie AZORG-YASA-2026-001 levert die).
- Voor de standaardkeuze van vandaag (`breath` voor alle scoorders) betekent dit: op 13 van 20 diagnostische nachten geen AHI-verschil (10 ook gelijk in de tellingen), op 7 geeft `breath` de lagere AHI. Het risico zit precies op de nachten waar de thermistorpoort de thermistor goedkeurt: daar vertrouwt `breath` de apneus aan een sensor toe die er een fractie van vindt, en één nacht zakt daardoor van 103 naar 66 /u. `breath_dual` vangt dat op zonder een apneu weg te nemen, maar met een andere RDI (lager op 11/20) en een andere ventilatoire last op die nachten. Of dat de standaard moet worden op dual-sensor-montages, is Barts beslissing — de meting zegt niet welke telling juist is, alleen waar en waarom de twee uiteenlopen.
- Niets aan bibliotheek, profielen of configuratie is gewijzigd.

## Verificatie

Onafhankelijk nagerekend (meting-verificatie, 07-10) uit R01–R20.json, R*_meta.json en de code op v0.34.0 (profiles/pipeline/postprocess byte-gelijk aan 0.34.2): alle AHI's, tellingen, wissels, tertielen en de mechanisme-tabel kloppen op 20/20; de prereg is niet gewijzigd na de start (c59652f 13:30, analyse.py 13:33, uitvoer 14:12). Verwerkt uit de verificatie: "10 volledig identiek" → "10 gelijk in AHI en tellingen, 0 op elk veld"; de vier profielschakelaars; de RDI/RERA/FRI/VB-sectie; 1–7 i.p.v. 1–8; exacte Wilcoxon; `uncertain`-apneus; tertielgrens; post-hoc-label voor de herhaling. Niet verifieerbaar vanaf de werkstation: looptijd, parallelisme en app-bereikbaarheid (app.log op de server), de 159-kandidatenlijst en de seed-trekking, en de hypnogram-identiteit per arm (in run.py op de server).
