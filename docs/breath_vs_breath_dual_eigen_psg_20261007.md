# `aasm_v3_breath` tegen `aasm_v3_breath_dual` op 20 eigen diagnostische PSG's

Datum: 2026-10-07. Preregistratie `docs/breath_vs_breath_dual_eigen_psg_preregistratie_20261007.md` (c59652f, geschreven vóór de meting). In situ op de Hetzner-server in een losse container uit het productie-image YASAFlaskified 0.38.9 (psgscoring **0.34.0**), uploads read-only, 16 CPU's, 4 analyses parallel, 39 min; de app bleef bereikbaar. Per opname identieke invoer (opgeslagen YASA-hypnogram, opgeslagen artefact-epochs, dezelfde kanaalkeuze en inleesroute als de worker), alleen het profiel verschilt. Steekproef 20 van 159 kandidaten (seed 20261007, 7/7/6 per tertiel van de opgeslagen AHI, grenzen 7,1 en 23,2 /u). Koppeling naar de jobs blijft op de server; hier alleen R01–R20. Ruwe uitvoer: `/srv/CODE/docs/breath_dual_20261007/R*.json` (geen patiëntgegevens). **Verschilmeting zonder referentie**: er is geen menselijke scoring van deze nachten.

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

- **ΔAHI (dual − breath): mediaan 0,0, gemiddeld +2,8, bereik 0,0 … +37,3 /u; Wilcoxon p = 0.018.** Op 10 van 20 opnames zijn de twee armen op elk veld identiek; op de overige 10 scoort `breath_dual` méér (nooit minder).
- **Ernstklasse wisselt op 3/20:** R11 (licht → matig), R17 (normaal → licht), R20 (normaal → licht).
- **|ΔAHI| ≥ 5 /u op 3/20:** R02, R11, R18 (+5,9, +8,7 en +37,3).
- **Per tertiel** (opgeslagen AHI): laag n=7 gemiddeld +0,2 (max +1,2); midden n=7 gemiddeld +1,5 (max +8,7); hoog n=6 gemiddeld +7,2 (max +37,3). Het verschil is op elk tertiel mediaan nul en komt uit enkele opnames.
- De vooraf opgeschreven lezing "mediaan |ΔAHI| < 1 én geen ernstklasse-wissel → klinisch onverschillig" gaat **niet** op: de mediaan is nul, maar drie opnames wisselen van klasse en één opname verschilt 37 /u. De keuze is dus niet onverschillig; ze raakt een minderheid van de nachten hard.

## Mechanisme: de duale regel is een vereniging, geen bevestiging

`breath_dual` erft alles van `breath` en zet alleen `dual_sensor_apnea` aan: apneus worden op **beide** flowsensoren gedetecteerd en samengevoegd (`corroborate_apnea_events`, `corroboration_licensed = False`, `keep_thermistor_only = keep_pressure_only = True`: n_kept = beide + alleen-thermistor + alleen-druk). De tweede sensor is dus **additief** — hij voegt apneus toe en verwijdert er nooit een (`profiles.py` `_with_dual_apneas`). Welke sensor `breath` zelf voor de apneus gebruikt, beslist de thermistorpoort (`envelope_agreement`, drempel uit het profiel): poort **aan** → apneus op de thermistor (AASM-voorkeurssensor), poort **af** → apneus op de druk. Uit de herhaling van de breath-arm met vastgelegde `meta.flow_channels` (AHI op alle 20 gelijk aan de eerste run):

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
- **Poort af, thermistor levert toch iets (4/20: R01, R13, R17, R20):** `breath` scoort op de druk; `breath_dual` voegt 1–8 alleen-thermistor-apneus van de afgekeurde sensor toe (ΔAHI 0,0–1,2; R17 en R20 wisselen daardoor van normaal naar licht rond de grens van 5 /u).
- **Poort af, thermistor leeg (11/20):** beide armen identiek.
- De eerste run en de herhaling van de breath-arm geven op alle 20 dezelfde AHI: de meting is reproduceerbaar binnen het image.

## Wat dit betekent

- `breath_dual` kan op deze montages alleen apneus **toevoegen**, nooit wegnemen; hypopneeën nemen af doordat events naar apneu verschuiven, en de AHI stijgt waar de tweede sensor apneus ziet die de eerste niet zag. Of die extra apneus juist zijn, kan deze meting niet zeggen — daarvoor is een menselijke scoring van dezelfde nachten nodig (de EC-studie AZORG-YASA-2026-001 levert die).
- Voor de standaardkeuze van vandaag (`breath` voor alle scoorders) betekent dit: op 13 van 20 diagnostische nachten geen AHI-verschil (10 volledig identiek), op 7 geeft `breath` de lagere AHI. Het risico zit precies op de nachten waar de thermistorpoort de thermistor goedkeurt: daar vertrouwt `breath` de apneus aan een sensor toe die er een fractie van vindt, en één nacht zakt daardoor van 103 naar 66 /u. `breath_dual` vangt dat op zonder ergens iets weg te nemen. Of dat de standaard moet worden op dual-sensor-montages, is Barts beslissing — de meting zegt niet welke telling juist is, alleen waar en waarom de twee uiteenlopen.
- Niets aan bibliotheek, profielen of configuratie is gewijzigd.

