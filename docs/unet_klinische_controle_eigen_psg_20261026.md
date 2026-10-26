# Klinische aan/uit-controle van de U-Net-arousaldetector op de 20 eigen PSG's

Datum: 2026-10-26, 12:40–13:58 CEST. Preregistratie
`docs/unet_klinische_controle_eigen_psg_preregistratie_20261026.md` (49f1e6b, geschreven vóór de
meting). In situ op de Hetzner-server in een losse container uit het productie-image
YASAFlaskified 0.39.1 (psgscoring **0.37.0**, versiecontrole per nacht), uploads read-only, 16 van
32 CPU's, `nice`, 8 analyses parallel, 66 min voor 19 nachten na een rookproef op R01 (11 min); de
app antwoordde op alle 67 bereikbaarheidscontroles met 200, de jobqueue was leeg bij elke start.
Dezelfde 20 opnames (R01–R20) en dezelfde invoer per opname als op 07-10 en 10-10 (opgeslagen
YASA-hypnogram, opgeslagen artefact-epochs, dezelfde kanaalkeuze en inleesroute via de
`tasks.py`-helpers van het image); koppeling naar de jobs blijft op de server
(`/data/slaapkliniek/metingen/unet_klinisch_20261026/`, mode 600). Vier armen per opname, profiel
`aasm_v3_breath_dual` (productiestandaard): **P** productieroute ongewijzigd (laadplan zónder
EOG, profieldefault `unet_v1`), **A** `lgbm`, **B** `unet_v1`, **C** `unet_v2` — A, B en C op
dezelfde raw mét het geconfigureerde EOG, de detector per arm via `PSGSCORING_AROUSAL_DETECTOR`,
het werkpunt dat van het gewicht (v1 0,35, v2 0,30). Ruwe uitvoer zonder patiëntgegevens:
`/srv/CODE/docs/unet_klinisch_20261026/R*.json` (tellingen, samenvattingen, provenance, event-
lijsten als onset/duur/type), analyse `analyse.py`, uitvoer `analyse_uitvoer.md`,
`samenvatting.json`. **Verschilmeting zonder referentie**: er is geen menselijke scoring van deze
nachten; de meting zegt waar en hoeveel de ketens uiteenlopen, niet welke juist is.

## 1. Levering: de nieuwe standaard kwam in productie niet aan

| | uitkomst |
|---|---|
| arm P valt terug op de LGBM-keten | **20/20**, reden `geen EOG-kanaal (alleen-EEG is slechter dan de LGBM-keten)` |
| P ≡ A op elk veld (indices, tellingen, respiratoire én arousal-eventlijsten) | **20/20** |
| P ≡ B | 0/20 |
| terugval in A / B / C | 0 / 0 / 0 |
| B: detector / werkpunt / gewicht | `unet_v1` / 0,35 / sha `6fbea285…` op 20/20 |
| C: detector / werkpunt / gewicht | `unet_v2` / 0,30 / sha `9a084f88…` op 20/20 |
| U-Net-rekentijd per nacht (2 threads) | B 4,6–11,6 s, C 4,6–11,1 s |
| EOG in jobconfig / in EDF / in productielaadplan / in raw van A–C | 20 / 20 / **0** / 20 |

De leveringslezing uit de preregistratie is bevestigd: `myproject/tasks.py::_pneumo_load_plan`
laadt het geconfigureerde `eog_ch` niet in de pneumo-raw en `_pneumo_channel_map` geeft geen
sleutel `"eog"` door; psgscoring vindt dan geen EOG (`_pick_eog`) en valt met reden terug. Sinds
de uitrol van 10:40 (YASAFlaskified 0.39.1 / psgscoring 0.37.0) levert productie dus **op elk
PSG-verslag de arousalketen van vóór 25-10** — de Herkomst-rij zegt dat ook ("LGBM-keten (lgbm) —
U-Net niet gebruikt: geen EOG-kanaal"), maar niemand had sinds de uitrol een verslag gemaakt.
Dezelfde fout als de kin-EMG van 24-08 (0.34.1), één kanaal verder. Met het EOG in de raw draait
het U-Net op 20/20 zonder terugval. **Reparatie staat klaar in de YASAFlaskified-werkboom
(ongecommit, wacht op Barts akkoord voor 0.39.2 + uitrol):** `eog_ch` door `_pneumo_load_plan`,
`_pneumo_channel_map` en alle vijf aanroepen (klinisch pad, polygrafievenster, profielvergelijking);
`tests/test_eog_reaches_unet.py` (6 tests, faalden eerst: 5 van 6 op de oude code), suite 751
passed / 22 skipped, ruff 0.16.3 en mypy 2.3.0 schoon.

## 2. Primair: `unet_v1` (B) tegen `lgbm` (A) — wat er op een verslag verandert zodra het EOG aankomt

| R | tertiel | TST (h) | arousals A → B | AI A → B | ΔAI | RDI A → B | AHI A → B | ΔAHI | hypopneeën A → B (desat / arousal) | apneus O/C/M | ernst | F1 any | κ_1s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R01 | hoog | 3,9 | 47 → 77 | 12,0 → 19,7 | +7,7 | 59,5 → 60,3 | 59,2 → 58,5 | −0,7 | 28 → 25 (22/6 → 22/3) | 203/0/0 = | ernstig | 0,484 | 0,352 |
| R02 | hoog | 7,2 | 152 → 193 | 20,9 → 26,5 | +5,6 | 26,4 → 26,5 | 22,4 → 22,4 | 0,0 | 82 → 82 (37/45 → 38/44) | 80/0/0 = | matig | 0,591 | 0,391 |
| R03 | midden | 2,2 | 158 → 119 | 29,0 → 21,9 | −7,1 | 47,1 → 33,9 | 24,0 → 20,8 | −3,2 | 50 → 43 (25/25 → 25/18) | 3/0/0 = | matig | 0,477 | 0,294 |
| R04 | midden | 5,0 | 45 → 48 | 8,8 → 9,3 | +0,5 | 20,4 → 22,6 | 19,8 → 20,8 | +1,0 | 14 → 19 (5/9 → 5/14) | 85/0/0 = | matig | 0,516 | 0,378 |
| R05 | laag | 7,6 | 127 → 179 | 16,6 → 23,4 | +6,8 | 5,5 → 7,3 | 3,7 → 5,1 | +1,4 | 27 → 38 (14/13 → 15/23) | 1/0/0 = | **normaal → licht** | 0,621 | 0,429 |
| R06 | laag | 1,8 | 28 → 47 | 15,1 → 25,4 | +10,3 | 9,7 → 22,3 | 8,0 → 17,7 | **+9,7** | 12 → 29 (8/4 → 10/19) | 2/0/0 = | **licht → matig** | 0,293 | 0,198 |
| R07 | laag | 1,6 | 26 → 14 | 16,2 → 8,8 | −7,4 | 10,0 → 3,7 | 4,4 → 3,1 | −1,3 | 7 → 5 (5/2 → 5/0) | 0/0/0 = | normaal | 0,500 | 0,331 |
| R08 | midden | 6,1 | 136 → 158 | 18,7 → 21,7 | +3,0 | 15,7 → 19,7 | 12,4 → 15,0 | +2,6 | 52 → 68 (30/22 → 30/38) | 24/0/0 = | **licht → matig** | 0,510 | 0,368 |
| R09 | midden | 2,1 | 32 → 32 | 13,0 → 13,0 | 0,0 | 36,1 → 35,1 | 30,8 → 30,8 | 0,0 | 40 → 40 (39/1 → 39/1) | 24/0/0 = | ernstig | 0,594 | 0,387 |
| R10 | midden | 8,1 | 158 → 220 | 19,2 → 26,7 | +7,5 | 27,9 → 31,5 | 23,5 → 25,5 | +2,0 | 78 → 94 (41/37 → 41/53) | 113/0/0 = | matig | 0,571 | 0,369 |
| R11 | midden | 4,8 | 90 → 120 | 17,5 → 23,3 | +5,8 | 29,6 → 32,7 | 25,0 → 25,7 | +0,7 | 54 → 57 (30/24 → 30/27) | 64/3/0 = | matig | 0,733 | 0,582 |
| R12 | hoog | 1,8 | 65 → 66 | 35,3 → 35,8 | +0,5 | 63,3 → 64,3 | 59,5 → 60,5 | +1,0 | 66 → 68 (65/1 → 66/2) | 33/10/0 = | ernstig | 0,656 | 0,466 |
| R13 | midden | 2,8 | 32 → 10 | 11,3 → 3,5 | −7,8 | 15,7 → 11,5 | 10,0 → 8,6 | −1,4 | 11 → 7 (7/4 → 7/0) | 16/1/0 = | licht | 0,286 | 0,247 |
| R14 | hoog | 3,3 | 244 → 228 | 71,2 → 66,6 | −4,6 | 103,3 → 106,0 | 91,9 → 92,2 | +0,3 | 109 → 110 (109/0 → 109/1) | 198/0/0 = | ernstig | 0,725 | 0,505 |
| R15 | laag | 5,1 | 50 → 76 | 9,8 → 14,9 | +5,1 | 6,6 → 8,3 | 3,9 → 5,7 | +1,8 | 13 → 22 (4/9 → 4/18) | 7/0/0 = | **normaal → licht** | 0,524 | 0,369 |
| R16 | hoog | 4,9 | 112 → 111 | 22,6 → 22,4 | −0,2 | 32,7 → 31,9 | 29,9 → 29,1 | −0,8 | 95 → 91 (70/25 → 69/22) | 51/2/0 = | matig | 0,565 | 0,442 |
| R17 | laag | 6,1 | 30 → 55 | 4,8 → 8,8 | +4,0 | 9,8 → 11,6 | 9,1 → 9,6 | +0,5 | 52 → 55 (51/1 → 51/4) | 3/0/0 = | licht | 0,635 | 0,446 |
| R18 | hoog | 5,0 | 228 → 429 | 43,4 → 81,6 | **+38,2** | 123,4 → 133,1 | 109,3 → 110,1 | +0,8 | 103 → 107 (103/0 → 104/3) | 440/0/0 = | ernstig | 0,609 | 0,456 |
| R19 | laag | 9,0 | 98 → 77 | 10,6 → 8,3 | −2,3 | 6,9 → 6,2 | 2,2 → 2,2 | 0,0 | 19 → 19 (2/17 → 2/17) | 1/0/0 = | normaal | 0,514 | 0,368 |
| R20 | laag | 5,8 | 158 → 168 | 26,5 → 28,2 | +1,7 | 9,4 → 10,6 | 8,9 → 9,9 | +1,0 | 44 → 50 (23/21 → 22/28) | 8/0/0 = | licht | 0,558 | 0,408 |

Lezing volgens de preregistratie (n = 20, geen terugval):

- **Apneus O/C/M identiek op 20/20** — zoals verwacht: de arousallijst bereikt op `breath_dual`
  alleen de hypopneeën (arousalpoot van de 3 %-regel), de RERA/FRI-lijst en de bevestiging van
  enkelsensor-apneus.
- **ΔAHI (B − A): mediaan +0,6, gemiddeld +0,8, bereik −3,2 … +9,7 /u**; gelijk op 3/20; 12 van
  17 niet-nulverschillen positief (exacte tekentoets p = 0,14); **|ΔAHI| ≥ 5 /u op 1/20 (R06,
  +9,7)**. Tegen de verwachting uit de preregistratie: mediaan |ΔAHI| = 1,0 /u, precies op de
  grens van "< 1" (de getekende mediaan is +0,6); nachten ≥ 5: 1, binnen "0 à 1". Ventilatoire
  last: Δ 0,0 op 20/20 in alle vergelijkingen. Per tertiel van de
  opgeslagen AHI: laag n = 7 mediaan +1,0 (gemiddeld +1,9, max |Δ| 9,7), midden n = 7 +0,7
  (+0,2, 3,2), hoog n = 6 +0,15 (+0,1, 1,0): de AHI-verschuiving zit in de lage en middentertielen.
- **Ernstklasse wisselt op 4/20, alle vier omhoog en alle vier in het lage/middentertiel:** R05
  normaal → licht (3,7 → 5,1), R06 licht → matig (8,0 → 17,7), R08 licht → matig (12,4 → 15,0),
  R15 normaal → licht (3,9 → 5,7). Drie van de vier zijn grensgevallen (5,1; 15,0; 5,7).
- **Waar de AHI vandaan komt (post-hoc, §6 van `analyse_uitvoer.md`):** de desaturatie-
  hypopneeën zijn over de drie armen vrijwel gelijk (A 690 / B 694 / C 695 over 20 nachten); de
  **arousal-hypopneeën gaan van 266 naar 335 (+69, +26 %)**, mediaan +2 per nacht, op R06 +15, R08
  +16, R10 +16, R05 +10, R15 +9. De +9,7 /u van R06 is +15 arousal-hypopneeën op 1,8 h TST. De
  splitsing is een proxy (desaturatie ≥ 3 % tegen < 3 %); op de `rule1a_arousal`-vlag van de
  events is het 262 → 331 → 324, B − A eveneens +69 (per arm dragen 4 hypopneeën met desaturatie
  2,66–2,99 % geen van beide vlaggen: gegradeerde acceptatie).
- **Arousal-index: mediaan +2,35, gemiddeld +3,4, bereik −7,8 … +38,2 /u; hóger op 13 van 19
  niet-nulverschillen (p = 0,17)** — tegen de verwachting uit MESA (daar lager op 73/100). De
  LGBM-keten telt op deze Somnomedics-montages met werkpunt 0,70 en vier afleidingen uit drie
  regio's (C3:A2, C4:A1, F4:A1, O2:A1), de MESA-
  doorwerking liep op generieke afleidingen met 0,80; welke telling dichter bij een menselijke
  ligt, zegt deze meting niet (op PSG-IPA lag de mediane count-ratio van het U-Net op 1,02 tegen
  1,11 voor de LGBM-keten, index-bias +0,22 tegen +2,03 /u — replicatie 27-09; de arousal-F1 steeg
  er op 4/5 in de doorwerking van 09-10 en op 5/5 in de replicatie).
  Meer dan 50 % verandering op 6/20: R01 (12,0 → 19,7), R06 (15,1 → 25,4), R13 (11,3 → 3,5), R15
  (9,8 → 14,9), R17 (4,8 → 8,8), R18 (43,4 → 81,6). **R18 is de uitschieter: 228 → 429 arousals
  op een nacht met 440 apneus** (respiratoire arousal-index 32,5 → 58,0) — het U-Net koppelt daar
  aan bijna elke apneu een arousal, de LGBM-keten aan de helft. Zes nachten hebben een TST
  onder 3 h (R03, R06, R07, R09, R12, R13); daar beweegt een index met een handvol events.
- **ΔRDI: mediaan +1,45, bereik −13,2 (R03) … +12,6 (R06) /u**, lager op 6/20; de RDI volgt de
  arousaltelling via RERA (R18 70 → 114) en de hypopneeën.
- **Overeenstemming van de arousallijsten B tegen A (elke overlap): F1 mediaan 0,562, precisie
  0,536, recall 0,616, κ_1s 0,382.** De twee detectoren zien dus voor een flink deel andere
  events; dit is geen verfijning van dezelfde lijst.

## 3. Secundair: `unet_v2` (C) tegen `unet_v1` (B)

| maat | C − B |
|---|---|
| ΔAHI | mediaan 0,0, gemiddeld 0,0, bereik −1,7 … +1,4 /u; gelijk 5/20; geen nacht ≥ 5 |
| Δarousal-index | mediaan +1,8, gemiddeld +2,5, bereik −2,6 … +17,9; **hoger op 14/20** (p = 0,12); > 50 % alleen R13 (3,5 → 9,5, TST 2,8 h) |
| Δarousals | mediaan +6, gemiddeld +10, bereik −19 … +94 (R18 429 → 523) |
| ΔRDI | mediaan +0,15, bereik −2,9 … +6,0 |
| apneus | identiek 20/20; hypopneeën mediaan 0, bereik −6 … +4 |
| ernstklasse | 1 wissel: R05 licht → normaal (5,1 → 4,7) |
| overeenstemming C tegen B | F1 mediaan 0,753, precisie 0,734, recall 0,776, κ_1s 0,691 |

`unet_v2` op zijn eigen werkpunt 0,30 telt op deze nachten iets méér dan `unet_v1` op 0,35
(mediaan +6 arousals, +1,8 /u) — de verwachting "telling niet systematisch hoger of lager" gaat
dus niet helemaal op: het tellingsneutrale werkpunt van v2 is op SHHS1/MESA tegen NSRR bepaald,
niet op deze montages. De twee U-Nets lijken wél veel meer op elkaar (F1 0,75, κ 0,69) dan elk op
de LGBM-keten (0,56 / 0,38). C tegen A samengevat: ΔAHI mediaan +0,5 (bereik −3,6 … +8,0; R06
+8,0), Δarousal-index mediaan +3,45 (R18 +56,1), 3 ernstwissels (R06, R08, R15), F1 0,588. De
per-nacht-tabellen van C tegen B en C tegen A, met respiratoire arousal-index en RERA-telling,
staan in `analyse_uitvoer.md` §3–4 (de RERA-index is niet apart gerapporteerd).

## 4. Drift van de LGBM-keten sinds 10-10 — en een bevinding buiten de preregistratie

Arm A (0.37.0, `lgbm`) tegen arm B van 10-10 (0.35.1, `breath_dual`, strictness 0,30):
**arousaltelling en arousal-index gelijk op 20/20** (de LGBM-keten is tussen 0.35.1 en 0.37.0
niet veranderd), maar **AHI gelijk op slechts 5/20 en apneus O/C/M op 4/20**. ΔAHI (nu − 10-10):
mediaan 0,0, gemiddeld +0,2, bereik −3,4 … +10,8 /u; |Δ| ≥ 2 op 6/20 (R02 −2,7, R04 **+10,8**,
R09 −3,4, R10 +2,3, R12 +4,4, R16 −2,3); ernstklasse-wissels 2/20 (R04 licht → matig 9,0 → 19,8,
R16 ernstig → matig 32,2 → 29,9). Op R04 gaan de apneus van 28 naar 85 (beide sensoren 9 → 54),
op R14 van 158 naar 198 (hypopneeën 150 → 109), op R02 van 105/2/0 naar 80/0/0. Op 8 van de 10
nachten die op 10-10 CSR-vlaggen droegen gaat de telling naar 0 (R02 108, R03 16, R04 4, R05 8,
R08 22, R09 17, R12 43, R14 114, R18 114 → 0; R01 43 → 42, R11 53 → 57) — de meest systematische
0.36.x-signatuur in de data (past bij `csr_period_rectified` uit 0.36.0), raakt
`ahi_csr_corrected`, niet `ahi_total`. Invoer identiek (zelfde epochs, artefacten, kanaalrollen,
flowkanalen; alleen het EOG is in A toegevoegd en dat raakt de respiratoire stap niet). De
thermistor is in 0.37.0 op R03 en R19 door de poort afgekeurd (`flow_thermistor_rejected`) waar
0.35.1 hem accepteerde (ΔAHI daar −0,9 en 0,0); op R04 is hij níét afgekeurd (alleen-thermistor
12 → 18), dus de R04-sprong heeft een andere drager.

**Post-hoc driftcontrole (§5b van `analyse_uitvoer.md`):** dezelfde arm A op het vorige
productie-image 0.38.12 (psgscoring **0.36.2**) voor R02, R04 en R14 is op AHI, apneus,
hypopneeën, arousals en de duale tellingen **gelijk aan 0.37.0 op 3/3**. De verschuiving zit dus
niet in de uitrol van vandaag maar in **0.36.0 (12-10) en/of 0.36.2 (18-10)**, en staat sinds dan
in productie. 0.36.0 verwerkte de codereview met vijftien profielvelden waarvan een deel
"een getal verschuift" op de klinische profielen (o.a. thermistorpoort zonder aliasing,
SpO2-nadir op de tijdas, CSR-periode zonder gelijkrichting, continue herstelbasislijn, gemengde-
apneudecompositie op de effort-envelope) en werd vóór de release gemeten op **MESA n = 30: "op
hele nachten niet meetbaar"**; 0.36.2 weert een dode thermistor ook additief. Op de eigen PSG's is
die stap nooit gemeten. Welk veld R04 van 9,0 naar 19,8 /u brengt (beide sensoren vinden er nu
ruim drie keer zoveel apneus) is uit deze meting niet af te leiden — dat vraagt een decompositie
per profielveld op R02/R04/R09/R14/R16 (één nacht, ~3 min per arm, in situ). Dit staat buiten de
preregistratie en is hier alleen vastgesteld, niet verklaard.

## 5. Wat dit betekent

- **Vandaag in productie draait de oude arousalketen.** De uitrol van 10:40 veranderde de
  arousal-index op geen enkel verslag, want het EOG bereikt het U-Net niet. Wie dat wil
  repareren, zet de klaarstaande 0.39.2 uit; daarna verandert een PSG-verslag zoals in §2:
  arousal-index mediaan +2,35 /u (meestal omhoog, op één nacht bijna verdubbeld), AHI mediaan +0,6
  /u via arousal-hypopneeën, 4 van 20 nachten een klasse omhoog rond de grenzen van 5 en 15 /u,
  apneus onveranderd.
- **Geen default-wissel uit deze meting, zoals vooraf vastgelegd:** "niet slechter én verklaarbaar"
  is het hoogst haalbare zonder referentie. Verklaarbaar is het (arousalpoot van de 3 %-regel,
  RERA via de arousallijst); of de hogere arousaltelling van het U-Net op deze montages juister is
  dan die van de LGBM-keten, weet alleen een scoorder. De replicaties (SHHS1, MESA, PSG-IPA) zeggen
  van wel op F1; op PSG-IPA lag de count-ratio van het U-Net op 1,02 tegen 1,11 voor de LGBM-keten.
- **`unet_v2`** verschilt op deze nachten weinig van v1 (AHI mediaan 0,0, arousal-index +1,8,
  F1 0,75 onderling); de keuze blijft Barts en hoort bij de schaalmeting, niet hier.
- **De 0.36.x-drift is het dringender punt voor de detectiekwaliteit:** AHI anders op 15/20
  eigen nachten, R04 licht → matig, R16 ernstig → matig, sinds 12/18-10 in productie, nooit op
  eigen nachten gemeten. Voorstel: decompositie per profielveld (aparte preregistratie).
- Niets aan bibliotheek, profielen of productieconfiguratie is gewijzigd; de YASAFlaskified-
  reparatie staat ongecommit klaar.

## 6. Verificatie

Onafhankelijk nagerekend (meting-verificatie, 26-10, eigen script uit R01–R20.json, `drift_0362/`
en de 10-10-uitvoer; alleen `agreement.compare_event_sets` uit de bibliotheek): de leverings-
tabel (terugval 20/20, P ≡ A op elk veld inclusief volledige eventlijsten en provenance 20/20),
alle cellen van de B-tegen-A-tabel, alle samenvattende cijfers van §2–§4, de 0.36.2 = 0.37.0-
controle (hele summary, eventlijsten en kanalen gelijk op 3/3) en de codeclaims van §1 (tasks.py
op 8b45b4c, pipeline.py/arousal.py op v0.37.0, Herkomst-tekst in de rapportlaag) repliceren; de
preregistratie is na de start niet gewijzigd. **Besluit staat.** Verwerkt uit de 14 bevindingen:
mediaan |ΔAHI| = 1,0 (niet "< 1"); de PSG-IPA-tellingsclaim was een F1-cijfer → count-ratio
1,02 tegen 1,11; CSR-signatuur op 8 van 10 nachten; thermistor afgekeurd op R03/R19 in 0.37.0;
vier afleidingen i.p.v. drie; R02 105/2/0; "vrijwel gelijk"; hypopneesplitsing als proxy met
de `rule1a_arousal`-telling ernaast; exacte halven (+1,45, +0,15, +2,35, +3,45); ventilatoire
last Δ 0,0 op 20/20; verwijzing naar de per-nacht-tabellen. Twee losse observaties: arm A
(LGBM-pad) draagt geen `summary.detector`-sleutel — 0.37.0 schrijft die alleen op het U-Net-pad
en bij terugval, de Herkomst-rij leunt dus op afwezigheid (bibliotheekgat, niet gerepareerd); en
de werkboom miste de tag v0.37.0 lokaal (nu opgehaald). Niet verifieerbaar vanaf de werkstation:
looptijd (plausibel: Σ t_total R02–R20 / 8 ≈ 58 min), de 67 bereikbaarheidscontroles, de lege
jobqueue, het containerrecept, mode 600 van de koppeling, en de YF-suite (alleen het nieuwe
testbestand opnieuw gedraaid: 6/6).
