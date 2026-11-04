# DOMINO-analyse-export lezen en per studie vergelijken (04-11-2026)

Vervolg op `somnomedics_domino_edfplus_20261102.md`: de EDF(+) van DOMINO draagt geen scoring,
de analyse-export wél. Op 04-11-2026 leverde de kliniek van één klinische PSG op de
server zowel de ASCII-export (`docs/psg.zip`) als de Excel-export (`docs/Map2.xlsx`). Dit stuk
legt vast wat erin zit, hoe de lezer (`psgscoring/compare/domino_reader.py`) ermee omgaat en
wat de eerste vergelijking mens–machine op eventniveau geeft. Alleen getallen; geen naam, geen
opnamedatum, geen job-id (dat laatste staat alleen op de server).

## 1. Wat de export is

**ASCII** ("Export analysis data → ASCII, alle signalen"): 45 tekstbestanden, één per signaal,
cp1252 (`µV` is één byte), CRLF, decimale komma. Kopblok `Signal ID` / `Start Time` / `Unit` /
`Signal Type` (+ `Events list`, `Rate` waar van toepassing), lege regel, rijen.

| type | voorbeeld | rijvorm |
|---|---|---|
| `Impuls` (20) | Flow Events, SpO2 Events, PLM Events, Classification Arousals, Snore Events, REM (blinks) | `22:00:14,246-22:00:29,121; 15;Hypopnea` |
| `Discret` (4) | Sleep profile (`Rate: 30 s`, `Wake/N1/N2/N3/REM/A`), Position (alleen wissels) | `22:00:30,000; Wake` |
| `Analog` (18) | SpO2 (~8 s), Heart Rate (1 Hz), PTT, HRV, FFT-banden, … | `22:00:01,250; 95` (`A` = artefact) |
| markers | `Signal ID: User markers`: `Start`, `Lights Off`, `End` | `22:15:05,230; Lights Off` |

**Excel** (`Map2.xlsx`, heropgeslagen in Excel): dezelfde 45 signalen als bladen, zelfde kopblok
in rijen 1–6, kolomkop in rij 8, datetime-cellen. Eventbladen rij voor rij gelijk aan ASCII
(≤ 1 ms), met een extra kolom **`Sleep stage`** = het DOMINO-stadium van de epoch waarin het
event begint (164/164 gecontroleerd tegen het profiel). Continue signalen zijn er tot **30 s
afgevlakt** (SpO2 1319 rijen tegen 4273 in ASCII; hartfrequentie tegen 41 285). Alle
`Start Time`-cellen staan op het 30-s-raster (afgerond op :00 of :30), de ASCII-eventbestanden op de echte
start (op de seconde), en de 30-s-bladen missen de laatste epoch (1319 rijen tegen 1320 in
ASCII; hier een wake-epoch, dus zonder gevolg voor de TST). Geen rapportblad, geen
analyse-instellingen, geen rapport-AHI. Het
xlsx-bestand draagt in `docProps/core.xml` de Windows-gebruikersnaam van wie het opsloeg:
nooit in git. Conclusie: **ASCII is de brondata; Excel voegt alleen de stage-kolom toe**, die
de lezer zelf afleidt.

In de eventlijsten staat ook wat buiten de slaap ligt: 26 van de 163 respiratoire events in
DOMINO-wake, 20 ervan vóór Lights Off, alle 20 met het kale label `Hypopnea` (het onbewerkte
auto-event; de scoorder bewerkt naar `Obstructive Hypopnea`). `Classification Arousals` bevat
413 arousals waarvan 379 `(Autonomic)` (pleth) en 34 EEG-arousals: op deze nacht zijn
EEG-arousals nauwelijks gescoord, dus geen arousalreferentie.

## 2. Tijdbasis — gecontroleerd, geen verschuiving

De marker `Start` is gelijk aan de EDF-kop van de upload. Controle op data:
kruiscorrelatie van DOMINO's SpO2-reeks (vooruit gevuld, 1 Hz) met `pneumo.spo2.timeseries`
van de machine geeft de beste r = 0,996 bij lag 0 s (37 432 paren; per uur apart 0 à +1 s);
hypnogram-κ 0,674 bij lag 0 en 0,41 bij −600 s. Een eerdere melding van "601 s verschil"
(ochtend 04-11) was een rekenfout in het nulpunt (de starttijd met de hand 600 s te hoog in seconden
omgezet) en is ingetrokken; de EDF+-exports van YF staan dus niet verkeerd. De lezer neemt de
marker `Start` als nulpunt en het script controleert dat op elke studie opnieuw
(`estimate_clock_offset`), omdat die controle goedkoop is en een verkeerd nulpunt anders een
F1 oplevert die er echt uitziet.

Het DOMINO-slaapprofiel ligt op :00/:30 van de klok; de machine-epochs beginnen bij de
EDF-start (hier 28 s later). `hypnogram_epochs` herbemonstert het profiel op het
machine-raster (midden van elke epoch), `hypnogram_1s` geeft de secondereeks; buiten het
profiel `"?"`.

## 3. De lezer

```python
from psgscoring.compare import read_domino_export
ex = read_domino_export("psg.zip")          # map, .zip of .xlsx
ev = ex.respiratory_events()                 # type/onset_s/duration_s/label/stage/value
ex.events_in_sleep(ev); ex.tst_minutes(); ex.hypnogram_epochs(n_epochs=1319)
ex.desaturations(); ex.limb_events(); ex.arousals(); ex.markers_s(); ex.series_1hz("SpO2")
ex.verify_label_mapping()                    # gekend / uitgesloten / ONBEKEND
```

Labels → psgscoring-typen (`DOMINO_RESP_LABELS`): Obstructive/Central/Mixed Apnea →
`obstructive`/`central`/`mixed`; Obstructive Hypopnea en kale Hypopnea → `hypopnea`;
Central/Mixed Hypopnea → `hypopnea_central`/`_mixed`; Apnea → `apnea`; RERA → `rera`;
`Body event` → geen respiratoir event (uitgesloten, geteld). Een label dat niet in de tabel
staat wordt NIET stil meegeteld en NIET stil weggelaten: `respiratory_events` laat het weg en
`verify_label_mapping` meldt het. PLM-labels: `P-Marker` → plm, `I-Marker` → lm, `LM Resp`,
`LM Body Position`. Stadia: Wake → W, REM → R, N4 → N3, A/Artefact → A.

Tests: `tests/test_domino_reader.py` (27, synthetische export over middernacht; de xlsx-test
vraagt openpyxl en skipt anders). Een onafhankelijke verificatie (04-11) bevestigde alle cijfers
op ASCII én Excel en vond vijf defecten die dezelfde dag gerepareerd zijn (een onleesbare regel
liet een bestand stil verdwijnen; een gat > 12 u tussen rijen viel een dag te vroeg; een kopblok
zonder lege regel gaf stil nul rijen; Excel `Rate [ms]` en `time`-cellen werden niet gelezen). Script: `scripts/compare_domino.py --export psg.zip
--results <job>_results.json [--json uit.json] [--t0 HH:MM:SS] [--offset-s N] [--auto-align]
[--matcher iou|any] [--iou 0.2]`.

## 4. Eerste studie: scorekaart

Laatst geanalyseerde PSG op de server, psgscoring 0.38.0, profiel `aasm_v3_breath_dual`,
`index_denominator=sleep`; mens = DOMINO-export (scoorder van de kliniek, AASM regel 1A).
Koppelregel IoU ≥ 0,20, één-op-één, symmetrisch (`agreement.compare_event_sets`).

| | DOMINO (mens) | psgscoring | opmerking |
|---|---|---|---|
| TST (min) | 457,5 | 388,5 | YASA 139 wake-epochs méér (542 tegen 403) |
| staging | κ 0,674, acc 0,767 op 1319 epochs | | 773 epochs noemen beide slaap |
| resp. events | 163 (137 in slaap) | 92 | mens: 43 OA, 10 MA, 3 CA, 81 H; machine: 9 OA, 1 CA, 82 H |
| AHI | 18,0 (eigen TST) / 21,2 (machine-TST) | 14,2 / 12,1 (DOMINO-TST) | rapport-AHI DOMINO = 16, zie §5 |
| F1 events in DOMINO-slaap | 0,524 (P 0,652 R 0,438; TP 60 FN 77 FP 32) | | elke-overlap 0,541 |
| F1 in epochs die beide slaap noemen | **0,645** (P 0,659 R 0,632; TP 60 FN 35 FP 31) | | ijkpunt scoorders onderling 0,667 |
| F1 alle 163 (ook wake) | 0,471 | | |
| typewissels bij gekoppelde paren | 19 | | 15 menselijke apneus bij ons hypopneu |
| gemiste events zonder enige overlap | 73 | | machine-stadium bij onset: W 42, N2 22, R 8, N3 1 |
| desaturaties | 223 (163 in slaap), ODI 21,4 | 129, ODI 19,9 | duur mediaan 28,0 tegen 10,9 s; koppeling (elke overlap) F1 0,562 |
| PLM in slaap | 19 (index 2,5) | 9 (index 1,4) | DOMINO 131 P-markers, 112 in wake |
| arousals | 21 EEG in slaap (index 2,8), 308 autonoom | 90 (index 13,9) | geen referentie op deze nacht |

Lezing: het staging-verschil is de grootste bron van gemiste events (42 van de 73 gemiste
liggen in machine-wake); binnen de slaap die beide zien ligt de F1 op het niveau van
menselijke scoorders onderling. Wat overblijft is typering (apneus die wij hypopneu noemen) en
hypopneeën zonder desaturatie — dezelfde richting als §12 van `laatste10_humaan_vs_machine`.
De desaturatiekoppeling is zwak omdat DOMINO's desaturaties bijna drie keer zo lang zijn als
de onze (definitieverschil, open).

## 5. Waarom het DOMINO-rapport 16 zegt en de lijst 18,0

De exportlijst is de volledige eventlijst. Het rapport past daar zelf het hypopneucriterium op
toe (desaturatie ≥ 3 % gekoppeld binnen een venster van enkele tientallen seconden): apneus
plus hypopneeën mét zo'n desaturatie geeft 117–125 events (venster 30–45 s) = 15,3–16,4 /u;
voor exact 16,0 zijn 122 events nodig. 12–20 van de 81 menselijke hypopneeën in slaap dragen
geen ≥ 3 %-desaturatie. De 'kolom mens' van §12 bestaat dus uit rapport-AHI's met DOMINO's
koppelregel; wie uit de export een rapport-AHI wil reproduceren moet dat venster kennen
(analyse-instellingen in DOMINO) of de rapport-AHI apart invoeren.

## 6. Open

- Bevestiging dat `Flow Events` de definitieve menselijke scoring is (aangenomen: ja).
- Het koppelvenster van de hypopneuregel in de DOMINO-analyse-instellingen (voor §5).
- Desaturatiedefinitie DOMINO (duur 28 s mediaan) tegen de onze (11 s).
- Volgende bouwstap (niet gedaan): upload "humane scoring toevoegen" in YASAFlaskified met
  deze scorekaart in het rapport, en een cohortoverzicht zodra er meer exports zijn.
