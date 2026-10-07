# Kandidaat — 1D-U-Net voor respiratoire events (apneu/hypopneu) op flow + effort + SpO2

**Status: dsp-scout-bench, 2026-10-07, in aanbouw.** Methodeonderzoek onder `bench/`, geen
bibliotheekcode; evalueert tegen /srv/DATA-referenties met de projectmatcher (IoU 0,20,
typeonbewust). Aanleiding: `docs/breath_dual_vervolg_denkstuk_20261007.md` §4.1 en het
precedent van 01-09 (`docs/nacht_20260901_bevindingen.md`: 462k-parameter-U-Net, 113
trainingsnachten, PSG-IPA mediaan F1 0,539 met steile ziektelasthelling). Menselijk plafond
op PSG-IPA: 0,667 gepoold, per nacht 0,55–0,95 (`docs/respiratoir_menselijk_plafond_20261007.md`).

| | |
|---|---|
| Architectuur | zelfde BSD-schone `UNet1D` als `bench/eeg/unet50` (encoder 16-32-64-128-256, pooling (2,4,4,4), kernel 21), nu **5 invoerkanalen op 8 Hz** en **2 uitgangen** (apneu, hypopneu). Bottleneck 1/16 Hz; receptief veld minuten. |
| Invoer | neusdruk `Pres`, thermistor `Therm`, RIP `Thor`, RIP `Abdo`, `SpO2` (MESA-namen via `psgscoring.utils.detect_channels`), `mne.filter.resample` naar 8 Hz. Flow/effort: lopend gemiddelde + RMS over 18 min verwijderd, clip ±20 (als het arousalmodel). SpO2: daling t.o.v. een lopend maximum over 10 min, gedeeld door 3 (−1 = 3 % desaturatie), clip [−10, 1]; uitval (< 50 %) voorwaarts gevuld. |
| Kanaal-uitval-augmentatie | per venster neusdruk op nul met p 0,30, thermistor met p 0,20 (nooit beide), elke effortband p 0,10; amplitudefactor U(0,8; 1,3) op flow/effort. Doel: één model voor dual-sensor- (AZORG/MESA), druk-alleen- en thermistor-alleen-montages (SHHS1 heeft alleen een thermokoppel) — het model leert de sensorarbitrage waar de regels nu een poort en een vereniging voor nodig hebben. |
| Labels | NSRR `aasm15` uit `scripts/validate_mesa.parse_nsrr` (dezelfde referentie als het harnas): per sample apneu-masker (obstructive/central/mixed) en hypopneu-masker. |
| Training | BCE-with-logits op beide koppen, AdamW 1e-3, warm-up + cosinus, vensters van 30 min rond de slaapperiode, batch 16, 400 stappen/epoch, ≤ 30 epochs, bf16-autocast, vroegtijdig stoppen op de gepoolde **event-F1 (IoU 0,20, typeonbewust)** op de validatienachten. |
| Nabewerking | p_event = max(p_apneu, p_hypopneu), 1 s-gemiddelde, drempel τ (raster), gaten ≤ 2 s samengevoegd, **≥ 10 s** (AASM), type = kop met het hoogste gemiddelde binnen het event, slaappoort (onset in slaap-epoch). |
| Data | MESA-nachten die NIET in de standaard-n150-set (seed 20260801) en NIET in de 140 van de breath_dual-MESA-run van 07-10 zitten (die twee blijven onaangeraakte testsets), geschud met seed 20261007: **400 train / 100 validatie**; geregistreerd in `/srv/DATA/MESA/gebruikte_mesa_ids.txt` onder `## dsp-scout resp 2026-10-07`. Signalen alleen in het geheugen. |
| Evaluatie (later, met preregistratie) | baseline `psgscoring` `aasm_v3_breath_dual` en `rec` op dezelfde validatienachten; PSG-IPA SN1–5 náást het plafond per nacht; **SHHS1 150 verse nachten (thermokoppel-only) als beslissende externe replicatie**; AHI-bias per tertiel. |

Afwijkingen t.o.v. het arousalmodel, bewust: 8 Hz i.p.v. 50 Hz (ademband < 1 Hz; eventgrenzen
op 0,125 s), minimumduur 10 s i.p.v. 3 s (AASM), twee koppen i.p.v. één, kanaal-uitval-
augmentatie. Wat dit NIET doet: subtypering (obstructief/centraal/gemengd), desaturatiekoppeling
per event, RERA — dat blijft nabewerking van de regelketen zodra het netwerk als kandidaatgenerator
in de bibliotheek zou komen (hybride, zie denkstuk §4.1).

`model_best.pt` komt niet in git (zie `../.gitignore`); herbouw met
`OMP_NUM_THREADS=1 ../eeg/_venv/bin/python train.py --n-train 400 --n-val 100 --epochs 30 --steps 400 --workers 6`.
