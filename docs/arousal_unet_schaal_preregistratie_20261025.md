# Preregistratie — arousal-U-Net opschalen (meer nachten, tweede cohort, flowkanaal)

Datum: 2026-10-25, avond. **Geschreven vóór enige training of inferentie.** Bart besliste
op 25-10 ("doe") dat dit het volgende meetproject is; de sha256 van dit bestand staat in het
trainingslog op het moment van de start (`/srv/CODE/docs/arousal_unet_schaal_20261025/chain.log`).

## Aanleiding

Het bevroren 1D-U-Net (`model_frozen_cc91bb86.pt`, 317 MESA-nachten, EEG + EOG + kin-EMG @
50 Hz; replicatie 27-09: SHHS1-150 F1 0,543 → 0,676, MESA-76 0,687, PSG-IPA 0,745) is sinds
0.35.1 opt-in in de bibliotheek. De literatuurcheck van 25-10
(`docs/literatuur_update_20261025.md` §1.1): het Sleep-Revolution-model (Pitkänen e.a., SLEEP
2026) traint op 1847 MESA-nachten en voedt naast EEG en kin-EMG ook **nasale druk** mee. Wij
trainden op 317 nachten zonder respiratoir kanaal. Twee hypothesen, elk met een eigen arm:

1. **Schaal**: vier keer zoveel trainingsnachten en een tweede cohort (SHHS1) verhogen de
   event-F1 op een nooit gezien SHHS1-cohort.
2. **Flow**: een vierde ingang (nasale druk waar aanwezig, anders thermistor) helpt de
   detector de respiratoire context van een arousal te zien.

## Wat vast ligt (niets wordt tijdens de meting gewijzigd)

- **Architectuur** `bench/eeg/unet50/model.py` (`UNet1D`, base 16, pools 2-4-4-4, kernel 21);
  alleen `in_ch` verschilt (3 of 4). **Voorbewerking** als `bench/eeg/unet50/data.py`:
  anti-alias + resample naar 50 Hz, per kanaal lopend gemiddelde en RMS over 18 min
  verwijderd, clip ±20, float16. **Nabewerking** `bench/eeg/common/postproc.py`
  (1 s-gemiddelde, drempel τ, gaten ≤ 1 s dichten, ≥ 3 s, slaappoort op het
  referentie-hypnogram). De 10 s-samenvoeging van de keten komt pas in de evaluatie
  (bibliotheekfunctie), identiek voor alle armen én het bevroren model.
- **Trainingsrecept** als unet50: BCE-with-logits, AdamW lr 1e-3 (wd 1e-5), warm-up 200
  stappen + cosinus, vensters 10 min rond de slaapperiode, batch 32, 600 stappen per epoch,
  bf16-autocast, gradient-clip 5, seed 20261025. Augmentatie: per venster willekeurig
  EEG-kanaal (MESA EEG3 0,5 / EEG1 0,25 / EEG2 0,25; SHHS1 EEG 0,5 / EEG(sec) 0,5 — of het
  enige aanwezige), willekeurig EOG (L/R), amplitudefactor U(0,8; 1,3) per kanaal; in arm MSF
  per venster willekeurig een beschikbaar flowkanaal (MESA `Pres` of `Therm`; SHHS1 `NEW AIR`
  of `AIRFLOW` — SHHS1 heeft géén drukcanule, dus daar altijd thermistor-type). Bij
  validatie/inferentie: vaste kanalen (MESA EEG3/EOG-L/EMG, SHHS1 EEG/EOG(L)/EMG, PSG-IPA
  C4-M1 of Cz-M1 / E1-M2 / EMG chin) en als flow **druk als die er is, anders thermistor**
  (MESA `Pres`, SHHS1 `NEW AIR` > `AIRFLOW`, PSG-IPA `Resp nasal`). MESA `Flow` is de
  getransformeerde `Pres` (r = −0,91 op mesa-sleep-0001) en wordt niet gebruikt; de
  bibliotheek wijst `Pres` als `flow_pressure` aan (`utils.detect_channels`).
- **Vroegtijdig stoppen en werkpunt** uitsluitend op de **validatieset van 80 nachten
  (40 MESA + 40 SHHS1)**, gepoolde event-F1 op IoU 0,20, raster τ 0,15 … 0,85 (stap 0,05).
  Epochs: M ≤ 50, MS/MSF ≤ 60, geduld 10. Lering van 27-09: geen τ-premisse uit een ander
  cohort; voor ELK model (ook het bevroren) wordt τ op dezelfde 80 validatienachten gekozen.

## Nachten

Pools (gereedschap `bench/eeg/unet_scale/select_ids.py`, seed 20261025; id-lijsten in die map;
registerregels geschreven vóór de training):

- **MESA-pool = 1318 nachten** = alle 2055 nachten met EDF + XML minus de
  standaard-n150-validatieset (seed 20260801; 150), de resp-bench-validatieset
  (`bench/resp/ids_val.txt`; 100), dsp-scout eeg 27-09 (400, incl. de 317 trainingsnachten van
  het bevroren model), de U-Net-replicatieset MESA-76 (76) en de kandidaatpoort-40 (40)
  (uitsluiting 737 unieke id's). **Kanttekening die in het register komt:** de eerste vijf
  secties van `gebruikte_mesa_ids.txt` (133-run, 450-run, arousalsets 474, eerste-200,
  herstel-anker) delen één ongescheiden blok van 1409 id's; de 474 "arousalsets"
  (afleiding/replicatie van de LGBM-arousalregels) zijn daarin niet te onderscheiden en zitten
  dus in de pool. Dat raakt géén testset van deze meting (SHHS1-150b, MESA-76 en PSG-IPA zijn
  er disjunct van), maar maakt deze 1318 nachten voortaan **arousal-getraind**: ze zijn nooit
  meer een verse arousal-evaluatieset, en respiratoire metingen erop dragen de kanttekening
  dat de U-Net-arousals (via de hypopneu-koppeling) er licht optimistisch kunnen zijn.
  Verdeling: **val 40**, **train 1278**.
- **SHHS1-pool = 5333 nachten** (5792 minus 458 in `gebruikte_shhs_ids.txt`). Trekking in deze
  volgorde uit de geschudde pool: **test SHHS1-150b (150, vers, nooit gezien)**, **val 40**,
  **train 1278** (gebalanceerd met MESA).
- **Testsets** (nooit in training of validatie): SHHS1-150b (beslissend), MESA-76
  (`/srv/CODE/docs/arousal_unet_20260927/ids_mesa76.txt`; zelfde cohort als een deel van de
  training, ander sample), PSG-IPA SN1–5 (referentie `bench/eeg/ref/`).
- Nachten zonder NSRR-arousal of met < 60 slaapepochs vallen af (zoals unet50); het aantal
  wordt gelogd.

## Armen

| arm | training | ingangen |
|---|---|---|
| bevroren | 317 MESA (27-09) | EEG, EOG, EMG |
| **M** | 1278 MESA | EEG, EOG, EMG |
| **MS** | 1278 MESA + 1278 SHHS1 | EEG, EOG, EMG |
| **MSF** | 1278 MESA + 1278 SHHS1 | EEG, EOG, EMG, **flow** |

Eén seed per arm; variantie tussen seeds is op 27-09 gemeten (±0,004 val-F1) en wordt hier
niet opnieuw gemeten. Het bevroren model loopt door dezelfde inferentiecode op dezelfde
testnachten (sha256-controle), zodat elk verschil aan het gewicht ligt en niet aan de keten.

## Beslisregel (vooraf vastgelegd)

Gepaard per nacht, projectmatcher IoU 0,20 (`validate_psgipa.LEGACY_MATCHER` via
`scripts/arousal_unet_replicatie.py eval` of gelijkwaardig), na de 10 s-samenvoeging, elk
model op zijn eigen τ uit de validatieset van 80.

- **Primair, SHHS1-150b, arm tegen bevroren:** mediane ΔF1 ≥ +0,02 **én** Wilcoxon p < 0,01
  **én** ≥ 60 % van de nachten beter **én** mediane |count-ratio − 1| niet slechter (arm ≤
  bevroren + 0,05; count-ratio = eigen events / NSRR-arousals per nacht).
- **Secundair:** MESA-76 mediane ΔF1 ≥ 0 en niet significant slechter (eenzijdige Wilcoxon
  "slechter" p ≥ 0,05); PSG-IPA op ≥ 4 van 5 nachten niet lager dan het bevroren model.
- **Flow-hypothese:** MSF tegen MS, dezelfde vier primaire voorwaarden met mediane ΔF1 ≥
  +0,01. Alleen als die halen, is het flowkanaal gerechtvaardigd (een vierde ingang maakt de
  bibliotheek afhankelijk van een flowkanaal voor de arousals).
- **Rapportage altijd met** het menselijk plafond 0,679 (PSG-IPA) als richtpunt, per tertiel
  van de referentie-arousalindex, REM/NREM apart, en de count-neutrale τ (ratio dichtst bij 1
  op de validatieset) als tweede werkpunt naast de F1-optimale τ. Het bevroren model krijgt
  twee extra rijen: τ 0,20 (eigen bench-keuze 27-09) en τ 0,35 (bibliotheekwerkpunt).

**Gevolg bij slagen:** de winnende arm (MS of MSF, anders M) wordt kandidaat voor een
`unet_v2`-gewicht in de bibliotheek (ONNX, sha256-wacht, zelfde terugval), met klinische
aan/uit-controle en Barts beslissing over default en werkpunt — niet in deze meting.
**Gevolg bij falen:** het bevroren model blijft; cijfers in CHANGELOG en
`docs/third_party_comparison.md`; de pool blijft als arousal-getraind geregistreerd.

## Wat NIET uit deze meting volgt

Geen uitspraak over RERA/RDI, de hypopneu-koppeling of PLM-associatie (de bench-inferentie
voedt de pipeline niet); geen uitspraak over CPU-kost van een 4-kanaalsmodel in productie
(wordt pas gemeten als de arm wint); geen hertuning van de nabewerking.

## Rekenplan en bewaking

Laden ≈ 2 636 trainings- en validatienachten in het geheugen (float16, ~70 GB RSS; 151 GB
beschikbaar; niets van MESA/SHHS naar schijf). 8 laadprocessen met OMP_NUM_THREADS=1;
training op de RTX A4000; ~2,4 min per epoch → M ≤ 2,0 u, MS en MSF ≤ 2,4 u elk; inferentie
op 231 testnachten × 4 modellen ≈ 15 min. Keten onder `setsid` met de thermische bewaker
(`thermal_pauze.sh`, pauze bij 3× ≥ 78 °C, hervat ≤ 66 °C, log elke 10 s). Uitvoer buiten git:
`/srv/CODE/docs/arousal_unet_schaal_20261025/` (per arm `model_best.pt`, `train_log.json`,
`KLAAR`; `out/<cohort>/` met per nacht en model `_prob10hz.npz`, `_pred.csv` op de val-τ
zonder samenvoeging, `_grid.json` met de events op het hele τ-raster; `_ref.csv` en
`_hypno.json` eenmaal per nacht; `tau.json` per model; marker `ALLES_KLAAR`).
