# Preregistratie — de laatste tien eigen PSG's: ligt psgscoring boven de menselijke scoring, en welke oktoberwijziging draagt dat?

Datum: 2026-11-02, geschreven vóór de meting. Status: wacht op Barts akkoord voor de in-situ-run
én op de menselijke cijfers (zie §3). Aanleiding: Barts indruk na de laatste tien uploads op Hetzner
("ik heb de indruk dat psgscoring boven de menselijke scoring zit").

## 1. Vraag

1. Hoe groot is het verschil machine − mens (AHI) op de tien meest recente diagnostische PSG's, en in
   welke richting?
2. Welke van de vier wijzigingen van oktober draagt dat verschil? Alle vier duwen de AHI of de telling
   omhoog en zaten in de scoring van deze tien nachten:

   | datum | wijziging | gemeten effect tot nu toe |
   |---|---|---|
   | 07-10 | standaardprofiel `aasm_v3_breath_dual` (vereniging thermistor- + drukapneus) | 20 eigen nachten: ΔAHI mediaan 0,0, gemiddeld +2,8, R18 +37 (`docs/breath_vs_breath_dual_eigen_psg_20261007.md`) |
   | 10-10 | 0.35.0: `hypopnea_strictness` 0,50 → 0,30 op breath/breath_dual | MESA n=140: ΔF1 +0,029, bias −3,28 → −0,05 (`docs/breath_dual_mesa_20261007.md`) |
   | 12-10 | 0.36.0: stadiumbasislijn draait nu ook op YASA-hypnogrammen | R04 28 → 85 apneus; MESA n=30: 3 van 4 normale nachten → matig/ernstig (`docs/inkortingstoets_20261026.md`) |
   | 26-10 | 0.37.0 + YF 0.39.2: `unet_v1` als arousaldetector, EOG bereikt het U-Net | PSG-IPA/MESA: AHI ±0,6 /u, RDI volgt de arousaltelling (`docs/unet_doorwerking_20261009.md`) |

3. Is wat overblijft een regelverschil (hypopneeregel 3 % of arousal tegenover 4 %; TST-noemer van
   YASA tegenover de menselijke staging) in plaats van een detectorfout?

## 2. Steekproef en invoer

- De **tien meest recente** jobs in `/data/slaapkliniek/uploads/` (op mtime van `<job>_config.json`)
  met `study_type` diagnostische PSG (geen polygrafie), `split_night` uit, aanwezige `_results.json`
  met hypnogram en respiratoire samenvatting, en een EDF die nog op schijf staat. Codes L01–L10 (L01 =
  meest recent). Koppeling job-id ↔ L blijft op de server in
  `/data/slaapkliniek/metingen/laatste10_20261102/mapping.json`, mode 600. Geen namen, paden,
  kloktijden of job-id's verlaten de server; alleen getallen.
- Per nacht dezelfde invoer als de worker (recept 26-10): opgeslagen YASA-hypnogram, opgeslagen
  artefact-epochs, kanaalconfig uit de jobconfig, laadplan mét EOG (0.39.2), positiecodekaart uit de
  productie-`.env`. Losse container uit `yasaflaskified:0.39.2` (psgscoring 0.37.0), uploads
  read-only, `--network slaapkliniek_kliniek_net`, `--cpus 16`, 8 nachten parallel. Jobqueue leeg bij
  de start; app-bereikbaarheid elke minuut gelogd; de scripts raken geen productiebestand.
- Scripts: `docs/laatste10_20261102/maak_mapping.py` (host, selectie + sjabloon voor de menselijke
  cijfers), `run.py` (container), `analyse.py` (lokaal, op L*.json + de menselijke cijfers).

## 3. Menselijke referentie

Bart vult op de server `humaan.csv` in (sjabloon met L, kort job-id en bestandsnaam, zodat de
koppeling op de server blijft): per nacht `ahi_h` (verplicht), `n_apneu_h`, `n_hypopneu_h`,
`tst_h_min` (optioneel) en `regel_h` (`3pct_of_arousal` | `4pct` | anders). `maak_mapping.py --cijfers`
strijkt namen en job-id's eruit en schrijft `humaan_cijfers.json`; alleen dat bestand komt naar de
Z6. De menselijke scoring is één scoorder per nacht, geen goudstandaard (PSG-IPA: respiratoire
F1-plafond 0,667 tussen scoorders); de AHI-ICC tussen scoorders is op PSG-IPA niet gemeten.

## 4. Armen (profiel `aasm_v3_breath_dual` tenzij anders vermeld; alles overig productie)

| arm | wat anders | isoleert |
|---|---|---|
| A0 | niets | productie 0.37.0; moet de opgeslagen AHI reproduceren |
| SB | `STAGE_BASELINE_PARTIAL_EPOCHS=False` (op een YASA-vloerlengte-hypnogram = stadiumbasislijn uit, zoals vóór 12-10; drager volgens `docs/drift_036x_decompositie_20261026.md`) | 12-10 |
| ST | `HYPOPNEA_STRICTNESS=0.50` | 10-10 |
| BR | profiel `aasm_v3_breath` | 07-10 (vereniging) |
| LG | env `PSGSCORING_AROUSAL_DETECTOR=lgbm` | 26-10 |
| ALL | SB + ST + LG op `aasm_v3_breath` | alle vier terug (additiviteit) |
| REC | profiel `aasm_v3_rec` + SB + LG | de route die de kliniek vóór 07-10 voor de meeste gebruikers draaide |

Per arm en nacht: AHI (totaal, incl. uncertain, OAHI), ernstklasse (AASM: < 5, 5–15, 15–30, ≥ 30),
tellingen per eventtype, duale tellingen, hypopneeën met ≥ 3 % desaturatie en met arousalvlag,
arousalaantal en -index, RDI, TST uit het hypnogram, `stage_baseline_used`, en de kale eventlijst
(onset, duur, type) zonder kloktijden.

## 5. Vooraf vastgelegde lezing

1. **Identiteit:** A0 reproduceert de opgeslagen AHI op 10/10 (|Δ| ≤ 0,1 /u). Zo niet, dan meet de run
   iets anders dan productie en wordt eerst dát verklaard.
2. **Vraag 1 (beschrijvend, n = 10):** Δ = AHI_machine − AHI_mens per nacht voor A0: mediaan, bereik,
   aantal positief/negatief, exacte tekentoets (tweezijdig; p < 0,05 heet hier "richting aannemelijk",
   meer niet), ernstconcordantie (exact en binnen één klasse), en Δ in % van AHI_mens. Als `tst_h_min`
   is ingevuld: AHI_machine herrekend op de menselijke TST, zodat de noemer afgesplitst wordt.
3. **Vraag 2 (drager):** voor elke één-wijziging-terug-arm (SB, ST, BR, LG) de verschuiving tegen A0 per
   nacht; de drager is de arm die de mediaan van (A0 − mens) het meest naar 0 brengt. Additiviteit:
   A0 − ALL tegen de som van de vier enkelvoudige verschuivingen (verschil > 1 /u op een nacht =
   interactie, benoemd). SB wordt vooraf als drager verwacht (R04 ×3, MESA 0111 3 → 183); BR als tweede
   op nachten met een afgekeurde thermistor.
4. **Vraag 3 (regelverschil):** blijft de mediaan van (REC − mens) of (ALL − mens) boven +2 /u, dan
   ligt de rest niet aan de oktoberwijzigingen maar aan de regel of de noemer; `regel_h` = `4pct`
   verklaart dan een deel per constructie (regel 1A telt hypopneeën met arousal mee). Dan geen
   detectorconclusie, wel een profielvraag (een 4 %-profiel bestaat niet; bouwen is een aparte stap).
5. **Beslisrelevantie voor de open keuze van 27-10:** brengt SB alleen de mediaan van (machine − mens)
   binnen ±2 /u zonder op meer dan één nacht een ernstklasse verder van de mens te landen dan A0, dan
   steunt dit `stage_baseline=False` op de klinische profielen (optie 1). Blijft A0 − SB klein (mediaan
   < 1 /u), dan is de stadiumbasislijn op deze tien geen drager en vervalt die steun.
6. n = 10 eigen nachten, één scoorder, geen uitspraak buiten deze beschrijving; een p-waarde uit tien
   tekens zegt alleen iets over de richting.

## 6. Uitvoer

`/data/slaapkliniek/metingen/laatste10_20261102/` (mapping.json 600, humaan.csv 600, L*.json,
run.log, app_check.log); kopie zonder koppeling naar `/srv/CODE/docs/laatste10_20261102/`; verslag
`docs/laatste10_humaan_vs_machine_20261102.md`; verificatie-agent vóór CHANGELOG/geheugen. Looptijd:
7 armen × 10 nachten, 8 parallel, ±1 u; geen uitrol, geen productiewijziging.
