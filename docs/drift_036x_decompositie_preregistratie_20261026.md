# Preregistratie — decompositie van de 0.36.x-drift op de eigen PSG's, per profielveld

Datum: 2026-10-26, geschreven vóór de meting. Opdracht van Bart: "doe de decompositie van de
0.36.x-drift, je hebt permissie" (dekt het lezen van de eigen opnames op de server).

## Vraag

De klinische controle van 26-10 (`docs/unet_klinische_controle_eigen_psg_20261026.md`, §4) vond dat
de LGBM-arm van psgscoring 0.37.0 op 15 van 20 eigen nachten een andere AHI geeft dan dezelfde arm op
0.35.1 (10-10) bij identieke invoer, terwijl de arousaltelling op 20/20 gelijk is; 0.36.2 geeft op
3/3 gecontroleerde nachten dezelfde cijfers als 0.37.0. De verschuiving zit dus in 0.36.0 (12-10,
codereview: "15 nieuwe profielvelden, gemeten op MESA n = 30: op hele nachten niet meetbaar") en/of
0.36.2 (18-10: dode thermistor ook additief geweerd). **Welk veld (of welke code buiten de velden)
draagt de verschuiving per nacht, en verklaren de velden samen het hele verschil?**

## Kandidaten (uit `constants.SCORING_PROFILES` 0.37.0 tegen 0.35.1; oude waarde = de pin op
`mesa_shhs`/`chicago_1999`, volgens CHANGELOG v0.36.0 "bevroren profielen gepind op oud")

| nr | veld | 0.37.0 klinisch | oud (0.35.1-gedrag) | kan `ahi_total` raken? (vooraf, uit de code) |
|---|---|---|---|---|
| F1 | `ANCILLARY_TST_EXCLUDES_ARTIFACTS` | True | False | nee (noemer arousal-index/CSR-dichtheid) |
| F2 | `BREATH_PHASE_SPLIT` | zero_crossing | midpoint | nee (afvlakking → RERA/RDI) |
| F3 | `CSR_PERIOD_RECTIFIED` | False | True | nee (CSR-vlaggen, `ahi_csr_corrected`) |
| F4 | `INDICES_ON_FINAL_EVENT_LIST` | True | False | nee (positie-AHI, PLM-uitsluiting) |
| F5 | `MIXED_DECOMPOSITION_ON_ENVELOPE` | True | False | O/C/M-verdeling, niet de telling |
| F6 | `PLM_RESP_EXCLUSION` | aasm | end | nee (PLM-index) |
| F7 | `RECOVERY_BASELINE_MODE` | interpolated | block | **ja** (basislijn na apneu → genormaliseerde flow) |
| F8 | `REM_AHI_ALIAS_EXCLUDES_UNCERTAIN` | True | False | nee (REM-AHI-alias) |
| F9 | `SPECTRAL_EFFORT_RESOLUTION` | event | 4s | O/C/M-verdeling (effortaanwezigheid) |
| F10 | `SPO2_INDEX_DENOMINATOR` | artifact_free_sleep | sleep | nee (ODI/T90) |
| F11 | `SPO2_NADIR_TIME_AXIS` | True | False | **ja** (desaturatiebevestiging van hypopneeën) |
| F12 | `STAGE_BASELINE_PARTIAL_EPOCHS` | True | False | **ja** (stadium-basislijn draaide op echte opnames niet) |
| F13 | `THERMISTOR_BAND_POWER_ANTIALIAS` | True | False | **ja** (thermistorpoort → sensorkeuze; R03/R19 afgekeurd in 0.37.0) |
| F14 | constante `THERMISTOR_BAND_POWER_DEAD` (0.36.2) | 0,20 | geen vloer (0,0) | **ja** (dode thermistor niet meer als apneusensor) |

Niet als arm: `AROUSAL_DERIVATION_MODE` (multi was al het gedrag, alleen als veld vastgelegd) en de
twee reparaties die in 0.36.0 bewust default-oud staan (`HYPOPNEA_MASK_EXCLUDES_APNEA_BAND=True`,
`LINEARISE_AFTER_DC_REMOVAL=False`). Code buiten de velden die een getal kan verschuiven
(CHANGELOG 0.36.0): SpO2-kruiscontaminatie vergelijkt met het voorafgaande event; snappen naar
ademteuggrenzen onder de minimumduur; MMSD-afgewezen apneus apart; kanaalkwaliteit per sensor
(`signal_quality_channels`); dode effortpoort in regel 5b weg. Die zijn niet per arm uit te zetten
en verschijnen als **residu** (arm ALL tegen 10-10).

## Steekproef en invoer

Acht van de twintig nachten: de zes met |ΔAHI| ≥ 2 /u tussen 0.35.1 en 0.37.0 (R02, R04, R09, R10,
R12, R16), R14 (apneus 158 → 198, hypopneeën 150 → 109 bij gelijke AHI) en R03 (thermistor in 0.37.0
afgekeurd). Dezelfde invoer als op 10-10 en 26-10 (opgeslagen hypnogram en artefact-epochs, zelfde
kanaalkeuze; inleesroute van het image 0.39.2, dus mét EOG — raakt de respiratoire stap niet, en
arm A0 toetst dat). Losse container uit `yasaflaskified:0.39.2` (psgscoring **0.37.0**,
versiecontrole), uploads read-only, 16 van 32 CPU's, `nice`, 8 nachten parallel, armen per nacht
sequentieel, `OMP_NUM_THREADS=2`. Profiel `aasm_v3_breath_dual`, `PSGSCORING_AROUSAL_DETECTOR=lgbm`
(de arousalketen is tussen 0.35.1 en 0.37.0 ongewijzigd: arousaltelling 20/20 gelijk).

## Armen (per nacht, zelfde raw)

- **A0**: 0.37.0 zoals het is — moet op AHI, apneus O/C/M, hypopneeën, duale tellingen en
  arousals gelijk zijn aan arm A van 26-10 op 8/8 (anders stopt de meting).
- **F1 … F14**: één veld (of de constante) teruggezet op de oude waarde, in-process via
  `constants.SCORING_PROFILES[...]` voor `aasm_v3_breath_dual`, `aasm_v3_breath` en `aasm_v3_rec`
  (de duale tweede pas leest een tweede profiel) en `pipeline.THERMISTOR_BAND_POWER_DEAD`; de
  teruggelezen waarde wordt per arm vastgelegd.
- **ALL**: alle veertien tegelijk terug — de reconstructie van 0.35.1.

## Vooraf vastgelegde lezing

1. **Identiteit eerst.** A0 ≡ arm A van 26-10 op 8/8. Zo niet: stoppen en uitzoeken.
2. **Volledigheid.** ALL tegen 10-10 (0.35.1, strictness 0,30) per nacht op AHI, apneus O/C/M,
   hypopneeën, `n_both`/`n_pressure_only`/`n_thermistor_only`, CSR-vlaggen. Gelijk op 8/8 → de
   velden verklaren alles. Niet gelijk → het residu (ALL − 10-10) wordt per nacht gerapporteerd en
   aan de code buiten de velden toegeschreven, zonder dat deze meting zegt aan welke regel.
3. **Drager per nacht.** Voor elk veld f: ΔAHI_f = AHI(F_f) − AHI(A0) en de verklaarde fractie
   ΔAHI_f / (AHI_10-10 − AHI_A0) waar de noemer ≥ 0,5 /u is. Een veld is **drager** op een nacht als
   het ≥ 50 % van het gat verklaart met hetzelfde teken; **mede-drager** bij 20–50 %. Zelfde
   decompositie voor de apneutelling (totaal, beide/alleen-druk/alleen-thermistor) en de
   hypopneetelling.
4. **Additiviteit.** Σ_f ΔAHI_f tegen ΔAHI_ALL per nacht; een verschil > 1 /u betekent interactie
   tussen velden en wordt benoemd (de meting lost die niet op).
5. **Verwachting (geen slaagcriterium):** de AHI-dragers zitten in {F7, F11, F12, F13, F14};
   F1, F2, F4, F6, F8, F10 bewegen `ahi_total` op geen enkele nacht (0,0 op 8/8 — wordt getoetst);
   F3 verklaart de CSR-vlaggen (108 → 0 op R02, 114 → 0 op R14); F5 en F9 verschuiven alleen
   O/C/M. R04 (9,0 → 19,8, beide sensoren 9 → 54 apneus) is géén dode-thermistornacht
   (thermistor-only 12 → 18, niet afgekeurd), dus F14 is daar niet de drager; de peer-sessie
   vermoedt F14 wél voor nachten waar de thermistor te laag telde — R03 is de toets.
6. **Prominent in het verslag:** per nacht de drager(s) met fractie; elk veld dat tegen de
   verwachting in de AHI raakt; het residu; en wat dat betekent voor de 0.36.0-claim "op hele
   nachten niet meetbaar".
7. **Geen beslissing uit deze meting.** Of een drager "juist" is, zegt alleen een referentie; de
   meting zegt welk veld welke nacht verplaatst, zodat Bart per veld kan kiezen (terugzetten
   achter het bestaande profielveld is een één-regel-wijziging, geen reparatie).

## Uitvoer en verificatie

Per nacht `R*.json` zonder patiëntgegevens: per arm AHI, ernst, tellingen per type, duale tellingen,
thermistorpoort-uitkomst, CSR-vlaggen, RDI/RERA, hypopneeën desat/arousal, arousals, teruggelezen
veldwaarden; eventlijsten alleen voor A0 en ALL. Kopie naar `/srv/CODE/docs/drift_036x_20261026/`,
analyse `analyse.py`, daarna de meting-verificatie-agent. Verslag:
`docs/drift_036x_decompositie_20261026.md`.
