# Decompositie van de 0.36.x-drift op de eigen PSG's: één veld draagt de apneus, één de hypopneeën

Datum: 2026-10-26, 16:06–18:24 CEST. Preregistratie
`docs/drift_036x_decompositie_preregistratie_20261026.md` (33157ff, geschreven vóór de meting).
In situ op de Hetzner-server in een losse container uit het productie-image YASAFlaskified 0.39.2
(psgscoring **0.37.0**, versiecontrole per nacht), uploads read-only, 16 van 32 CPU's, `nice`,
8 nachten parallel met 16 armen sequentieel per nacht (67–125 min per nacht, 2 u 5 min totaal na een
rookproef van 13 min op R04); de app antwoordde op elke bereikbaarheidscontrole met 200, de
jobqueue was leeg bij beide starts. Dezelfde invoer als op 10-10 en 26-10; profiel
`aasm_v3_breath_dual`, LGBM-arousalketen. Per nacht: **A0** (0.37.0 zoals het is), **F1 … F14**
(één 0.36.0-profielveld, of de 0.36.2-constante, in-process terug op de 0.35.1-waarde — de
teruggelezen waarde is per arm vastgelegd: 112/112 zoals bedoeld) en **ALL** (alle veertien
terug). Ruwe uitvoer zonder patiëntgegevens `/srv/CODE/docs/drift_036x_20261026/R*.json`, analyse
`analyse.py`, uitvoer `analyse_uitvoer.md`, `samenvatting.json`. Verschilmeting zonder referentie.

## 1. Identiteit en volledigheid

- **A0 ≡ arm A van 26-10 op 8/8** (AHI, apneus O/C/M, hypopneeën, duale tellingen, arousals).
- **ALL = 10-10 (0.35.1) op 8/8** op AHI, apneus O/C/M, hypopneeën, `n_both`/`n_pressure_only`/
  `n_thermistor_only` én CSR-vlaggen; residu-AHI 0,0 op elke nacht. **De veertien velden verklaren
  het hele verschil; de code buiten de velden (kruiscontaminatie, snappen, MMSD-lijst,
  kanaalkwaliteit, effortpoort 5b) draagt op deze nachten niets bij.**

## 2. Drager per nacht

ΔAHI per veld (F − A0), vet = drager (≥ 50 % van het gat 10-10 − A0, zelfde teken):

| R | gat 10-10 − A0 | F2 fasesplitsing | F12 stadium-basislijn | overige 12 velden | Σ_f | ΔALL |
|---|---|---|---|---|---|---|
| R02 | +2,7 | 0,0 | **+2,7 (100 %)** | 0,0 | +2,7 | +2,7 |
| R03 | +0,9 | **+0,9 (100 %)** | 0,0 | 0,0 | +0,9 | +0,9 |
| R04 | −10,8 | +0,2 | **−11,0 (102 %)** | 0,0 | −10,8 | −10,8 |
| R09 | +3,4 | +0,5 | **+2,9 (85 %)** | 0,0 | +3,4 | +3,4 |
| R10 | −2,3 | −0,1 | **−2,1 (91 %)** | 0,0 | −2,2 | −2,3 |
| R12 | −4,4 | **−3,3 (75 %)** | −1,7 (39 %, mede-drager) | 0,0 | −5,0 | −4,4 |
| R14 | +0,3 | 0,0 | +0,6 | 0,0 | +0,6 | +0,3 |
| R16 | +2,3 | 0,0 | **+2,1 (91 %)** | 0,0 | +2,1 | +2,3 |

- **F12 `STAGE_BASELINE_PARTIAL_EPOCHS` (True → False) draagt de apneus.** Alleen deze arm
  reproduceert op alle acht nachten de apneutelling van 10-10, inclusief de duale verdeling:
  R02 80 → 107, R04 85 → 28 (beide sensoren 54 → 9), R09 24 → 36, R10 113 → 101, R12 43 → 34,
  R14 198 → 158 (hypopneeën 109 → 151), R16 53 → 73, R03 3 → 3. Hij raakt `ahi_total` op 7/8
  nachten en is drager op 5/8, mede-drager op R12.
- **F2 `BREATH_PHASE_SPLIT` (zero_crossing → midpoint) draagt de hypopneeën en de RERA's.** Hij
  raakt `ahi_total` op 5/8 (R03 +0,9, R12 −3,3 via hypopneeën 66 → 60, R09 +0,5, R04 +0,2, R10
  −0,1) en de RERA-telling op 7/8 (R14 38 → 9, R10 36 → 29, R02 29 → 25); ΔRDI tot −8,7 (R14).
- **F3 `CSR_PERIOD_RECTIFIED` (False → True) draagt de CSR-vlaggen**: A0 0 op alle acht nachten,
  F3 100/16/12/15/0/47/118/0 tegen 10-10 108/16/4/17/0/43/114/0 — bijna, niet exact; ALL exact.
  Raakt `ahi_csr_corrected`, niet `ahi_total`.
- **De elf andere velden en de constante bewegen `ahi_total` op geen enkele nacht** (0,0 op 8/8):
  F1, F4, F5, F6, F7 herstelbasislijn, F8, F9, F10, F11 SpO2-nadir, F13 anti-alias, F14 dode-
  thermistorvloer. De verwachting uit de preregistratie dat F7, F11, F13 en F14 kandidaat-dragers
  waren, is weerlegd; alleen F12 van de vijf kandidaten draagt. F14 raakt wel R03 (de thermistor
  is daar in 0.37.0 "dood" en afgekeurd, zonder de vloer niet) — AHI onveranderd, want de
  thermistor levert daar 0 apneus.
- **Additiviteit:** Σ_f en ΔALL verschillen ≤ 0,6 /u op elke nacht (R12 −5,0 tegen −4,4: F2 en
  F12 raken deels dezelfde events); geen interactie > 1 /u.

## 3. Wat F12 is, en waarom MESA en PSG-IPA het niet zagen

`stage_baseline_partial_epochs` (0.36.0): de stadium-specifieke basislijn van de flow-envelope
gooide `IndexError` zodra `len(hypno)·samples_per_epoch < n` — het hypnogram korter dan het
signaal — en de aanroeper slikte dat op DEBUG, zodat "de stage-specifieke basislijn uit README en
paper op de meeste echte opnames niet draaide". Sinds 0.36.0 draait hij ook dan. De meting van
12-10 (`docs/reviewvelden_hermeting_20261012.md`) vond het veld op MESA n = 30 en PSG-IPA n = 5
"gelijk in F1, AHI en eventtelling (0/0/30)" met als verklaring: **"MESA-hypnogrammen dekken de
opname exact"** — daar faalde de oude code dus niet en draaide de stadium-basislijn al. Op de eigen
PSG's (YASA-hypnogram, EDF loopt door na het laatste epoch) faalde hij stil tot 12-10. Gevolg:

- **Tot 0.36.0 scoorde de kliniek haar eigen nachten zónder stadium-basislijn, een pad dat nooit
  op MESA of PSG-IPA is gevalideerd**; alle MESA-replicaties (F1, bias, ernst) gelden voor de
  detector mét stadium-basislijn. 0.36.0 bracht de eigen nachten op het gevalideerde pad.
- Dat pad telt op deze nachten fors anders: R04 ×3 apneus (9,0 → 19,8 /u, licht → matig), R14
  +25 % apneus bij −27 % hypopneeën, R02/R09/R16 −25 … −33 % apneus (R16 ernstig → matig).
- De claim "op hele nachten niet meetbaar" was juist voor de cohorten waarop hij gemeten is en
  verkeerd voor de montage waarvoor hij bedoeld was; de meting van 12-10 kon het per constructie
  niet zien.

## 4. Wat dit betekent

- **De drift is verklaard en herleidbaar tot twee velden**: `stage_baseline_partial_epochs`
  (apneus, 7/8 nachten) en `breath_phase_split` (hypopneeën/RERA, 5/8); `csr_period_rectified`
  verklaart de verdwenen CSR-vlaggen. Elk is een profielveld: terugzetten is één regel in
  `profiles.py`, geen reparatie.
- **Welke apneutelling juist is, zegt deze meting niet.** De stadium-basislijn is de gevalideerde
  configuratie (MESA F1/bias zijn ermee gemeten), maar of de partial-epoch-implementatie op een
  YASA-hypnogram dat korter is dan de EDF hetzelfde doet als op een dekkend hypnogram, is een
  aparte vraag — te toetsen door op MESA het hypnogram kunstmatig in te korten (zelfde nacht,
  hypnogram −2 %, −5 %) en te meten of de apneutelling stabiel blijft; stabiel → de nieuwe
  eigen-PSG-cijfers zijn de gevalideerde detector, instabiel → de implementatie is het probleem.
  Dat is een kleine prereg (MESA-nachten uit de al gebruikte n150, geen verse trekking nodig).
- `breath_phase_split` is op MESA neutraal en op PSG-IPA 4/5 i.p.v. 3/5 in ernstbereik gemeten;
  op de eigen nachten verschuift hij RERA's (R14 38 → 9) — RDI-lezers merken dat.
- Geen default-wissel uit deze meting (prereg punt 7). Beslissingen bij Bart: (a) F12 laten
  staan (gevalideerd pad) of terugzetten tot de inkortingstoets is gedaan; (b) F2 idem;
  (c) de inkortingstoets op MESA preregistreren.
- Niets aan bibliotheek, profielen of productieconfiguratie is gewijzigd.

## 5. Verificatie

[Meting-verificatie volgt; bevindingen worden hieronder verwerkt.]
