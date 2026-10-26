# Decompositie van de 0.36.x-drift op de eigen PSG's: één veld draagt apneus én hypopneeën, een tweede de hypopneegradering en RERA's

Datum: 2026-10-26, 16:06–18:24 CET. Preregistratie
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
  `n_thermistor_only` én CSR-vlaggen — en ook op arousaltelling, RERA-telling, RDI, ernstklasse en
  totale eventtelling; residu-AHI 0,0 op elke nacht (A0 ≡ A van 26-10 bovendien op eventniveau; voor
  ALL tegen 10-10 is eventniveau niet toetsbaar, de 10-10-uitvoer draagt geen eventlijsten). **De veertien velden verklaren
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

- **F12 `STAGE_BASELINE_PARTIAL_EPOCHS` (True → False) draagt de apneus én de hypopneeën.** Alleen
  deze arm reproduceert op alle acht nachten de apneutelling van 10-10, inclusief de duale
  verdeling en O/C/M: R02 80 → 107, R04 85 → 28 (beide sensoren 54 → 9), R09 24 → 36, R10 113 →
  101, R12 43 → 34, R14 198 → 158, R16 53 → 73, R03 3 → 3. Hij raakt `ahi_total` op 7/8 nachten
  en is AHI-drager op 5/8, mede-drager op R12. Dezelfde decompositie op de hypopneetelling
  (prereg punt 3): het hypopneegat 10-10 − A0 is −7/+2/+3/−5/−7/+1/+41/−9 en F12 alleen geeft
  −7/0/+2/−6/−5/+6/+42/−10 → **hypopnee-drager op 6/8** (R14 109 → 151): de stadium-basislijn
  verschuift de grens tussen apneu en hypopneu, niet alleen het aantal apneus. Op de RERA-telling
  raakt F12 7/8 nachten en draagt hij het gat op 4/8 (R02, R04, R12, R16).
- **F2 `BREATH_PHASE_SPLIT` (zero_crossing → midpoint) draagt de hypopneegradering en een deel van de
  RERA's.** Hij raakt `ahi_total` op 5/8 (R03 +0,9, R12 −3,3 via hypopneeën 66 → 60, R09 +0,5,
  R04 +0,2, R10 −0,1) — **tegen de verwachting van de preregistratie in** (daar stond F2 bij de
  velden die `ahi_total` nergens bewegen) — en de RERA-telling op 7/8 (R14 38 → 9, R10 36 → 29,
  R02 29 → 25; ΔRDI tot −8,7 op R14), maar hij is drager van het RERA-gat op slechts 3/8 (R09, R10,
  R14) en reproduceert de 10-10-hypopneetelling alleen op R03 (mede-drager R04). Terugzetten van
  F2 alleen brengt de hypopneetellingen dus niet terug naar 0.35.1.
- **F3 `CSR_PERIOD_RECTIFIED` (False → True) draagt de CSR-vlaggen**: A0 0 op alle acht nachten,
  F3 100/16/12/15/0/47/118/0 tegen 10-10 108/16/4/17/0/43/114/0 — bijna, niet exact; ALL exact.
  Raakt `ahi_csr_corrected`, niet `ahi_total`.
- **De tien andere velden, F3 en de constante — twaalf van de veertien armen — bewegen `ahi_total`
  op geen enkele nacht** (0,0 op 8/8): F1, F3, F4, F5, F6, F7 herstelbasislijn, F8, F9, F10, F11
  SpO2-nadir, F13 anti-alias, F14 dode-thermistorvloer (F13 bereikte de poort wél: bandvermogen
  verschuift op de derde decimaal; F7 beweegt alleen FRI, F8 alleen de REM-AHI-alias). De
  verwachting uit de preregistratie dat F7, F11, F13 en F14 kandidaat-dragers waren, is weerlegd;
  alleen F12 van de vijf kandidaten draagt. F14 raakt wel R03 (de thermistor is daar in 0.37.0
  "dood" en afgekeurd, zonder de vloer niet) — AHI onveranderd, want de thermistor levert daar 0
  apneus, maar de RDI daalt er van 47,1 naar 38,0 (RERA 51 → 31): op R03 draagt F14 het RERA-gat.
- **Additiviteit:** Σ_f en ΔALL verschillen ≤ 0,6 /u op elke nacht (R12 −5,0 tegen −4,4: F2 en
  F12 raken deels dezelfde events); geen interactie > 1 /u.

## 3. Wat F12 is, en waarom MESA en PSG-IPA het niet zagen

`stage_baseline_partial_epochs` (0.36.0): de stadium-specifieke basislijn van de flow-envelope
gooide `IndexError` zodra `len(hypno)·samples_per_epoch < n` — het hypnogram korter dan het
signaal — en "de aanroeper slikte dat op DEBUG"; CHANGELOG v0.36.0: 'De "stage-specifieke
basislijn" uit README en paper draaide dus op de meeste echte opnames niet.' Sinds 0.36.0 draait hij
ook dan. Diezelfde CHANGELOG-passage en de docstring van `signal.compute_stage_baseline` noemen het
vloerlengte-geval "het normale geval met YASA/NSRR" — volgens de bron faalde de oude code dus óók
op MESA. Dat klopt niet voor de harnassen: `validate_mesa.parse_nsrr` én
`validate_psgipa.parse_scorer_file` bouwen het hypnogram op **ceil**-lengte
(`n_epochs = ceil(duur / 30)`), zodat het de envelope dekt en beide codepaden samenvallen; YASA in
YASAFlaskified levert een **vloerlengte**-hypnogram. De meting van 12-10
(`docs/reviewvelden_hermeting_20261012.md`) vond het veld op MESA n = 30 en PSG-IPA n = 5 "gelijk in
F1, AHI en eventtelling (0/0/30)" en noemt "MESA-hypnogrammen dekken de opname exact" als een van
drie ontwerpredenen waarom de meeste velden daar byte-identiek zijn. Dat de basislijn op de eigen
PSG's tot 12-10 stil faalde, is afgeleid, niet waargenomen: geen van de drie runs legde
`respiratory.stage_baseline_used` vast, maar bij exacte dekking zijn de twee codepaden identiek,
dus een F12-effect impliceert dat het oude pad daar een `IndexError` gooide (op R04 en R09 is de
afgeronde duur precies n_epochs × 0,5 min; de overige nachten lopen 6–24 s door). Gevolg:

- **Tot 0.36.0 scoorde de kliniek haar eigen nachten zónder stadium-basislijn, een pad dat nooit
  op MESA of PSG-IPA is gevalideerd** (harnaseigenschap: ceil-hypnogram); alle MESA-replicaties
  (F1, bias, ernst) gelden voor de detector mét stadium-basislijn. 0.36.0 bracht de eigen nachten
  op het gevalideerde pad. Nuance: PSG-IPA is éénkanaals, dus de hypopnee-tak draait daar zonder
  eigen stadium-basislijn (`reuse`); PSG-IPA valideerde alleen de hoofdflow-basislijn. Welke van de
  twee basislijnen de verschuiving op de eigen PSG's draagt, zegt deze meting niet; de
  inkortingstoets op MESA (twee sensoren) dekt beide.
- Dat pad telt op deze nachten fors anders: R04 ×3 apneus (9,0 → 19,8 /u, licht → matig), R14
  +25 % apneus bij −27 % hypopneeën, R02/R09/R16 −25 … −33 % apneus (R16 ernstig → matig).
- De claim "op hele nachten niet meetbaar" was juist voor de cohorten waarop hij gemeten is en
  verkeerd voor de montage waarvoor hij bedoeld was; de meting van 12-10 kon het per constructie
  niet zien.

## 4. Wat dit betekent

- **De drift is verklaard en herleidbaar tot twee velden**: `stage_baseline_partial_epochs`
  (apneus én hypopneeën, 7/8 nachten; RERA-gat 4/8) en `breath_phase_split` (hypopneegradering,
  AHI 5/8; RERA-gat 3/8); `csr_period_rectified` verklaart de verdwenen CSR-vlaggen en de
  dode-thermistorvloer het RERA-gat op R03. Elk is een profielveld: terugzetten is één regel in
  `profiles.py`, geen reparatie — maar F2 alleen terugzetten brengt de hypopneeën niet terug.
- **Welke apneutelling juist is, zegt deze meting niet.** De stadium-basislijn is de gevalideerde
  configuratie (MESA F1/bias zijn ermee gemeten), maar of de partial-epoch-implementatie op een
  YASA-hypnogram dat korter is dan de EDF hetzelfde doet als op een dekkend hypnogram, is een
  aparte vraag — te toetsen door op MESA het hypnogram kunstmatig in te korten (zelfde nacht,
  hypnogram −2 %, −5 %) en te meten of de apneutelling stabiel blijft; stabiel → de nieuwe
  eigen-PSG-cijfers zijn de gevalideerde detector, instabiel → de implementatie is het probleem.
  Dat is een kleine prereg (MESA-nachten uit de al gebruikte n150, geen verse trekking nodig).
- `breath_phase_split` is op MESA neutraal gemeten; op PSG-IPA haalde de oude waarde (midpoint)
  4/5 ernstklassen tegen 3/5 voor zero_crossing (n = 5, één opname wisselt; "binnen bereik" 2/5 in
  beide). Op de eigen nachten verschuift hij RERA's (R14 38 → 9) — RDI-lezers merken dat.
- Geen default-wissel uit deze meting (prereg punt 7). Beslissingen bij Bart: (a) F12 laten
  staan (gevalideerd pad) of terugzetten tot de inkortingstoets is gedaan; (b) F2 idem;
  (c) de inkortingstoets op MESA preregistreren.
- Niets aan bibliotheek, profielen of productieconfiguratie is gewijzigd.

## 5. Verificatie

Onafhankelijk nagerekend (meting-verificatie, 26-10, eigen script uit de ruwe JSON's): identiteit
A0 ≡ A (8/8, ook op eventniveau), volledigheid ALL = 10-10 (8/8 op AHI, O/C/M, hypopneeën, duale
drietallen, CSR, arousals, RERA, RDI, ernst), alle cellen en fracties van §2, de F12-apneutellingen
inclusief duale drietallen, de F3-CSR-reeks, de twaalf nul-armen, de 128/128 teruggelezen
veldwaarden, looptijden en percentages repliceren; de prereg is vóór de run gecommit en daarna niet
gewijzigd. **Besluit staat, met herformulering.** Verwerkt uit de 11 bevindingen: F12 draagt ook de
hypopneeën (6/8) en het RERA-gat op 4/8, F2 het RERA-gat op 3/8 en F14 op R03 — de kopclaim "F2
draagt de hypopneeën en de RERA's" is vervangen; F14 verlaagt op R03 de RDI 47,1 → 38,0; de
weerlegde prereg-verwachting voor F2 is benoemd; "elf andere velden" → twaalf van veertien armen;
CET; CHANGELOG letterlijk geciteerd en de tegenspraak ("het normale geval met YASA/NSRR") met de
ceil-hypnogrammen van beide harnassen benoemd; "faalde stil" als afleiding gemarkeerd;
PSG-IPA-nuance (éénkanaals); de breath_phase_split-zin op PSG-IPA in de juiste richting. De
commitboodschap van f773a06 en het geheugen droegen de oude F2-lezing (geheugen bijgewerkt). Niet
verifieerbaar vanaf de werkstation: image, CPU-toewijzing, bereikbaarheidscontroles, jobqueue,
rookproefduur.
