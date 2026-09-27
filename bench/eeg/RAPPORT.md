# EEG-arousaldetectie: halen recente, vrij gelicentieerde methodes op PSG-IPA een hogere event-F1 dan psgscoring?

**Onafhankelijk geverifieerd op 27-09 (meting-verificatie-agent): kerncijfers reproduceren exact; correcties 1–8 verwerkt.**

*dsp-scout, 2026-09-27. Werkmap `bench/eeg/`. Alles hieronder is gemeten met de
scripts in deze map; niets in `psgscoring/`, `scripts/`, `docs/` of `tests/` is
aangeraakt, er is niets gecommit, en onder `/srv/DATA` is alleen de registerregel
`## dsp-scout eeg 2026-09-27 (400)` bijgeschreven.*

**Menselijk plafond voor arousal-F1 op PSG-IPA: 0,679 (330 scoorderparen,
`docs/arousal_menselijk_plafond.md`). Elk F1-cijfer hieronder hoort naast dat getal.**

## 1. Vraag

Welke arousaldetectiemethodes van de laatste drie jaar, met een licentie die in een
BSD-3-bibliotheek past, halen op PSG-IPA een hogere event-niveau-F1 dan de huidige
detector (`psgscoring.arousal.detect_arousals_multi`: multi-derivatie-union +
LightGBM `arousal_classifier_v3` op werkpunt 0,70, 10 s-minimuminterval, onset-offset
+2 s, autonome re-ranker waar Pleth/HR bestaat)?

## 2. Wat al beoordeeld was (niet opnieuw geprototypeerd)

| idee | waar | stand |
|---|---|---|
| CAISR/ABED (SLEEP 2025, Nat Commun 2026) | `docs/benchmark_caisr_abed_20260904.md:5,24`; `docs/third_party_comparison.md` (kop) | **CC BY-NC 4.0 → niet porteerbaar**; op MESA n=50 lokaal gedraaid voor respiratoire events (F1 0,234 tegen onze 0,410); hun arousalmodule viel op MESA in de 2-kanaals-terugval. Arousal-CSV's liggen onder `/home/bart/caisr_mesa/` (voor deze sessie onleesbaar). Niet opnieuw gedaan. |
| PhysioNet-2018/DeepSleep (U-Net) als referentiepunt | `docs/literatuur_algoritmes_20260904.md:11,44-45` | genoemd (AUPRC 0,55) maar nooit op PSG-IPA gedraaid — hier gedaan via de MIT-opvolger DeepSleep 2.0 (kandidaat 1). |
| Eigen 1D-U-Net (462k par., 113 nachten) | `docs/nacht_20260901_bevindingen.md:113` | dat was een detector voor **respiratoire** events, geen arousals. |
| Commerciële arousaldetectie (EnsoSleep PA 73,6 %, MICHELE κ 0,54, Somnolyzer autonoom) | `docs/commerciele_autoscoring_20260907.md:37,55,69-70,78` | proprietair; alleen index/epoch-PA's, geen event-F1 — niet vergelijkbaar en niet beschikbaar. |
| Eigen classifiervarianten v4/v5/v6, drempelorakel, kandidaatdrempels | `docs/arousal_waar_het_gat_zit.md`, `docs/arousal_foutanalyse_verworpen_kandidaten.md`, `docs/arousal_v4/v5/v6_preregistratie.md`, `docs/arousal_kandidaatdekking_bevinding.md` | alle weerlegd; het gat zit in de SELECTIE (pool-orakel 0,896 tegen 0,514). Dat is precies de plek waar een extern model iets anders kan doen. |
| Spectrale verschuiving, hysterese, EOG-reject, brede alfaband, REM-alfa-basislijn, event-locked werkpunt, wake-arousals | `psgscoring/profiles.py:547,580,848,871,1184,1200,1458`; `CHANGELOG.md:1131,1926` | gebouwd, gemeten, default uit — regelvarianten van de eigen detector, geen externe methodes. |
| Multi-derivatie, artefactlijst negeren, classifier default, 10 s-regel, re-ranker | `CHANGELOG.md:1078,1578,1712,663,112`; `profiles.py:746,813,855` | de huidige baseline ís de som van die beslissingen. |

## 3. Literatuur en licenties (≤ 3 jaar, tenzij referentiestandaard)

| # | methode | bron | licentie | claim (eigen harnas van de auteurs) | hier |
|---|---|---|---|---|---|
| K1 | **DeepSleep 2.0** — 1D-U-Net, 13 kanalen @200 Hz, per-sample segmentatie | Fonod, *AI* 2022, doi:10.3390/ai3010010; github.com/rfonod/deepsleep2 (Zenodo 2024) — compacte opvolger van de PhysioNet-2018-winnaar DeepSleep (Li & Guan, Commun Biol 2020) | **MIT** (code + gewichten) | gross AUPRC 0,450 / AUROC 0,901 op 249 PhysioNet-2018-nachten (doel: *niet-apneu*-arousals) | geprototypeerd: voorgetraind `model_2`, plus eerlijke MESA-ijking |
| K2 | **MSED** — SplitStreamNet (DOSED/SSD-stijl ankers 3/15/30 s + bi-GRU + aandacht), 10 kanalen @128 Hz, gezamenlijk arousal/LM/SDB | Zahid, Jennum, Mignot, Sørensen, *IEEE TBME* 70(9) 2023, doi:10.1109/TBME.2023.3252368; github.com/neergaard/msed | **MIT** (code + gewichten) | arousal-F1 **0,70** (P 0,76 / R 0,67) op 1000 MrOS-hold-outnachten, matching IoU 0,5; drempel 0,64 uit hun eval-set | geprototypeerd: voorgetraind `splitstream`, upstream-drempel |
| K3 | **U-Net 50 Hz** (EEG+EOG+kin-EMG), kernel 21 | Ehrlich e.a., *Sci Rep* 14, 2024, doi:10.1038/s41598-024-67022-9; gitlab.com/sleep-is-all-you-need/arousaldetector | **GPL → NIET-PORTEERBAAR**; gewichten niet gebruikt | AUPRC 0,82 / F1 0,81 op MESA (elke overlap = TP), 0,83/0,80 SHHS, 0,71/0,74 op 3423 klinische PSG's | **eigen schone herimplementatie** uit het artikel, getraind op 317 verse MESA-nachten |
| — | FullSleepNet (FCN+RNN+aandacht, één EEG-kanaal, multitask arousal+staging) | Zan & Yildiz, *J Neural Eng* 20:056034, 2023 (arXiv 2406.01834); github.com/hasanzan/FullSleepNet (Keras-gewichten voor SHHS en MESA) | **geen licentie** in de repo (GitHub API: `license: null`) → NIET-PORTEERBAAR/onbekend | AUPRC 0,70 arousal op SHHS/MESA | niet geprototypeerd (licentie); een herimplementatie zou een tweede U-Net-achtige zijn zonder gepubliceerde hyperparameters |
| — | Multi-task objectdetectie voor events + hypnogram | Anido-Alonso & Alvarez-Estevez, arXiv 2501.09519 (2025) | geen code gevonden | geen arousal-F1 in het abstract | overgeslagen |
| — | CAISR (arousal-module) | Thomas/Westover e.a., *SLEEP* 2025 | CC BY-NC 4.0 → niet porteerbaar | κ 0,45 ≈ technoloog | zie §2 |
| — | SleepFM (multimodaal foundation-model) | Thapa e.a., *Nat Med* 2025; github.com/zou-group/sleepfm-clinical | gewichten **CC BY-NC 4.0** → niet porteerbaar | geen arousal-taak | overgeslagen |
| — | PSG-MAE (maskerende autoencoder, staging + apneu) | Wang e.a., *IEEE TNSRE* 2025; github.com/yfw-scut/PSG-MAE | geen licentie, geen gewichten | geen arousal-taak | overgeslagen |
| — | SLEEPYLAND / U-Sleep / YASA | npj Digit Med 2026; MIT / BSD-3 | bruikbaar | alleen staging + spindels/K-complexen/SO — **geen arousaldetector** | n.v.t. |
| — | Somnolyzer autonome arousals (Philips, Front Physiol 2023), MICHELE/ORP, EnsoSleep | proprietair | — | zie `docs/commerciele_autoscoring_20260907.md` | n.v.t. |

Andere PhysioNet-2018-inzendingen (2018–2020) en DOSED (Chambon 2019, MIT — waar MSED op
voortbouwt) vallen buiten de drie jaar en zijn door K1/K2 gedekt.

## 4. Opzet

- **Referentie.** PSG-IPA SN1–SN5, de scoorder-1-kopie van de gedeelde EEG-arousalannotatie
  van de Resp_events-subboom, gelezen uit `Resp_events/Annotations/manual/SNx_Respiration_manual_scorer1.edf`
  ("eeg arousal", `common/export_ref.py`) — identiek aan de referentie van de orakelmeting
  (`docs/orakel_rule1a_psgipa.json`). De twaalf kopieën verschillen ≤ 1 event per opname
  (SN1/SN5 12/12 identiek; SN2 scoorder 7 heeft 95 i.p.v. 96; SN3 scoorder 1 heeft 241 waar
  tien scoorders 242 hebben; SN4 scoorder 10 heeft 59): 37 / 96 / 241 / 58 / 164 arousals (596 totaal;
  arousal-index 6,4 / 19,8 / 39,8 / 9,6 / 23,4 per uur), mediane duur 6,5–9,7 s. Hypnogram
  van scoorder 1 (`validate_psgipa.parse_scorer_file`). Signalen: de Resp_events-EDF's
  (256 Hz; F4-M1, C4-M1 — Cz-M1 op SN5 —, O2-M1, E1/E2-M2, kin, ECG, LAT/RAT, nasale druk,
  buik, borst, SaO2).
- **Baseline.** `psgscoring.run_pneumo_analysis(raw, hypno, scoring_profile="aasm_v3_rec")`
  uit de geïnstalleerde bibliotheek (0.34.2), `result["arousal"]["events"]`
  (`common/run_baseline.py`).
- **Evaluatie.** `bench/evaluate.py`, `--matcher project` (IoU 0,20, typeonbewust — de
  matcher van alle gepubliceerde cijfers van dit project) én `--matcher onset --threshold 5`;
  sensitiviteit, PPV, F1 per opname en gepoold (som van TP/FP/FN over de vijf). Ook de
  mediaan van de vijf opname-F1's staat erbij, omdat SN3 40 % van de referentie draagt.
- **Gelijke nabewerking** voor de kansmodellen (`common/postproc.py`): 1 s-gemiddelde,
  drempel, gaten ≤ 1 s dicht, ≥ 3 s (AASM), slaappoort op het scoorder-1-hypnogram (psgscoring
  scoort zelf ook geen arousals in W). MSED levert zelf events; daar alleen de slaappoort.
- **Werkpunten zonder PSG-IPA.** DeepSleep2: 0,50 vooraf (geen eigen drempel) + ijking op
  MESA; MSED: de upstream-drempel 0,64 + ijking op MESA; U-Net: drempel en vroegtijdig
  stoppen op MESA-validatienachten. Drempelvegen op PSG-IPA staan er wél, maar als
  **orakel** gelabeld.
- **MESA als tweede cohort en trainingsbron.** 400 nachten die in geen enkele eerdere sectie
  van `/srv/DATA/MESA/gebruikte_mesa_ids.txt` stonden (seed 20260927): 320 training / 80
  validatie (`unet50/ids_train.txt`, `ids_val.txt`); geregistreerd als
  `## dsp-scout eeg 2026-09-27 (400)`. Vier nachten vielen af (0 NSRR-arousals of te weinig
  slaap): 317 train / 79 validatie. Op de validatienachten draaien ook psgscoring
  (`common/run_baseline_mesa.py`) en MSED (`msed/run_msed_sweep.py --cohort mesa`), zodat de
  drie tegen dezelfde NSRR-referentie staan (één scoorder per nacht; absolute F1 lager dan
  op PSG-IPA — het gaat om het paar).
- **Rekenomgeving.** Aparte uv-venv `bench/eeg/_venv` (torch 2.11+cu128, mne 1.13, MSED als
  pakket); niets in de projectomgeving geïnstalleerd. GPU RTX A4000; CPU-werk met
  `OMP_NUM_THREADS=1`, maximaal 8 (baseline PSG-IPA 5, MESA-laden 8, MESA-baseline 6)
  processen tegelijk.

## 5. Resultaten op PSG-IPA (n = 5, 596 referentie-arousals)

Volledige tabellen (incl. onset-matcher per opname en alle drempelvegen): `results/tabel.md`,
ruwe cijfers `results/resultaten.json`. Gepoold = som van TP/FP/FN over de vijf opnames.

### 5.1 Hoofdtabel — projectmatcher (IoU 0,20) en onset ±5 s, gepoold

| arm | werkpunt vastgelegd op | n_pred / n_ref | ratio | sens | PPV | **F1 IoU 0,20** | mediaan F1 per opname | F1 onset ±5 s | menselijk plafond |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **baseline** psgscoring 0.34.2 `aasm_v3_rec` | productie (0,70) | 640 / 596 | 1,07 | 0,576 | 0,536 | **0,555** | 0,598 | 0,526 | 0,679 |
| K1 DeepSleep2 model_2, τ 0,50 | vooraf | 224 / 596 | 0,38 | 0,008 | 0,022 | **0,012** | 0,023 | 0,012 | 0,679 |
| K1 DeepSleep2, τ 0,05 uit MESA-val | MESA (59 nachten) | 1064 / 596 | 1,79 | 0,186 | 0,104 | **0,134** | 0,137 | 0,027 | 0,679 |
| K1 DeepSleep2, τ 0,05 + onset +28,2 s / offset +0,3 s uit MESA-val | MESA (59 nachten) | 669 / 596 | 1,12 | 0,475 | 0,423 | **0,447** | 0,275 | 0,372 | 0,679 |
| K2 MSED splitstream, τ 0,64 (upstream), C3 := C4-M1 | upstream (MrOS-eval) | 527 / 596 | 0,88 | 0,574 | 0,649 | **0,609** | 0,564 | 0,645 | 0,679 |
| K2 MSED, τ 0,64, C3 := F4-M1 (gevoeligheid) | upstream | 512 / 596 | 0,86 | 0,560 | 0,652 | **0,603** | 0,532 | 0,625 | 0,679 |
| K2 MSED, τ 0,30 uit MESA-val | MESA (80 nachten) | 934 / 596 | 1,57 | 0,614 | 0,392 | **0,478** | 0,412 | 0,473 | 0,679 |
| **K3 U-Net-50Hz** (eigen, 317 MESA-nachten), τ 0,20 uit MESA-val | MESA (79 nachten) | 735 / 596 | 1,23 | 0,819 | 0,664 | **0,733** | 0,719 | 0,723 | 0,679 |

### 5.2 Per opname — F1 (IoU 0,20), met sens / PPV en aantal voorspelde events

| arm | SN1 (ref 37) | SN2 (ref 96) | SN3 (ref 241) | SN4 (ref 58) | SN5 (ref 164) | beter dan baseline op | gepoold ΔF1 |
|---|---|---|---|---|---|---:|---:|
| baseline | 0,68 / 0,61 / **0,641** (41) | 0,64 / 0,45 / **0,524** (137) | 0,58 / 0,69 / **0,630** (200) | 0,59 / 0,23 / **0,335** (145) | 0,51 / 0,72 / **0,598** (117) | — | — |
| K1 DeepSleep2 τ 0,50 | **0,023** (49) | **0,000** (2) | **0,000** (117) | **0,026** (19) | **0,030** (37) | 0/5 | −0,543 |
| K1 DeepSleep2 MESA-geijkt + verschuiving | 0,57 / 0,18 / **0,275** (116) | 0,17 / 0,41 / **0,237** (39) | 0,75 / 0,66 / **0,699** (274) | 0,34 / 0,20 / **0,253** (100) | 0,28 / 0,33 / **0,303** (140) | 1/5 | −0,108 |
| K2 MSED τ 0,64 | 0,57 / 0,44 / **0,494** (48) | 0,50 / 0,69 / **0,578** (70) | 0,71 / 0,81 / **0,754** (210) | 0,48 / 0,29 / **0,361** (97) | 0,46 / 0,74 / **0,564** (102) | 3/5 | +0,054 |
| K2 MSED τ 0,30 (MESA) | 0,68 / 0,19 / **0,294** (133) | 0,51 / 0,35 / **0,412** (142) | 0,69 / 0,54 / **0,607** (309) | 0,43 / 0,13 / **0,202** (190) | 0,61 / 0,62 / **0,617** (160) | 1/5 | −0,077 |
| **K3 U-Net τ 0,20** | 0,86 / 0,62 / **0,719** (52) | 0,70 / 0,56 / **0,623** (119) | 0,93 / 0,81 / **0,867** (276) | 0,74 / 0,33 / **0,455** (131) | 0,74 / 0,78 / **0,760** (157) | **5/5** (+0,078 … +0,236) | **+0,178** |

*Richtpunt, geen rij in de vergelijking:* het menselijke plafond per opname is 0,642 / 0,492 /
0,692 / 0,767 / 0,766 (alle paren 0,679; `docs/arousal_menselijk_plafond.md`). Dat is
scoorder-tegen-scoorder gemeten in de **EEG_arousals**-subboom — een andere export met een
andere duur (SN3: 8,13 u tegen 6,57 u hier) en twaalf onafhankelijke arousalsets — terwijl
alle F1's hierboven tegen één vaste annotatie in de Resp_events-subboom staan. Geen scoorder
is ooit tegen die Resp_events-annotatie gemeten; de plafonds zijn dus een richtpunt naast de
cijfers, geen vergelijkbare rij (`docs/arousal_menselijk_plafond.md`, §Beperkingen).

Onset ±5 s per opname (F1): baseline 0,590 / 0,498 / 0,571 / 0,355 / 0,584; MSED τ 0,64
0,494 / 0,578 / 0,789 / 0,400 / 0,632; U-Net 0,674 / 0,586 / 0,867 / 0,487 / 0,735.

### 5.3 Drempelvegen op PSG-IPA — ORAKEL (alleen om de robuustheid van het werkpunt te tonen)

- **U-Net**: F1 0,722–0,748 over τ 0,15–0,50 (maximum 0,748 op 0,25; op 0,35 is de telling
  neutraal: ratio 1,01, F1 0,743). Het MESA-gekozen 0,20 zit op dat plateau; de winst hangt
  niet aan de drempel.
- **MSED**: maximum 0,614 op τ 0,60; het upstream-werkpunt 0,64 (0,609) ligt daar vlak
  onder. De MESA-gekozen 0,30 is voor PSG-IPA te laag (PPV 0,39).
- **DeepSleep2**: zonder verschuiving nooit boven 0,147 (τ 0,10).

### 5.4 Wat de fouten zeggen

- **DeepSleep2 is het verkeerde doel, geen kapotte pijplijn.** Sample-AUROC tegen de
  referentie 0,71–0,82 op SN1/2/4/5 (0,47 op SN3, de OSA-nacht), maar de kans piekt
  systematisch **vóór** de menselijke arousal: kruiscorrelatie-lag −14 … −22 s op alle vijf
  opnames, en op MESA een mediane onsetfout van −28 s bij een offsetfout van −0,3 s. Het
  model markeert de respiratoire spanne die op de arousal uitloopt — precies wat de
  PhysioNet-2018-"target arousals" (RERA-spannen inbegrepen, apneu-arousals gemaskeerd)
  zijn. Zelfs met MESA-geleerde onsetcorrectie blijft 0,447 (F1) en zit de beste drempel
  op de rand van het raster (0,05).
- **MSED wint precisie, niet dekking.** Zelfde referentiedekking als psgscoring (0,574 tegen
  0,576) bij hogere PPV (0,649 tegen 0,536) en 12 % ondertelling. De twee vinden voor een
  deel andere arousals: 245 door beide, 98 alleen psgscoring, 97 alleen MSED, 156 door geen
  van beide; de unie dekt 0,738 (`results/complementariteit.md`). MSED's winst komt van
  SN3 (0,754 tegen 0,630; 40 % van de referentie); op SN1 verliest het 0,147, en de mediaan
  per opname ligt ónder de baseline (0,564 tegen 0,598).
- **De U-Net wint op elke opname en op beide matchers**, met de grootste sprong op SN3
  (+0,236) en de kleinste op SN1 (+0,078, de opname met 37 arousals). Zijn zwakste opname
  blijft SN4 (0,455; PPV 0,33) — dezelfde opname waar de baseline (0,335) en MSED (0,361)
  het slechtst zijn en waar de referentie het dunst is (58 arousals in 8,4 u). De telling
  ligt op τ 0,20 23 % te hoog (SN4: 131 tegen 58; SN2: 119 tegen 96) — op MESA-val was
  dezelfde drempel 11 % te laag. Het werkpunt is dus cohortafhankelijk gekalibreerd; de F1
  niet (§5.3).

## 6. Tweede cohort: de 80 MESA-validatienachten (NSRR, één scoorder per nacht)

Zelfde referentie en matcher voor de drie detectoren; psgscoring via
`run_pneumo_analysis(aasm_v3_rec)` met het NSRR-hypnogram. Hypnogram én referentie-arousals
komen voor alle drie de armen uit dezelfde parser, `unet50/data.parse_mesa_xml` (niet
`scripts/validate_mesa.parse_nsrr`); die knipt het einde van een arousal op de EDF-duur, wat
op mesa-sleep-5733 en -6741 ≤ 1 event raakt, gelijk voor alle armen. Tabel `results/mesa_val.md`.
De U-Net-rij telt 79 nachten (mesa-sleep-5721 heeft 0 NSRR-arousals en is bij het laden
afgevallen); psgscoring staat daarom óók op die 79 gemeenschappelijke nachten.

| arm | n_pred / n_ref | sens | PPV | **F1 gepoold** | mediaan F1 per nacht | nachten beter / slechter dan psgscoring | Wilcoxon (gepaard) |
|---|---:|---:|---:|---:|---:|---:|---:|
| psgscoring 0.34.2 (alle 80) | 11 675 / 12 432 | 0,546 | 0,581 | **0,563** | 0,556 | — | — |
| K2 MSED τ 0,64 (n 80) | 1 672 / 12 432 | 0,099 | 0,736 | **0,174** | 0,017 | 5 / 74 | p = 6·10⁻¹⁴ (slechter) |
| psgscoring 0.34.2 op de 79 gemeenschappelijke nachten | 11 459 / 12 432 | 0,546 | 0,592 | **0,568** | 0,560 | — | — |
| **K3 U-Net τ 0,20 (n 79)** | 11 004 / 12 432 | 0,643 | 0,726 | **0,682** (Δ gepoold +0,114) | 0,673 | **74 / 5** (Δmed +0,112) | **p = 1,28·10⁻¹²** |

- U-Net tegen psgscoring per nacht: ΔF1 p10/p50/p90 = +0,02 / +0,11 / +0,21 (min −0,30,
  max +0,38). Per tertiel van arousallast — snede op de gesorteerde n_ref per nacht van de
  79 gemeenschappelijke nachten, 26 / 27 / 26 nachten, n_ref 32–119 / 126–172 / 185–370 —:
  psgscoring 0,490 / 0,553 / 0,611 → U-Net 0,621 / 0,672 / 0,712 (Δ +0,131 / +0,119 /
  +0,102) — de winst is er in elke laag en het grootst waar weinig te vinden is. Count-ratio per nacht p10/p50/p90: psgscoring
  0,63 / 0,91 / 1,38, U-Net 0,62 / 0,92 / 1,21.
- **Voorbehoud bij dit cohort voor K3:** de validatienachten zijn dezelfde nachten waarop
  drempel én stop-epoch gekozen zijn (een validatie-, geen testset), en MESA is het
  trainingscohort. PSG-IPA (§5) is de externe test; MESA-val zegt vooral dat het effect
  groot en consistent is en niet aan vijf opnames hangt.
- **MSED valt op MESA weg door zijn voorbewerking, niet door zijn detector.** Met de
  upstream-pijplijn (z-score per kanaal over de héle opname) haalt het recall 0,10 (F1 0,174;
  ook bij τ 0,30 maar 0,250): MESA-nachten zijn 10–12 u met lange wakkere randen, PSG-IPA en
  MrOS zijn op lights-off/on geknipt. Diagnose-run achteraf (`msed/run_msed_sweep.py
  --crop-sleep`, `msed/out_sweep_mesa_crop/`, `results/msed_mesa_crop.json`; n = 80): invoer
  geknipt op [eerste slaapepoch − 5 min, laatste + 5 min] en daarbinnen opnieuw
  gestandaardiseerd → F1 **0,534** op τ 0,64 (sens 0,41 / PPV 0,77) en 0,552 op τ 0,50
  (0,47 / 0,67), tegen psgscoring 0,563 op dezelfde 80 nachten. Dat is een post-hoc
  reparatie (het knipvenster is niet vooraf vastgelegd) en brengt MSED hooguit op
  psgscoring-niveau op MESA; het verandert de conclusie niet.
- DeepSleep2 op dezelfde nachten (59 van de 80 pasten in 2^23 samples): F1 0,164 ruw,
  0,292 met verschuivingscorrectie, beide op de rasterrand τ 0,05 (`deepsleep2/calibratie_mesa.json`).

## 7. Conclusie

1. **Ja, er is één methode die het beter doet, en één die het mogelijk beter doet.**
   - **K3, de eigen BSD-schone herimplementatie van de 50 Hz-U-Net van Ehrlich e.a. (2024),
     getraind op 317 verse MESA-nachten, haalt op PSG-IPA F1 0,733 (IoU 0,20) tegen 0,555
     voor de huidige detector, beter op 5 van 5 opnames en op beide matchers (onset ±5 s:
     0,723 tegen 0,526), zonder dat PSG-IPA voor enige keuze is gebruikt.** Op de 79
     gemeenschappelijke MESA-validatienachten is het beeld hetzelfde (0,682 tegen 0,568,
     74/5, p = 1,3·10⁻¹²). Het getal 0,733 ligt numeriek boven het richtpunt 0,679, maar dat
     richtpunt is scoorder-tegen-scoorder gemeten in een andere export (EEG_arousals-subboom,
     andere duur, twaalf onafhankelijke sets) terwijl hier tegen één vaste annotatie in de
     Resp_events-subboom wordt gemeten; de twee zijn niet als rijen van één tabel te lezen
     (`docs/arousal_menselijk_plafond.md`, §Beperkingen). "Boven het plafond" is hier dus geen
     vergelijking met een mens.
   - **K2, MSED (MIT, voorgetraind op MrOS), is op PSG-IPA gepoold +0,054 beter (0,609),
     maar niet robuust:** 3/5 opnames, mediaan per opname lager dan de baseline, en op MESA
     0,174 met de upstream-voorbewerking — 0,534 (τ 0,64) tot 0,552 (τ 0,50) na een post-hoc
     knip op de slaapperiode, nog altijd niet boven psgscoring (0,563). Het levert wél iets wat de huidige detector mist: hogere precisie
     bij gelijke dekking en voor een deel andere arousals (unie 0,738). Als tweede getuige
     naast de eigen kandidatenpool is het interessant; als vervanger niet.
   - **K1, DeepSleep2 (MIT), is niet beter en kan het ook niet worden:** het is op een andere
     doelvariabele getraind (niet-apneu-arousals met RERA-spannen) en vuurt 15–28 s te vroeg.
   - **Niet-porteerbaar en dus niet gemeten:** CAISR (CC BY-NC), SleepFM (CC BY-NC),
     Ehrlich's eigen GPL-gewichten, FullSleepNet (geen licentie), alle commerciële systemen.
2. **Waar de winst vandaan komt.** Het project wist al dat het gat in de SELECTIE zit
   (pool-orakel 0,896 tegen 0,514) en dat drie hertrainingen van dezelfde 50-feature-
   classifier niets opleverden. Een sequentiemodel dat de golfvorm van EEG+EOG+EMG op
   50 Hz leest, zonder kandidaatstap en zonder handgemaakte features, doet die selectie
   beter — én sensitiever (0,819 tegen 0,576) — met 4,6 M parameters, ~50 min training op
   één GPU en 2 s CPU-inferentie per nacht. Dat weerspreekt de aanname uit
   `docs/literatuur_algoritmes_20260904.md` §3 dat waveform-modellen pas op duizenden
   opnames winnen: voor arousals volstaan ~300 nachten met een goede normalisatie.
3. **Geen aanbeveling tot uitrol — dit is een bench-resultaat.** Wat een pre-geregistreerde
   meting (huisregels, `docs/*preregistratie*.md`) vooraf moet vastleggen voordat hier iets
   van in `psgscoring` komt:
   - het werkpunt en de telling: τ 0,20 telt op PSG-IPA 23 % te veel en op MESA 11 % te
     weinig; het plateau van de F1 (0,72–0,75 over 0,15–0,50) laat ruimte om een
     tellingsneutraal werkpunt te kiezen, maar dat moet op een derde set;
   - de gevolgen stroomafwaarts: RERA/RDI op de vier `arousal_limb_wired`-profielen, de
     PLM-arousalindex, de koppeling aan hypopneus, en de 10 s-regel/onset-offset die de
     huidige detector wél en de U-Net niet toepast;
   - variantie: één trainingsrun (seed 20260927); herhaal met andere seeds en een andere
     MESA-steekproef voordat een getal geclaimd wordt;
   - montages zonder EOG of kin-EMG, andere bemonstering, REM-arousals (EMG op 50 Hz draagt
     weinig) en de generieke MESA-kanaalnamen;
   - de afhankelijkheid: de bibliotheek heeft nu geen torch; 4,6 M parameters passen in
     een ONNX-/numpy-pad, maar dat is een ontwerpbeslissing;
   - de MESA-referentie is één scoorder per nacht en de trainingsscoorder is niet
     gestratificeerd (`docs/nacht_20260901_bevindingen.md` §4).

## 8. Gebruikte data, bestanden, stand

- **PSG-IPA** (publiek): SN1–SN5, `Resp_events`; alleen gelezen.
- **MESA** (DUA, ter plekke gelezen, niets gekopieerd): 400 nachten in
  `/srv/DATA/MESA/gebruikte_mesa_ids.txt` onder `## dsp-scout eeg 2026-09-27 (400)`
  (ids mesa-sleep-4535 … -6812; seed 20260927 over de 476 nog verse nachten):
  `unet50/ids_train.txt` (320; 3 zonder NSRR-arousals afgevallen: mesa-sleep-4854, -5623,
  -6476 → 317 getraind), `unet50/ids_val.txt` (80; mesa-sleep-5721 zonder arousals
  afgevallen voor de U-Net → 79; alle 80 voor psgscoring en MSED; 59 voor DeepSleep2 —
  de nachten ≤ 2^23 samples, ids in `deepsleep2/calibratie_mesa.json`). Er zijn nu nog 76
  verse MESA-nachten.
- **Code**: `common/` (referentie-export, baseline PSG-IPA/MESA, nabewerking, evaluatie,
  complementariteit), `deepsleep2/`, `msed/`, `unet50/` met elk een README (bron,
  licentie, wat precies is geïmplementeerd) en `requirements.txt`; venv `bench/eeg/_venv`
  (niet in git; `.gitignore` in deze map sluit ook `*/upstream/`, `out*/` en `*.pt` uit — de
  README's noemen de upstream-commits/Zenodo-versie en het ophaalcommando, en hoe
  `model_best.pt` te regenereren). Modelgewichten: `unet50/model_best.pt` (18 MB,
  eigen), `deepsleep2/upstream/models/model_2/my_checkpoint_21.pth.tar` (MIT, commit 9d27032),
  `msed/upstream/models/splitstream/weights.pth` (MIT, commit 9b6eedc).
- **Uitvoer**: `results/tabel.md`, `results/resultaten.json`, `results/mesa_val.md/json`,
  `results/complementariteit.md/json`; per arm de pred-CSV's en `run_log.json`.
- **Upstream MSED is NIET gewijzigd** (`msed/upstream/msed/predict_events.py:149` staat nog op
  `ev[-1] - 1`, waardoor hún CSV arousals in de sdb-rij zet). De correctie zit uitsluitend in
  de bench-eigen inferentielus: `msed/run_msed.py:73` en `msed/run_msed_sweep.py:85`
  (`mask[ev[-1]]`; zie `msed/README.md`). De eerste MSED-run, die de upstream-rij nog volgde,
  gaf F1 0,118 — die staat nergens meer in de tabellen.
- **Afgerond na de verificatie**: de MSED-diagnoserun met slaapgeknipte MESA-invoer
  (`msed/sweep_mesa_crop.log`, `results/msed_mesa_crop.json`; cijfers in §6) — post-hoc,
  verandert de conclusie over MSED niet.
