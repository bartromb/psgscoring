# Replicatie — bevroren U-Net-arousaldetector (preregistratie 27-09-2026)

Datum meting: 2026-09-27 (avond). Preregistratie: `docs/arousal_unet_preregistratie_20260927.md`
(commit 7fcc4f4, geschreven vóór enige meting). Harnas: `scripts/arousal_unet_replicatie.py`
(stappen `unet`, `baseline` met doorwerking, `eval`). Ruwe uitvoer buiten git:
`/srv/CODE/docs/arousal_unet_20260927/out/<cohort>/` (per nacht `_ref.csv`, `_hypno.json`,
`_base.csv`, `_base.json`, `_unet.csv`, `_unet_nomerge.csv`, `_door.json`, `eval*.json|md`).

## Wat er precies gemeten is
- **Artefact:** `bench/eeg/unet50/model_best.pt`, bevroren kopie
  `/srv/CODE/docs/arousal_unet_20260927/model_frozen_cc91bb86.pt`, sha256
  `cc91bb867c167d7387cf619c8dd005ba294d30f640d0fe0a59c2eaeb3a798566` (het harnas weigert een
  andere hash). Invoer EEG + EOG + kin-EMG @ 50 Hz, 18 min-normalisatie (bench-`data.py`).
- **Nabewerking (vast):** 1 s-gemiddelde van de kans, **τ = 0,35**, gaten ≤ 1 s dichten, runs
  < 3 s weg (`bench/eeg/common/postproc.prob_to_events`), slaappoort op de onset-epoch
  (`gate_sleep`; let op: 13,9 % van de SHHS-referentie-arousals en 18,6 % van MESA-76
  (MESA-val 18,2 %, PSG-IPA 0,3 %) beginnen in een W-epoch van hetzelfde NSRR-hypnogram en zijn
  voor béide slaapgepoorte armen onbereikbaar — haalbare sensitiviteit ≤ 0,86 op SHHS, ≤ 0,81 op
  MESA, en count-ratio en index-bias liggen daardoor mechanisch lager), daarna de
  10 s-samenvoeging van de huidige keten
  (`psgscoring.arousal.enforce_min_arousal_interval`, profielwaarde 10,0 s van `aasm_v3_rec`).
  De samenvoeging raakt < 1 % van de events (SHHS 16 758 → 16 610; MESA-val 8 767 → 8 683).
- **Baseline:** psgscoring 0.34.2, `aasm_v3_rec`, `run_pneumo_analysis(raw, hypno, …)` met het
  NSRR-hypnogram (PSG-IPA: scoorder 1). SHHS met `channel_map` NEW AIR / THOR RES / ABDO RES
  (zoals `/srv/DATA/SHHS-validation/score_shhs.py`); de arousalstap koos daar de afleidingen
  `['EEG(sec)', 'EEG']` (145/150; 5 nachten `EEG2`/`EEG 2` + `EEG`, generieke terugval,
  werkpunt 0,80). MESA: EEG1 + EEG2 + EEG3 (0,80). PSG-IPA: F4-M1 + C4-M1 + O2-M1 (0,70).
- **Montage U-Net:** SHHS `EEG` (C4-A1) + `EOG(L)` + `EMG`; MESA `EEG3` (C4-M1) + `EOG-L` +
  `EMG`; PSG-IPA C4-M1 (SN5: Cz-M1) + E1-M2 + kin.
- **Matcher:** `validate_psgipa.match_events`, IoU 0,20, typeonbewust (via `bench/evaluate.py`
  `project`); tweede maat onset ±5 s. Per nacht F1, gepaard Wilcoxon; tertielen op de
  referentie-arousalindex; REM/NREM op de stage van de onset-epoch (gepoold).
- **Referentie:** NSRR-`Arousal`-annotaties (SHHS/MESA, één scoorder); PSG-IPA de
  bench-referentie `bench/eeg/ref/SNx_ref_arousal.csv`. Menselijk plafond op PSG-IPA: **0,679**
  (330 scoorderparen) — richtpunt, geen criterium.

## Twee correcties op mijn preregistratie
1. De zin "τ 0,35 is tellingsneutraal op MESA-val (ratio 1,01)" was **fout toegeschreven**: de
   1,01 komt uit de PSG-IPA-veeg (`bench/eeg/results/tabel.md` r. 80: 602 events tegen 596).
   Op MESA-val telt τ 0,35 mediaan **0,69** van de NSRR-referentie (bench-conventie zonder
   samenvoeging: hetzelfde, 8 767 tegen 12 432). De regel en τ zijn **niet** aangepast; het
   gevolg staat hieronder in de count-ratio's.
2. Drie SHHS-nachten (`shhs1-200641`, `201775`, `204054`) dragen **nul** NSRR-arousals terwijl
   beide detectoren er 4–88 vinden (waarschijnlijk niet gescoord). Zij vallen buiten de gepaarde
   toets (ΔF1 ongedefinieerd); de teller "≥ 90" blijft op de 150 getrokken nachten staan. Het
   harnas toetste eerst óók `n == 150` (een voorwaarde die niet in de preregistratie staat) en
   zette het criterium daardoor op NEE; die codefout is hersteld **ná de eerste eval-uitvoer**,
   de uitslag is ongewijzigd: 134 ≥ 90, en ook met de drie nachten als "gelijk" geteld.
   Correctie 1 is vastgesteld vóór de SHHS-uitslag bekend was (MESA-val-eval 21:1x), correctie 2
   erna.

## 1. SHHS1, 150 verse nachten (147 met referentie) — beslissend
Seed 20260927, 8 eerder gebruikte nachten uitgesloten, geregistreerd in
`/srv/DATA/SHHS/gebruikte_shhs_ids.txt`. 19 713 NSRR-arousals.

| arm | F1 gepoold | mediaan F1 | F1 onset ±5 s | sens | PPV | events | mediaan count-ratio | bias index /u | REM F1 | NREM F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| psgscoring 0.34.2 | 0,543 | 0,525 | 0,523 | 0,415 | 0,787 | 10 386 | 0,51 | −10,85 | 0,271 | 0,640 |
| U-Net τ 0,35 | **0,676** | **0,667** | **0,657** | 0,621 | 0,742 | 16 485 | **0,85** | **−3,85** | **0,655** | **0,737** |

Gepaard (U-Net − baseline): ΔF1 gemiddeld **+0,128** (mediaan +0,125), beter/slechter/gelijk
**134/12/1**, Wilcoxon **p = 2,0·10⁻²²**; onset-maat +0,130, 135/11/1, p = 8,5·10⁻²².
Per-nacht F1 p10/p50/p90: baseline 0,39/0,53/0,67 → U-Net 0,52/0,67/0,77. Grootste verlies
−0,299 (`shhs1-203303`), daarna −0,126 en −0,081.

| tertiel ref-index | n | bereik /u | F1 base | F1 U-Net | ΔF1 | beter | bias base | bias U-Net | count-ratio base | count-ratio U-Net |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 49 | 4,0–17,6 | 0,516 | 0,621 | +0,105 | 43 | −6,13 | −0,65 | 0,52 | 0,98 |
| T2 | 49 | 17,8–24,8 | 0,514 | 0,645 | +0,131 | 46 | −9,67 | −3,04 | 0,52 | 0,84 |
| T3 | 49 | 25,5–96,3 | 0,548 | 0,695 | +0,147 | 45 | −16,74 | −7,85 | 0,50 | 0,82 |

REM: baseline sens 0,163 / PPV 0,808 → U-Net sens 0,747 / PPV 0,583 (F1 0,271 → 0,655).

**Beslisregel SHHS (vooraf):** ΔF1 > 0 op ≥ 90/150 → **134: JA**; Wilcoxon p < 0,05 → **JA**;
mediane count-ratio in [0,80; 1,25] → **0,85: JA**; geen tertiel met gemiddelde ΔF1 < −0,02 →
**JA** (laagste +0,105). **Alle vier gehaald.**

Doorwerking (psgscoring met de U-Net-events als `arousal_events`, n = 147, rapportage): de
arousalindex stijgt van 11,95 naar 18,95 /u (Δ +7,0, mediaan +6,3; NSRR-referentie ≈ 22,8 /u);
AHI en respiratoire eventlijst zijn **identiek** op 144/147 nachten met AHI (drie SHHS-nachten
hebben geen bruikbaar flowkanaal: AHI `None` in beide armen; Δ 0,00; de Rule-1A-arousaltak
staat uit). RERA/RDI en PLM-arousal-associatie zijn langs deze weg **niet meetbaar**: de externe
arousal-ingang van `run_pneumo_analysis` bouwt een leeg arousalblok en slaat de RERA-detectie en
de koppeling over (`rdi = None`, geen `coupling`-sleutel), en `plm_arousal_index` ontbreekt in
beide armen op 76/76 MESA-nachten — dat vraagt een bibliotheekpad, geen harnas.

### Gevoeligheid voor τ op SHHS (post hoc, informatief — geen onderdeel van de regel)
Na de uitslag op τ 0,35 zijn dezelfde 147 nachten met hetzelfde bevroren model op drie lagere
werkpunten geëvalueerd (`--thr`, zelfde keten). Dit is rapportage achteraf op een nu geziene set
en mag de uitslag niet bepalen; het laat zien wat τ doet met telling en precisie.

| τ | F1 gepoold | mediaan F1 | sens | PPV | events | mediaan count-ratio | bias index /u | REM F1 | beter/slechter | p |
|---|---|---|---|---|---|---|---|---|---|---|
| 0,20 | 0,669 | 0,670 | 0,699 | 0,641 | 21 501 | 1,07 | +1,86 | 0,590 | 132/15 | 4,0·10⁻²¹ |
| 0,25 | 0,675 | 0,676 | 0,675 | 0,676 | 19 672 | 0,99 | −0,22 | 0,616 | 132/15 | 2,4·10⁻²² |
| 0,30 | 0,677 | 0,673 | 0,647 | 0,710 | 17 950 | 0,91 | −2,18 | 0,639 | 135/12 | 1,4·10⁻²² |
| **0,35 (vooraf)** | 0,676 | 0,667 | 0,621 | 0,742 | 16 485 | 0,85 | −3,85 | 0,655 | 134/12 | 2,0·10⁻²² |

F1 is vlak over τ 0,20–0,35 (0,669–0,677, alle vier halen de regel); τ ruilt telling tegen
precisie. Tegen de NSRR-referentie is τ 0,25 tellingsneutraal op SHHS (ratio 0,99, bias
−0,2 /u), waar τ 0,35 dat op PSG-IPA is (1,02). Welk werkpunt de klinische index moet dragen,
is een aparte beslissing (§5), geen reden om deze uitslag te herzien.

## 2. MESA, de 76 resterende verse nachten — tweede replicatie
Alle nog niet gebruikte MESA-nachten (geregistreerd in `/srv/DATA/MESA/gebruikte_mesa_ids.txt`,
totaal nu 2 155). Zelfde cohort als de training, ander sample; 12 009 NSRR-arousals,
referentie-index gemiddeld 27,2 /u.

| arm | F1 gepoold | mediaan F1 | F1 onset ±5 s | sens | PPV | events | mediaan count-ratio | bias index /u | REM F1 | NREM F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| psgscoring 0.34.2 | 0,556 | 0,548 | 0,541 | 0,533 | 0,581 | 11 013 | 0,92 | −2,78 | 0,322 | 0,643 |
| U-Net τ 0,35 | **0,687** | **0,670** | **0,686** | 0,597 | 0,809 | 8 858 | 0,72 | −7,37 | **0,698** | **0,757** |

Gepaard: ΔF1 gemiddeld **+0,134** (mediaan +0,124), beter/slechter **72/4**, Wilcoxon
**p = 1,3·10⁻¹³**; onset-maat +0,146, 72/4, p = 8,6·10⁻¹⁴. Per-nacht F1 p10/p50/p90: baseline
0,36/0,55/0,67 → U-Net 0,59/0,67/0,77. De vier verliezen zijn klein (−0,071, −0,057, −0,044,
−0,014). REM: sens 0,21 → 0,66 bij PPV 0,69 → 0,74.

| tertiel ref-index | n | bereik /u | F1 base | F1 U-Net | ΔF1 | beter | bias base | bias U-Net | count-ratio base | count-ratio U-Net |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 26 | 3,0–19,1 | 0,471 | 0,633 | +0,162 | 24 | +1,53 | −2,42 | 1,12 | 0,81 |
| T2 | 25 | 19,4–28,7 | 0,555 | 0,677 | +0,122 | 24 | +0,43 | −5,42 | 0,95 | 0,72 |
| T3 | 25 | 28,9–117,4 | 0,569 | 0,686 | +0,117 | 24 | −10,46 | −14,46 | 0,82 | 0,66 |

**Beslisregel MESA (vooraf):** meerderheid beter → **72/76: JA**; p < 0,05 → **JA.**
Maar let op de telling: tegen de MESA-referentie telt het model op τ 0,35 mediaan 0,72 en de
index-bias verslechtert van −2,8 naar −7,4 /u (zie correctie 1 en §5). Doorwerking: arousalindex
24,4 → 19,8 /u (Δ −4,6); AHI en respiratoire events identiek op 76/76.

## 3. PSG-IPA n = 5 — bevestiging
Baseline hier 0,555 gepoold = het bench-cijfer (zelfde aanroep).

| arm | F1 gepoold | mediaan F1 | F1 onset ±5 s | sens | PPV | events | mediaan count-ratio | bias index /u | REM F1 | NREM F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| psgscoring 0.34.2 | 0,555 | 0,598 | 0,526 | 0,576 | 0,536 | 640 | 1,11 | +2,03 | 0,290 | 0,567 |
| U-Net τ 0,35 | **0,745** | **0,709** | **0,740** | 0,747 | 0,743 | 599 | **1,02** | **+0,22** | **0,656** | **0,757** |

Per opname beter op **5/5** (ΔF1 +0,145 gemiddeld; Wilcoxon p = 0,0625 is het minimum bij
n = 5). Tertielen (2/2/1): +0,102 / +0,134 / +0,253. Het menselijk plafond 0,679 (gemiddelde
paarsgewijze scoorder-F1) ligt eronder; het referentiebestand is dat van de bench (scoorder 1),
dus dit cijfer is een bevestiging op een al geziene set, geen bewijs; het plafond is bovendien
mens-tegen-mens op een andere export en dus niet like-for-like met deze rij. **Criterium (≥ 4/5 niet
lager): JA.** Doorwerking: arousalindex 21,8 → 20,0 /u (mediaan +0,15), AHI identiek.

## 4. Bewakers
- **(a) Variantie, seeds 20260928/29 op dezelfde 400 ids** (`bench/eeg/unet50/train_seed.py`,
  zelfde 317 train / 79 val als de bevroren run; uitvoer
  `/srv/CODE/docs/arousal_unet_20260927/seed_<seed>/`). Regel: elk ΔF1 > +0,05 tegen de
  baseline op MESA-val. Twee ketens: de preregistratieketen (τ 0,35 + samenvoeging, zoals
  hierboven) en de bench-conventie (eigen validatie-τ, zonder samenvoeging, zoals het
  bench-cijfer 0,682).

  | run | beste epoch (val-F1 train.py) | ketens: F1 τ 0,35 | ΔF1 vs base | beter | bench-conventie F1 (τ) | ΔF1 vs base | beter |
  |---|---|---|---|---|---|---|---|
  | bevroren, seed 20260927 | 16 (0,682) | 0,664 | +0,094 | 70/79 | 0,682 (0,20) | +0,114 | 74/79 |
  | seed 20260928 | 15 (0,686) | 0,658 | **+0,090** | 69/79 | 0,686 (0,20) | **+0,117** | 75/79 |
  | seed 20260929 | 18 (0,678) | 0,632 | **+0,062** | 58/79 | 0,678 (0,15) | **+0,111** | 73/79 |

  Beide seeds: **JA** (elk > +0,05 in beide ketens); de bevroren run is geen uitschieter naar
  boven — de drie runs liggen op 0,678–0,686 validatie-F1. Wel zichtbaar: op de VASTE τ 0,35 loopt
  de telling per seed uiteen (count-ratio 0,69 / 0,68 / 0,61; seed 29 koos zelf 0,15), dus τ hoort
  bij het gewicht en niet bij de architectuur — een nieuw getraind model vraagt zijn eigen
  werkpunt. Seed 29 is op 29-09 afgerond (de evaluatie stierf met de sessie en is opnieuw gedraaid;
  `seed29_eval.sh`).
- **(b) CPU-inferentie zonder GPU** (SHHS, 5 nachten, inclusief EDF-laden en resamplen):
  4 threads **4,5–5,4 s/nacht** (forward 3,3–4,1 s), 1 thread **11,6–15,2 s/nacht** — ver
  onder de 60 s. **JA.**
- **(c) Montage-ablaties op MESA-val** (79 nachten, baseline uit de bench, zelfde keten;
  kanaal op nul gezet):

  | variant | F1 gepoold | mediaan F1 | sens | PPV | count-ratio | bias /u | REM F1 | beter dan baseline | p |
  |---|---|---|---|---|---|---|---|---|---|
  | baseline psgscoring | 0,568 | 0,560 | 0,546 | 0,592 | 0,91 | −2,70 | 0,266 | — | — |
  | U-Net volledig | **0,664** | 0,661 | 0,560 | 0,817 | 0,69 | −9,00 | 0,697 | 70/79 | 2,2·10⁻¹⁰ |
  | zonder EOG | 0,620 | 0,613 | 0,495 | 0,828 | 0,59 | −11,45 | 0,687 | 56/79 | 8,9·10⁻⁵ |
  | zonder kin-EMG | 0,624 | 0,618 | 0,500 | 0,832 | 0,62 | −11,33 | 0,634 | 59/79 | 9,3·10⁻⁶ |
  | alleen EEG | 0,513 | 0,512 | 0,365 | 0,864 | 0,43 | −16,28 | 0,575 | 35/79 | 0,0087 (slechter) |

  Alleen-EEG is **slechter dan de huidige detector** → verplichte terugval. Zonder één van
  EOG/EMG blijft het model boven de baseline maar telt een derde te laag; de preregistratie
  schrijft óók dan terugval voor (geen stille degradatie) — dat blijft zo.
- **(d) REM/NREM apart:** in de tabellen; de winst zit voor het grootste deel in REM
  (sensitiviteit 0,16 → 0,75 op SHHS, 0,17 → 0,66 op MESA-val).

Rekenkundig: SHHS-baseline 150 nachten met 20 workers in ≈ 18 min (60–198 s per nacht,
mediaan 99 s), piek 79 °C op SHHS én MESA (bewaker op de pgid, geen pauze; op MESA hing de
eerste bewaker aan een dode pgid en waren de eerste 5 min onbewaakt). U-Net-inferentie op de
GPU 1–10 s per nacht inclusief EDF-laden (forward zelf ≤ 1 s).

## 5. Lezing
- **De drie cohorten halen hun vooraf vastgelegde regel**, met dezelfde richting en grootte:
  gepaarde ΔF1 +0,13 (SHHS, extern en nooit gezien), +0,13 (MESA-vers), +0,15 (PSG-IPA);
  slechter op 12/147, 4/76 en 0/5 nachten. De winst zit niet in één tertiel en zit voor het
  grootste deel in REM: sensitiviteit ×3–4,6; de REM-PPV stijgt op MESA (0,69 → 0,74) maar
  daalt op SHHS (0,81 → 0,58) en PSG-IPA (0,77 → 0,57) — precies waar de huidige detector zijn
  zwakste plek heeft (REM-F1 0,27–0,32).
- **Telling is een τ-keuze plus de slaappoort, geen modelfout.** Op τ 0,35 telt het model 0,85
  (SHHS), 0,72 (MESA) en 1,02 (PSG-IPA) van de referentie; F1 is vlak over τ 0,20–0,35. Een
  deel van het tekort op de NSRR-cohorten is de poort (14–19 % van de referentie ligt in W en is
  voor beide armen onbereikbaar), niet τ. De preregistratie
  koos τ 0,35 op een verkeerd toegeschreven premisse (correctie 1). De huidige detector telt op
  SHHS 0,51 en verliest 10,9 /u op de index; het model verliest daar 3,9 /u (τ 0,35) of 0,2 /u
  (τ 0,25). Op MESA daarentegen verslechtert de bias (−2,8 → −7,4 /u op τ 0,35).
- **Wat NIET is aangetoond:** doorwerking op RERA/RDI (externe ingang slaat die over) en op de
  hypopneu-koppeling (tak uit); de hoge PPV-onderkant van de NSRR-referentie (één scoorder) blijft
  de onzekere maat — het PSG-IPA-plafond 0,679 laat zien dat de referentie zelf ruis draagt.
- **Gevolg volgens de preregistratie (alle bewakers gehaald):** bouwen als profielveld
  `arousal_detector = "unet_v1"` (default `"lgbm"`), ONNX + `onnxruntime` onder `[ml]`, gewichten
  met sha256-wacht in `psgscoring/data/`, verplichte terugval naar `lgbm` zonder EOG of kin-EMG,
  re-ranker uit onder `unet_v1` met reden in de provenance, bevroren profielen gepind, golden 9/9
  met de vlag uit. **Default aanzetten op `aasm_v3_rec` is een aparte beslissing van Bart**, net
  als het werkpunt: τ 0,35 (PSG-IPA-neutraal, hoogste PPV) of τ 0,25 (NSRR-neutraal). Wie
  0,25 kiest, maakt de SHHS-set tot afstelset en heeft voor dát werkpunt een verse cohort nodig
  (SHHS1 heeft er nog 5 634); τ als profielparameter met 0,35 als default kost geen extra
  meting.
