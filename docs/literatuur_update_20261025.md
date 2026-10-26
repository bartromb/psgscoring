# Literatuurupdate psgscoring — stand 2026-10-25

*Vervolg op `literatuur_algoritmes_20260904.md`. Venster: publicaties 2025–2026, met nadruk op
wat sinds 4 september verscheen of toen nog niet gelezen was. Methode: gerichte websearches,
systematische titel-sweeps in Europe PMC (automatische respiratoire/arousal/PLM-scoring,
inter-scoorder-overeenstemming, HSAT-autoscoring, Physiological Measurement 2025–2026),
controle van AASM-, FDA- en GitHub-pagina's. Abstracts via de Europe PMC-API; twee papers
(Sleep Revolution) in volledige tekst gelezen. Cijfers hieronder zijn uit de abstracts en
tabellen overgenomen, niet herberekend.*

## 0. In één alinea

Het veld rapporteert nu standaard tegen **multi-scoorder-referenties** en claimt "beter dan
de mediane individuele scoorder" — precies onze PSG-IPA-framing. Voor arousals staat er een
direct vergelijkbaar open model (Pitkänen e.a., SLEEP 2026; MIT-code) met F1 0,61 tegen een
scoorderplafond van 0,57 op een onafhankelijke 10-scoorder-set; onze U-Net-replicatie
(0,745 op PSG-IPA tegen plafond 0,679; 0,676 op 150 verse SHHS1-nachten) ligt in lijn, maar
de koppelregels verschillen (zij: élke overlap; wij: IoU 0,20). Voor respiratoire events
bevestigt dezelfde groep (JSR 2025) met seconde-per-seconde-kappa's van 0,05–0,26 per
subtype wat wij op PSG-IPA maten: hypopneu-subtypen zijn geen stabiel doel. ABED (Nat Commun,
feb 2026) is verschenen en **ontbreekt in de bibliografie van paper v44**. Onafhankelijke
HSAT-studies (n = 3144 en een retrospectieve reeks) laten zien dat ingebouwde
polygrafie-autoscoring zonder manuele nalezing klinisch tekortschiet — relevant voor de
hPG-conventie vanaf 2027. Nieuw in de index-discussie: event-gebonden hartslagstijging als
prognostische maat náást hypoxic burden. Geen nieuwe AASM-manual (nog v3, 2023), wel een
geactualiseerd AASM-AI-standpunt (JCSM, nov 2025).

## 1. Arousals

### 1.1 Pitkänen e.a., SLEEP 2026 — "agrees with ten expert scorers in unseen data"

| | hun model | scoorders onderling | wij (U-Net-replicatie 27-09) |
|---|---:|---:|---:|
| training | 1847 MESA-nachten | — | 317 MESA-nachten |
| MESA-test (event-F1 / 1-s κ / ArI-ICC) | 0,77 / 0,67 / 0,83 (205 nachten, 1 scoorder) | — | 0,687 op MESA-76 |
| onafhankelijke multi-scoorder-set | Sleep Revolution 50 PSG × 10 scoorders: **0,61** / 0,50 / 0,66 | **0,57** / 0,46 / 0,59 | PSG-IPA 5 PSG × 12 scoorders: **0,745** tegen plafond **0,679** |
| verse enkelscoorder-cohort | — | — | SHHS1 150 nachten: 0,543 → 0,676 |
| koppelregel event-F1 | **elke overlap telt** | idem | **IoU ≥ 0,20** (greedy, symmetrisch) |
| invoer | C4–M1 + kin-EMG + neusdruk, 64 Hz | | drie afleidingen (multi-derivatie), kin-EMG |
| architectuur | U-Time-achtige FCN + SE-attentie + ASPP (Huttunen e.a.) | | 1D-U-Net |
| nabewerking | alleen AASM-regels (≥ 3 s, 10 s stabiele slaap vooraf) | | τ 0,35 tellingsneutraal |

Details die ertoe doen: hun automatische arousals zijn systematisch **langer** (mediaan 9,0 s
tegen 6,9 s manueel), beginnen 1,3 s vroeger en eindigen 0,9 s later — dezelfde
onset-verschuiving die wij in `arousal_onsetverschuiving_bevinding.md` beschreven. Prestatie
het hoogst in N3, het laagst in wake. Ze rapporteren expliciet dat eerdere modellen alleen
event-F1 gaven (0,70–0,81 in-domein, 0,57–0,76 onafhankelijk) en zetten daar 1-s-κ naast;
dat is een goed argument om in onze paper **beide** te geven.

**Code: publiek, MIT** (github.com/UEF-SmartSleepLab/slf-arousalscoring), TensorFlow,
traint uit MESA via hun sleeplab-format. **Geen gewichten in de repo** (41 kB; het README
verwijst naar een checkpoint op hun cluster). Een eerlijke externe benchmark zoals met
CAISR vraagt dus hertraining op onze MESA-kopie (GPU-werk) — of, veel goedkoper, onze
U-Net óók met "elke overlap" scoren zodat de getallen vergelijkbaar worden.

### 1.2 Kaneda e.a., SLEEP Advances 2026 — LightGBM met handgemaakte kenmerken

Hiërarchische pijplijn (staging → arousals → respiratoir) op **LightGBM met tijd- en
frequentiekenmerken**, dus dezelfde familie als onze regel/LGBM-aanpak. Japanse
kwaliteitsgecontroleerde annotaties (vier gecertificeerde scoorders, AASM v2.6),
koppelregel IoU ≥ 0,3. Arousal-F1 0,733 (recall 0,725, precisie 0,742), ArI-LoA ± 15/u; op
de consensus-subset (27 opnames) F1 0,807 **binnen de scoorderspreiding 0,770–0,827**.
Respiratoir F1 0,818, maar één klasse (geen apneu/hypopneu, geen obstructief/centraal) en
alleen neusdruk; op de consensus-subset (5 opnames) 0,767 **onder** de scoorders
(0,842–0,888) met recall 0,944/precisie 0,648. Geen code, gefinancierd door ACCELStars.
Betekenis: een GBDT op handgemaakte kenmerken haalt scoorderniveau voor arousals; ons
LGBM-spoor (`arousal_lgbm_preregistratie.md`) is methodisch niet achterhaald.

### 1.3 Foundation models: SleepFM (Nat Med, jan 2026) en SleepFM-2 (arXiv, sep 2026)

SleepFM-2 (Thapa, Mignot, Zou): ~2 miljoen uur PSG, 24 cohorten, 1-s-tokens, bevroren
encoder + LSTM-koppen. Tegen leave-one-out-consensus: arousals **0,601 tegen gemiddelde
scoorder 0,546** (WSC, 9 scoorders), beenbewegingen 0,604 tegen 0,530, respiratoir
per-seconde macro-F1 0,462 tegen 0,404 (DREEM, 5 scoorders); geen AHI, geen subtypen
apart. Code-link in het preprint (`zou-group/sleepfm-v2-public`) gaf op 25-10 een 404; de
SleepFM-1-code (`zou-group/sleepfm-clinical`) staat onder **CC BY-NC 4.0** met checkpoints
via BDSP. Stanford Sleep Bench (arXiv dec 2025, 17 467 PSG's) belooft gewichten en code.
Bilal/Mehra (Nat Commun 2026, Cleveland Clinic, > 10 000 opnames met EMR): respiratoire
event-F1 0,65 in 30-s-epochs; vijf risicogroepen voorspellen mortaliteit waar
AHI-categorieën dat nauwelijks doen.

### 1.4 AASM-ISR-benchmark (Tripathi e.a., JCSM nov 2025) — ter herinnering

CAISR op 57 ISR-opnames: staging κ 0,70, arousal κ 0,45, respiratoir κ 0,34,
beenbewegingen **κ 0,11** — "ervaren technoloog", onder de crowd (κ 0,83/0,78/0,86).

## 2. Respiratoire events en het menselijke plafond

### 2.1 ABED (Kjaer e.a., Nat Commun, 10 feb 2026) — nu gepubliceerd

5456 trainings-PSG's, 1099 test-PSG's uit zes cohorten; AHI r² 0,84; F1 0,78 over
apneutypen, OA 0,71 / **CA 0,51** / hypopneu 0,65; "vergelijkbaar of beter dan individuele
experts" op twee multi-scoorder-sets; "apnotyping" correleert met loop gain en
spiercompensatie. Onze CAISR/ABED-benchmark van 04-09 (`benchmark_caisr_abed_20260904.md`)
blijft gelden met de kanttekeningen van toen. **Staat niet in de bibliografie van v44**
(wel CAISR en Nassi 2022) — toevoegen.

### 2.2 Pitkänen e.a., J Sleep Res 2025 — seconde-per-seconde-overeenstemming, 10 scoorders

50 PSG's, 10 experts uit zeven centra, AASM v2.6, apneus op **neusdruk** (niet thermistor),
Noxturnal met autoscoring vooraf (zoals in de kliniek). Fleiss' κ seconde-per-seconde:

| classificatie | Fleiss' κ |
|---|---:|
| event / geen event | 0,32 |
| apneu / hypopneu / geen | 0,29 |
| obstructieve apneu | 0,21 |
| centrale apneu | 0,12 |
| gemengde apneu | 0,05 |
| hypopneu met alleen desaturatie | 0,17 |
| hypopneu met alleen arousal | **0,09** |
| hypopneu met beide | 0,11 |
| per stadium N1 / N2 / N3 / R | 0,12 / 0,23 / 0,22 / 0,23 |

ICC's: AHI 0,93, AI 0,95, HI 0,86, **CAI 0,63**; paarsgewijze AHI-ICC 0,77–0,99. In slechts
**34 %** van de deelnemers bleef de ernstcategorie gelijk over alle scoorders; 22 % van de
AHI's lag binnen ± 1/u van een drempel. Dit is de externe bevestiging van onze bevinding
`hypopnee_subtypering_geen_referentie.md` (κ ≈ 0 voor subtypen op PSG-IPA) en van de
noodzaak om respiratoire F1 altijd naast een gemeten plafond te zetten
(`reference_respiratoir_plafond`: 0,667 op 330 scoorderparen). Citeerbaar naast Magalang
2013 en Rosenberg & Van Hout 2014.

Contrast: Choi e.a. (Life 2026, 3 scoorders, 30 opnames, **30-s-epochs**) melden apneu
κ 0,955 en hypopneu κ 0,93 — epoch-gebaseerde maten blazen de overeenstemming op; nuttig
als voorbeeld waarom wij event-niveau rapporteren.

### 2.3 Physiological Measurement 2025–2026 (doeltijdschrift)

- **Cerina e.a. 2025** — AHI-onzekerheid: bootstrap over inter-eventintervallen faalt bij
  lange waakbouts en overdispersie; negatief-binomiaal model voorgesteld. Sluit aan bij
  onze p_scored/verwachtingswaarde-AHI-variant; citeren.
- **Elmi & Elmi 2026 (ApneaTime)** — ankervrije temporele lokalisatie van apneus/hypopneus
  uit alleen EEG/EOG met state-space-model; SHHS→MESA-transfer; getallen in het abstract
  weggevallen; auteurs zelf: "geen diagnostische gereedheid".
- **Crosbie e.a. 2026** — vier staging-algoritmes op 204 PSG's (atleten): GSSC macro-F1
  0,76, U-Sleep 0,71, **YASA 0,70**, Luna 0,69.
- **Nurminen e.a. 2026** — PPG-golfvorm verandert per eventtype en -duur (520 PSG's,
  14 442 events): sterker bij apneus dan hypopneus.
- Daarnaast veel PPG-staging en radar/PPG-apneu-screening. Een open scorer met
  multi-scoorder-validatie past in het recente profiel van het tijdschrift.

## 3. hPG / HSAT-autoscoring — relevant voor de conventie 2027

Onafhankelijke studies tegenover vendorvalidaties:

| studie | opzet | uitkomst |
|---|---|---|
| Roncero e.a., Sleep Breath 2025 | **n = 3144** polygrafieën, automatisch vs manueel, prospectief | diagnostische concordantie **44,5 %**; hoge sensitiviteit/NPV voor uitsluiten, lage specificiteit; "manuele analyse blijft essentieel" |
| Khadadah e.a., Sleep Med X 2026 | level-3 HSAT, retrospectief, één centrum | autoscoring onderschat REI en hypopneu-index; tot 44,5 % van matige gevallen fout geclassificeerd; fouten rond drempels |
| Nat Sci Sleep 2025 (type 3, Alice NightOne) | automatisch vs manueel | r 0,976 maar LoA −5,3…+7,3; Somnolyzer dichter bij manueel dan default-autoscoring |
| Sousa e.a., JCSM 2026 (DeepRESP, Nox) | 88 AF-patiënten, type-II-PSG, alleen HSAT-signalen | AHI-ICC 0,92, **ArI-ICC 0,83 en TST-ICC 0,82 uit RIP-banden**; epoch-OPA arousal 0,80; geen RERA |
| Watson e.a., JCSM 2026 (EnsoSleep ePPG) | 225 patiënten, één PPG-kanaal, 3 RPSGT's | 3 %-AHI bias +0,74/u; sens 90,9 %, **spec 72,3 %** |
| Rohrscheib e.a., Sleep Med 2026 (Wesper) | 44 simultaan PSG + 139 klinisch | AHI r 0,90 (PSG), CAI r 0,82 |
| Attia e.a., JSR 2026 (SleepAI, oximetrie) | 53 patiënten | ernstklasse-accuratesse 89 %; 37,5 % wisselt één klasse tussen thuisnachten |
| Touil e.a., Sleep Med 2026 (Nox BodySleep, kinderen) | 88 kinderen | 3-stadia κ 0,68; OAHI-onderschatting −1,4/u i.p.v. −1,9/u zonder algoritme |

Lezing: autoscoring **mét** PSG-gelijkwaardige validatie (DeepRESP, ePPG) haalt
correlaties ≥ 0,9, maar de twee onafhankelijke reeksen op gewone ingebouwde
polygrafie-autoscoring zijn ronduit slecht rond de klinische drempels. Voor de Slaapkliniek
betekent dat: manuele nalezing van elke hPG blijft nodig (ook onder de conventie), en een
transparante scorer die op hPG-kanalen draait (geen EEG; arousalpoot via Pleth/RIP) vult
een reëel gat. DeepRESP's ArI-ICC 0,83 uit alleen RIP is de externe bevestiging dat die
poot haalbaar is (zie `project_pleth_ptt_lead`, Philips 2023).

## 4. PLM: geen nieuwe referentie, wel bevestiging van onze AASM-duurbevinding

Geen multi-scoorder-validatie 2025–2026 gevonden. Eguchi e.a. (EMBC 2025): een strikte
lezing van de scoringshandleiding gaf op sEMG **recall en precisie 0** — manuele scoorders
volgen de letter niet; hun benchmark haalt F1 0,751 (recall 0,895, precisie 0,604) op 20
personen. Dat is precies ons `plm_offset_aasm`-verhaal (0/5 → 4/5 binnen de spreiding). Li
e.a. (Bioengineering 2025, AttenCRF-U): gezamenlijke detectie SDB + beenbewegingen, PLMS-F1
0,778, onderscheid respiratoir-gerelateerde beenbewegingen vs PLMS (125 patiënten). CAISR
op ISR: κ 0,11. Ons PLM-resultaat tegen 12 scoorders is in dit veld zeldzaam en als korte
mededeling publiceerbaar.

## 5. Indices voorbij de AHI: hypoxic burden, hartslagstijging, ventilatory burden

- **Heart rate surge** (Huang e.a., JACC Asia 2026; 4790 deelnemers, 11,6 jaar follow-up):
  piekhartslag in het event-venster minus 60-s-baseline; ρ met HB slechts 0,08; hoogste
  kwintiel HR 3,00 voor cardiovasculaire sterfte; C-index 0,77 → 0,81 bóven HB. Wij
  berekenen die hartslagrespons al in de autonome re-ranker (`arousal_autonomic_rerank`,
  de HR-stijging droeg de winst) — als rapportmaat is dit bijna gratis.
- **SASHB in acuut hartfalen** (Masaki/Kasai, medRxiv sep 2026, n = 178): voorspelt
  samengesteld eindpunt waar AHI dat niet consistent doet.
- **Ventilatory burden** (Wu e.a., Sleep Med 2026, APPLES): hoge VB + goede therapietrouw →
  vroege ochtend-SBP-daling onder CPAP; exploratief.
- HoneyNaps kondigt HB, arousal burden en ventilatory burden aan als volgende
  software-uitvoer; het veld beweegt naar rapportage van deze maten
  (`hypoxic_burden_bevinding.md` hebben wij al).

## 6. Staging-backend en afhankelijkheden

- YASA bracht **0.7.0 (8 mrt 2026)**, **0.8.0 (2 okt 2026)** en **0.8.1 (25 okt 2026)** uit.
  0.7: `predict()` geeft een `Hypnogram`. 0.8: Python ≥ 3.11, numpy ≥ 2.2.2,
  scikit-learn ≥ 1.6.1, lightgbm ≥ 4.6, `predict_proba` verwijderd, de 0.4.0-classifiers
  verwijderd (altijd de nieuwste classifier). Beide repo's pinnen alleen naar onder
  (YF `yasa>=0.7`, psgscoring-extra `yasa>=0.6`); de productie-image krijgt vandaag nog
  0.7.0 **uitsluitend** omdat YF `scikit-learn>=1.3,<1.4` en `numpy<2` pint. Zodra die pin
  wordt losgelaten, wisselt de staging-classifier stil — dan opnieuw ijken, want elke
  index deelt door de TST.
- Vergelijkende staging-literatuur plaatst YASA niet meer bovenaan (Crosbie 2026: 0,70 tegen
  GSSC 0,76; SLEEPYLAND/SOMNUS, npj Digit Med 2025, open-source ensemble dat op DOD-H/DOD-O
  de beste menselijke scoorder overtreft en waarvan de ensemble-onenigheid scoorderambiguïteit
  voorspelt, AUC 0,83). Een backend-wissel is meetbaar op onze 20 eigen PSG's + MESA-deelset,
  maar staat achter detectiekwaliteit in prioriteit.

## 7. Regelgeving en markt

- **AASM-manual**: nog **v3 (feb 2023)**, erratum 2024, geen v3.1/v4 aangekondigd.
  Hypopneu 3 % "recommended", 4 % "optional". Niets te wijzigen in `aasm_v3_*`.
- **AASM AI-standpunt, geactualiseerd** (Oks e.a., JCSM 2025;21(11):1953): AI als partner,
  niet vervanger; eisen aan klinische validatie, **voortdurende accuratesse na
  implementatie**, transparantie. Er bestaat een AASM-certificeringsprogramma voor volledige
  PSG-autoscoring (details niet op de publieke pagina).
- **FDA**: HoneyNaps SOMNUM v3.0, K253390 (juli 2026): staging, arousals, apneu/hypopneu
  met OSA/CSA/gemengd, beenbewegingen; alleen **OPA** gerapporteerd (97,4–99,9 %) — door de
  negatieven opgeblazen, zelfde kritiek als in `commerciele_autoscoring_20260907.md`.
- Europa: geen ERS/ESRS-richtlijn 2025–2026 over HSAT-diagnostiek; wel het ERS-e-Sleep-
  telegeneeskundestatement (ERJ 2025) en het ERS/ESRS-ASV-statement voor centrale apneu.

## 8. Wat dit betekent voor paper v44 → v45

1. **Bibliografie aanvullen**: Kjaer 2026 (ABED), Pitkänen 2026 (arousals), Pitkänen 2025
   (respiratoire overeenstemming), Tripathi 2025 (ISR/CAISR), Kaneda 2026, Cerina 2025
   (Physiol Meas), Oks 2025 (AASM-standpunt); SleepFM-2 alleen als preprint.
2. **Framing is actueel**: "boven de mediane scoorder op een multi-scoorder-set" is de
   standaardclaim geworden; wij hebben er een expliciet gemeten plafond bij. Vermeld altijd
   de koppelregel (IoU 0,20 is strenger dan "elke overlap") en geef voor arousals ook een
   1-s-κ of de onset/offset-verschuiving.
3. **Hypopneu-subtypen**: Pitkänen 2025 (κ 0,09–0,17 per subtype) is de externe steun voor
   "geen stabiel doel"; de Simpson/basiskans-analyse van 02-09 blijft nergens elders
   gerapporteerd.
4. **Centrale apneus**: ABED CA-F1 0,51, Pitkänen CAI-ICC 0,63, κ 0,12 — onze zwakke hoek
   is die van het hele veld; zo benoemen.

## 9. Roadmap-kandidaten (Bart beslist; detectie eerst)

| | kandidaat | kost | waarom nu |
|---|---|---|---|
| A | paper-citaties bijwerken (§8) | klein | ABED ontbreekt; reviewers kennen deze papers |
| B | U-Net óók met "elke overlap" scoren; optioneel UEF-model hertrainen op onze MESA-kopie en op PSG-IPA/SHHS1 met onze matcher draaien | klein / groot (GPU, TF) | enige eerlijke externe arousal-benchmark naast CAISR |
| C | hPG-profiel meten: psgscoring op hPG-kanaalset (geen EEG) met Pleth-arousalpoot, tegen onze 20 PSG's | middel | conventie 2027; onafhankelijke literatuur zegt dat ingebouwde autoscoring niet volstaat |
| D | event-gebonden HR-surge, HB en VB als rapportmaten | klein | we berekenen ze (bijna) al; het veld beweegt erheen |
| E | PLM-korte mededeling (12-scoorder-referentie, AASM-duurbevinding) | klein–middel | geen vergelijkbare benchmark in 2025–2026 |
| F | staging-backend-meting (SOMNUS/GSSC vs YASA) | middel | pas na D/C; TST is de noemer van alles |
| G | bovengrens zetten op de YASA-pin of bewust naar 0.8 met herijking | klein | voorkomt een stille classifierwissel bij de volgende build |

## 10. Niet gevonden / open

- Geen 2025–2026 PLM-validatie tegen meerdere scoorders; geen nieuwe AASM-manual; geen
  Europese HSAT-richtlijn 2025–2026.
- CHEST-2026-congresabstract (cardiopulmonaal algoritme, 140 000 uur, PPA 76,8 %) niet
  verifieerbaar; niet opgenomen.
- Stanford Sleep Bench: event-F1's (arousal 0,70 / been 0,63 / SDB 0,62 volgens een
  secundaire bron) staan niet in het abstract; niet geverifieerd.
- PSG-IPA-controle: de PhysioNet-set telt 20 opnames = 5 per taak (staging, arousals,
  respiratie, beenbewegingen); de "n = 5, 12 scoorders" in v44 klopt dus.

## Bronnen (DOI)

- Pitkänen H e.a. Automated deep learning-based arousal detection agrees with ten expert scorers in unseen data. *Sleep* 2026. 10.1093/sleep/zsag191 (PMC13658263). Code: github.com/UEF-SmartSleepLab/slf-arousalscoring (MIT).
- Pitkänen M e.a. Temporal and sleep stage-dependent agreement in manual scoring of respiratory events. *J Sleep Res* 2025. 10.1111/jsr.14391 (PMC12069727).
- Kjaer MR e.a. Expert-level probabilistic breathing event detector informs phenotyping of sleep apnea. *Nat Commun* 2026. 10.1038/s41467-026-69163-z.
- Tripathi A e.a. Automated analysis of the AASM Inter-Scorer Reliability gold-standard polysomnogram dataset. *J Clin Sleep Med* 2025;21(11):1821. 10.5664/jcsm.11848.
- Kaneda M e.a. Automated sleep stage and event detection algorithms using quality-controlled polysomnography annotations. *SLEEP Advances* 2026;7(2):zpag054. 10.1093/sleepadvances/zpag054.
- Thapa R e.a. Learning transferable human physiology from two million hours of sleep with SleepFM-2. arXiv:2609.06849 (sep 2026). Thapa R e.a. A multimodal sleep foundation model for disease prediction. *Nat Med* 2026;32:752. 10.1038/s41591-025-04133-4.
- Kjaer MR e.a. Stanford Sleep Bench. arXiv:2512.09591.
- Bilal E e.a. A foundation model for sleep-based risk stratification and clinical outcomes. *Nat Commun* 2026. 10.1038/s41467-026-75326-9.
- Rossi AD e.a. SLEEPYLAND. *npj Digit Med* 2025. 10.1038/s41746-025-02237-2.
- Choi JH e.a. Interrater reliability … across clinical subgroups. *Life* 2026;16(4):669. 10.3390/life16040669.
- Cerina L e.a. Estimation of AHI uncertainty in the presence of long wake bouts and overdispersion. *Physiol Meas* 2025. 10.1088/1361-6579/ae2c3c.
- Elmi Z, Elmi S. Anchor-free temporal localization of apnea events from EEG/EOG. *Physiol Meas* 2026. 10.1088/1361-6579/ae8ca8.
- Crosbie F e.a. Four automated sleep scoring algorithms on 204 PSG recordings. *Physiol Meas* 2026. 10.1088/1361-6579/aea9ea.
- Nurminen J e.a. Respiratory event type and duration modulate PPG waveforms in OSA. *Physiol Meas* 2026. 10.1088/1361-6579/ae7feb.
- Roncero A e.a. Automatic polygraphy analysis versus manual analysis. *Sleep Breath* 2025. 10.1007/s11325-025-03374-4.
- Khadadah S e.a. Agreement between automated and manual scoring of level 3 HSAT devices. *Sleep Med X* 2026. 10.1016/j.sleepx.2026.100193.
- Sousa S e.a. Validation of DeepRESP in atrial fibrillation patients. *J Clin Sleep Med* 2026. 10.1007/s44470-026-00145-0 (PMC13433715).
- Watson NF e.a. Validation of an AI-based single-channel PPG HSAT (EnsoSleep ePPG). *J Clin Sleep Med* 2026. 10.1007/s44470-025-00030-2.
- Rohrscheib C e.a. Wesper Lab HSAT autoscoring. *Sleep Med* 2026. 10.1016/j.sleep.2026.108828.
- Attia S e.a. SleepAI (oximetrie/PPG). *J Sleep Res* 2026. 10.1111/jsr.70093.
- Touil Y e.a. Pediatric respiratory polygraphy with built-in algorithm. *Sleep Med* 2026. 10.1016/j.sleep.2026.109130.
- Validation of automated respiratory event scoring in type 3 HSAT. *Nat Sci Sleep* 2025 (PMC12264349).
- Eguchi K e.a. Benchmark algorithm for sEMG-based PLM detection. *EMBC* 2025. 10.1109/embc58623.2025.11252797.
- Li Q e.a. AttenCRF-U: joint detection of SDB and leg movements. *Bioengineering* 2025;12(6):571. 10.3390/bioengineering12060571.
- Huang Z e.a. Heart rate surge improves cardiovascular risk stratification beyond hypoxic burden. *JACC Asia* 2026. 10.1016/j.jacasi.2026.08.040.
- Masaki M e.a. Hypoxic burden predicts cardiovascular outcomes in heart failure. medRxiv 2026. 10.64898/2026.09.29.26364138 (preprint).
- Wu W e.a. Ventilatory burden and blood pressure response to CPAP (APPLES). *Sleep Med* 2026. 10.1016/j.sleep.2026.109296.
- Thuptimdang W e.a. PWA, PPI and SpO2 for home OSA screening. *Sleep Breath* 2026. 10.1007/s11325-026-03592-4.
- Oks M e.a. Artificial intelligence in sleep medicine: an updated AASM position statement. *J Clin Sleep Med* 2025;21(11):1953. 10.5664/jcsm.11832.
- AASM. Scoring Manual v3 (feb 2023), pagina geraadpleegd 25-10-2026; FDA K253390 (HoneyNaps SOMNUM v3.0, juli 2026) via aasm.org-nieuwsbericht 17-07-2026.
- YASA releases v0.7.0 / v0.8.0 / v0.8.1: github.com/raphaelvallat/yasa/releases.
