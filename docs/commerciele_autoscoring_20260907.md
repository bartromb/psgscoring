# Hoe commerciële systemen apneus, hypopneus en arousals scoren

*Onderzoek 2026-09-07. Bronnen: FDA 510(k)-dossiers (primair), peer-reviewed
validaties, vendordocumentatie. Vraag: wat is er publiek bekend over de
methode én de gemeten kwaliteit, en wat betekent dat voor psgscoring?*

## Het landschap: twee families

1. **Ingebouwde autoscoring van PSG-vendoren** — Philips Sleepware
   (+ Somnolyzer), SOMNOmedics DOMINO, Nihon Kohden Polysmith, Compumedics
   ProFusion, Nox Noxturnal, OSG BrainRT. Meestal regelgebaseerd op
   flow/SpO2/EEG; methodedetails vrijwel nooit gepubliceerd.
2. **Standalone (AI-)scorers** die EDF's van elk systeem lezen — EnsoSleep
   (EnsoData), MICHELE/MY Autoscoring (Younes/Cerebra), Neurobit PSG,
   Somnolyzer als dienst. Dit is waar de publieke validatie zit, omdat de
   FDA-route event-niveaucijfers afdwingt.

Daarnaast een derde paradigma: **surrogaatsignaal-systemen** (WatchPAT/PAT:
perifere arteriële tonus + hartslag + SpO2 + actigrafie) die geen flow of
EEG zien en events uit de autonome respons afleiden (pAHI/pRDI; r ≈ 0,88-0,89
tegen PSG-AHI, maar diagnostische accuratesse 50-70 % in onafhankelijke
series).

## Wat er bekend is over de methode, per systeem

### Philips Somnolyzer (24x7 → Sleepware G3 → HSAT)
- Ontstaan als **expertsysteem** (Anderer/Siesta-groep, >40 experts, 12
  universiteiten): statistische classifiers (LDA-familie) plus
  regelgebaseerde nabewerking voor staging; respiratoir volgens
  AASM-drempels — apneu ≥90 % flowreductie ≥10 s (thermistor), hypopneu
  30-90 % reductie (neusdruk) + 3 %-desat óf arousal; type via effort;
  arousal als EEG-frequentieverschuiving ≥3 s. Basislijndefinitie:
  niet publiek.
- FDA-cleared (K202142), non-inferioriteit tegen experts voor stadia,
  AHI én arousal-index; door AASM "gecertificeerd" als autoscoring.
- **Nieuwste richting (2023, Frontiers Physiol)**: deep-learning
  **autonome-arousaldetector** op ruwe PPG + flow (residuele
  conv-blokken; hartslag 2 Hz + flow 10 Hz → arousalkans, ≥2 s = event)
  om de arousal-poot van Regel 1A op HSAT's te herstellen: arousal-index-
  ICC 0,73 tegen corticale arousals, en de AHI-bias van HSAT's gaat van
  −4 naar 0,0/u (ICC 0,86 → 0,94). **Externe bevestiging van ons
  Pleth/PTT-spoor** — Philips bouwt exact de brug die in
  [[project-pleth-ptt-lead]] staat.

### MICHELE / Cerebra MY Autoscoring (Younes)
- Regelgebaseerd naar AASM 2007, expliciet ontworpen om een
  technoloog na te doen; kern-IP is de **Odds Ratio Product** (ORP,
  EEG-microanalyse als continue slaapdiepte), o.a. gebruikt rond
  arousals. Bedoeld als score-dan-bewerk (technoloog-editing hoort bij
  het product).
- **510(k) K112102 (2011) is de rijkste publieke validatie**: 30 nachten
  (Foothills, Calgary — vooral split-nights!), 3 technologen, referentie
  = 2/3-consensus. Event-niveau PPA: hypopneu 76,3, obstructieve apneu
  57,1, centrale 64,9, gemengde 79,4 (κ respiratoir 0,74); arousals PPA
  60,0 (κ 0,54); **arousal-index-ICC 0,566 met bias −9/u** waar de
  technologen onderling 0,94-0,96 halen. AHI-ICC 0,97-0,98.
- Zelfde dossier toont wat de toenmalige **ingebouwde vendor-autoscoring
  (Alice 5)** waard was: staging-κ 0,06, arousal-κ 0,10, respiratoir
  κ 0,25, hypopneu-PPA 9,3 — en in 27/30 nachten géén REM gevonden.
  Het beeld "autoscoring is waardeloos" bij clinici stamt uit dit
  tijdperk.

### EnsoSleep (EnsoData)
- **Deep learning op de ruwe signalen** (EEG, EOG, EMG, ECG, flow, RIP,
  PVDF, SpO2, PPG, snurk, actigrafie); regelt AASM-conforme events incl.
  CSR en periodieke ademhaling; CSA moet handmatig gereviewd.
- **510(k) K210034 (2021), n=100 PSG's, referentie 2/3-consensus,
  gepoolde epochs**: SDB-events PA 75,4 %, **hypopneu PA 66,3 %**,
  obstructieve apneu PA 74,1 %, centrale PA 65,3 %, **arousal PA
  73,6 %**, beenbewegingen 82,0 %; staging overall 86,6 %. Diagnostische
  agreement AHI≥5: PPA 94,4/NPA 89,7.
- Marketing zegt ">93 % AHI-agreement, >95 % hypopneas, >93 % arousals" —
  dat zijn indexcijfers; de event-niveau-PA's hierboven zijn het echte
  verhaal.

### Neurobit PSG (benchmark Frontiers Neurol 2023, klinische populatie)
- DL-staging + **DL+regels voor respiratoire events**.
- **AHI r = 0,97 náást hypopneu-ICC 0,31 en arousal-ICC 0,46** (experts
  onderling: 0,85 en 0,78). Na handmatige review: 0,62 en 0,89. De
  perfecte demonstratie dat een AHI-correlatie event-kwaliteit verbergt.

### Vendor-autoscoring zonder publieke methode
- **DOMINO (SOMNOmedics)**: AASM-"gecertificeerd" (naast Somnolyzer en
  EnsoSleep), telt A/H "volgens type en in correlatie met desaturaties";
  geen publieke event-niveau-validatie gevonden.
- **Polysmith (Nihon Kohden) v11**: HSAT-validatie REI r = 0,96,
  "95,4 % agreement" hypopneus (methode niet publiek).
- **ProFusion (Compumedics)**: AHI r = 0,96 tegen manueel; methode niet
  publiek.
- **Noxturnal (Nox)**: regels + RIP-terugval als de canule uitvalt;
  BodySleep-AI voor AHI-schatting uit ademhaling+actigrafie; validaties
  op AHI-niveau.

## De vijf lessen voor psgscoring

1. **De validatievaluta is index-correlatie of epoch-PA tegen een kleine
   consensus — vrijwel nooit event-F1 met IoU.** Onze event-F1-cijfers
   (arousal 0,546 tegen menselijk plafond 0,679; respiratoir tegen
   12-scoorder-PSG-IPA) zijn strenger en NIET 1-op-1 vergelijkbaar met
   "PA 73,6 %" (gepoolde epochs, 2/3-consensus van drie). Bij elke
   vergelijking de metriek erbij zeggen.
2. **Hypopneu en arousal zijn overal de zwakke plek**, ook bij de
   FDA-toppers: hypopneu PA 66 % (EnsoSleep), ICC 0,31 (Neurobit);
   arousal κ 0,54 en index-bias −9/u (MICHELE), ICC 0,46 (Neurobit).
   Dezelfde twee assen waar ons werk zit (precisie-gat arousals;
   hypopneu-strictness-dossier). Wij zijn dus niet een zwak systeem aan
   het bijspijkeren — dit is de moeilijke helft van het vak.
3. **Iedereen is score-dan-bewerk.** Geen enkel systeem claimt autonomie;
   MICHELE's 510(k) en de Neurobit-benchmark kwantificeren zelfs hoeveel
   de editing goedmaakt (arousal-ICC 0,46 → 0,89). Onze
   "screening/second reader"-positionering en de /review-weergave staan
   precies in die traditie.
4. **De trend is DL voor staging/arousals, regels voor respiratoire
   events** (Neurobit expliciet; Somnolyzer de facto; EnsoSleep als
   uitzondering volledig DL). Ons hybride ontwerp (regels + LGBM-arousal)
   zit op de hoofdstroom.
5. **Philips' autonome-arousalrichting valideert ons Pleth/PTT-spoor**:
   PPG+flow-DL herstelt de gemiste arousal-hypopneus en haalt de
   HSAT-AHI-bias naar 0. MESA draagt Pleth+EKG in 2056 opnames — de
   meetopzet ligt klaar.

## Vergelijkbaarheidswaarschuwing

De consensusreferenties verschillen fundamenteel: 2/3 van 3 technologen
(EnsoSleep, MICHELE) tegen onze 12-scoorder-spreiding (PSG-IPA) of
1-scoorder-MESA. Een PA tegen een mildere consensus oogt systematisch
hoger dan een F1 tegen een strenge parenverdeling. Wie onze cijfers naast
de commerciële legt zonder deze voetnoot, vergelijkt appels met de
verpakking van peren.

## Bronnen

- MICHELE 510(k) K112102 (FDA, 2011) — event-niveau tabellen 6-1 t/m 6-3
- EnsoSleep 510(k) K210034 (FDA, 2021) — event-niveau PA/NA/OA, n=100
- Bakker/Anderer e.a., Frontiers in Physiology 2023 (Somnolyzer HSAT +
  autonome arousals, PMC10484584)
- Somnolyzer-methode: Anderer e.a. 2005/2010 (Neuropsychobiology);
  kinderen-validatie 2025 (PMC12213436)
- Neurobit-benchmark: Frontiers in Neurology 2023 (fneur.2023.1123935)
- Polysmith v11 HSAT-validatie: J Clin Med 2024 (PMC11277620)
- WatchPAT: Itamar PAT-technologie; multicenter validatie SLEEP 2022;
  Ioachimescu 2020 (accuratesse 50-70 %)
- DOMINO: vendordocumentatie; AASM-certificering (BRN Reviews 2026)
