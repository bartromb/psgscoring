# Hermeting van de reviewvelden (12-10-2026) — verslag

Preregistratie: `docs/reviewvelden_hermeting_preregistratie_20261012.md` (dd69d3e, vóór de
run). Code: `main` bd4ee7f (harnas met `--field-arms`), bibliotheek identiek aan 7814130.
Uitvoer en scripts: `docs/reviewvelden_hermeting_20261012/`.

## 1. Herijking `THERMISTOR_BAND_POWER_MIN` op de productie-uploads — KLAAR

In situ op de Hetzner-server (losse container uit `yasaflaskified:0.38.10`, `--network
none`, uploads read-only, code van `main` via `PYTHONPATH`), alle 456 EDF-uploads;
434 droegen een thermistorkanaal (`Flow Th.`), 22 niet (andere montages), 1 BDF
overgeslagen (het script leest alleen EDF). Dezelfde opname staat vaak onder meerdere
job-id's: op identieke getallen ontdubbeld blijven **210 unieke montages** over
(32 Hz, enkele 256 Hz). Alleen tellingen verlaten de server
(`ijk_edf_zonder_namen.csv`, `ijk_uitvoer.txt`).

| | `x[::q]` (zoals geijkt) | FIR-anti-alias (nu default) |
|---|---|---|
| mediaan ademband-fractie | 0,690 | 0,690 |
| onder 0,70 | 108 van 210 (51 %) | 107 van 210 (51 %) |
| verschil anti − alias | mediaan +0,0000, max \|Δ\| 0,036 | |
| montages die van oordeel wisselen bij 0,70 | 1 (0,699 → 0,700) | |

**Twee bevindingen, in volgorde van gewicht.**

1. **Op de productiemontage is aliasing geen factor.** De thermistor wordt op 32 Hz
   bemonsterd; `q = sf // 8 = 4` en het kanaal draagt vrijwel niets boven 4 Hz, dus
   `x[::q]` en het FIR-filter geven dezelfde fractie (max |Δ| 0,036; één montage wisselt
   van kant, op de drempel). De reparatie uit de review is juist voor hoog bemonsterde
   kanalen met netbrom (de synthetische 0,671 → 1,000), maar **verandert op de eigen
   uploads niets**. De 07-10-meting (+5,9/+8,7/+37,3 /u waar de poort de thermistor
   goedkeurt) wordt er dus niet door verklaard en niet door geraakt.
2. **De drempel 0,70 heeft op dit cohort geen natuurlijke scheiding.** De negen
   augustus-montages toonden een leeg gat van 0,53 (0,970 | 0,441); op 210 montages is de
   verdeling **continu** van 0,00 tot 1,00 met 0,70 precies op de mediaan. Dichtstbijzijnde
   waarden onder en boven 0,70: 0,697 en 0,700 (gat 0,003); de breedste lege opening in
   [0,40; 0,95] is 0,022 (0,793–0,815). Volgens de vooraf vastgelegde regel is er dus
   **geen** nieuwe drempel af te leiden; 0,70 blijft staan **als niet-geijkte waarde**, en
   de poort keurt met die waarde de helft van de productiethermistors af. Dat is dezelfde
   conclusie als de MESA-steekproef van 22-08 (geen gat), nu op de eigen montage.

**Wat dit betekent voor `aasm_v3_breath_dual`** (poort `respiratory_band`, additief):
onder de additieve vereniging blokkeert de poort niet, hij bepaalt welke sensor de
primaire apneupas draait en wordt in `flow_channels.thermistor_check` gerapporteerd. Op de
helft van de nachten is dat de druk, op de andere helft de thermistor, en de grens
daartussen is een mediaan, geen fysische scheiding. **Voor Bart:** de enige ijking die
hier helpt is een referentie — welke thermistors scoren apneus die de druk mist en de
mens wél ziet — en dat is precies wat de EC-studie met beide sensoren levert. Tot dan:
0,70 laten staan en in het verslag blijven melden, of de poort onder de additieve
vereniging op "rapporteer, beslis niet" zetten (druk als primaire sensor, thermistor als
aanvulling). Niet door mij beslist.

Overgeslagen: de ene BDF-upload (`read_raw_edf` weigert BDF; het script hoort
`read_raw_bdf` te proberen — klein, open), en de 22 montages zonder thermistorkanaal.

## 2. MESA n=30 en PSG-IPA n=5: reviewvelden tegen hun oude waarde — KLAAR

Twee MESA-runs op dezelfde 30 held-out nachten (run 1: `aasm_v3_breath_dual`, 13 armen;
run 2: `aasm_v3_rec`, 9 armen; addendum 11:05 van de preregistratie), PSG-IPA SN1–SN5 met
alle armen op beide profielen. Thermische bewaker: piek 68 °C. Volledige tabellen in
`reviewvelden_hermeting_20261012/analyse_uitvoer.md`; onafhankelijk nagerekend (zie §3).
ΔF1 is steeds **nieuw − arm** tegen NSRR `aasm15`.

### 2.1 De default-aan-velden: neutraal, geen enkele ter heroverweging (regel 1)

| veld | `aasm_v3_breath_dual` (run 1) | `aasm_v3_rec` (run 2) |
|---|---|---|
| nadir-tijdas, CSR-periode, spectrale resolutie, PLM-venster, ODI-noemer, stadiumbasislijn, anti-alias | **gelijk in F1, AHI en eventtelling** (0/0/30; zie §3 voor de subtypelabels) | byte-identiek (nadir, CSR, spectraal); PLM/ODI/stadium/anti-alias niet gedraaid (inert per ontwerp, zie addendum) |
| herstelbasislijn (`recovery_baseline_mode`) | gelijk in F1/AHI (1/30 nachten een ander hypopneesubtypelabel) — de ademteugdetector leest de envelope niet | mediaan ΔF1 0,0000 (gem. +0,0005), 10/9/11, p = 0,79; bias −5,84 → −5,97; ernst 16 → 15; events 3625 → 3605 |
| fasesplitsing (`breath_phase_split`) | mediaan −0,0014 (gem. −0,0036), 13/16/1, p = 0,80; bias −0,53 → −0,44; ernst 18 → 18; events 4606 → 4625 | mediaan 0,0000 (gem. +0,0017), 11/11/8, p = 1,0; bias −5,84 → −5,80; ernst 16 → 16 |
| mixed-decompositie | F1 byte-identiek (typeonbewust); **subtypes**: centraal 222 ← 209, gemengd 34 ← 47 | F1 byte-identiek; centraal 206 ← 202, gemengd 8 ← 12 |
| **alles_oud** (netto review) | mediaan −0,0014, 13/16/1, p = 0,80; bias −0,44 → −0,53; ernst 18 → 18 | mediaan 0,0000 (gem. +0,0022), 14/10/6, p = 0,58; bias −6,00 → −5,84; ernst 15 → 16 |

Per AHI-tertiel van de referentie (nieuw − alles_oud): `breath_dual` laag −0,0044 (3/6/1), midden
+0,0001 (5/5/0), hoog +0,0013 (5/5/0), bias laag +10,73 → +10,81, midden −2,79 → −3,01, hoog
−9,26 → −9,38; `rec` laag +0,0008 (5/2/3), midden 0,0000 (4/4/2), hoog +0,0011 (5/4/1), bias
laag +6,69 → +6,99, midden −6,97 → −6,87, hoog −17,73 → −17,64, ernst laag 3 → 4.

**Lezing.** Geen enkel veld haalt een criterium van regel 1: de grootste mediane ΔF1 is 0,0014,
geen p < 0,05, de grootste biasverschuiving 0,16 /u, de ernstconcordantie beweegt hooguit één
nacht. De review verandert op hele MESA-nachten dus vrijwel niets aan event-overeenstemming of
AHI; de ene systematische verschuiving is de **subtypering** door de decompositie op de
effort-envelope (13 gemengde apneus worden centraal op `breath_dual`, 4 op `rec`) — zonder
NSRR-subtypereferentie in dit harnas is dat een verschuiving, geen oordeel. Dat de meeste velden
op `breath_dual` byte-identiek zijn, is ontwerp: de ademteugdetector leest het ruwe signaal en
heeft een eigen desaturatieroute (`breath_scoring._desat_at`, niet `get_desaturation`), MESA-
hypnogrammen dekken de opname exact, de thermistor staat op 32 Hz.

PSG-IPA bevestigt het beeld: alle `breath_dual`-armen identiek aan `nieuw` (bias +2,95, 2/5
binnen bereik, F1 0,479) behalve de fasesplitsing (bias +2,61, 4/5 ernst i.p.v. 3/5, F1 0,476 —
één opname wisselt van ernstklasse op n = 5); op `rec` liggen `oud_fase` en `oud_herstel`
op F1 0,355/0,327 tegen 0,318 voor `nieuw`, maar MESA n = 30 wijst dat als neutraal uit
(p = 1,0 en 0,79). Plafond respiratoire F1 op PSG-IPA: 0,667.

### 2.2 De default-uit-velden (regel 2)

**`hypopnea_mask_excludes_apnea_band=False` (`aan_masker`)** — op `breath_dual` byte-identiek
(het masker is een envelope-eigenschap). Op `rec`:

| cohort | ΔF1 (nieuw − aan) | beter/slechter | p | bias nieuw → aan | ernst | binnen bereik |
|---|---|---|---|---|---|---|
| MESA n = 30 | **−0,0754** (aan hoger) | 2/27/1 | 6,2e-6 | −5,84 → **−0,37** | 16 → 17 | — |
| PSG-IPA n = 5 | F1 0,318 → 0,387 | | | +1,51 → **+8,95** | 4/5 → 2/5 | 3/5 → **1/5** |

MESA haalt alle drie de MESA-criteria van regel 2 (ΔF1 > 0 met p < 0,05, |bias| kleiner,
ernst niet lager); PSG-IPA verlaagt het aantal binnen het scoordersbereik van 3 naar 1. **Volgens
de vooraf vastgelegde regel blijft het veld uit.** De lezing is dezelfde als bij de
duurtolerantie (29-08): het effect volgt het teken van de bestaande bias — MESA onderdetecteert
(−5,84 /u) en krijgt er events bij die het bereik dichten; PSG-IPA overdetecteert al (+1,51)
en schiet door naar +8,95. Een compensatieknop, geen reparatie. Events 3625 → 4626 (+28 %).

**`linearise_after_dc_removal=True` (`aan_lineariseer`)** — `breath_dual` byte-identiek;
`rec` op MESA: mediaan 0,0000 (gem. −0,0057), 6/7/17, p = 0,44, bias −5,84 → **−6,88**, ernst
16 → **13**, events 3625 → 3430. Faalt regel 2 op bias en ernst: **blijft uit.** PSG-IPA is
onbeslisbaar (één flowkanaal, geen linearisatie). De DC-verwijdering doet wat ze belooft
(lagere gemeten reducties) en kost op een cohort dat al onderdetecteert precies daardoor.

### 2.3 Conclusie voor de release

- Geen van de tien default-aan-velden hoeft terug (regel 1); de netto-review is neutraal op
  F1, bias en ernst over 30 held-out nachten en twee profielen.
- `aan_masker` en `aan_lineariseer` blijven uit, beide volgens regel 2; `aan_masker` is een
  echte beslissing voor Bart (cohortconflict, groot effect), `aan_lineariseer` niet.
- De subtypeverschuiving van de decompositie (gemengd → centraal) hoort in de CHANGELOG als
  gedragsverandering vermeld; het aantal is klein (13 op 2 370 apneus).
- Prestatie: de andere sessie profileerde +17 % looptijd op `breath_dual` (decompositie
  rekent de effort-enveloppen opnieuw); reparatie = enveloppen uit `_precomp` doorgeven; `rec`
  gelijk. Geen regressie in productie (pint 0.35.1).

## 3. Onafhankelijke nalezing (12-10, aparte agent, eigen herberekening uit de ruwe JSON)

- **Alle gerapporteerde getallen kloppen** (beide runs, alle armen, tertielen, subtypes, de 28
  PSG-IPA-regels). Provenance: elke arm draagt op 30/30 opnames exact de bedoelde
  sleutel/waarde-paren; anker `aasm_v3_rec` = `aasm_v3_rec|nieuw` op 30/30 in beide runs;
  steekproef = eerste 30 van `opnames.txt` in beide runs; tussen prereg-sha 7814130 en de
  run-sha is niets in `psgscoring/` gewijzigd. Run 1 bevatte ook alle rec-armen en die zijn
  op 30/30 identiek aan run 2 (f1, ahi): de herhaling reproduceert.
- **Eén fout in `analyse.py`, gerepareerd:** regel 2 stond met omgekeerd teken, waardoor de
  kolom "oordeel" verzweeg dat `rec|aan_masker` op MESA alle drie de criteria haalt (ΔF1
  +0,0754, 27/30, p = 6,2e-6; |bias| 5,84 → 0,37; ernst 16 → 17). §2.2 hierboven was al met de
  hand juist geschreven; het eindoordeel (blijft uit, op PSG-IPA) verandert niet.
- **"Byte-identiek" was te sterk.** F1, AHI en eventtelling zijn gelijk, maar op eventniveau
  wisselen op `breath_dual` hypopnee-subtypelabels (`hypopnea` ↔ `hypopnea_mixed`/
  `_central`): oud_herstel 1/30 nachten, oud_mixed 6/30, aan_masker 8/30 (48 events),
  aan_lineariseer 10/30 (41 events); onzichtbaar voor de typeonbewuste matcher. Het verslag
  zegt nu "gelijk in F1, AHI en eventtelling".
- Wilcoxon: de normale benadering van het harnas tegen scipy-exact verschilt hooguit in de
  derde decimaal (aan_masker 6,2e-6 tegen 9,3e-8); geen oordeel verschuift.
- Een F1-gelijkspel kan een AHI-sprong verbergen: mesa-sleep-1862 onder `rec|aan_lineariseer`
  F1 exact 8/17 in beide armen, AHI 19,1 → 12,2, events 95 → 61 — telt als "gelijk".
- **`aan_masker` per tertiel (rec, nieuw − aan):** laag −0,0477 (2/7/1, bias +8,37 → +6,99),
  midden −0,0681 (0/10/0, −4,75 → −6,87), hoog **−0,1305** (0/10/0, −4,73 → −17,64 zonder
  masker). De winst van het masker groeit met de ernst en haalt de onderdetectie in het hoge
  tertiel van −17,6 naar −4,7 /u; op het laag-prevalente PSG-IPA (bias al positief) jaagt
  hetzelfde masker de bias naar +8,95. Dat is het duurtolerantiepatroon: het effect volgt het
  teken van de bestaande bias per cohort. **Voor Bart:** als de klinische populatie op de
  ernstige kant van MESA lijkt, is dit de grootste nog beschikbare F1-winst in de bibliotheek;
  de prijs is overdetectie op lichte nachten. Een tertiel-afhankelijke beslissing is geen
  optie zonder hermeting; de regel van het huis zegt: uit, tot een cohort met beide tekens het
  beslist.
