# Respiratoire 1D-U-Net (bench/resp) — eerste evaluatie volgens de preregistratie

Datum: 2026-10-07/08 (nacht). Preregistratie `docs/resp_unet_preregistratie_20261007.md`
(a46a228 → 9f24c48 formulering → 2be0f06 bevriezing, alle vóór de eerste evaluatie buiten
de validatienachten). Bench `bench/resp/` (README, data, model, postproc, train, eval_cohort,
baseline, analyse). Bevroren model sha256 39b1c095… (epoch 14 van 21, vroegtijdig gestopt
op MESA-val), kopie `/srv/CODE/docs/resp_unet_20261007/model_frozen_39b1c095.pt`; ruwe
uitvoer in diezelfde map (rows-JSON's per cohort, ids, consensus-diagnostiek). Werkpunt
τ = 0,20 (uit de checkpoint; MESA-val vlak 0,79 over τ 0,15–0,30).

**Status: primaire regel (SHHS1) GEHAALD tegen alle baselines; PSG-IPA-regel NIET gehaald (3/5); bewakers gehaald; niets ingebouwd, geen beslissing — zie §6.**

## 1. Training (geen bewijs, keuzeset)
399 MESA-trainingsnachten / 100 validatienachten buiten de standaard-n150-set en buiten de
140 van de breath_dual-MESA-run (seed 20261007, geregistreerd), 4,58 M parameters, 21
epochs à ~20 s + ~65 s validatie op de A4000; tijdfractie apneu 2,0 % / hypopneu 7,1 %.
Gepoolde event-F1 op MESA-val (IoU 0,20, typeonbewust, slaappoort): epoch 0 al 0,751,
beste 0,795 (epoch 14). Variantie: seeds 20261008 en 20261009 op dezelfde nachten halen
0,794 (epoch 17, τ 0,25) en 0,793 (epoch 16, τ 0,30) — binnen 0,002 van de bevroren run; de
bewaker (elk > +0,05 tegen `breath_dual` op MESA-val) wordt in §2 afgerekend zodra de
baseline er is.

## 2. MESA-validatienachten (keuzeset, beschrijvend)
U-Net n=100: F1 mediaan 0,753 (p25 0,653), gepoold 0,795, count-ratio mediaan 1,02,
AHI-bias gemiddeld +0,10 /u. Gepaard tegen psgscoring `aasm_v3_breath_dual@0,50` en `rec`
op dezelfde nachten (psgscoring op het NSRR-hypnogram, artefact-epochs leeg; de
baseline-run is drie keer afgebroken — 08-10 01:04, 20:06 en de bevriezing van 21:48 — en
hervat; 16 profiel-nachten missen daardoor een gelogde AHI en krijgen hem uit de
eventlijst (afwijking ≤ 0,4 /u door `uncertain`-events), en vijf door de bevriezing
leeg achtergelaten CSV's zijn op 09-10 opnieuw berekend nadat de verificatie ze vond):

| | F1 mediaan | F1 gepoold | ΔF1 mediaan / gemiddeld | beter / slechter | p | AHI-bias gemiddeld | ΔF1 per tertiel |
|---|---:|---:|---:|---:|---:|---:|---|
| U-Net | 0,753 | 0,795 | | | | +0,10 | |
| `aasm_v3_breath_dual` | 0,511 | 0,579 | +0,207 / +0,224 | 98 / 2 | 4,2e-17 | −4,95 | +0,27 / +0,20 / +0,20 |
| `aasm_v3_rec` | 0,366 | 0,459 | +0,338 / +0,337 | 97 / 3 | 8,3e-18 | −9,40 | +0,33 / +0,33 / +0,36 |

Bewaker (a), variantie: de seeds 20261008 en 20261009 halen gepoold 0,794 en 0,793 tegen
0,579 voor `breath_dual` — elk +0,21, ruim boven de +0,05 die de preregistratie eist.
Dit is de keuzeset (het werkpunt en het vroegtijdig stoppen zijn hierop gekozen), dus
géén bewijs; het laat wél zien dat het verschil met de regelketen op MESA niet aan één
trainingsrun hangt.

**Ablaties (MESA-val, kanaal op nul bij inferentie):**

| zonder | F1 mediaan | count-ratio | AHI-bias |
|---|---:|---:|---:|
| niets (volledig) | 0,753 | 1,02 | +0,10 |
| neusdruk | 0,741 | 0,98 | −0,58 |
| thermistor | 0,755 | 1,01 | −0,24 |
| effortbanden | 0,738 | 1,03 | −0,01 |
| **SpO2** | **0,394** | **0,29** | **−14,90** |

De kanaal-uitval-augmentatie werkt voor de flowsensoren (zonder druk of zonder thermistor
nauwelijks verlies — het model heeft de sensorarbitrage geleerd die de regelketen met een
poort en een vereniging doet), maar het model **leunt op de SpO2**: zonder desaturatie
vindt het minder dan een derde van de events. Dat is de referentie zelf (NSRR `aasm15`:
hypopneu = ≥ 3 % desaturatie óf arousal, en arousals ziet dit model niet) en dus de
bibliotheekregel voor een inbouw: zonder SpO2 terug naar de regelketen.

## 3. SHHS1, 150 verse nachten (beslissend; thermokoppel + RIP + SaO2, geen neusdruk)
Ids `/srv/CODE/docs/resp_unet_20261007/shhs1_ids.txt` (seed 20261007, geregistreerd in
`gebruikte_shhs_ids.txt`). U-Net: F1 mediaan **0,583** (p25 0,410), gepoold 0,650,
count-ratio mediaan 0,91, AHI-bias gemiddeld −1,37 /u; per NSRR-AHI-tertiel laag (AHI 4,3)
F1 0,384 / count-ratio 0,76 / bias −0,29, midden (12,9) 0,562 / 0,95 / −0,94, hoog (30,4)
0,694 / 0,93 / −2,88 — dezelfde ziektelasthelling als mensen en regels.
**Primaire regel (gepaard tegen `rec` én tegen `breath`, psgscoring op hetzelfde
NSRR-hypnogram, thermokoppel op de drukplaats zoals `SHHS-validation/score_shhs.py` en de
arousal-replicatie van 27-09):**

| | F1 mediaan | ΔF1 mediaan / gemiddeld | beter / slechter | Wilcoxon p | AHI-bias gemiddeld | ΔF1 per tertiel laag / midden / hoog |
|---|---:|---:|---:|---:|---:|---|
| U-Net | 0,583 | | | | −1,37 | |
| `aasm_v3_rec` | 0,081 | +0,372 / +0,406 | 146 / 0 | 1,0e-25 | −10,69 | +0,31 / +0,45 / +0,46 |
| `aasm_v3_breath` | 0,200 | +0,262 / +0,289 | 142 / 6 | 1,3e-24 | −9,55 | +0,21 / +0,34 / +0,32 |

Count-ratio mediaan 0,91 (in [0,80; 1,25]), |bias| 1,37 < 10,69, geen tertiel onder −0,02:
**alle vier de onderdelen van de primaire regel gehaald, tegen beide baselines.** Kanttekening
die de lezing kleurt: de regelketen is op dit cohort zelf zwak — een thermokoppel uit de
jaren negentig op de drukplaats haalt de apneugrens van 0,90 zelden (het dossier van
22-08: 13 % op een thermistor) en de hypopneeroute mist evenveel, vandaar AHI-bias −10
en F1 0,08–0,20. Dat is de werkelijke stand van de bibliotheek op SHHS1 (de paper-
reproductie daar gebruikte het dataset-profiel `mesa_shhs`); daarom staat hieronder
post-hoc ook `mesa_shhs` als derde baseline.

| post-hoc | F1 mediaan | ΔF1 mediaan / gemiddeld | beter / slechter | p | AHI-bias | tertielen |
|---|---:|---:|---:|---:|---:|---|
| `mesa_shhs` | 0,173 | +0,317 / +0,339 | 145 / 2 | 1,1e-25 | −10,58 | +0,25 / +0,37 / +0,40 |

De psgscoring-profielen vinden op SHHS1 een mediaan van 11 (`rec`), 16,5 (`breath`) en
25 (`mesa_shhs`) events per nacht tegen 77 in de NSRR-referentie (AHI-mediaan 1,9 / 2,7 /
4,1 tegen 12,9): de regelketen is op dit cohort vrijwel blind, ongeacht profiel. **Deel
daarvan is kanaaltoewijzing, en dat raakt ook de vergelijking:** `SHHS_CMAP` kent alleen
`NEW AIR`. Op 47 nachten lazen U-Net en baseline hetzelfde kanaal (`NEW AIR`); op 55
nachten met `NEW AIR` én `AIRFLOW` nam het U-Net `AIRFLOW` (de generieke flowrol) en de
baseline `NEW AIR`; op 43 nachten zónder `NEW AIR` zette psgscoring `AIRFLOW` op de
thermistorplaats (rec F1-mediaan 0,000, 2 events); op 3 nachten (`NEWAIR` zonder spatie)
vond psgscoring géén flowkanaal (0 events, bias −10,69 i.p.v. −10,79 doordat die drie als
nul tellen) terwijl het U-Net het wel las; op 2 nachten met dubbele kanaalnaam
(`AIRFLOW-0/-1`) draaide het U-Net zónder flow (F1 0,44 en 0,66 op effort + SpO2 alleen).
De U-Net-F1 per groep is 0,59 / 0,59 / 0,57 tegen rec 0,09 / 0,00 / 0,20 — de winst hangt
dus niet aan de kanaalkeuze, en zonder de 5 afwijkende nachten (n = 145) blijft de regel
gehaald (vs `rec` +0,373, 141/0, p = 6,9e-25; vs `breath` +0,261, 137/6; vs `mesa_shhs`
+0,319, 140/2). Maar "zelfde montage" geldt op 102 van 150 nachten, en het eigen dossier
over psgscoring op SHHS1 (kanaalkeuze, de 0,90-grens op een thermokoppel, SaO2 op 1 Hz)
hoort vóór elke verdere claim. De winst is vooral "het U-Net kan thermokoppel-montages aan
waar de regels dat niet kunnen", niet "het nadert de NSRR-scoorder" (F1 0,58, tegen 0,75
op MESA-val met neusdruk).

## 4. PSG-IPA SN1–5 (12 scoorders; neusdruk + RIP + SaO2, geen thermistor)
Scoorder-mediaan F1 (IoU 0,20) per nacht, náást het menselijk plafond
(`docs/respiratoir_menselijk_plafond_20261007.md`) en de psgscoring-baselines op hetzelfde
hypnogram:

| nacht | U-Net | `breath_dual` | `rec` | plafond | U-Net/plafond | AHI U-Net | AHI scoordermediaan |
|---|---:|---:|---:|---:|---:|---:|---:|
| SN1 | 0,486 | **0,678** | 0,470 | 0,826 | 0,59 | 8,8 | 6,0 |
| SN2 | **0,557** | 0,388 | 0,317 | 0,549 | 1,01 | 5,8 | 4,3 |
| SN3 | 0,882 | **0,900** | 0,886 | 0,948 | 0,93 | 45,6 | 54,0 |
| SN4 | **0,471** | 0,252 | 0,286 | 0,553 | 0,85 | 4,8 | 3,8 |
| SN5 | **0,580** | 0,476 | 0,349 | 0,556 | 1,04 | 12,0 | 10,0 |

- **Regel "op ≥ 4/5 nachten niet lager dan `breath_dual`": NIET gehaald (3/5).** Het model
  wint fors op de drie lichte nachten SN2/SN4/SN5 (+0,17 / +0,22 / +0,10; op SN2 en SN5
  zit het op of boven het menselijk plafond) en verliest op SN1 (−0,19) en nipt op SN3
  (−0,02).
- **SN1 ontleed** (consensus = events die ≥ 6 van 12 scoorders markeren, 34 stuks): het
  U-Net zet 51 events waarvan **22 door geen enkele scoorder** worden gezien (precisie
  0,41 tegen consensus, recall 0,62); `breath_dual` zet 31 events met 7 zonder steun
  (precisie 0,71, recall 0,65). Op SN3 is het omgekeerd: U-Net precisie 0,97 maar recall
  0,82 (276 tegen 326 consensus-events; AHI 45,6 tegen 54,0), `breath_dual` 0,91 / 0,90.
  Op SN5 wint het U-Net op beide (0,56/0,72 tegen 0,54/0,57). Het model telt op de
  lichte nachten te veel (AHI +1,0 tot +2,8 boven de scoordermediaan) en op de zware te
  weinig; de apneukop blijft op deze druk-alleen-montage vrijwel leeg (0 apneus op
  SN1/SN2/SN4, 1 op SN5, 91 op SN3) — alles wordt hypopneu, wat de typeonbewuste F1 niet
  raakt maar de rapportage wel zou raken. Bewaker (d), typebewuste F1 (apneu/hypopneu als
  type, IoU 0,20): mediaan 0,679 op MESA-val en 0,424 op SHHS1 tegen 0,753 / 0,583
  typeonbewust — het type klopt op MESA redelijk en op SHHS1 (thermokoppel) slecht.
- Duur: U-Net-events mediaan 18,8–30,1 s tegen scoorders 14,4–26,5 s (iets lang).
- **Post-hoc (niet in de preregistratie):** de consensus-diagnostiek hierboven, de
  SN1-kenmerken hieronder en de `mesa_shhs`-baseline in §3.
- **Waarom SN1 misgaat (signaalkenmerken van de 22 onbesteunde U-Net-events):** het zijn
  echte stroomdalingen op de neusdruk — mediaan 77 % op de robuuste omhullende-maat (p90
  over ±120 s), duur 15,6 s — **zonder desaturatie** (SpO2-daling mediaan 1,1 %, 0 van 22
  ≥ 3 %), de helft in N3. De 21 wél besteunde U-Net-events zakken dieper (0,94) en
  desatureren vaker (7 van 21 ≥ 3 %); de 22 consensus-events van `breath_dual` desatureren
  in 12 van 22. Lezing: het model heeft op NSRR geleerd dat een diepe daling zónder
  desaturatie soms toch een (arousal-)hypopneu is, maar het ziet geen EEG om dat te toetsen,
  en scoort die dalingen op SN1 systematisch waar twaalf scoorders dat niet doen. Dat is
  precies het gat dat een **hybride** inbouw dicht: het netwerk als kandidaatgenerator en
  de regelketen (desaturatie- en arousalkoppeling) als poort — zonder die poort hoort dit
  model niet in een rapport.

## 5. Bewakers
- CPU-inferentie (4 threads, PSG-IPA): voorwaarts 0,6–2,1 s, totaal 7–14 s per nacht
  (inclusief inlezen en resamplen) — ruim onder de 60 s. Tellingen en F1 identiek aan de
  GPU-run op 5/5; vier eventgrenzen verschillen 0,125 s (één sample op 8 Hz).
- Variantie: zie §1. Ablaties: zie §2 (SpO2 is de kritische ingang).
- Niet gemeten: subtypering (het model levert alleen apneu/hypopneu), RERA, desaturatie-
  koppeling per event; menselijk plafond alleen op PSG-IPA.

## 6. Lezing
1. **Beslissend cohort (SHHS1, 150 verse nachten, thermokoppel-only): regel gehaald** tegen
   `rec`, `breath` en post-hoc `mesa_shhs` — maar op een cohort waar de regelketen zelf
   vrijwel niets vindt (F1 0,08–0,20, AHI-bias −10). De winst is dus vooral "het model kan
   een thermokoppel-montage aan en de regels niet", en de absolute kwaliteit (F1 0,58) is
   lager dan op MESA met neusdruk (0,75).
2. **MESA-val (keuzeset): +0,21 F1 tegen `breath_dual` op 98 van 100 nachten**, bias +0,1
   tegen −5,0 /u, stabiel over drie seeds. Dat is de grootste afstand tot de regelketen
   die in dit project ooit is gemeten, maar het blijft de set waarop τ gekozen is.
3. **PSG-IPA (12 scoorders, onze eigen referentie): regel NIET gehaald.** Winst op de drie
   lichte nachten tot op het menselijk plafond, verlies op SN1 (0,49 tegen 0,68) en SN3.
   Het SN1-mechanisme is precies benoemd: diepe drukdalingen zonder desaturatie die geen
   scoorder scoort, omdat het model geen EEG ziet en zonder arousal-/desaturatiepoort elke
   diepe daling meetelt. Op MESA/SHHS valt dat niet op omdat de NSRR-scoorder dezelfde
   neiging heeft; op PSG-IPA wel.
4. **Afhankelijkheid van SpO2** (zonder: F1 0,39) en een lege apneukop op druk-alleen-
   montages: het model levert een AHI, geen apneu/hypopneu-verdeling.
5. **Gevolg volgens de preregistratie:** het SHHS1-criterium is gehaald en het PSG-IPA-
   criterium niet. De preregistratie koppelde de bouwstap aan "slagen"; dat is hier half.
   Mijn lezing: **niet inbouwen als zelfstandige detector**, wél als kandidaatgenerator in
   een hybride keten waarin de regels de poort vormen (desaturatie- en arousalkoppeling,
   minimumduur, subtypering) — dat dicht het SN1-gat per constructie en houdt elk event
   verklaarbaar. Dat vraagt een nieuwe preregistratie met PSG-IPA als beslissend cohort
   (de 4/5-regel opnieuw) en de DUA-vraag beantwoord vóór enige release. Bart beslist.

## 7. Wat dit niet is
Geen bibliotheekcode, niets uitgerold, geen beslissing. De DUA-vraag (gewichten uit
MESA/SHHS verspreiden) is niet beantwoord. De vergelijking op PSG-IPA gebruikt het
hypnogram van de arousal-replicatie (scoorder 1) voor alle drie de systemen.

## 8. Verificatie
Onafhankelijk nagerekend (meting-verificatie, 09-10) uit de CSV's met `bench/evaluate.py`,
de scoordersets van PSG-IPA en de logs: prereg na de bevriezing van het model niet
gewijzigd, elke evaluatie begon erna met de juiste sha; SHHS1-, PSG-IPA- en ablatiecijfers,
seeds, CPU-tijden, registraties en de kopieën buiten git kloppen. Verwerkt: vijf lege
baseline-CSV's op MESA-val uit de bevriezing (opnieuw berekend; ΔF1 tegen `breath_dual`
+0,207/+0,224 i.p.v. +0,215/+0,232, bias −4,95; tegen `rec` 0,366/0,459, −9,40); de
SHHS1-biases van `rec` en `mesa_shhs` (drie `NEWAIR`-nachten tellen als nul); de
kanaaltoewijzing op SHHS1 (zelfde kanaal op 102/150; regel houdt zonder de 5 afwijkende
nachten); `mesa_shhs` vindt 25 events, niet 11–17; typebewuste F1; post-hoc-labels; AHI-
bereik op PSG-IPA (+1,0 tot +2,8), scoorderduur 14,4–26,5 s, vier GPU/CPU-grenzen 0,125 s;
de orakelcurve over τ (§9). Niet exact verifieerbaar: de "drukdaling 0,77" van de
SN1-kenmerken (omhullende-definitie niet in het verslag; eigen herberekening 0,71, zelfde
rangorde); de consensus-telling op SN5 wijkt ±1 event af.

## 9. Orakelcurve over τ (informatief, zoals de preregistratie belooft; geen regel)
**Correctie 09-10 22:10:** de τ-runs hieronder (0,15 / 0,25 / 0,30 / 0,35) draaiden door een
fout in `eval_cohort.py` op een **tweede, onbedoeld getrokken set van 150 verse
SHHS1-nachten** (de eerste stond al in het register, dus de trekking sloeg hem over en
overschreef `ids.txt`); alleen de rij τ 0,20 is de hoofdset. De SHHS1-kolommen zijn dus geen
curve op dezelfde nachten; de PSG-IPA-kolommen wel. De tweede set is alsnog geregistreerd,
de script-fout hersteld (vaste id-lijst), en de curve wordt op de hoofdset opnieuw gedraaid
(zie §10 zodra klaar). τ 0,20 blijft het gerapporteerde werkpunt.

| τ | SHHS1 F1 mediaan | gepoold | count-ratio | AHI-bias | PSG-IPA SN1 / SN2 / SN3 / SN4 / SN5 (scoorder-mediaan F1) |
|---|---:|---:|---:|---:|---|
| 0,15 | 0,521 | 0,654 | 1,00 | +0,22 | 0,486 / 0,523 / 0,884 / 0,440 / 0,581 |
| **0,20** | **0,583** | 0,650 | 0,91 | −1,37 | 0,486 / 0,557 / 0,882 / 0,471 / 0,580 |
| 0,25 | 0,506 | 0,647 | 0,77 | −2,89 | 0,544 / 0,565 / 0,861 / 0,461 / 0,585 |
| 0,30 | 0,491 | 0,635 | 0,65 | −4,41 | 0,553 / 0,578 / 0,848 / 0,416 / 0,592 |
| 0,35 | 0,449 | 0,615 | 0,55 | −5,76 | 0,566 / 0,540 / 0,835 / 0,426 / 0,546 |

Op SHHS1 is de gepoolde F1 vlak (0,62–0,65) en is τ 0,15 tellingsneutraal (ratio 1,00,
bias +0,2); de mediaan piekt op het vooraf gekozen 0,20. Op PSG-IPA wisselt de rangorde per
nacht: SN1 wint bij strenger τ (0,49 → 0,57, minder overtelling), SN3 verliest (0,88 →
0,84). Geen drempel maakt de 4/5-regel goed (bij 0,25–0,30 blijft SN3 onder `breath_dual`
en SN1 ook) — de SN1-fout is niet met τ te repareren, wat de hybride-poortlezing in §6
ondersteunt.
