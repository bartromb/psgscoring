# `breath`, `breath_dual` en de voorwaardelijke vereniging op MESA (stap 2)

Datum: 2026-10-08. Preregistratie `docs/breath_dual_mesa_preregistratie_20261007.md` (2d64e19,
vóór de run). Run gestart 07-10 20:26 op de Z6 (20 workers), onderbroken door een bevriezing
van de machine 08-10 01:46 (123/140 nachten als checkpoint), hervat 08-10 07:11 en opnieuw
om 19:51 na een dagpauze, klaar 20:41 (17 workers; checkpoint per nacht, resume van het
harnas). Bibliotheek bevroren op be78002 (voorwaardelijke vereniging, opt-in) — `git diff`
tegen HEAD op `psgscoring/` en `scripts/validate_mesa.py` leeg vóór elke herstart; de merge
van `unet-v1` kwam pas ná `mesa.json`. 140 nachten (standaard-n150 minus 10 kalibratienachten),
0 fouten, referentie NSRR `aasm15`, matcher IoU 0,20 typeonbewust, artefact-epochs leeg.
Ruwe uitvoer `docs/breath_dual_mesa_20261007/` (`mesa.json` 32 MB en de checkpoints buiten
git; `samenvatting.json`, `opnames.txt`, `gevoeligheid_93.txt`, `analyse_mesa.py`,
`posthoc_robuust.py` erin). Thermische bewaker: geen pauzes; piek vannacht 82 °C (zie
§6).

## 1. Primair — voorwaardelijke vereniging (`dual+conf@0,50` tegen `dual@0,50`)

| | F1 mediaan | ΔF1 mediaan / gemiddeld | beter / slechter / gelijk | Wilcoxon p | AHI-bias gemiddeld | MAE | ernstklasse juist |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dual+conf` | 0,513 | +0,000 / +0,007 | 62 / 19 / 59 | 1,1e-6 | **−4,30** | 8,34 | 82/140 |
| `dual` | 0,507 | | | | −2,74 | 9,18 | 78/140 |

Per NSRR-AHI-tertiel ΔF1 gemiddeld +0,011 / +0,008 / +0,004 (laag / midden / hoog); bias
`conf` tegen `dual`: laag +3,15 tegen +5,73, midden −4,25 tegen −2,87, hoog −11,79 tegen
−11,08. Gevoeligheidsset (93 nachten uit posities 51–150): ΔF1 gemiddeld +0,006, 36/15, p =
0,004, bias −4,34 tegen −2,71.

**Regel: NIET gehaald.** ΔF1 > 0 op meer nachten dan slechter én p < 0,05 kloppen, maar de
bewaker faalt: de gemiddelde bias ligt 1,56 /u verder van nul (grens 1,0). De
voorwaardelijke vereniging blijft **gebouwd-uit** (`dual_sensor_confirmation` default None).
Wat ze wél doet, staat in §4: ze verwijdert bijna uitsluitend events die geen NSRR-event
zijn, maar op een cohort waar `breath_dual` al 2,7 /u ondertelt, maakt elk verwijderd
event de bias slechter — het patroon van een compensatieknop (vergelijk de duurtolerantie
van 29-08): de MAE en de ernstklasse-overeenstemming verbeteren juist (9,18 → 8,34; 78 →
82), omdat de verwijdering vooral de lichte nachten raakt waar `dual` overtelt (bias laag
tertiel +5,73 → +3,15).

## 2. Baseline — `breath_dual` tegen `breath` (geen regel; productie is al omgezet)

| | F1 mediaan | ΔF1 gemiddeld | beter / slechter / gelijk | p | AHI-bias | MAE | ernstklasse juist |
|---|---:|---:|---:|---:|---:|---:|---:|
| `breath_dual@0,50` | 0,507 | −0,001 | 51 / 54 / 35 | 0,84 | **−2,74** | 9,18 | 78 |
| `breath@0,50` | 0,521 | | | | −5,66 | 9,51 | 78 |
| `aasm_v3_rec` (anker) | 0,441 | dual +0,077 | 95 / 40 / 5 | 1,1e-9 | −5,08 | 10,00 | 79 |

Per tertiel `dual` tegen `breath`: ΔF1 −0,009 / −0,005 / **+0,012**, bias laag +5,73 tegen
+4,00, midden −2,87 tegen −5,59, hoog −11,08 tegen −15,38. **Het teken van 0.17.0 (14-08:
bias −2,34 tegen −5,18, F1 −0,006) houdt stand op de huidige versie:** de duale as kost geen
F1 (p = 0,84) en halveert de onderschatting, met de winst op de zware nachten. Tegen `rec`
wint `breath_dual` op F1 (+0,077, 95/40) én bias. Dat ondersteunt de productiekeuze van
07-10 op de enige referentie die er is; op de eigen 20 PSG's (07-10) zat het verschil in
de thermistor-goedgekeurde nachten en daar is MESA geen uitspraak over (andere
thermistor, één scoorder).

## 3. Secundair — `hypopnea_strictness` 0,30 (regel van 24-08: ΔF1 ≥ +0,010, p < 0,05, |bias| ≤ +1,0 slechter)

| arm | F1 mediaan 0,30 vs 0,50 | ΔF1 mediaan / gemiddeld | beter / slechter | p | bias 0,30 vs 0,50 | MAE | ernst juist | regel |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `breath_dual` | 0,557 vs 0,507 | +0,029 / +0,035 | 113 / 25 | 1,2e-16 | **+0,52** vs −2,74 | 8,00 vs 9,18 | 83 vs 78 | **JA** |
| `breath_dual+conf` | 0,558 vs 0,513 | +0,029 / +0,035 | 111 / 26 | 1,2e-16 | −1,03 vs −4,30 | **6,89** vs 8,34 | 82 vs 82 | **JA** |
| `breath` (replicatie 24-08) | 0,561 vs 0,521 | +0,030 / +0,038 | 115 / 22 | 2,7e-17 | −2,21 vs −5,66 | 7,58 vs 9,51 | 85 vs 78 | JA |

Per tertiel wint 0,30 overal (+0,03 tot +0,04), met de bias in het lage tertiel +7,72
(tegen +5,73) en in het hoge −6,60 (tegen −11,08). De herijking van 24-08 repliceert dus
op een verse versie en op een drie keer zo grote set, en ze erft door naar `breath_dual`:
strictness 0,30 onder `breath_dual` is op MESA de arm met de beste F1 (0,557), een bias
rond nul (+0,52) en de beste ernstklasse-overeenstemming (83/140). De combinatie met de
voorwaardelijke vereniging heeft de laagste MAE (6,89) maar een bias van −1,03.
**Niets verandert automatisch** — strictness 0,30 verhoogt de AHI op de lichte nachten
(bias laag tertiel +7,7) en is een klinische beslissing; zie §5.

## 4. Per event — wat de vereniging toevoegt en wat de voorwaardelijke regel verwijdert
Alle 5406 alleen-druk-apneus van de `dual+conf@0,50`-arm (behouden + vervallen) tegen de
NSRR-events (IoU ≥ 0,20): klasse op de thermistordaling van de bibliotheekmaat
(mediaan 60 s ervoor; A ≥ 0,72 / C 0,30–0,72 / B < 0,30) en op de robuuste maat
(p90 over ±120 s, post-hoc, `posthoc_robuust.py`).

| bevestiging | klasse (bibliotheekmaat) | n | NSRR-apneu | enig NSRR-event |
|---|---|---:|---:|---:|
| thermistor (d ≥ 0,72) | A | 678 | **0,68** | 0,84 |
| desaturatie | C | 1581 | 0,26 | **0,88** |
| desaturatie | B | 2014 | 0,07 | 0,56 |
| arousal | C | 91 | 0,14 | 0,69 |
| arousal | B | 240 | 0,02 | 0,25 |
| **vervallen** | C | 119 | 0,08 | 0,34 |
| **vervallen** | B | 674 | **0,01** | **0,07** |
| pending (lek) | A/B/C | 8 | | |

- **De thermistorbevestiging werkt:** 678 druk-apneus met een thermistordaling ≥ 0,72 zijn
  in 68 % een NSRR-apneu (84 % enig event); dat zijn de events waarvoor de vereniging
  bestaat. Op de robuuste schaal halen 1776 events A, maar daarvan is alleen de door de
  bibliotheek bevestigde deelverzameling precies (0,68); de rest (desat/arousal-bevestigd met
  A op de robuuste schaal) zit op 0,23–0,37.
- **De gevolgbevestiging houdt vooral hypopneeën:** 3595 desaturatie-bevestigde events zijn
  in 7–26 % een NSRR-apneu maar in 56–88 % een NSRR-event. Als apneu zijn ze fout getypeerd,
  voor de AHI tellen ze terecht mee. Arousal-bevestiging (331) is de zwakste tak (25–69 %
  enig event).
- **Wat vervalt, is bijna nooit een event:** 793 vervallen events matchen in 1–8 % een
  NSRR-apneu en in 7–34 % enig event. De regel verwijdert dus wat ze moet verwijderen; de
  bias-bewaker faalt omdat die non-events op dit cohort de onderschatting elders
  compenseerden.
- **Drempelveeg op de robuuste schaal** (events zonder gevolg-bevestiging, n = 1538, 2 %
  NSRR-apneu-prevalentie… precisie/recall van "d_robuust ≥ τ" voor NSRR-apneu): τ 0,72
  n = 746, precisie 0,63, recall 0,99; τ 0,80 n = 691, 0,66 / 0,96; τ 0,85 n = 529, 0,68 /
  0,76; τ 0,90 n = 247, 0,70 / 0,36. Op de robuuste schaal is 0,80–0,85 het knikpunt;
  dat is afgeleid op deze 140 nachten en dus geen validatie.
- Boekhouding: 8 "pending"-events bleven staan (fase 2 niet bereikt op die nachten — klein
  lek, te repareren) en 1 event zonder thermistordaling.

## 5. Lezing en wat eruit volgt
1. De **voorwaardelijke vereniging** haalt haar vooraf vastgelegde regel niet (bias) en blijft
   uit. Ze doet wél wat de diagnostiek van stap 1 voorspelde: ze verwijdert non-events en
   bevestigt echte apneus via de thermistor. Wie haar wil, moet de regel op bias of MAE
   herformuleren vóór een nieuwe meting — niet achteraf.
2. **`breath_dual` als standaard** staat op MESA: gelijke F1 als `breath`, bias gehalveerd,
   gelijk aan 0.17.0. Het eigen-PSG-risico (thermistor-goedgekeurde nachten) blijft een
   vraag voor de EC-studie.
3. **Strictness 0,30** haalt onder `breath_dual` alle drie criteria op 140 nachten en
   repliceert 24-08. Het is de grootste meetbaar-gewonnen knop die er nu ligt (ΔF1 +0,03,
   bias naar nul, ernst 78 → 83) en hij staat nog steeds niet aan: aanzetten verhoogt de AHI
   op lichte nachten — Barts beslissing, met de PSG-IPA-replicatie van 24-08 (zwak, 3/5)
   ernaast.
4. De **typefout** (desaturatie-bevestigde druk-"apneus" die hypopneeën zijn) is
   AHI-neutraal maar raakt de apneu/hypopneu-verhouding in het rapport; een thermistor-
   bevestigingsdrempel op de robuuste schaal (0,80–0,85) voor het *type* in plaats van voor
   het *behoud* is de logische volgende knop — als nieuwe preregistratie.

## 6. Rekenkundige metadata
Twintig workers, ~65 min per ronde van 20 nachten; de eerste nacht kwam na 62 min. Machine
bevroren 01:46 (package-temperatuur piek 82 °C om 01:19, 110 metingen ≥ 78 °C, bewaker op
81 °C vuurde nooit; RAM tussen 01:00 en 01:20 vol door verweesde baseline-workers, daarna
48 GB vrij), herstart met checkpoint. Hervatting 's avonds met 17 workers en bewaker op
78/66 °C: geen pauze. Geen nacht met fout.
