# v6 (duur zwaarder) — gemeten en verworpen

Gemeten 2026-08-26 tegen `arousal_v6_preregistratie.md`, die vóór de training is
vastgelegd. Aanleiding: `arousal_foutanalyse_verworpen_kandidaten.md`.

## Wat er getraind is

Eén getal verschil met v3: LightGBM `feature_contri` schaalt de split-gain van
`duration_s` met factor *f*. Geen nieuwe features, dezelfde trainingsdata,
dezelfde hyperparameters. Werkpunt per arm afgeleid op MESA — de drempel die
dezelfde OOF-precisie haalt als v3 bij 0,80 (0,6345), zodat PSG-IPA schoon blijft.

| arm | *f* | OOF-AP (q≥5∖q7) | **holdout-AP (q7)** | werkpunt |
|---|---:|---:|---:|---:|
| v3 | — | 0,7211 | 0,7329 | 0,800 |
| v6f2 | 2 | 0,7224 | **0,7362** | 0,800 |
| v6f4 | 4 | 0,7215 | **0,7337** | 0,805 |
| v6f8 | 8 | 0,7178 | **0,7305** | 0,810 |

**Poort gehaald door alle drie** (≥ 0,7229). Op MESA is duur zwaarder wegen
neutraal: f = 2 iets beter dan v3, f = 8 iets slechter.

## PSG-IPA (n = 5), event-F1 tegen twaalf scoorders

| opname | mens | v3 | v6f2 | v6f4 | v6f8 |
|---|---:|---:|---:|---:|---:|
| SN1 | 121 | **0,514** | 0,498 | 0,507 | 0,498 |
| SN2 | 46 | 0,396 | 0,403 | 0,394 | 0,424 |
| SN3 | 142 | 0,419 | **0,427** | 0,403 | 0,408 |
| SN4 | 104 | 0,601 | 0,602 | 0,596 | 0,603 |
| SN5 | 202 | 0,708 | 0,720 | 0,716 | 0,709 |
| **mediaan** | | **0,514** | 0,498 | 0,507 | 0,498 |
| beter dan v3 op | | — | 4/5 | 1/5 | 3/5 |
| telratio mediaan | | 1,01 | 1,04 | 1,01 | 1,02 |

## Toetsing aan de criteria

| | poort MESA | mediaan > 0,514 | ≥ 4/5 beter | SN3 stijgt | bewaker |
|---|---|---|---|---|---|
| v6f2 | ✅ 0,7362 | ❌ **0,498** | ✅ 4/5 | ✅ 0,419→0,427 | ✅ 1,04 |
| v6f4 | ✅ 0,7337 | ❌ 0,507 | ❌ 1/5 | ❌ 0,403 | ✅ 1,01 |
| v6f8 | ✅ 0,7305 | ❌ 0,498 | ❌ 3/5 | ❌ 0,408 | ✅ 1,02 |

**Geen arm haalt alle criteria. v6 wordt niet gebundeld; productie blijft v3.**

v6f2 komt het dichtst: hij verbetert op vier van de vijf opnames én op SN3, en
haalt als enige de twee criteria die ik had bedacht om cherry-picking uit te
sluiten. Hij valt op de mediaan.

### Waarom "4/5 beter" en "lagere mediaan" naast elkaar kunnen staan

De mediaan van vijf opnames ís één opname. Bij zowel v3 als v6f2 is dat SN1, en
SN1 is precies de opname die achteruitgaat (−0,016). De vier verbeteringen
(+0,007, +0,008, +0,001, +0,012) liggen om de mediaan heen en verplaatsen hem
niet. Opgeteld over vijf opnames is het effect **+0,012** — bij n = 5 is dat ruis.

Dit is een tekortkoming van mijn eigen maat, niet van de meting: bij n = 5 volgt
een mediaan één opname. Voor een volgende voorregistratie is de gemiddelde
paarsgewijze Δ met een tekentoets de betere primaire maat. Dat verandert de
uitkomst hier niet — een effect van +0,012 over vijf opnames is hoe dan ook niets.

## Wat dit toevoegt aan v4 en v5

Drie modelvarianten op rij weerlegd: v4 (extra features), v5 (herschaalde
features), v6 (andere weging). Samen met de orakelmetingen:

| grens | winst |
|---|---:|
| beste vaste drempel (orakel) | +0,008 |
| duurbewuste regel (orakel) | +0,047, maar **−0,062 op SN3** |
| perfecte selectie uit de pool (orakel) | 0,514 → **0,896** |

De eerste twee zijn klein, de derde is enorm — en de derde is geen selectie- maar
een **kandidaatgrens**. SN3 vat het samen: zijn telratio blijft over alle vier
de modellen 0,52–0,55. Het model kan niet kiezen wat de kandidaatstap niet
aanlevert.

**Conclusie: het resterende gat zit niet meer in de selectie.** De drie
modelvarianten, de twee drempelorakels en de vier telratio's van SN3 wijzen
allemaal dezelfde kant op. Vervolg: de kandidaatstap
(`arousal_kandidaatdekking.json`).

## Reproductie

`train_arousal_classifier_v6.py --boost-feature duration_s --boost-factor {2,4,8}
--match-precision 0.6345 --n-jobs 16` in `~/MESA-ab-test`; PSG-IPA met
`scripts/eval_arousal_model_psgipa.py`; het trainingsscript zelf staat buiten
de repo in `~/MESA-ab-test/train_arousal_classifier_v6.py`.

Twee valkuilen die onderweg zijn gerepareerd en die de tabel stil hadden kunnen
bederven:

1. **Modelbesmetting tussen taken.** De workers worden hergebruikt en houden de
   booster in hun module-globals. Een v3-taak op een worker die eerder een
   v6-arm draaide, zou met v6 gemeten zijn. Elke taak zet nu expliciet zijn
   model, ook v3.
2. **Draden tegen het CPU-quotum.** LightGBM negeert `OMP_NUM_THREADS` bij
   `n_jobs=-1` en opent een pool over alle 56 kernen; onder een `CPUQuota` van
   1200 % knijpt de cgroup die af en gaan de OpenMP-barrières spinnen. Wat 23
   minuten niet afkwam, was met `--n-jobs` gelijk aan het quotum in 40 seconden
   klaar.
