# Preregistratie — begrenzing van de stadiumbasislijn (reparatie van de foutmodus uit de inkortingstoets)

Datum: 2026-10-27, geschreven vóór de meting. Status: **concept, wacht op Barts akkoord** (raakt de
bibliotheek: nieuw profielveld, meting op 30 MESA-nachten + 20 eigen PSG's in situ).

## Vraag

De inkortingstoets (`docs/inkortingstoets_20261026.md`) en de diagnose van 27-10 (§3b) laten zien dat
de stadium-specifieke basislijn op drie van dertig MESA-nachten honderden valse apneus maakt doordat
de per-stadium-P90-constante onbegrensd is ten opzichte van de dynamische basislijn (0111: 488×;
0225: 97× in REM), terwijl hij op apneu-rijke nachten apneus binnenhaalt (1892: apneu-F1 0,14 →
0,40). Welke begrenzing houdt de winst en neemt de foutmodus weg?

## Armen (zelfde 30 MESA-nachten als de inkortingstoets, dekkend hypnogram; profiel `aasm_v3_breath_dual`)

| arm | `stage_baseline` | begrenzing (nieuw veld `stage_baseline_ratio_max`) | rol |
|---|---|---|---|
| UIT | False | — | referentie "zonder" (= H0 met het nieuwe veld uit) |
| AAN | True | geen (huidig gedrag) | referentie "met" (= H0_new van de inkortingstoets) |
| KLEM2 | True | stadiumconstante geklemd op [1/2, 2] × mediaan van de dynamische basislijn in dat stadium | kandidaat 1 |
| SKIP2 | True | stadium waarvan de constante buiten [1/2, 2] valt wordt overgeslagen (dynamische basislijn alleen voor dat stadium) | kandidaat 2 |
| KLEM1.5 | True | idem met [2/3, 1,5] | gevoeligheid |

Implementatie: `signal.compute_stage_baseline(..., ratio_max=None|float, ratio_mode="clamp"|"skip")`,
profielveld met default None (= huidig gedrag, byte-identiek), gelezen op de drie aanroepplaatsen;
provenance per stadium (`stage_baseline_ratio`, `stage_baseline_clamped`/`skipped`). Tests die eerst
falen; golden ongewijzigd met default None.

## Vooraf vastgelegde lezing

1. **Foutmodus weg:** op 0111, 0225 en 1305 (referentie-AHI < 2) hoort de ernstklasse van de kandidaat
   gelijk te zijn aan de referentie (Normal); een kandidaat die ergens Normal → Moderate/Severe
   houdt, valt af.
2. **Winst behouden:** apneu-F1 van de kandidaat tegen `aasm15` niet lager dan UIT op de mediaan en
   niet lager dan AAN op 1892/0583/1589 (de nachten waar AAN won) met meer dan 0,02.
3. **Primair (huisregel), kandidaat tegen UIT:** mediaan ΔF1 > 0 met Wilcoxon p < 0,05 en |bias|
   niet groter → kandidaat wint; anders gelijkspel. Ernstconcordantie kandidaat ≥ max(UIT, AAN).
4. **Eigen PSG's (in situ, R01–R20, beschrijvend):** kandidaat tegen A0 (0.37.0) en tegen 10-10
   (0.35.1): R04 moet tussen 9,0 en 19,8 /u landen met een verklaarbare sensorverdeling; geen nacht
   mag meer dan één klasse van beide referentiearmen afwijken. Geen criterium zonder referentie,
   wel prominent.
5. Beslissing: de best scorende kandidaat die 1–3 haalt gaat als default op de klinische profielen in
   een volgende release; haalt geen kandidaat 1–3, dan blijft `stage_baseline=False` op de klinische
   profielen het veiligste advies. Bart beslist.

## Uitvoer

`docs/stadiumbasislijn_begrenzing_20261027/` (rijen.json, analyse.py, run.log, thermal_guard.log),
verslag `docs/stadiumbasislijn_begrenzing_20261027.md`, verificatie-agent. Looptijd ±2,5 u op de Z6
(5 armen × 30 nachten, 8 workers) + ±1 u in situ (2 armen × 20 nachten).
