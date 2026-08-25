# Preregistratie — v4-arousalmodel met begrensde fractie-features

**Datum:** 2026-08-25, **vóór** enige training
**Aanleiding:** `docs/arousal_sn3_amplitude_bevinding.md`, `issues/v4_fractie_features.md`
**Trainingswerkruimte:** `/home/bart/MESA-ab-test/`

---

## Hypothese

`beta_ratio` is het zwaarste feature van v3 (gain 2 533 753, 3,6× boven nummer
twee) en is een VERHOUDING tegenover de NREM-basislijn. Die veronderstelt dat
arousals multiplicatief schalen met de achtergrond. Op PSG-IPA SN3 — 2,2× het
absolute beta-achtergrondvermogen van SN1 — leest het model daardoor te laag:
telling 0,52 van de menselijke tegen 0,93–1,10 elders, recall 0,31 tegen een
menselijk plafond van 0,71, bij een normale precisie (0,61 tegen 0,72).

**Verwachting:** features die per constructie begrensd én amplitude-invariant
zijn, verkleinen die opname-tot-opname spreiding.

## Wat er verandert

Vier afgeleide kolommen, berekend uit de bandvermogens die al in de dataset
staan — geen enkel EDF hoeft opnieuw door de featureextractie:

    r_w = (alpha_w + theta_w + beta_w)
          / (delta_w + alpha_w + theta_w + beta_w + sigma_w)      voor w in {pre, cand, post}
    r_delta = r_cand - r_pre

`r` ligt per constructie in [0, 1] en verandert niet wanneer het EEG met een
constante wordt geschaald. `r_delta` is de verschuiving die de AASM-definitie
beschrijft: een abrupte verschuiving NAAR alfa/thèta/>16 Hz.

De vier kolommen komen **achteraan**, zodat de bestaande vijftig hun positie
houden — het model draagt generieke kolomnamen en `psgscoring` bouwt de
featurematrix POSITIONEEL uit `_AROUSAL_LGBM_FEATURE_ORDER`. Die lijst wordt
mee uitgebreid; een test bewaakt dat trainings- en runtimevolgorde gelijk
blijven.

Trainingsopzet ongewijzigd: GroupKFold op `mesaid`, 5 folds, 500 bomen,
31 bladeren, lr 0,05, q7 als holdout. Alleen de featureset verschilt, zodat
een verschil aan de features toe te schrijven is.

## Beslisregel — VOORAF

Een kandidaat-v4 wordt alleen gebundeld als **alle vier** kloppen:

1. **Poort — geen regressie op MESA.** Holdout average precision van v4 ≥ die
   van v3 (v3: OOF AP 0,7211, AUC 0,9549; holdout in
   `arousal_classifier_v3_holdout_eval.json`).
2. **Primaire maat — de SPREIDING krimpt.** Het bereik van de count-ratio
   (onze telling / menselijke mediaan) over de vijf PSG-IPA-opnames moet
   kleiner worden dan het huidige **0,52–1,10 (bereik 0,58)**. Dít is de
   grootheid die het issue moet verbeteren, niet de mediane F1.
3. **Bewaking — geen verlies op de mediaan.** Mediane PSG-IPA-F1 van v4 ≥
   **0,514** (v3, tegen de twaalf scoorders in `EEG_arousals`).
4. **Bewaking — SN5 loopt niet weg.** SN5's count-ratio blijft ≤ 1,10.

Elk werkpunt wordt op MESA-OOF gekozen, op dezelfde manier als bij v3 (het
punt met de beste balans), en pas daarna op PSG-IPA geëvalueerd. Het werkpunt
kiezen op PSG-IPA zou dat cohort in de training trekken.

**Faalt criterium 2, dan is de hypothese weerlegd** en blijft v3 gebundeld, ook
als de mediane F1 toevallig stijgt. Een model dat gemiddeld beter is maar even
onvoorspelbaar per opname lost het gemeten probleem niet op.

## Wat deze preregistratie NIET dekt

- **EMG-dropout-augmentatie** (het tweede spoor uit het issue). Aparte
  wijziging, aparte meting; samen veranderen maakt niet toewijsbaar wat werkt.
- **Vervanging van de ratio-features.** De fracties komen ERBIJ. Weglaten van
  `beta_ratio` is een tweede experiment; het model mag zelf wegen.
- Een modelwissel in productie. Die vraagt naast deze criteria een
  gebruikersbeslissing, want een gebundeld model is niet met een profielvlag
  terug te draaien.

## Bekende beperking, vooraf genoteerd

PSG-IPA heeft **n = 5**. De spreidingsmaat is daarmee ruw: één opname die
meebeweegt verschuift het bereik zichtbaar. De maat is toch primair omdat het
de grootheid is die het klinische probleem beschrijft — maar een positieve
uitkomst vraagt replicatie op MESA per-opname vóór er iets uitgerold wordt.
