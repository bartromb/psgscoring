# Voorregistratie v6 — duur zwaarder wegen

Vastgelegd 2026-08-26, **vóór** de training. Vervolg op
`arousal_foutanalyse_verworpen_kandidaten.md`.

## Aanleiding

De foutanalyse wees één aangrijpingspunt aan: `duration_s` scheidt de ten
onrechte verworpen arousals van de terecht verworpen ruis met **Cohen's
*d* = 1,197** (9,2 s tegen 4,6 s; menselijke arousals duren op PSG-IPA mediaan
8,6 s). Het zwaarste feature van v3, `beta_ratio` (gain-rang 1), haalt de top-10
van *d* niet — het onderscheidt de gemiste arousals **niet** van de ruis.

De informatie zit dus in de features; het model weegt de verkeerde. v4 (extra
fracties) en v5 (z-scores) faalden omdat ze informatie toevoegden in plaats van
de weging te veranderen.

## Ingreep

LightGBM `feature_contri`: vermenigvuldigt de split-gain per feature bij het
bouwen van de bomen (`gain[i] = max(0, feature_contri[i]) * gain[i]`).
Alle features 1,0 behalve `duration_s`, die factor *f* krijgt.

Geen nieuwe features, geen andere trainingsdata, geen andere hyperparameters.
Enige verschil met v3: één getal.

Verkend: *f* ∈ {2, 4, 8}. Dit is **exploratief** — drie armen op n=5 kan ik
niet als bevestiging tellen. Daarom staan de criteria hieronder strenger dan bij
v4/v5, en moet het winnende arm alle drie halen.

## Werkpunt — afgeleid zonder PSG-IPA

Een hertrainde model is anders gekalibreerd; 0,80 overnemen zou willekeurig
zijn, en het werkpunt op PSG-IPA kiezen is precies het orakel dat ik moet
vermijden. Daarom: het v6-werkpunt is de drempel waarbij de **OOF-precisie op
MESA q≥5∖q=7 gelijk is aan die van v3 bij 0,80 (0,6345)**. Volledig op MESA
afgeleid, PSG-IPA blijft schoon.

## Criteria — alle drie moeten gehaald

1. **Poort (MESA).** Holdout-AP op q=7 ≥ **0,7229** (v3 = 0,7329, marge 0,01).
   Duur zwaarder wegen mag MESA niet slopen.
2. **Primair (PSG-IPA).** Mediane event-F1 > **0,514** (v3) **én** verbetering
   op **≥ 4 van de 5** opnames. Het tekencriterium is wat v4 en v5 niet haalden
   en is niet te bereiken door één opname te bevoordelen.
3. **Doelgericht (SN3).** De F1 van **SN3** moet stijgen. SN3 heeft het grootste
   gat (0,42 tegen plafond 0,692) en de laagste pooldekking (50 %). Bij v4 was
   mijn criterium ("het bereik moet krimpen") haalbaar zónder het doel te raken;
   door SN3 bij naam te noemen kan dat hier niet.

**Bewaker.** Mediane telratio tegen de menselijke referentie blijft in
[0,85; 1,15]. Een model dat wint door meer te tellen wint niet.

## Vooraf: wat een negatief resultaat betekent

Haalt geen enkele *f* alle drie, dan is de conclusie **niet** "duur is
irrelevant" — *d* = 1,197 staat vast. De conclusie is dan dat de weging binnen
dit modeltype niet te sturen valt met split-gain, en dat het volgende
aangrijpingspunt de **kandidaatstap** is (SN3's pool dekt maar 50 % van de
menselijke arousals — die helft kan geen enkel model terugverdienen).

Bij een negatief resultaat wordt v6 **niet** gebundeld en blijft productie op
v3. Geen enkel model gaat mee in een release zonder uitdrukkelijke goedkeuring:
een gebundeld model is niet met een profielvlag terug te draaien.
