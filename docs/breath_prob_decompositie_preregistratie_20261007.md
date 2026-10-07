# Preregistratie — decompositie van de `breath`/`prob`-daling op PSG-IPA (env-armen op 0.34.2)

Datum: 2026-10-07. **Geschreven vóór de meting**, na de claim-trace van dezelfde dag
(`/srv/CODE/docs/profielvergelijking_psgipa_20261007.md` §8): de daling van `aasm_v3_breath`
en `aasm_v3_prob` tussen 9 augustus (0.15.2) en vandaag stond al op 0.27.0 (24-08) en zit dus
in het venster 0.15.2 → 0.27.0. Twee kandidaten uit dat venster zijn via omgevingsvariabelen
terug te zetten zonder bibliotheekwijziging: de EOG-afwijzing die 0.24.0 uitzette, en de
LightGBM-classifier die 0.27.0 op deze profielen aanzette (werkpunt toen 0,80).

## Vraag
Welke van de twee stappen draagt de daling van de hypopneetelling op `breath` (SN4 22 → 12,
SN1 21 → 16), of ligt de drager buiten wat env-overrides kunnen terugzetten (afleidingsset
F4-M1 + O2-M1 van 0.15.2, harnasverschil)?

## Opzet (vast)
Harnas `scripts/profile_comparison_psgipa.py`, psgscoring 0.34.2 + f9ce415 (werkboom
schoon), hypnogram van scoorder 1, geen artefact-epochs, referentie = mediaan van twaalf
scoorders; profielen `aasm_v3_rec` (controle), `aasm_v3_breath`, `aasm_v3_prob`; alle vijf de
opnames. Vijf armen, elk een eigen proces met de env gezet vóór de start (ProcessPool-workers
erven de omgeving; de pipeline leest de variabelen zelf, `pipeline.py` r. 854, 896, 2069, 2047):

| arm | env | wat het terugzet |
|---|---|---|
| A | — (default) | anker: vandaag, moet §3 van het rapport van 07-10 reproduceren |
| B | `PSGSCORING_AROUSAL_LGBM=0` | classifier uit → regelgebaseerd pad (stand vóór 0.27.0) |
| C | `PSGSCORING_AROUSAL_EOG_REJECT=1` | EOG-afwijzing aan (stand vóór 0.24.0) |
| D | B + C | beide, dichtst bij de 0.15.2-arousallijst die env kan halen |
| E | D + `PSGSCORING_AROUSAL_ONSET_OFFSET_S=0` + `PSGSCORING_AROUSAL_MIN_INTERVAL_S=0` | D plus de stappen ná 0.27.0 ongedaan (+2 s, 10 s-regel); bovengrens van wat 0.34.2 terug kan |

Wat géén arm terugzet: de afleidingsset (vandaag F4-M1 + C4-M1 + O2-M1 — SN5: Cz-M1; de
prereg schreef hier eerst "Cz-M1", gecorrigeerd ná de meting als feitelijke fout zonder
beslisgevolg —, op 0.15.2 F4-M1 + O2-M1; geen env), het werkpunt (op het regelgebaseerde pad niet van toepassing) en het harnas
van de 24-08-meting (artefact-epochs). Verschil tussen E en de 0.15.2-cijfers is dus de som
van die drie.

## Beslisregel (vooraf)
Maat: de hypopneetelling van `breath` per opname (noemer-onafhankelijk), met als tekort
t.o.v. 0.15.2: SN4 22 − 12 = 10, SN1 21 − 16 = 5 (uit de claim-trace; SN2/SN3/SN5 veranderden
≤ 1 hypopneu en tellen niet mee in de regel).
- Een arm **draagt** de daling als hij op SN4 ≥ 50 % van het tekort terughaalt (≥ 17
  hypopneeën) **en** op SN1 in dezelfde richting beweegt (≥ 18).
- Voldoet B wel en C niet → de classifier draagt het; C wel en B niet → de EOG-afwijzing;
  beide → gedeeld (rapporteer de aandelen uit B, C en D); geen van beide maar D wel →
  interactie; ook D niet → de drager zit buiten de env-stappen (afleidingsset of harnas) en
  dit dossier wijst naar een checkout-replay van v0.15.2/v0.27.0.
- **Controle (vooraf):** `aasm_v3_rec` moet in alle vijf armen op elk veld identiek zijn aan arm
  A. Beweegt `rec`, dan lekt een env in het AHI-pad en is de meting ongeldig.
- `prob` wordt in dezelfde armen gerapporteerd (eerste tussenliggende cijfers ooit), zonder
  eigen regel.

## Wat deze meting NIET is
- Geen afstelling: "welk profiel of welke arm ligt het dichtst bij de scoorders" wordt per
  arm gerapporteerd (|bias| en scoorderrang zoals §7 van het rapport van 07-10) maar is
  **geen selectiecriterium** — PSG-IPA is n = 5 en al gezien; `breath` blijft afgewezen als
  standaard op de MESA-replicatie van 03-09.
- Geen bibliotheekwijziging, geen profielwijziging, niets wordt uitgerold.

## Rekenplan
Vijf armen parallel, elk 5 workers (één per opname), `OMP_NUM_THREADS=1`; ≈ 20 min.
Temperatuurbewaker (pauze bij 81 °C, hervat bij 68 °C) gebonden aan de pgid van de wrapper,
logregel geverifieerd na 20 s. Uitvoer buiten git:
`/srv/CODE/docs/profielvergelijking_20261007/decompositie/arm_<A..E>.json` + logs. Rapport:
`psgscoring/docs/breath_prob_decompositie_20261007.md`, daarna meting-verificatie.
