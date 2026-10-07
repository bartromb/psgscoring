# Decompositie van de `breath`/`prob`-daling op PSG-IPA — de classifier van 0.27.0 draagt haar

Datum: 2026-10-07. Preregistratie: `docs/breath_prob_decompositie_preregistratie_20261007.md` (9d6adae, geschreven vóór de meting). psgscoring 0.34.2 + f9ce415, harnas `scripts/profile_comparison_psgipa.py`, hypnogram scoorder 1, geen artefact-epochs, vijf armen parallel (5 workers elk, `OMP_NUM_THREADS=1`), 13 min, piek 78 °C, geen pauze. Ruwe uitvoer: `/srv/CODE/docs/profielvergelijking_20261007/decompositie/arm_A..E.json`, `analyse.py`, `oordeel.json`.

## Controle (vooraf): `aasm_v3_rec` identiek in alle vijf armen

Op elk veld (AHI, events, apneus, hypopneeën) en elke opname gelijk aan arm A. Geen env lekt in het AHI-pad van `rec`; de meting is geldig.

## `aasm_v3_breath` — AHI (hypopneeën) per arm

| arm | SN1 | SN2 | SN3 | SN4 | SN5 | \|bias\| | binnen range | ernst | scoorderrang SN1…SN5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.15.2 (09-08) | 6,2 (21) | 5,2 (22) | 54,0 (47) | 3,7 (22) | 9,8 (52) | 0,29 | 5/5 | 4/5 | — |
| A default (vandaag) | 5,4 (16) | 4,9 (21) | 53,5 (43) | 2,0 (12) | 9,7 (51) | 0,74 | 5/5 | 5/5 | 6, 3, 4, 10, 3 |
| B classifier uit (`LGBM=0`) | 6,6 (23) | 5,4 (23) | 55,3 (54) | 3,3 (20) | 11,0 (60) | 0,91 | 4/5 | 4/5 | 10, 6, 8, 5, 3 |
| C EOG-afwijzing aan (`EOG_REJECT=1`) | 5,4 (16) | 4,7 (20) | 53,3 (42) | 2,0 (12) | 9,5 (50) | 0,78 | 5/5 | 5/5 | 6, 1, 6, 10, 3 |
| D B + C | 6,6 (23) | 5,2 (22) | 54,8 (51) | 3,3 (20) | 11,0 (60) | 0,77 | 4/5 | 4/5 | 10, 6, 6, 5, 3 |
| E D + onsets 0 s + geen 10 s-regel | 6,7 (24) | 5,2 (22) | 54,5 (49) | 3,7 (22) | 10,7 (58) | 0,59 | 4/5 | 4/5 | 10, 6, 5, 1, 3 |

## `aasm_v3_prob` — AHI (hypopneeën) per arm (eerste tussenliggende cijfers ooit)

| arm | SN1 | SN2 | SN3 | SN4 | SN5 | \|bias\| | binnen range | ernst | scoorderrang SN1…SN5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.15.2 (09-08) | 5,7 | 3,7 | 53,1 | 2,3 | 8,8 | 0,89 | 5/5 | 5/5 | — |
| A default (vandaag) | 4,8 (13) | 4,3 (18) | 52,5 (37) | 1,7 (10) | 8,7 (44) | 1,21 | 5/5 | 4/5 | 12, 1, 8, 11, 3 |
| B classifier uit (`LGBM=0`) | 5,9 (19) | 4,1 (17) | 54,0 (46) | 2,3 (14) | 9,4 (49) | 0,48 | 5/5 | 5/5 | 1, 1, 3, 9, 3 |
| C EOG-afwijzing aan (`EOG_REJECT=1`) | 4,8 (13) | 4,1 (17) | 52,5 (37) | 1,7 (10) | 8,6 (43) | 1,27 | 5/5 | 4/5 | 12, 1, 8, 11, 3 |
| D B + C | 5,9 (19) | 3,9 (16) | 53,5 (43) | 2,3 (14) | 9,4 (49) | 0,61 | 5/5 | 5/5 | 1, 3, 4, 9, 3 |
| E D + onsets 0 s + geen 10 s-regel | 6,2 (21) | 3,9 (16) | 53,3 (42) | 2,3 (14) | 9,4 (49) | 0,69 | 5/5 | 5/5 | 4, 3, 6, 9, 3 |

## Beslisregel (vooraf): SN4 ≥ 17 hypopneeën én SN1 ≥ 18 op `breath`

| arm | SN4 (tekort t.o.v. 22) | SN1 (tekort t.o.v. 21) | draagt? |
|---|---|---|---|
| A | 12 (10) | 16 (5) | nee |
| B | 20 (2) | 23 (-2) | **ja** |
| C | 12 (10) | 16 (5) | nee |
| D | 20 (2) | 23 (-2) | **ja** |
| E | 22 (0) | 24 (-3) | **ja** |

**Oordeel volgens de regel: de classifier (0.27.0) draagt de daling.** B (classifier uit) haalt op SN4 8 van de 10 ontbrekende hypopneeën terug en op SN1 meer dan het tekort; C (EOG-afwijzing aan) verandert op SN1 en SN4 niets en op SN2/SN3/SN5 één hypopneu. D ≡ B op de regel-opnames. De stap van 0.24.0 (EOG-afwijzing uit) is dus **geen** drager; de classifier van 0.27.0, die op `breath`/`prob` het regelgebaseerde arousalpad verving, is het wel.

## Wat E leert over de rest

E (classifier uit, EOG-afwijzing aan, onsets 0 s, geen 10 s-regel) is de verste terugzetting die 0.34.2 via env toelaat. Op SN2 en SN4 landt E **exact** op 0.15.2 (22 en 22 hypopneeën); op SN1, SN3 en SN5 schiet E erover heen (24/21, 49/47, 58/52). Het enige wat E niet terugzet is de afleidingsset: vandaag F4-M1 + Cz-M1 + O2-M1 (0.27.4/0.27.6), op 0.15.2 F4-M1 + O2-M1. De extra centrale afleiding levert méér arousals en dus meer gegradeerde hypopneeën op precies de opnames waar E boven 0.15.2 uitkomt — consistent met de claim-trace, maar hier **afgeleid, niet gemeten** (geen env voor de set; dat vraagt een checkout). Voor `prob` geldt hetzelfde patroon (B/D/E 2,3 op SN4 = 0.15.2).

## Dichtst bij de scoorders — beschrijvend, geen selectie

De preregistratie sluit afstelling op deze cijfers uit (n = 5, al gezien). Ter beschrijving: op `prob` geeft de classifier-uit-arm (B) |bias| 0,48 met 5/5 binnen range en 5/5 ernstklasse, op `breath` geeft E |bias| 0,59 maar 4/5 binnen range (SN5 10,7 tegen 3,56–14,39 valt binnen; SN1 6,7 tegen 4,66–6,56 valt er net buiten). De productiestand (A) blijft op `breath` 5/5 en 5/5 met |bias| 0,74. Welke arousallijst `breath` of `prob` het dichtst bij deze vijf scoorders brengt, zegt niets over MESA — en op MESA is de classifier op `breath` juist de stap die de bias verslechterde (−2,09 → −3,28, CHANGELOG 0.27.0) terwijl hij de arousal-F1 verbeterde. Dat is dezelfde ruil als bij de 10 s-regel: betere localisatie, andere telling.

## Wat blijft staan

- De daling van `breath`/`prob` sinds augustus is verklaard: de LightGBM-classifier (0.27.0) laat op PSG-IPA minder arousals over, en `breath`/`prob` tellen hypopneeën via die arousals. `rec` is onaangetast (controle).
- Niets verandert aan de bibliotheek, de profielen of de uitrol; `breath` blijft afgewezen als standaard (MESA-replicatie 03-09).
- Open: de afleidingsset-bijdrage (SN1/SN3/SN5-overschot van E) is afgeleid, niet gemeten; een checkout-replay van v0.15.2 zou die sluiten. Alleen zinvol als iemand het exacte 0.15.2-gedrag wil reproduceren.

