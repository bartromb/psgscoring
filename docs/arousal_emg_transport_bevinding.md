# De kin-EMG bereikte de arousalclassifier nooit — en wat dat werkelijk kostte

**Datum:** 2026-08-24
**Aanleiding:** `docs/arousal-lgbm-no-emg-fix.md` (analyse), twee klinische
AZORG-opnames met een arousal-index die klinisch niet kan.
**Status:** gerepareerd in psgscoring 0.27.1 / YASAFlaskified 0.34.1.

---

## 1. Wat er kapot was

De LightGBM-arousalclassifier draaide vanaf v0.27.0 op de vier
`arousal_limb_wired`-profielen, werkpunt 0,80. Dat werkpunt is gekozen op
MESA-runs die het EDF **volledig** inlezen — met de chin-EMG erbij. De
klinische keten van YASAFlaskified leverde dat kanaal **nooit** aan, langs drie
onafhankelijke wegen:

1. `pneumo_needed = pneumo_ch_list + [eeg_ch]` — de geconfigureerde `emg_ch`
   stond er niet bij; `raw_pneumo` bevatte het kanaal per constructie niet.
2. `run_pneumo_analysis(..., channel_map=pneumo_channels)` — de respiratoire
   map, zonder sleutel `"emg"`. `ch.get("emg")` was dus altijd `None` vanuit
   de gebruikersconfig.
3. `CHANNEL_PATTERNS` kende geen `"emg"`-rol, dus autodetectie kon het gat
   niet vullen. Alleen de substringfallback in `_pick_emg` bleef over, en die
   zocht in de uitgeklede pneumo-raw.

Daarbovenop blokkeerde in `_pick_emg` een geconfigureerde-maar-**afwezige**
naam de fallback volledig: met `{"emg": "EMG1"}` en een kanaal `Chin1-Chin2`
in dezelfde raw gaf de functie `None`.

En het foutpad (`raw_pneumo = raw_staging`) had de EMG juist wél — de
classifier kreeg zijn features dus alléén na een mislukte load.

## 2. Waarom dat het model raakt (geverifieerd op het gebundelde model)

```
n_trees = 500
emg_var_ratio (kolom 47): 486 splits in 279 trees
                          min 0,0157   mediaan 1,86   max 884,4
                          drempels <= 0: 0
gain-rangorde: 4 (na beta_ratio, delta_beta_pre_cand, bp_post_beta)
emg_confirmed (kolom 10): 0 splits
```

Zonder EMG zet `_arousal_lgbm_features()` `emg_var_ratio` op constant 0,0.
Elke kandidaat gaat dan in alle 486 splits dezelfde kant op en de
kansverdeling schuift als geheel. Op een vast werkpunt is dat geen graduele
versoepeling maar een systematische verschuiving over de hele nacht.

Dat `emg_confirmed` nul splits heeft is de vergunning voor de tweede
wijziging: die semantiekreparatie kan de voorspellingen per constructie niet
bewegen. Beide eigenschappen staan nu vast in
`tests/test_arousal_model_emg_dependency.py`, zodat een v4-model dat ze niet
meer waarmaakt de guard meeneemt in plaats van hem stil te ondermijnen.

## 3. Wat het kost — gemeten, niet aangenomen

MESA n=10, dezelfde kandidatenlijst tweemaal door het model, werkpunt 0,80,
één keer met de echte chin-EMG en één keer met `emg_data=None`
(productiesimulatie):

| opname | kandidaten | met EMG | zonder EMG | AI met | AI zonder |
|---|---:|---:|---:|---:|---:|
| 0001 |  770 |  90 |  72 | 15,7 | 12,6 |
| 0002 |  909 |  96 |  89 | 15,4 | 14,3 |
| 0006 |  854 | 132 | 171 | 22,2 | 28,7 |
| 0010 |  332 |  74 |  72 | 40,5 | 39,5 |
| 0012 |  765 |  80 |  65 | 17,6 | 14,3 |
| 0014 |  893 |  68 |  70 |  9,7 | 10,0 |
| 0016 |  873 |  89 |  74 | 12,7 | 10,5 |
| 0021 | 1196 | 144 | 142 | 18,4 | 18,1 |
| 0027 | 1216 |  87 |  79 | 10,6 |  9,6 |
| 0028 |  826 | 109 | 121 | 19,7 | 21,9 |

**Mediane AI 16,65/u met EMG tegen 14,30/u zonder — ongeveer −14 %.** De
richting klopt (7 van de 10 verliezen events), maar de grootte niet: dit is
**geen** verklaring voor de klinische −80 % (AI 23,0 → 4,9 en 11,0 → 3,5).

Dat verschil blijft dus **open**. Wat de analyse voorspelde — dat de
EMG-degeneratie het gat tussen de MESA-voorspelling (−45 %) en de kliniek
(−80 %) dicht — houdt op MESA geen stand. Wie het restant wil verklaren, moet
bij de twee AZORG-opnames zelf beginnen, niet bij dit mechanisme.

## 4. Waarom de guard er tóch komt

Niet vanwege de grootte, maar vanwege de aard van de fout:

- Het werkpunt 0,80 is **gekalibreerd op runs mét EMG**. Zonder EMG draait het
  model op een invoer die in de trainingsdata niet voorkomt (`emg_var_ratio`
  exact 0 náást een gezette `emg_confirmed`). Wat er dan uitkomt is
  ongekalibreerd — de −14 % mediaan zegt niets over de spreiding per opname,
  en 0006 en 0028 gaan de andere kant op.
- Terugvallen op het regelgebaseerde pad herstelt een **gemeten, vastgelegd**
  gedrag (v0.22), in plaats van een ongemeten.
- Een apart werkpunt voor EMG-loze montages is de betere uitkomst, maar die
  meting bestaat nog niet (T5), en het model kan het beter zelf leren
  (EMG-dropout-augmentatie, T7).

## 5. Wat er nog openstaat

- **T5** — drempelsweep op een EMG-loze simulatie, om een
  `arousal_lgbm_threshold_no_emg` af te leiden in plaats van de classifier
  helemaal over te slaan.
- **T7** — v4-model met EMG-dropout-augmentatie en een expliciet
  `emg_present`-feature.
- **De klinische −80 %** — niet verklaard door dit mechanisme. Vereist de twee
  AZORG-opnames zelf.
- **Los van code:** de twee AZORG-exports bevatten géén kin-EMG terwijl de
  montage die normaal heeft. Controleer het exportprofiel van de recorder —
  geen enkele reparatie hierboven raakt die schakel.
