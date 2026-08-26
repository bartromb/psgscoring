# Foutanalyse op de verworpen kandidaten (PSG-IPA, n=5)

Gemeten 2026-08-25. Vervolg op `arousal_waar_het_gat_zit.md`, dat drie openstaande
vragen noteerde; dit is de eerste.

## Vraagstelling

Het pool-orakel gaf F1 0,896 tegen onze 0,514 — de kandidaten zijn er dus, de
selectie faalt. De vraag: **verwerpt het model de echte arousals met overtuiging,
of net-aan?** Vooraf vastgelegd, vóór de meting:

> Grootste *d* klein (< 0,3) → het onderscheid zit niet in deze features, en
> hertrainen op dezelfde features is zinloos. Groot (> 0,5) op een bekend
> feature → de informatie ís er en het model onderweegt hem. Dan is hertrainen
> wél de weg, met dat feature als aangrijpingspunt.

## Methode

Elke gepoolde kandidaat gelabeld op scoordersstemmen (12 scoorders per opname):

- **echt** = ≥ 6 van 12 → majority-arousal
- **onecht** = 0 van 12 → niemand zag iets
- **dubieus** = 1–5 → uitgesloten uit het contrast

Referentie: uitsluitend de `EEG_arousals`-subboom (zie
`arousal_menselijk_plafond.md` — `Resp_events` is een ándere export met andere
duur en andere annotaties; dat is hier tweemaal misgegaan).

Contrast: ten onrechte verworpen (**FN**) tegen terecht verworpen (**TN**),
per feature Cohen's *d*.

## Uitkomst — de tellingen

| opname | pool | echt | dubieus | onecht | TP | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| SN1 | 1163 | 100 | 130 | 933 | 53 | 47 | 15 |
| SN2 | 1188 | 27 | 120 | 1041 | 20 | 7 | 6 |
| SN3 | 1453 | 68 | 139 | 1246 | 41 | 27 | 6 |
| SN4 | 1410 | 74 | 65 | 1271 | 57 | 17 | 15 |
| SN5 | 1601 | 195 | 111 | 1295 | 144 | 51 | 9 |

Gepoold: van 464 majority-arousals in de pool houdt het model er **315 (68 %)**,
en verwerpt het er **149 (32 %)**. Terecht verworpen: 5735.

## Uitkomst 1 — de verworpen arousals liggen *net* onder de drempel

| | mediaan kans | p75 | aandeel > 0,50 |
|---|---:|---:|---:|
| ten onrechte verworpen (149) | **0,545** | 0,674 | **53,7 %** |
| terecht verworpen (5735) | 0,018 | 0,069 | — |

Dit weerlegt de verwachting waarmee ik de meting inging. Het model **rangschikt
de gemiste arousals goed** — mediaan 0,545 tegen 0,018 voor de ruis — het krijgt
ze alleen niet over 0,80.

Dit verklaart ook waarom het drempelorakel maar +0,005 gaf (zie
`arousal_v5_preregistratie.md`): zakken naar 0,50 haalt ~80 van de 149 binnen,
maar laat tegelijk een deel van 5735 ruiskandidaten door. De precisie stort in
vóór de recall iets oplevert. **Een drempel kan dit niet oplossen; het is geen
kalibratieprobleem maar een weging.**

## Uitkomst 2 — het onderscheid zit in de duur

| feature | Cohen's *d* | mediaan FN | mediaan TN |
|---|---:|---:|---:|
| **`duration_s`** | **1,197** | **9,2 s** | 4,6 s |
| `stage_n3` | 0,843 | 0,000 | 0,000 |
| `delta_total_pre_cand` | 0,694 | 1,062 | 0,346 |
| `stage_code` | 0,690 | 2,000 | 2,000 |
| `delta_alpha_pre_cand` | 0,684 | 1,450 | 0,382 |
| `stage_n2` | 0,626 | 1,000 | 1,000 |
| `td_cand_kurt` | 0,574 | 0,864 | 0,204 |
| `bp_cand_beta` | 0,569 | 10,867 | 5,084 |
| `ratio_arousal_to_sigma` | 0,538 | 7,663 | 9,857 |
| `bp_cand_sigma` | 0,505 | 10,055 | 5,296 |

*d* = 1,20 is een groot effect, en het getal is klinisch herkenbaar: de ten
onrechte verworpen kandidaten duren mediaan **9,2 s**, de ruis 4,6 s, en
**menselijke arousals duren op PSG-IPA mediaan 8,6 s**. Het model gooit precies
de kandidaten weg waarvan de duur het best bij een menselijke arousal past.

## Uitkomst 3 — de weging staat systematisch scheef

Nagemeten over alle 50 features, met de gain-rangorde van v3 ernaast. De vier
features waar v3 het zwaarst op leunt, onderscheiden vrijwel niets:

| gain-rang | feature | Cohen's *d* |
|---:|---|---:|
| 1 | `beta_ratio` | **0,135** |
| 2 | `delta_beta_pre_cand` | 0,428 |
| 3 | `bp_post_beta` | 0,169 |
| 4 | `emg_var_ratio` | 0,215 |

En omgekeerd staan sterke onderscheiders laag in de weging:

| feature | Cohen's *d* | gain-rang |
|---|---:|---:|
| `duration_s` | 1,197 | 5 |
| `delta_total_pre_cand` | 0,694 | **44** |
| `delta_alpha_pre_cand` | 0,684 | **33** |
| `td_cand_kurt` | 0,574 | **34** |
| `bp_cand_sigma` | 0,505 | **37** |

`beta_ratio` — het zwaarste feature van het model — haalt *d* = 0,135. Het
onderscheidt de arousals die v3 ten onrechte weggooit **niet** van de ruis die
het terecht weggooit.

Dit is breder dan "duur telt te licht": de hele rangorde staat scheef op dit
cohort. Een verklaring die past is domeinverschuiving — v3 is op MESA getraind
en daar goed gekalibreerd (AP 0,73); `beta_ratio` is een contrast tegen een
NREM-basislijn, en dat contrast hoeft zich op andere apparatuur niet hetzelfde
te gedragen. Gemeten is dat niet; het blijft een hypothese.

Volledige lijst: `arousal_foutanalyse_features.json`.

## Uitkomst 4 — het plafond van een duurbewuste regel

Vóór het hertrainen: hoeveel valt er maximaal te halen met de **bestaande**
kansen plus duur? Regelfamilie *houd als p ≥ t_kort, óf duur ≥ D en p ≥ t_lang*,
met de parameters gekozen op de meetset zelf — een bovengrens, geen werkpunt.
Alle drie op dezelfde maat (kandidaatniveau, echt = ≥ 6 stemmen):

| beslisregel | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|
| v3 zoals hij draait (p ≥ 0,80) | 0,7590 | 315 | 51 | 149 |
| beste **vaste** drempel (orakel, t = 0,77) | 0,7674 | 330 | 66 | 134 |
| **duurbewust** (orakel: p ≥ 0,80 óf duur ≥ 9 s en p ≥ 0,28) | **0,8061** | 370 | 84 | 94 |

Duurbewustzijn is **ruim vijf keer** zoveel waard als de beste denkbare vaste
drempel (+0,047 tegen +0,008). Dat bevestigt de richting.

**Maar het orakel faalt precies waar het moet slagen:**

| opname | v3 | duurbewust | Δ |
|---|---:|---:|---:|
| SN1 | 0,631 | 0,720 | +0,089 |
| SN2 | 0,755 | 0,781 | +0,027 |
| **SN3** | **0,713** | **0,652** | **−0,062** |
| SN4 | 0,781 | 0,839 | +0,058 |
| SN5 | 0,828 | 0,894 | +0,067 |

SN3 heeft het grootste gat en de laagste pooldekking (50 %), en is de énige
opname die van een duurbewuste regel slechter wordt. Daar zitten de gemiste
arousals blijkbaar niet ín de pool, zodat langere kandidaten toelaten alleen
vals-positieven oplevert.

## Conclusie

De vooraf vastgelegde regel selecteert de tweede tak: **de informatie zit in de
features, en het model weegt de verkeerde.** Hertrainen is de weg — maar niet
met nieuwe features, en niet met een andere drempel.

Dat verklaart achteraf waarom v4 (begrensde fracties) en v5 (z-scores binnen
opname) beide faalden: beide voegden informatie tóe aan een model dat de
informatie die het al had verkeerd woog.

Aangrijpingspunt voor v6: `duration_s` zwaarder laten wegen bij het bouwen van
de bomen. Voorregistratie in `arousal_v6_preregistratie.md`.

Wel met een waarschuwing die uit uitkomst 4 volgt: zelfs het **orakel** haalt
criterium 3 (SN3 moet stijgen) niet. Als v6 zich als de orakelregel gedraagt,
valt hij op SN3. Dat is geen reden het niet te meten — het is de reden dat
SN3 in de criteria staat.

## Reproductie

`scripts/diag_arousal_rejected_candidates.py`, pool uit
`sweep_arousal_threshold_psgipa.py`, referentie `EEG_arousals`-subboom.
