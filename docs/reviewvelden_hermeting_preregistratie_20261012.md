# Preregistratie — de reviewvelden van 11-10-2026 tegen hun oude waarde (MESA n=30, PSG-IPA n=5)

Datum: 2026-10-12, **geschreven vóór de run.** Code: `main` op 7814130 (werkboom schoon
op dit document en de harnasuitbreiding `--field-arms` na). Vervolg op de codereview
(`docs/code_review_20261011.md`, CHANGELOG "Unreleased").

## Vraag

De review zette tien reparaties die een gescoord getal verschuiven **default aan** op de
klinische profielen (oud gedrag gepind op `mesa_shhs`/`chicago_1999`) en twee **default
uit**. De synthetische golden kan de meeste niet raken. Vóór een release: wat doet elk
veld op echte opnames tegen de referentie, en in welke richting?

## Cohort

- **MESA**: de eerste 30 id's van `docs/breath_dual_mesa_20261007/opnames.txt` (de
  held-out set van Validation 4; MESA is op, dus hergebruik is onvermijdelijk en wordt
  benoemd). Geen parameter van een arm hieronder is op deze nachten gekozen. Referentie
  NSRR `aasm15` (Rule 1A-reconstructie, één scoorder), matcher `LEGACY_MATCHER` (IoU 0,20,
  typeonbewust), artefact-epochs leeg (harnasdefault), hypnogram uit de nsrr-xml.
- **PSG-IPA**: SN1–SN5, 12 scoorders, hypnogram scoorder 1 (`profile_comparison_psgipa.py`).
  Referentie: scoordermediaan-AHI en het bereik; event-F1 tegen elke scoorder apart,
  mediaan over scoorders, met het plafond 0,667 (330 paren) erbij.

## Armen (één run per cohort; `--field-arms`, elke arm vertrekt van de pristine registry)

Profiel `aasm_v3_breath_dual` (productiestandaard):

| arm | velden |
|---|---|
| `nieuw` | alle defaults van `main` |
| `alles_oud` | de tien hieronder tegelijk op hun oude waarde |
| `oud_nadir` | `spo2_nadir_time_axis=False` |
| `oud_csr` | `csr_period_rectified=True` |
| `oud_plm` | `plm_resp_exclusion=end` |
| `oud_odi` | `spo2_index_denominator=sleep` |
| `oud_spectraal` | `spectral_effort_resolution=4s` |
| `oud_herstel` | `recovery_baseline_mode=block` |
| `oud_mixed` | `mixed_decomposition_on_envelope=False` |
| `oud_stadium` | `stage_baseline_partial_epochs=False` |
| `oud_fase` | `breath_phase_split=midpoint` |
| `oud_antialias` | `thermistor_band_power_antialias=False` |
| `aan_masker` | `hypopnea_mask_excludes_apnea_band=False` (de reparatie, nu uit) |
| `aan_lineariseer` | `linearise_after_dc_removal=True` (de reparatie, nu uit) |

Profiel `aasm_v3_rec` (bibliotheekdefault): `nieuw` en `alles_oud`. De rec-basislijn van
het harnas is identiek aan `aasm_v3_rec|nieuw` en dient als anker.

Niet in deze meting (raken geen event of AHI): `indices_on_final_event_list`,
`ancillary_tst_excludes_artifacts`, `rem_ahi_alias_excludes_uncertain`.

## Wat per veld meetbaar is (vooraf benoemd)

| veld | meetbaar op MESA | meetbaar op PSG-IPA |
|---|---|---|
| nadir-tijdas, herstelbasislijn, stadiumbasislijn, fasesplitsing, antialias (poort van `breath_dual`), masker, linearisatie | event-F1, AHI-bias, ernstklasse | AHI t.o.v. mediaan/bereik, event-F1 |
| CSR-periode, mixed-decompositie, spectrale resolutie | alleen apneu-**subtypes** (uit `apneas[].type`), niet de typeonbewuste F1 | idem |
| PLM-uitsluiting | niet (geen beenreferentie; MESA gediskwalificeerd) | niet (respiratoire set) |
| ODI-noemer | niet in F1/AHI; alleen de ODI zelf, zonder referentie | idem |

## Beslisregels, vooraf

1. **Default-aan-velden.** Per veld gepaard `ΔF1 = nieuw − oud` per nacht; rapporteer
   mediaan, beter/slechter/gelijk, Wilcoxon-p, Δbias (gemiddeld), Δernstconcordantie.
   Een veld wordt **ter heroverweging aan Bart voorgelegd** (terug naar oud als default)
   als mediaan ΔF1 < 0 met p < 0,05, **of** als de AHI-bias er ≥ 1,0 /u door verslechtert
   (verder van nul), **of** als de ernstconcordantie met ≥ 3/30 daalt. Anders blijft de
   reparatie de default; dit is geen promotie maar een controle op schade.
2. **Default-uit-velden** (`aan_masker`, `aan_lineariseer`): kandidaat voor default-aan
   alleen als mediaan ΔF1 > 0 met p < 0,05, |bias| niet groter en ernstconcordantie niet
   lager op MESA, **én** op PSG-IPA het aantal opnames binnen het scoordersbereik niet
   lager (5/5 blijft 5/5). Anders blijven ze uit. De beslissing zelf is Barts.
3. **Subtype-velden**: rapporteer de verschuiving in O/C/M-tellingen en, tegen de NSRR-
   types, de subtype-concordantie over gematchte apneus; geen criterium (één scoorder,
   zie `project_enriched_reclist`-les: per opname bewaren).
4. `alles_oud` tegen `nieuw`: het netto-effect van de review op F1/bias/ernst, per
   AHI-tertiel van de referentie (regel: profielvergelijkingen per tertiel).

## Verwachting (eerlijk opgeschreven)

Op MESA zonder artefact-epochs en met NSRR-hypnogrammen van exacte lengte verwacht ik dat
`oud_stadium`, `oud_odi` en `oud_plm` byte-gelijk zijn aan `nieuw`; `oud_fase` en
`oud_herstel` kleine F1-verschillen; `oud_nadir` iets meer hypopneeën onder `nieuw`
(dropouts aan de onset komen voor); `oud_antialias` alleen op nachten met netbrom op de
thermistor. `aan_masker` voegt kandidaten toe (eenzijdig); `aan_lineariseer` verlaagt de
gemeten reducties op drukkanalen met offset en dus het aantal hypopneeën.

## Rekenplan

Z6: MESA 17 configuraties × 30 nachten à ~3 min, 14 workers ≈ 2 h; thermal guard op 78 °C
gebonden aan de systemd-unit; checkpoint per nacht. PSG-IPA 16 configuraties × 5 opnames,
3 workers, ≈ 30 min, parallel. Uitvoer `docs/reviewvelden_hermeting_20261012/`
(`mesa.json`, `psgipa.json`, `analyse.py`, `analyse_uitvoer.md`), verslag
`docs/reviewvelden_hermeting_20261012.md`, daarna onafhankelijke nalezing.

## Addendum, vóór de run: herijking `THERMISTOR_BAND_POWER_MIN` op de server

De oorspronkelijke negen montages (augustus; deels werkstationbestanden op paden die de
sessie-user niet meer kan lezen) zijn niet exact te reproduceren. In plaats daarvan: **alle
uploads op de Hetzner-server die zowel een drukkanaal als een thermistorkanaal dragen**
(productiemontage Somnomedics, `Pressure Flow` + `Flow Th.`; afwijkende namen worden
gemeld, niet geraden), in situ, uploads read-only, losse container uit
`yasaflaskified:0.38.10` met de code van `main` (bd4ee7f) via `PYTHONPATH`; geen
patiëntgegevens verlaten de server, alleen bestandsnaam-hashes en getallen komen terug.

Per montage: mediaan van de ademband-fractie over tien vensters, in beide decimatiemodi
(`x[::q]` zoals geijkt, en FIR-anti-alias zoals nu default). Rapporteer de gesorteerde
verdelingen, het aantal onder 0,70 per modus, en de montages die van kant wisselen.

Beslisregel, vooraf: 0,70 **blijft** als onder anti-alias de verdeling een leeg gat heeft
dat 0,70 bevat (zoals de oorspronkelijke 0,53-brede opening). Valt 0,70 onder anti-alias
midden in een bezette zone, dan wordt het midden van de breedste lege opening tussen 0,40 en
0,95 als nieuwe drempel **voorgesteld** (niet doorgevoerd) en aan Bart voorgelegd, met de
lijst van montages die daardoor van oordeel wisselen. Zonder menselijke referentie per
montage is dit een drempel-in-het-gat-argument, geen validatie.
