# Codereview psgscoring 0.35.2 — 11-10-2026

Leesreview van de hele bibliotheek (22 200 regels in `psgscoring/`, plus tests
en packaging) door vier parallelle controleurs met elk een deel van de modules,
daarna door mij op de ernstigste punten nagelezen in de bron. **Er is niets
gewijzigd**; dit is een lijst met voorstellen. De suite stond bij aanvang op
1492 geslaagd / 18 overgeslagen / 1 xfail (204 s).

Legenda: **[bron]** = door mij in de code nagelezen; **[repro]** = door de
controleur met een wegwerpscript numeriek gereproduceerd; **[lezing]** = alleen
uit de code afgeleid. Niets hieronder is al een meting: wat gescoorde getallen
verandert, hoort achter een vlag, met preregistratie, zoals altijd.

Gedocumenteerd-en-weerlegde zaken (Rule-1A-arousaltak, duurtolerantie,
event-locked drempel, strictness-knop dood op `rec`, enz.) staan er bewust
niet in.

---

## A. Raakt gescoorde of gerapporteerde getallen (hoogste prioriteit)

### A1. Thermistorpoort `respiratory_band` decimeert zonder anti-aliasfilter — [bron] [repro]
`signal_quality.py:904-905`: `d = x[::q]` met `q = sf // 8`. Alles boven ~4 Hz
vouwt terug in de 0,02–4 Hz-noemer die het docstring juist uitsluit. Meting van
de controleur: ademhaling 0,20 Hz + 50 Hz-toon op 0,7× amplitude → fractie
**0,671** op elke bemonsterfrequentie, tegen 1,000 met Welch op volle rate.
Drempel is 0,70 → de thermistor wordt afgekeurd "draagt geen ademhaling".
Dit is de poort van `aasm_v3_breath_dual` (productiestandaard sinds 07-10); de
07-10-meting laat zien dat precies deze poortbeslissing de AHI met +5,9/+8,7/
+37,3 /u verlegt. De invariantietoets ziet het niet (witte ruis aliast
uniform). **Voorstel:** `scipy.signal.decimate(..., ftype="fir",
zero_phase=True)` of Welch op volle rate; drempel 0,70 opnieuw afleiden op
dezelfde 9 montages (die ijking is mét aliasing gemeten); smalbandige
buiten-band-toon toevoegen aan `test_gate_metrics_are_file_invariant.py`
(faalt eerst: 0,671 ≠ 1,000).

### A2. Autonome re-ranker verliest `stage` → NREM/REM-arousalindex 0,0 — [bron] [repro]
`arousal.py:1785` bouwt de geselecteerde events met alleen
`onset_s/duration_s/autonomic_score/lgbm_proba`; `_recompute_arousal_summary`
(`:347`) leest `a.get("stage","W")` → `nrem_arousal_index = rem_arousal_index =
0.0`, alle `n_*_dominant = 0`, naast een correcte totale index. Live op
`aasm_v3_rec` (re-ranker aan sinds 0.34.0 op montages met Pleth + HR); niet op
`breath_dual`. **Voorstel:** volledige kandidaatdict doorgeven (`end_s, stage,
epoch, dominant_band, emg_confirmed`); test `nrem + rem ≈ arousal_index` als
`prov["active"]`. Telling blijft byte-gelijk.

### A3. SpO₂ "vroege nadir"-test loopt op het gecompacteerde array — [bron] [repro]
`spo2.py:83,106,111`: `seg` wordt gefilterd op 50–100 % vóór `argmin`; de index
meet daarna geen tijd meer. 10 s dropout (0 % of NaN) bij event-onset →
een echte 4 %-desaturatie met nadir op +11 s wordt `None` (de controleur
reproduceerde `(4.0, 92.0)` → `(None, 92.0)`). Onder de 3 %-regel valt de
hypopnee dan weg. **Voorstel:** `valid`-masker, `nadir_i =
flatnonzero(valid)[argmin(seg[valid])]`; op MESA n=30 meten vóór promotie.

### A4. Cheyne-Stokes-periode uit een gelijkgerichte band — [bron] [repro]
`ancillary.py:413`: `np.abs(filtfilt(...))` op een al nulgemiddelde band
verdubbelt de grondfrequentie; de autocorrelatie piekt op T/2, T, 3T/2. Repro:
88 s → **44**, 60 s → **90**, 70/100/110 correct. `flag_csr_events`
(`respiratory.py:242-258`) matcht intervallen op k·periode ± 12 s, dus de
CSR-trogcorrectie over- of ondervlagt afhankelijk van de cycluslengte.
`csr_detected` zelf blijft staan. **Voorstel:** `abs()` weg; golden
herijken op CSR-positieve nachten; CSR-herclassificatiearm hermeten.

### A5. Twee REM-AHI's in dezelfde summary met verschillende eventsets — [bron]
`respiratory.py:2832` schrijft `ahi_rem/ahi_nrem` zonder kale `uncertain`
(de 0.27.7-reparatie van "AHI 10,1 / NREM-AHI 30,9"); `pipeline.py:2493-2500`
schrijft daarna `rem_ahi/nrem_ahi` mét alle events in dezelfde dict, en
`_compute_phenotypes` (`:2328`) leest die laatste voor `rem_predominant`.
De tegenspraak van 0.27.7 staat dus één sleutelnaam verderop nog.
**Voorstel:** `rem_ahi/nrem_ahi`-schrijfregels in `_compute_rera_rdi`
verwijderen, fenotype op `ahi_rem/ahi_nrem`; in YF controleren wie `rem_ahi`
leest.

### A6. `confidence = p_scored` op breath-events verandert de betrouwbaarheidsgestratificeerde OAHI die de PDF toont — [bron]
`breath_scoring.py:587` zet `confidence = p` (0,30–1,0); `_compute_summary`
(`respiratory.py:2705`, commentaar "hypopneeën krijgen altijd conf=0,70 en
tellen dus altijd mee") stratificeert op `CONF_THRESHOLD = 0.60`. Op
`breath_dual` vallen hypopneeën met p < 0,60 dus uit `oahi_conf60`,
`confidence_bands`, `oahi_thresholds`, `ahi_excl_noise` — en YF leest precies
die velden (`generate_pdf_report.py:2737, 2760, 3255`). **Voorstel:** alleen
apneus op confidence stratificeren (hypopneeën altijd meetellen), of
`confidence = 0.70` op breath-events en `p_scored` apart laten; test op een
breath-profiel: `oahi_thresholds["0.60"] == oahi_all` als alle apneus ≥ 0,85.

### A7. Hypopneemasker snijdt alles onder `APNEA_THRESHOLD` weg — [bron]
`respiratory.py:709-712`: `& ~(smooth < APNEA_THRESHOLD)`. Bevestigde apneus
zijn al uitgesloten via `excl` (±5 s); dit extra lid splitst een hypopnee met
een korte (< 10 s) bijna-stilstand in twee te korte runs. Voorbeeld: 7 s op
0,5 / 4 s op 0,05 / 7 s op 0,5 → geen apneu (4 s) én geen hypopnee (2× 7 s),
terwijl de hele 18 s ≥ 50 % gereduceerd is. Geen CHANGELOG-rechtvaardiging
gevonden. Dit is iets anders dan de weerlegde duurtolerantie (die overbrugt
hérstel, dit splitst op díepere reductie). **Voorstel:** lid schrappen en op
`excl` vertrouwen; eenzijdig effect (alleen extra kandidaten die over hun hele
duur ≥ 30 % halen) → preregistreren en meten.

### A8. Fix 1 (herstelbasislijn na apneu) plakt vlakke 10 s-blokken in een geïnterpoleerde, stagegemiddelde basislijn — [lezing]
`respiratory.py:147-197`: `bl[center:center+step] = value` (trap in plaats van
interpolatie), ruwe P95 i.p.v. het 50/50-mengsel met de stageconstante
(sprong ≈ (dyn − stage)/2 per blok), en `win=300 s / P95 / 0,3·p95`
hardgecodeerd terwijl `mesa_shhs` 120 s / P85 gebruikt. Het docstring zegt
"~4 % van de ankers"; bij apneu-index 15–20 /u kwalificeert vrijwel elk anker.
Geen unittest. **Voorstel:** schaarse ankers herberekenen met de
profielparameters, dan één `np.interp`, dan dezelfde stage/positie-menging;
continuïteitstoets op een synthetische enveloppe.

### A9. PLM-respiratoire uitsluiting alleen ±0,5 s rond het event-EINDE — [bron]
`plm.py:410-427` test `end−0,5 ≤ lm_onset ≤ end+0,5`; AASM sluit uit van 0,5 s
vóór onset tot 0,5 s ná einde. Een beenbeweging 12 s in een apneu van 30 s telt
als PLM-kandidaat → PLMI opgeblazen op ernstige OSA-nachten. `test_plm.py`
codeert de huidige semantiek. Omdat `plm_offset_aasm` op PSG-IPA 4/5 haalde:
achter een vlag en hermeten tegen de 12 scoorders.

### A10. Apneudesaturatievenster negeert `nadir_search_s` — [lezing]
`respiratory.py:2074` geeft geen `post_win_s` door (hypopneelus op `:2331`
wel). Onder `aasm_v3_strict` (30 s) krijgen apneus 45 s en hypopneeën 30 s.
Eén regel, byte-gelijk op alle 45 s-profielen.

### A11. ODI/T90-noemer ≠ AHI-noemer — [lezing]
`spo2.py:545` deelt door alle slaapsamples incl. artefact-epochs en
dropout; AHI, arousalindex en PLMI sluiten artefact-epochs uit; hypoxische
last gebruikt valid ∧ sleep. Twee getallen in één tabel op twee noemers.
`analyze_spo2` heeft geen `artifact_epochs`-parameter.

### A12. Gemengde-apneudecompositie meet "geen inspanning" op `|ruwe RIP|` — [bron]
`postprocess.py:167-174, 223-235`: DC-offset (de 145 mV-opnames) →
`low_effort` nooit waar → nooit gedecomposeerd; nulgemiddeld → `|x|` zakt bij
elke nuldoorgang onder 20 % van het max → centrale deel breekt na de eerste
ademteug af. Plus `confidence = max(conf, 0.80)` — hetzelfde patroon dat bij
CSR na 14-08 is verwijderd. Enige test bevat geen mixed-event met inspanning.
**Voorstel:** op de inspanningsenveloppen (`preprocess_effort`) met
`EFFORT_ABSENT_RATIO`; vloer schrappen; fixture 12 s vlak + 13 s inspanning.

### A13. Hypopneesubtypering (§6.1 crit. 3) berekent de fasehoek op enveloppen — [lezing]
`respiratory.py:2408-2410` geeft `thorax_env/abdomen_env` aan
`_compute_phase_angle`, dat voor ruwe oscillaties is bedoeld. Opt-in
(`hypopnea_subtype_aasm` is nergens aan); de test voedt ruwe sinussen onder de
naam `_env`, dus groen op invoer die productie nooit levert.

---

## B. Provenance en reproduceerbaarheid (de uitvoer zegt niet wat er draaide)

### B1. Onbekende profielnaam scoort stil als `aasm_v3_rec` met de verkeerde naam in `meta` — [lezing]
`pipeline.py:164-176`: `SCORING_PROFILES.get(naam, SCORING_PROFILES["aasm_v3_rec"])`,
daarna `meta["scoring_profile"] = naam`. Een tikfout in een harnas
(`aasm_v3_breath_daul`) meet 150 nachten `rec` onder het etiket `breath_dual`.
`get_profile()` gooit wél `KeyError`. **Voorstel:** `SCORING_PROFILES[naam]`.

### B2. Falen van de breath-detector valt stil terug op de enveloppe-hypopneeën — [lezing]
`pipeline.py:1163-1165`: `except Exception: logger.warning(...)`, niets in de
uitvoer. De nacht staat dan onder `aasm_v3_breath_dual` met enveloppe-events
op een ander werkpunt. **Voorstel:** `breath_detector = {"status":
"exception", "fallback": "envelope"}` in de uitvoer en in `ahi_rule`.

### B3. `ahi_rule`, `hypopnea_criterion` worden uit het **profiel** afgeleid, niet uit wat **liep** — [lezing]
`arousal_limb_is_effective(profile)` geeft `True` voor elk breath-profiel,
ook als de arousalstap faalde (`:1008-1013`; `from .arousal import` vangt
álle excepties, dus een kapotte lightgbm-wheel schakelt de hele arousalmodule
uit) en de hypopneeën alleen op desaturatie bevestigd zijn. De PDF drukt dan
het v3-criterium. Dit is `feedback_verify_the_delivery_surface` in een andere
jas. **Voorstel:** één `ExecutionRecord` waar elke stap aan toevoegt; labels
daaruit afleiden; test die een kapotte optionele dependency het *label* laat
veranderen.

### B4. ~55 omgevingsvariabelen, 3 in `meta.env_overrides` — [lezing]
Volledige inventaris staat in het controleursverslag: 40 reads in
`pipeline.py`, 8 in `respiratory.py`, 5 in `arousal.py` (waarvan 2 op
**importtijd**, `:50-58`), 1 in `classify.py`, 1 in `utils.py`, 2 in
`arousal_unet.py`. Drie gevaren: importtijd-vastlegging (variabele ná `import
psgscoring` zetten doet niets); `PSGSCORING_UNET_SKIP_SHA=1` slaat de
checksum over terwijl `unet_onnx_sha256` de **metadata**-hash rapporteert, niet
die van het geladen bestand; `PSGSCORING_AROUSAL_LGBM` en
`PSGSCORING_APNEA_MAX_DUR_S` worden op twee plaatsen gelezen. **Voorstel:**
één `_env(name, kind, default)` die registreert; test dat `os.environ` nergens
anders voorkomt; voor de U-Net de echte bestands-hash loggen.

### B5. Positie en PLM op de stap-1-eventlijst; latere stappen vervangen die lijst — [lezing]
`pipeline.py:769-773, 817-821` nemen een momentopname; de lijst wordt daarna
vervangen op `:1040, 1134 (7b, productiestandaard), 1245, 1300, 1344, 1375`.
Alleen 7b herberekent positie; PLM nooit. `plm_index`/`n_resp_associated`
beschrijven een hybride lijst die geen verslag toont — dezelfde klasse als de
split-night-bug van 0.28.x. **Voorstel:** alle event-afgeleide blokken één
keer, na de laatste mutatie.

### B6. Per-kanaal bemonsterfrequentie = `raw.info["sfreq"]` — [lezing]
`pipeline.py:217-222`. Gevolg: `spo2_low_samplerate` (`:759-762`) kan nooit
afgaan; RIPsum-resample (`:1768`) is een no-op; `:1394-1400` bouwt de
CSR-enveloppe op `sf_ref` maar leest periodiciteit met `sf_apnea` (latent,
vandaag gelijk); `:347` valt terug op een hardgecodeerde 256.

### B7. Drie definities van slaaptijd en twee letterlijke `30` — [lezing]
Artefact-gecorrigeerd (AHI, RDI), niet-gecorrigeerd (externe arousalindex
`:848`, CSR-dichtheid `:2736`, ML-feature `:1292`), letterlijke `30.0`
(`:1328`). Eén `_tst_hours(hypno, artifact_epochs)` in `indices.py`.

### B8. Modelidentiteit — [lezing]
Alleen de U-Net heeft een geverifieerde sha256 in de uitvoer. De
arousal-LightGBM (`data/arousal_classifier_v3.txt`) draagt `Column_0…49`, het
report-JSON met de kenmerkvolgorde zit niet in repo of sdist, en de enige test
controleert `len == 50`. `PSGSCORING_AROUSAL_LGBM_MODEL` mag naar elk bestand
wijzen zonder spoor. **Voorstel:** `data/models.json` met
`{file, sha256, n_features, feature_names, trained_on, frozen_date}`, een test
die het doorloopt, en `summary["models"]`.

### B9. Profielvelden die maar op één detectorpad gelezen worden — [lezing]
Zelfde patroon als `HYPOPNEA_STRICTNESS`: `HYPOPNEA_AROUSAL_WEIGHT`,
`HYPOPNEA_DESAT_WIDTH`, `HYPOPNEA_AROUSAL_LATENCY`, `DESAT_LOW_BASELINE_RELAXATION`
(alleen breath-pad); omgekeerd `DESAT_OR_AROUSAL`/`DESAT_REQUIRED` alleen in
provenance, nooit door `score_hypopneas_breathwise` (een breath-4 %-profiel
zou stil op arousals bevestigen). Twee dode velden ontsnappen aan
`test_profile_fields_have_a_consumer` door naamcollisie:
`ApneaRules.flow_reduction_threshold` (de echte drempel is de moduleconstante
0,10; `aasm_v3_sensitive` pint 0,90 op een knop die niet bestaat) en
`SpO2Rules.baseline_window_s` (`spo2.py` hardcodeert 120).

### B10. Bevroren profielen zijn met de hand bevroren — [lezing]
Tien defaultwissels sinds augustus, vijf keer hetzelfde pin-blok gekopieerd
(`profiles.py:2340, 2386, 2435, 2496, 2621`); `AROUSAL_DERIVATION_MODE` is
geen veld maar afgeleid van `family`, zodat `chicago_1999` multi draait terwijl
de rest van zijn pins single-tijdperk is. **Voorstel:** `tests/golden/
profile_mesa_shhs.json` + `profile_chicago_1999.json` als snapshot, één
`_FROZEN_PINS`-dict.

---

## C. Compliance

### C1. Patiëntgegevens op INFO-niveau in het log — [bron]
`pipeline.py:3291-3293`: `logger.info("[pneumo] EDF patient: %s, sex=%s,
dob=%s, ...")`. Op de server staan 8 workers met stdout in `logs/`; elke EDF+
met gevulde Patient-ID schrijft naam en geboortedatum weg. Handboek §13 zegt
"PHI nooit in logs". **Voorstel:** alleen `name_present=True` loggen; dit is
een eenregelige wijziging die ik meteen zou doorvoeren.

---

## D. Tests en harnas

- **D1.** `FLOAT_ATOL = 0.25` in `test_golden_output.py:66` maakt de 3-decimale
  digest cosmetisch: AHI-drift 0,24 /u, onsetverschuiving 0,2 s of confidence
  0,55→0,79 passeren. Per-bladtype tolerantie (1e-3 indices/onsets, 1e-2
  confidences, exact voor ints/strings). [bron]
- **D2.** `compute_stage_baseline` (`signal.py:871`) gooit `IndexError` als
  `len(hypno)·spe < n` — het normale geval met een floor-lengte-hypnogram van
  YASA/NSRR — en de aanroeper slikt het op DEBUG. De "stage-specifieke
  basislijn" uit README en paper draait dus op de meeste echte opnames niet;
  de golden-gevallen zijn 600 s en zien het nooit. [bron] [repro]
- **D3.** `signal_quality_channels.py:122-134` grades op absolute std/clip in
  ruwe eenheden: SpO₂ en positie altijd "poor", µV-gedeclareerde
  ademhalingskanalen "100 % flat". De golden heeft dit **gezegend**
  (`synthetic_baseline.json:54-63, 543-552`, inclusief het `mv_scale_effort`-
  geval dat juist eenheidsbugs moest vangen). Geen directe tests voor deze
  module. Nagaan of YF `channel_quality` toont. [bron]
- **D4.** `expected_scorer_agreement` (`indices.py:121-130`) gebruikt een
  cumulatief **maximum** en noemt dat conservatief; n=70 → `f1_human 0,83,
  band "laag"` tegelijk; n=33→34 springt 0,55→0,83. [lezing]
- **D5.** Zelf-overslaande tests: `test_split_night_indices_reach_the_output.py:222,236`,
  `test_pre_lgbm_candidates.py:87` skippen als de fixture niets oplevert;
  `test_smoke.py:80-100` slaat zijn eigen controle over zodra yasa/lightgbm
  geïnstalleerd zijn (altijd, in de poort); `test_plm.py:267` xfail zonder
  `strict=True`; wavelet-pad draait in geen enkele CI-job (`denoise`-extra
  niet geïnstalleerd).
- **D6.** Geen directe tests voor `compute_mmsd`, `compute_anchor_baseline`,
  `compute_stage_baseline`, `detect_position_changes`,
  `reset_baseline_at_position_changes`, `hours_of`, `signal_quality_channels`,
  `_spo2_cross_contaminated`, `_recompute_baseline_with_recovery_excluded`,
  `decompose_mixed_apneas` (met inspanning).
- **D7.** sdist bevat geen `tests/__init__.py` en geen `tests/golden/`; de
  golden kan vanuit de sdist niet draaien of herzegend worden. `hypothesis` in
  de `test`-extra wordt nergens geïmporteerd. `uv.lock` (29-07) documenteert
  een omgeving die niemand draait. `tests.yml` pint acties op tag,
  `publish.yml` op SHA.

---

## E. Kleinere correctheidspunten

- `respiratory.py:200-221` `_spo2_cross_contaminated` vergelijkt met het
  **laatst toegevoegde** event, niet het voorgaande in de tijd (eerst de
  laatste apneu van de nacht, daarna steeds de vorige hypopnee). Veld
  `n_spo2_cross_contaminated` staat op de PDF (`:3250`). [lezing]
- `flow_reduction` is een ratio op cascade-hypopneeën en een percentage op
  gereïnstateerde en breath-events (`respiratory.py:2425` vs `:1527`,
  `breath_scoring.py:586`); `_types.py:50` zegt ratio. [lezing]
- `arousal.py:1393-1404` LGBM-terugval roept `detect_arousals` opnieuw aan
  zonder `rem_alpha_baseline, score_wake_arousals, alpha_band_wide,
  min_interval_s, resp_event_ends, …` — stil ander profiel. [lezing]
- U-Net-pad: artefact-epochs niet op events toegepast (wel op de noemer),
  `PSGSCORING_AROUSAL_MIN_INTERVAL_S` genegeerd, EOG/EMG geforceerd op
  `sf_eeg`, integer-hypnogram → 0 arousals met `success=True`. [lezing]
- `sqrt`-linearisatie vóór DC-verwijdering (`signal.py:572-574`): met offset
  0,5× amplitude wordt een echte 50 %-daling gemeten als **80 %**. Raakt elk
  dual-montage; achter een vlag meten. [repro]
- `epoch_bounds` via `int(sf*30)` op zes plaatsen — het PLM-tijdbasispatroon,
  latent zolang `sf·30` geheel is. Eén helper.
- `_rule1a_gate_param` (`pipeline.py:2838`): `float(env)` ongevangen;
  `_thermistor_gate` (`:2783`) hoofdlettergevoelig zonder `strip()`.
- `read_caisr_resp_csv` leidt de tijdbasis af en retourneert dan toch
  hardgecodeerd `1.0`; de test kan het verschil niet zien.
- `channel_map_from_user` (`utils.py:216-230`) laat een ongeldige
  gebruikersoverride stil vallen.
- `spectral_effort_classifier` (`ecg_effort.py:347-356`): 4 s Welch → 0,25 Hz
  bins → ademband 2 bins tegen ~7 cardiale; neigt naar "centraal".
- Commentaar ≠ code: `classify.py:176-180` (regel 5b na regel 4 onbereikbaar,
  dus de "2×-verruiming" van v0.8.30 deed niets); `respiratory.py:623`
  (thermistordrempel is fractie, geen daling); event-locked venster
  gedocumenteerd als `[onset, end+15]`, code `[end−5, end+15]`;
  `arousal.py:669` ">2×" terwijl 1,5; `resolve_profile_name` waarschuwt
  "removed in v0.5.0" op 0.35.2; `_types.py` noemt `psgscoring.types`,
  `arousals` i.p.v. `arousal`, en mist ~10 uitvoersleutels.

---

## F. Grotere lijnen (vijf voorstellen, in volgorde van rendement)

1. **Provenance beschrijft de uitvoering, niet het verzoek.** B1–B4, B6 zijn
   één defect van vijf kanten. Eén `ExecutionRecord` dat elke stap vult
   (`detector_used, arousal_source, n_arousals, env_overrides, fallbacks,
   channel_rates, models`), waaruit `ahi_rule`, `hypopnea_criterion`,
   `scoring_label` en `env_overrides` aan het eind worden afgeleid; test die
   een bewust kapotte optionele dependency het label laat veranderen.
2. **Eén event-lijstlevenscyclus.** Negen mutatiepunten, zeven herberekeningen
   van `summary`, vier momentopnames op vier verschillende punten (B5). Eén
   `_derive(output)`-stap na de laatste mutatie.
3. **Eén invariantieharnas voor álle poortstatistieken**, met registratie (een
   nieuwe statistiek kan niet bestaan zonder erin te staan) en drie extra
   renderings: eenheidsschaal 1e-6, 50 Hz-toon, ECG-achtige 1 Hz-pulstrein. A1
   en D3 vallen er dan uit.
4. **Modelidentiteit als data** (B8): `data/models.json`, loader weigert
   mismatch, `summary["models"]` op elk verslag.
5. **Schema scheiden van labjournaal.** `profiles.py` is voor 60 %
   meetverhaal in velddocstrings (1 892 regels dataclass) en is al minstens
   één keer van de tests afgedreven (`csr_confidence_floor`). `pipeline.py`:
   `run_pneumo_analysis` is 1 570 regels; natuurlijke naden staan in het
   controleursverslag (overrides, kanalen, duale samenvoeging, sweep,
   arousalstap, 7b, split-night, verrijkingen). `arousal.py` splitst langs zijn
   eigen bannerkoppen (rules / lgbm / multi / autonomic / coupling); eerst de
   twee byte-gelijke refactors (één spectrogram i.p.v. vijf
   `_bandpower_instant`-passes; inline summary → `_recompute_arousal_summary`).

## G. Wat ik zelf als eerste zou doen

1. C1 (PHI-log) — direct, geen meting nodig.
2. B1 (onbekend profiel → `KeyError`) en A5 (`rem_ahi` dubbel) — klein, geen
   gescoorde waarde beweegt op geldige invoer.
3. A2 (re-ranker stage) en A10 (`post_win_s`) — byte-gelijk op tellingen.
4. D1 (golden-tolerantie) en D2/D3 (stage-basislijn, kanaalkwaliteit) — maken
   het harnas scherp vóór de rest.
5. A1, A3, A4, A6, A7, A9, A12 — elk een preregistratie waard, te beginnen met
   A1 omdat die de productiepoort van `breath_dual` raakt.
