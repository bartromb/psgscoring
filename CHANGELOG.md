# Unreleased — breath-graded hypopnea detector becomes the default

## Changed — the default scoring profile

`run_pneumo_analysis()` now defaults to **`aasm_v3_breath`** instead of
`aasm_v3_rec`. On PSG-IPA this is better on every aggregate measure
(median and mean event-F1, percentile within the inter-scorer distribution,
AHI bias, MAE) with equal severity concordance; see the detector entry under
*Added* for the full table and the known limitations.

**What does not move with it:**

- The legacy aliases `strict` / `standard` / `sensitive` still resolve to
  `aasm_v3_strict` / `aasm_v3_rec` / `aasm_v3_sensitive`. They document the
  historical profiles.
- `mesa_shhs` is untouched, so NSRR reproduction is unaffected.
- Paper v31/v37 reproduces by naming `aasm_v3_rec` explicitly, which
  `validate_psgipa.py` already does.
- YASAFlaskified passes `scoring_profile="standard"` explicitly, so
  **deployed behaviour does not change** until that pin is changed
  deliberately.

Reverting is one line: `scoring_profile: str = "aasm_v3_rec"` in
`pipeline.py`. `tests/test_profiles.py::TestDefaultProfile` pins both the
default and the alias stability.

### Held-out confirmation on MESA/NSRR

50 records, seed 20260801, drawn from the 2055 with both an EDF and an NSRR
annotation. **Nothing was re-tuned**: `hypopnea_strictness` stayed at the
0.50 chosen on PSG-IPA. Both profiles ran on the same record with the NSRR
hypnogram, so only respiratory scoring differs. `scripts/validate_mesa.py`.

The reference is reconstructed, not counted — see that script's docstring.
NSRR's `Hypopnea`/`Unsure` labels carry no desaturation requirement; it is
applied afterwards by linking to the separate `SpO2 desaturation` events.
Reconstruction validated against published `oahi35`/`oahi45`: bias +0.11 /
+0.19 AHI, MAE 0.84 / 0.66, **r = 0.998** over 60 records.

| ref | profile | F1 med | F1 mean | prec | recall | bias | r | LoA | severity |
|---|---|---|---|---|---|---|---|---|---|
| oahi4 | `aasm_v3_rec` | 0.379 | 0.361 | 0.332 | 0.484 | −2.24 | 0.731 | −17.3…+12.8 | 29/50 |
| oahi4 | `aasm_v3_breath` | **0.442** | **0.411** | **0.365** | **0.571** | **−0.36** | **0.754** | **−14.8…+14.1** | 25/50 |
| oahi3 | `aasm_v3_rec` | 0.414 | 0.399 | 0.471 | 0.404 | −7.95 | 0.751 | −27.9…+12.0 | 23/50 |
| oahi3 | `aasm_v3_breath` | **0.464** | **0.432** | **0.507** | **0.494** | **−6.07** | **0.784** | **−24.8…+12.7** | 22/50 |

Paired per record: **31 better / 17 worse** on oahi4 (Wilcoxon
**p = 0.0069**) and **31 better / 18 worse** on oahi3 (**p = 0.039**). The
PSG-IPA finding replicates on data that had no hand in choosing the
operating point.

**Superseded in part — read the second round below before using this
table.** The two references above (`oahi35`/`oahi45`) credit no hypopnea that
qualifies on arousal alone, while Rule 1A does. That made every arousal-only
event a false positive and depressed both precision and recall. A third
reference reconstructed from `nsrr_ahi_hp3r_aasm15` — literally the rule
these profiles implement — changes the conclusion about severity.

**The apparent trade-off.** Event-level agreement improves and so do
correlation and limits of agreement, but severity concordance against
MESA's 4% definition falls from 29/50 to 25/50, and the loss is almost
entirely one cell:

| error | `aasm_v3_rec` | `aasm_v3_breath` |
|---|---|---|
| **Normal → Mild** | 7 | **10** |
| Moderate → Mild | 3 | 5 |
| Severe → Moderate | 2 | 5 |
| Mild → Moderate | 5 | 3 |

Higher recall (0.484 → 0.571) tips three more normal recordings over
AHI 5 — clinically the least comfortable direction, a false-positive OSA
call. And the gain is distributed as the mirror image of that risk:

| reference severity | median ΔF1 (oahi4) | better |
|---|---|---|
| Normal (n=16) | −0.002 | 6/16 |
| Mild (n=19) | +0.033 | 12/19 |
| Moderate (n=10) | **+0.096** | **9/10** |
| Severe (n=5) | **+0.153** | 4/5 |

The detector wins where there is disease and is neutral-to-harmful in
normals. MESA is a community cohort with many normal recordings; a sleep-lab
population is enriched for disease, which shifts the balance — but that is an
argument to weigh, not a reason to discount the finding.

### Second MESA round — the severity trade-off was a reference artefact

A second sample of 50 records, seed 20260802, drawn from the 2005 that the
first round did not use (**overlap 0**, asserted by the harness). Three
configurations run on the *same* records so every comparison is paired:
`aasm_v3_rec`, breath at 0.50, breath at 0.55.

The reason for the round was a mistake worth recording. On the strength of
the `Normal → Mild` drift above, strictness was raised to 0.55 — a change
made *after* seeing the round-1 result, and therefore no longer held-out.
This round was meant to confirm it on unseen records. It refuted it.

Against **`aasm15`** (reconstructed from `nsrr_ahi_hp3r_aasm15`: all apneas
plus hypopneas with ≥30% reduction and (≥3% desaturation **or arousal**) —
AASM v3 Rule 1A, r = 0.999 against the published column):

| config | F1 med | F1 mean | prec | recall | bias | MAE | r | severity |
|---|---|---|---|---|---|---|---|---|
| `aasm_v3_rec` | 0.417 | 0.390 | 0.547 | 0.356 | −16.52 | 16.79 | 0.686 | 15/50 |
| **breath @ 0.50** | **0.471** | **0.450** | 0.567 | **0.407** | **−15.78** | **15.96** | 0.838 | **24/50** |
| breath @ 0.55 | 0.441 | 0.431 | **0.574** | 0.378 | −16.93 | 17.03 | **0.841** | 20/50 |

Paired against the previous default: **0.50 improves 36/50 records
(p = 0.0016)**; 0.55 improves 31/50 (p = 0.026).

**0.50 beats 0.55 on F1, recall, bias, MAE and severity — on all three
references.** Severity concordance, the very thing 0.55 was supposed to
protect: 24/50 vs 20/50 (`aasm15`), 28/50 vs 25/50 (`oahi4`), 24/50 vs 21/50
(`oahi3`). Strictness has therefore been returned to **0.50**.

Why the round-1 reading was wrong is visible in the direction of the errors:

```
aasm_v3_rec   Severe→Moderate:10, Mild→Normal:9, Severe→Mild:6, Moderate→Normal:4
breath @0.50  Moderate→Mild:7, Severe→Moderate:7, Severe→Mild:6, Mild→Normal:3
```

Against the rule these profiles actually implement, the problem was never
over-calling — it is substantial **under**-detection (bias −16.5;
`aasm_v3_rec` gets only 15/50 severities right). The breath detector reduces
it: `Mild → Normal` falls from 9 to 3, and severity concordance rises from
15/50 to 24/50. The `Normal → Mild` drift that motivated 0.55 was an artefact
of scoring a Rule 1A profile against a 4% reference.

The general lesson, which cost a wrong parameter change: **when a validation
reference encodes a different rule than the thing being validated, its
error-direction statistics are not interpretable — only the paired
difference is.** The paired result held throughout; the severity reading did
not.

**What is still not established.** Both rounds are MESA. Cross-cohort
transfer is untested — SHHS was considered and set aside on data-quality
grounds — so nothing here shows the result holds on different equipment,
a different scoring era, or a different population. And the absolute level
remains poor: a bias of −16.5 AHI against Rule 1A means both profiles miss a
large share of qualifying events. This work improves the gap; it does not
close it.

## Known issue — blocks enabling the arousal-limb work by default

**The arousal plumbing repair changes `mesa_shhs`, which is a hard
requirement violation, and the cause is the ML re-classifier.**

`mesa_shhs` is the only profile that runs the LightGBM candidate
re-classifier, and `apply_ml_reclassification()` takes the arousal list.
Its highest-gain features are arousal-based (`n_arousals_per_h`,
`n_arousals_within_30s`, `has_arousal_within_5s`). Because of issue #16
those features were **always zero at inference**. Feeding them real values
makes the model reclassify large numbers of rejected candidates:

| mesa-sleep-2408 | main | this branch |
|---|---|---|
| AHI | 30.4 | **54.9** |
| RDI | 30.4 | 84.4 |
| events | 291 | 465 |
| RERAs | 0 | 204 |
| arousals visible | 0 | 372 |

This happens with the arousal limb **disabled** — it is the classifier, not
the limb. It also means the model has a train/serve mismatch that predates
this work: whatever it learned about arousal features, it has never seen a
non-zero one in production. Re-validating or retraining the classifier is a
prerequisite for shipping the plumbing fix, and that is out of scope here.

## Fixed

- **The AASM Rule 1A arousal limb was dead, and so was FRI-based RERA
  detection.** `run_arousal_respiratory_analysis()` returns its events nested
  at `["arousals"]["events"]`, while step 8 read the flat
  `output["arousal"]["events"]` — a key the auto-detection path never
  populated (issue #16). On PSG-IPA SN3: 232 arousals detected, 0 seen.
  `_normalise_arousal_block()` now guarantees the flat key from either
  producer.

  This also repairs two consumers outside scoring: YASAFlaskified's EDF+
  export and event API read the same flat key, so exported EDF+ files
  contained no arousals and the score editor showed none.

  Repairing the plumbing alone already changes **RDI**, because
  `_compute_rera_rdi()` uses the same list — FRI-based RERAs could never
  couple. The golden case shows it: `rdi 0.0 -> 37.9`.

- **`rule1a_arousal_stats`** in the respiratory output: arousals available,
  candidates tested, coupled, qualified, and an explicit `skipped_reason`.
  A limb sitting silently at zero is what let this survive; zero now always
  comes with a reason.

## Added

- **`breath_scoring.py` — a hypopnea detector with the breath as the atom,
  behind the new profile `aasm_v3_breath`** (family `exploratory`, used by
  nothing automatically).

  The existing detector is a signal-processing model: envelope → sliding
  threshold → candidate → validate → classify, with a dozen corrections on
  top. Nearly every correction repairs the same gap with what a scorer
  actually does. This replaces that mismatch rather than patching it
  further, along five shifts:

  1. **The breath is the atom, not the sample.** The AASM speaks of *peak
     signal excursions*. Event boundaries land on breath transitions, and a
     recovery breath breaks the run by itself — so smoothing is gone, and
     with it the merge/split problem that depresses event-F1.
  2. **Two-pass patient calibration.** Pass 1 detects only the
     incontestable events and derives a template — typical depth, typical
     duration, and the patient's **own SpO₂ lag** via cross-correlation
     instead of one fixed 30–45 s window for everybody. Pass 2 judges the
     marginal candidates against that template.
  3. **Graded AASM predicates.** The rule structure stays literally the
     Rule 1A conjunction; what changes is that each threshold gets a
     tolerance instead of being infinitely sharp. A 29% reduction with a 6%
     desaturation should score; 31% with a doubtful 3.0% should not.
  4. **Each event carries `p_scored`** — "what fraction of scorers would
     mark this" — plus a `criteria` dict with each predicate's
     contribution. The audit trail gets richer, not more opaque.
  5. **Strictness is one axis** instead of three parameter combinations.

  Scope is hypopneas only. Apneas keep the existing detector (F1 0.83–0.93
  on PSG-IPA, with effort-based subtyping); there is no deficit there for
  this to fix.

  **Measured on PSG-IPA (5 recordings, 12 scorers each, legacy matcher).**

  The operating point was chosen with a **rule declared before the sweep**:
  take the strictness where the AHI bias crosses zero, *not* the strictness
  that maximises F1 — the latter would be fitting on the outcome measure.
  That rule lands on 0.50. F1 and percentile are therefore reported as
  non-fitted outcomes.

  | config | F1 median | F1 mean | pct | AHI bias | MAE | severity | hyp |
  |---|---|---|---|---|---|---|---|
  | `aasm_v3_rec` (previous default) | 0.343 | 0.460 | p6 | +1.77 | 1.84 | 4/5 | 208 |
  | breath, strictness 0.35 | 0.447 | 0.522 | p17 | +2.73 | 2.73 | 3/5 | 242 |
  | breath, strictness 0.45 | 0.430 | 0.515 | p17 | +1.13 | 1.13 | 4/5 | 192 |
  | **breath, strictness 0.50** | **0.434** | **0.511** | **p17** | **+0.17** | **0.29** | 4/5 | 163 |
  | breath, strictness 0.55 | 0.426 | 0.510 | p14 | −0.89 | 0.89 | **5/5** | 131 |
  | breath, strictness 0.62 | 0.438 | — | p9 | −1.57 | 1.57 | 5/5 | 110 |
  | breath, strictness 0.70 | 0.396 | 0.495 | p14 | −2.61 | 2.61 | 4/5 | 79 |

  Per recording at strictness 0.50, with the human-human distribution for
  scale (66 pairwise F1s per recording):

  | rec | human median | F1 previous | F1 breath | Δ | pct |
  |---|---|---|---|---|---|
  | SN1 | 0.826 | 0.470 | 0.662 | **+0.192** | p0 → p0 |
  | SN2 | 0.549 | 0.317 | 0.380 | +0.063 | p9 → **p23** |
  | SN3 | 0.948 | 0.886 | 0.897 | +0.011 | p2 → p5 |
  | SN4 | 0.553 | 0.286 | 0.184 | **−0.102** | p17 → p17 |
  | SN5 | 0.556 | 0.343 | 0.434 | +0.091 | p6 → **p18** |

  Better on every aggregate: median F1, mean F1, percentile, bias, MAE;
  severity concordance equal at 4/5. MAE drops from 1.84 to **0.29** AHI
  points.

  The detector diagnostics confirm the mechanism rather than just the
  outcome. The patient-specific SpO₂ lag comes out at 21/12/15/19/17 s —
  plausible and genuinely per-patient, which is the whole point of shift 2 —
  and the template duration is 15–21 s, i.e. actual hypopnea durations.

  **Known limitations.**

  - **SN4 regresses (0.286 → 0.184) and no strictness value fixes it.**
    Of its 24 events with ≥6/12 scorer consensus, **12 never become a
    candidate at all**: at those locations there is no breath-level flow
    reduction of ≥15% sustained over 10 s. Compare SN1, where 33 of 34
    consensus events do produce a candidate. This is a sensitivity ceiling
    in candidate formation, not a threshold that can be turned down. Its
    percentile is unchanged at p17 either way, because SN4's own scorers
    disagree strongly there (pairwise F1 p10 = 0.087, p25 = 0.462) — but
    that is context, not an excuse.
  - **n = 5, and the operating-point rule was applied to the same five
    recordings the result is reported on.** See the MESA held-out result
    below, which was run afterwards with nothing re-tuned.
  - On the recordings where scorers *do* agree, the algorithm remains at or
    below the bottom of their distribution (SN1 p0 against a human median
    of 0.826; SN3 p5 against 0.948).

- **`compute_pre_event_baseline()`** in `signal.py` — the AASM-conforming
  baseline: only breaths in the window *before* the event, breath-amplitude
  based. Stable breathing (CV below threshold) gives the mean amplitude;
  otherwise the mean of the N largest, which is the operationalisation the
  AASM gives when stable breathing cannot be determined. Returns `None`
  rather than a fabricated number when the window holds no usable breathing
  (start of recording, after a gap, or entirely in wake) so the caller can
  fall back deliberately.

- **`rule1a_arousal_enabled` in `PostProcessingRules`** (default **False**)
  and `rule1a_gap_max_breaths` (default 1, the previous hard-coded value).

  The plumbing is repaired, but *enabling* the limb is a behaviour change,
  so it stays off. Measured on PSG-IPA with the limb on:

  | rec | AHI off | AHI on | hypopneas | tested | coupled | qualified |
  |---|---|---|---|---|---|---|
  | SN1 | 8.1 | 9.3 | 32 -> 39 | 74 | 11 | 7 |
  | SN3 | 53.8 | 56.0 | 45 -> 58 | 70 | 20 | 13 |
  | SN5 | 11.4 | 14.8 | 63 -> 87 | 170 | 38 | 24 |

  Coupling is far from universal — 11/74, 20/70, 38/170 — which matches the
  phase-1 finding that only ~17% of rejected candidates have an arousal
  inside the 15 s window. Window and gap are now profile-tunable so phase 4
  can determine them empirically instead of by assumption.

- **`baseline_mode` in `PostProcessingRules`** — `"rolling"` (current) or
  `"pre_event"`. **Every profile keeps `"rolling"`**, including `mesa_shhs`;
  nothing switches until phase 4 justifies it. The rolling baseline stays in
  place for signal quality and the existing visualisations either way.

  Wired into the hypopnea threshold assessment: with `"pre_event"` both
  sides of the ratio come from the breath segmentation (candidate mean
  amplitude vs pre-event baseline), so they measure the same quantity. The
  envelope used by the rolling path is a different scale and is deliberately
  not mixed in. When the baseline is unavailable the event falls back to the
  rolling validation rather than being silently dropped.

  `PSGSCORING_BASELINE_MODE` overrides the profile, so phase 4 can measure
  both settings without mutating profiles (same pattern as
  `PSGSCORING_AROUSAL_DERIVATION`).

  First measurement on PSG-IPA SN1 (reference scorer median AHI 5.96):

  | mode | AHI | events | hypopneas | rejected | local_reduction rejects |
  |---|---|---|---|---|---|
  | rolling | 8.1 | 47 | 32 | 74 | **56** |
  | pre_event | 11.6 | 67 | 52 | 54 | **1** |

  The predicted by-catch is confirmed and it is large: Fix 6
  (`local_reduction`) drops from 56 rejections to 1, i.e. it becomes almost
  entirely redundant once the baseline is measured before the event instead
  of around it. Fix 1 and Fix 6 are not removed here — phase 4 measures what
  they still contribute first.

  AHI rises on this recording, away from the scorer median. That is not by
  itself a verdict: the decision measure for this work is event-level F1
  against human scorers and the percentile within the inter-scorer
  distribution, not AHI bias, precisely because a better bias can come from
  errors that cancel. Phase 4 settles it.

  Why it matters: `compute_dynamic_baseline()` uses a *centred* 5-minute
  window, so it includes the recovery hyperpnoea that follows an event. That
  inflates the baseline and shrinks the measured reduction. Fix 1 and Fix 6
  are patches on precisely that design.



**Rename with aliases — no behaviour change** (golden regression green;
PSG-IPA output byte-identical).

## Changed

- **The arousal qualifier is AASM Rule 1A, not 1B.** The AASM places the
  arousal criterion in Rule 1A (">=30% flow reduction AND (>=3% desaturation
  OR arousal)"); Rule 1B is the >=4%-desaturation variant that explicitly
  *excludes* arousals. The code had it the other way round.

  | was | is |
  |---|---|
  | `reinstate_rule1b_hypopneas()` | `reinstate_rule1a_arousal_hypopneas()` |
  | event `classify_detail.rule = "1B_arousal"` | `"1A_arousal"` (+ `rule_legacy`) |
  | event flag `rule1b` | `rule1a_arousal` |
  | `respiratory.rule1b_reinstated` | `respiratory.rule1a_arousal_reinstated` |

## Deprecated (kept working)

Every old name is retained as an alias and is still emitted in the output.
This is not cosmetic caution — three consumers depend on them:
YASAFlaskified imports `reinstate_rule1b_hypopneas` in `pneumo_analysis.py`
and reads `rule1b_reinstated` in both `generate_pdf_report.py` and
`generate_psg_report.py`, and `psgscoring.ml_classifier` uses the candidate
flag `rule1b` as the trained LightGBM feature `is_rule1b` — dropping it
would silently deprive the model of an input. `tests/test_rule1a_arousal_naming.py`
pins all of it.

Note this rename does not by itself make the arousal limb fire; that is a
separate plumbing defect (issue #16).
# v0.12.3 — 2026-07-29 — NSRR nasal pressure detected as flow_pressure, not pulse

**Channel-detection fix — no scoring-logic change** (golden regression byte-identical;
PSG-IPA output byte-identical, md5 5ddbcf42360d965cd20ce1b8796221da before and after).

## Fixed

- **`Pres` (the NSRR/MESA/SHHS nasal-pressure channel) was claimed by the `pulse` role.**
  Channel matching is case-insensitive substring, first-match-wins per role, so pattern
  *order* is semantics. The `pulse` list contained `"pr"` — a substring of `"pres"` — and
  no `flow_pressure` pattern matched a bare `"Pres"`. Two silent consequences on MESA:
  `flow_pressure` stayed empty, so `_resolve_flow_channels()` ran **both** apnea and
  hypopnea detection on the thermistor (amplitude ~1500× smaller than the nasal pressure);
  and the `pulse` role took the nasal pressure instead of `"HR"`.
  `"pres"` is now in `flow_pressure` (deliberately *after* the specific patterns, so an
  explicit `"Nasal Pressure"` still wins) and `"pr"` moved to the end of `pulse`.
  MESA now resolves `flow_pressure=Pres`, `flow_thermistor=Therm`, `pulse=HR`.

Recordings that already resolved their channels correctly are unaffected — PSG-IPA has
neither a `Pres` nor an `HR` channel, and its full 5-recording output is byte-identical.
On MESA the hypopnea F1 barely moves (median 0.304 → 0.321) even though the hypopnea
sensor switches off a 1500× smaller channel: detection normalises, so amplitude scarcely
matters. The value is correctness of sensor assignment, not a metric gain.

# v0.12.2 — 2026-07-27 — docs: absolute DISCLAIMER link (fix dead link on PyPI)

**Docs-only — no code change** (golden regression byte-identical).

- README: the `DISCLAIMER.md` link is now absolute
  (`https://github.com/bartromb/psgscoring/blob/main/DISCLAIMER.md`). Relative links
  break on the PyPI project page (404); this release re-renders the README on PyPI with
  the working link. No API or behaviour change.

# v0.12.1 — 2026-07-26 — ventilatory burden made breath-based (fixes over-count)

**Output-additive — no AHI/OSAS-grade change** (golden regression byte-identical).

## Fixed

- **Ventilatory burden is now breath-based.** v0.12.0 computed VB as the *time-fraction*
  of the normalized flow envelope below 0.5, which over-counted heavily (the smoothed
  Hilbert envelope dips between breaths even during normal breathing, and the baseline is
  the 95th percentile) — a real severe-OSA recording gave an implausible **82.9 %**.
  VB is now the **proportion of breaths whose peak amplitude is < 50 % of the eupneic
  baseline** (per breath, using `output["respiratory"]["_breaths"]`), matching the AJRCCM
  2023 definition and excluding inter-breath troughs. Signature:
  `compute_ventilatory_burden(flow_norm, sf_flow, breaths, hypno=None, threshold=0.5)`.

# v0.12.0 — 2026-07-26 — VB recalibration + saturation bands + arousal-aetiology fix

**Output-additive — no AHI/OSAS-grade change** (golden regression byte-identical).
Three report-driven fixes found while verifying a real v0.11.0/v0.15.0 report.

## Changed

- **Ventilatory burden recalibrated to the validated metric.** `summary["ventilatory_burden"]`
  is now **the percentage of sleep with airflow < 50% of the eupneic baseline** (proportion
  of "small breaths"), per *Ventilatory Burden*, AJRCCM 2023;208(11):1153 & 1216 — a bounded
  0–100% measure that predicts CV/all-cause mortality (normative ≈ ≤25%). Replaces the earlier
  unbounded %·min/h area integral (which produced implausible values, e.g. ~1074). Computed on
  the normalized flow envelope restricted to sleep. `compute_ventilatory_burden` signature is
  now `(flow_norm, sf_flow, hypno=None, threshold=0.5)`; `VB_NORMATIVE_MAX = 25.0` exported.
- **Arousal-aetiology indices now reconcile with the arousal index.** `respiratory_arousal_index`
  + `spontaneous_arousal_index` are derived by splitting `arousal_index` by aetiology fraction,
  so they **sum to the arousal index exactly** (previously divided the raw counts by a
  differently-computed TST, so resp + spont < AI). `plm_arousal_index` is reported as a subset
  of the spontaneous group.

## Added

- **Time-in-saturation-bands** in the SpO2 summary: `time_95_100_min`/`pct_95_100`,
  `time_90_95_min`/`pct_90_95`, `time_80_90_min`/`pct_80_90`, `time_70_80_min`/`pct_70_80`,
  `time_below_70_min` (the report's band table read these keys, which did not exist → all 0.0).

All output-additive; AHI/OSAS/CSAS grade and arousal index unchanged.

# v0.11.0 — 2026-07-26 — AASM v3 clinical enrichments (dual-AHI, CSR density, arousal aetiology)

**Output-additive — no AHI/OSAS-grade change** (golden regression byte-identical).
Derived from the manual re-read in `docs/aasm/` (AASM Scoring Manual v3, 2023).

## Added

- **Dual AHI (A1)** — `summary["ahi_dual"]`: AASM v3 hypopnea **Rule 1A** (≥30% flow +
  ≥3% desat OR arousal — the recommended standard) and **Rule 1B / CMS** (≥4% desat)
  AHI side by side, with severity per rule. Reuses the existing 3-profile confidence
  pass (`standard`=aasm_v3_rec, `strict`=aasm_v3_strict); no extra detection pass.
- **Cheyne-Stokes density criterion (A2)** — `cheyne_stokes` now also reports the AASM
  G.1(b) criterion: `central_events_per_h`, `monitoring_hours`, `density_criterion_met`
  (≥5 central apneas/hypopneas per hour over ≥2 h), and `criteria_met` (periodicity
  G.1(a) **and** density G.1(b)).
- **Arousal aetiology indices (A3)** — `arousal["summary"]`: `respiratory_arousal_index`,
  `spontaneous_arousal_index`, and `plm_arousal_index` (+ `n_plm_arousals`) per hour of
  sleep (AASM V.A Note 4). Counts existed; now exposed as clinical indices.
- **Hypopnea criterion string (A5)** — `meta["hypopnea_criterion"]`: the exact scoring
  rule in words (AASM v3 VIII.D Note 1 requires the criterion be stated in the report).
- **Hypoventilation scope statement (A6)** — `summary["hypoventilation"]` explicitly
  marks it *not assessed* (no PCO2/capnography channel), rather than silently omitting it.

## Changed

- **Apnea/hypopnea max-duration cap (A4)** — the split cap stays profile-default (90 s /
  60 s → **byte-identical**) but is now overridable per site via
  `PSGSCORING_APNEA_MAX_DUR_S` / `PSGSCORING_HYPOPNEA_MAX_DUR_S` (AASM imposes no maximum
  apnea duration; splitting a genuinely long *central* apnea over-counts). Apneas sitting
  at the cap are flagged in `summary["n_apneas_at_max_dur"]`.

All of the above are output-additive; the AHI, OSAS/CSAS grade and arousal index are
unchanged from v0.10.0.

# v0.10.0 — 2026-07-25 — clinical phenotypes (POSA, REM-predominant) + ventilatory burden

**Output-additive — no AHI/OSAS-grade change** (golden regression byte-identical;
validated on PSG-IPA: SN1 AHI 8.1 unchanged).

## Added

- **Phenotype flags** in the respiratory summary (`summary["phenotypes"]`), derived
  from the already-computed position + REM/NREM indices (`_compute_phenotypes()`):
  - **Positional OSA (Cartwright):** supine AHI ≥ 2× non-supine AHI, with OSA present
    (AHI ≥ 5) and ≥ 30 min sleep in both the supine and non-supine groups. Reports
    supine/non-supine AHI + ratio + positional-therapy candidacy (non-supine AHI < 5).
    Requires a body-position channel.
  - **REM-predominant OSA:** REM-AHI ≥ 2× NREM-AHI, with ≥ 30 min REM.
- **Ventilatory burden** (`summary["ventilatory_burden"]`, %·min/h;
  `compute_ventilatory_burden` in new module `ventilation.py`): total event-associated
  airflow deficit relative to the eupneic baseline — pairs with the hypoxic burden as a
  cardiovascular-risk duo beyond the AHI. Method after Labarca et al. 2023.
  ⚠️ The exact parameterization/scale should be confirmed against Labarca 2023 before
  clinical interpretation (no reference range asserted yet).

Both are output-additive; the AHI and all existing indices are unchanged.

# v0.9.0 — 2026-07-25 — multi-derivation arousal detection is now the DEFAULT

**Behaviour change (clinical profiles only).** Arousal detection now defaults to
**multi-derivation** (central + occipital + frontal, event-level union + EOG-reject)
for all `family="clinical"` profiles (`aasm_v3_rec`/`_strict`/`_sensitive` and their
aliases). `single` is retained as an explicit option.

## Changed

- **Default arousal mode per profile family** (`constants.py`,
  `AROUSAL_DERIVATION_MODE`): **clinical → `multi`**, **dataset → `single`**
  (`mesa_shhs` and any NSRR-reproduction profile stay single-channel — their
  reference was scored on one central derivation, so this keeps MESA/SHHS
  reproduction byte-identical), legacy/exploratory → `multi`.
- Choose explicitly at any time with env `PSGSCORING_AROUSAL_DERIVATION=single|multi`
  (overrides the profile) or the profile field. With < 2 usable EEG derivations,
  `multi` degrades to `single` automatically.

## Impact & reproducibility

- **The AHI is unchanged** by this flip on the validation cohort (arousals
  contribute 0 arousal-only events; the golden cases carry no EEG so they are
  unaffected). The **arousal index rises** (multi is more sensitive: PSG-IPA
  sens 0.38 → 0.47 vs the scorer consensus).
- **For exact single-channel reproduction of the paper's PSG-IPA numbers**, set
  `PSGSCORING_AROUSAL_DERIVATION=single`. MESA reproduction is unaffected
  (`mesa_shhs` is a dataset profile → already single).
- Rationale: clinicians score arousals by scanning the whole montage; multi
  (union + EOG-reject) is the more sensitive, more human-like operating point.
  Note the Fase-3 finding stands — the gain is *generic sensitivity*, not an
  occipital-specific recovery (see v0.8.1 notes) — so this is a deliberate
  sensitivity-first product choice, not an accuracy improvement.

# v0.8.1 — 2026-07-25 — multi-derivation arousal detection (opt-in; default unchanged)

**No default behaviour change — `single` mode is byte-identical to v0.8.0.**

## Added

- **Multi-derivation arousal scoring** (`arousal.detect_arousals_multi` + event-level
  `_union_arousals`; `pipeline._pick_eeg_multi`). Runs the mature single-channel
  detector independently on central (C4-M1) + occipital (O2-M1) + frontal (F4-M1)
  and unions overlapping events, keeping per-channel validation (spindle exclusion,
  K-complex, REM-EMG) intact. Events gain `derivation`/`derivations` provenance.
- Opt-in via env `PSGSCORING_AROUSAL_DERIVATION=multi` (or a future profile field
  `AROUSAL_DERIVATION_MODE`). **Default stays `single`**; with < 2 usable
  derivations it degrades to single automatically.
- **Hard invariant** (unit-tested): multi with one derivation == `detect_arousals`
  byte-for-byte.

### Fase 2 — human-like refinement (also opt-in)

- **Per-channel detection thresholds are now parameters** (`detect_arousals(...,
  ratio_thresh=, abrupt_thresh=)`), replacing module-global mutation — concurrency-safe
  (8 workers + the multi-derivation loop) and enabling per-derivation calibration.
  Defaults unchanged → **byte-identical** (unit-tested), incl. the LGBM hybrid path.
- **`detect_arousals_multi(..., per_channel_thresh=, eog_data=, eog_reject=)`**: each
  derivation can run at its own calibrated threshold, and an **EOG-based reject drops
  occipital-only events that coincide with a large eye movement** (EOG-doorslag) —
  mirroring a scorer cross-referencing the EOG. Cross-channel-confirmed events are
  never touched. All default-off.

## Validation (PSG-IPA, 5 recordings × 12 scorers, consensus ≥6/12)

- **Per-channel sweep vs scorer consensus** — best single-channel F1: C4-M1 0.28,
  O2-M1 0.32, **F4-M1 0.41** (frontal > occipital > central), i.e. the standard
  single central derivation is the *weakest* for arousals. F1 peaks near the current
  defaults; frontal benefits from ratio 2.5.
- **Payoff (single vs multi-default vs multi-tuned+EOG-reject):**
  sens **0.38 → 0.53 → 0.47**, PPV **0.36 → 0.30 → 0.32**, F1 **0.37 / 0.38 / 0.38**.
  The tuning+EOG-reject **cuts ~16% of the naïve-union detections (mostly false
  positives) and recovers PPV while keeping most of the sensitivity gain** — but no
  configuration dominates on F1 (arousal-vs-consensus agreement is fundamentally
  modest, consistent with low inter-scorer reliability).
- **Net:** multi (esp. tuned+EOG-reject) is a defensible, more sensitive, more
  human-like operating point (its value is *sensitivity* — catching arousals the
  single central channel misses — not overall accuracy). The **AHI is unaffected**
  (0 arousal-only events on this cohort). Kept **opt-in**; no default flip. The
  scorer consensus is itself an imperfect reference (low inter-scorer agreement),
  so F1 understates true performance.

### Fase 3 — pipeline-wired + the occipital question settled

- **Multi mode is now fully reachable through `run_pneumo_analysis`** (not just the
  low-level API): set `PSGSCORING_AROUSAL_DERIVATION=multi` (or profile field
  `AROUSAL_DERIVATION_MODE`). In multi mode the pipeline unions the available
  central/occipital/frontal derivations **and applies the EOG-reject by default**
  (picks an EOG channel; disable with `PSGSCORING_AROUSAL_EOG_REJECT=0`). New
  `pipeline._pick_eog`. **Default stays `single` → byte-identical** (re-verified:
  SN1 8.1/32 unchanged).
- **The original occipital hypothesis is NOT supported by the data.** Per-derivation
  recall on scorer-consensus arousals split by where scorers marked them
  (posterior/occipital-involving vs anterior): C4-M1 **0.29 post / 0.23 ant** (no
  posterior blind spot — central is if anything slightly *better* posteriorly),
  O2-M1 **0.31 / 0.31** (occipital is not specifically better at occipital-evident
  arousals), **F4-M1 0.39 / 0.46 (frontal is the strongest single channel)**, multi
  0.50 / 0.55 (a *generic* sensitivity gain across both classes). So multi-derivation
  is a coverage/sensitivity choice, not a fix for a central occipital blind spot.
- **Design decision:** ship multi as a well-engineered, EOG-aware, **opt-in
  high-sensitivity mode**; keep single-central as the reproducible default. No
  default flip is justified by the current evidence (F1 does not improve; the AHI is
  unchanged; occipital adds no specific class of arousals).

# v0.8.0 — 2026-07-25 — arousal & RERA detection moved in-package (byte-identical)

**Structural — no scoring change. AHI / ODI / events / hypoxic burden byte-identical.**
Validated on the real PSG-IPA cohort (5 recordings): AHI, arousal counts and
arousal index unchanged to the digit (SN1 8.1/32 arousals, SN3 53.8/185, …;
0/516 AH-events arousal-only — the AHI is desaturation-driven).

## Added / Changed

- **New module `psgscoring.arousal`** — EEG arousal detection, RERA detection,
  flow-limitation detection and respiratory-arousal coupling now live **inside
  psgscoring**. Ported verbatim from YASAFlaskified `arousal_analysis.py` so the
  library is self-contained; `run_pneumo_analysis` no longer depends on an
  external, optionally-importable module. New public API: `detect_arousals`,
  `detect_reras`, `run_arousal_respiratory_analysis`.
- `pipeline.py` imports the detector in-package (`from .arousal import …`) instead
  of the fragile `from arousal_analysis import …` optional import.
- Added `from __future__ import annotations` to the ported module — fixes a
  Python 3.9 crash (PEP 604 `X | None` unions evaluated at def-time).
- Optional LightGBM arousal re-classifier carried over; model path/env made
  psgscoring-neutral (`PSGSCORING_AROUSAL_LGBM[_MODEL/_THRESHOLD]`, old
  `YASAFLASKIFIED_*` env still honoured). Default off → pure rule-based, identical.
- YASAFlaskified `arousal_analysis.py` becomes a thin re-export shim
  (requires psgscoring ≥ 0.8.0; deploy the pair together).

This lands the groundwork for **multi-derivation arousal scoring** (central +
occipital + frontal), which will be added as an opt-in mode on top.

# v0.7.6 — 2026-07-22 — fix: hypoxic-burden TST denominator excludes invalid SpO2

**Numerics-changing — hypoxic burden only; no AHI/ODI/event change.** Closes #2
(PR #3).

## Fixed

- **Hypoxic-burden TST denominator now excludes invalid SpO2 samples**
  (`spo2.py`, `compute_hypoxic_burden`). The per-event desaturation-area
  numerator already drops NaN / out-of-[50,100] samples, but the TST
  denominator was built from the hypnogram alone (`sleep_mask`), so sensor
  dropouts during sleep inflated the denominator and **deflated** the burden.
  `tst_h` now uses `sleep_mask & ~np.isnan(spo2)`, matching the de Chazal
  `calcHB.m` reference (`HourSleep`). On a recording with ~32 min of in-sleep
  SpO2 < 50 the reference TST shifts 7.20 h → 6.67 h (higher, correct HB).
- Only `hypoxic_burden` changes, and only on recordings with invalid in-sleep
  SpO2; recordings with clean oximetry are byte-identical (`~np.isnan` is all
  True). AHI / ODI / events / all other fields are untouched. Golden 6/6
  unchanged (synthetic cases have no in-sleep dropouts). New unit test
  (`test_spo2_numeric.py`) asserts HB rises by exactly the TST reduction.
- **MESA A/B (2026-07-22):** on the same 16-recording sample, `hypoxic_burden`
  changed on 5 recordings — all **raised** (e.g. 156.18→167.52, 17.06→18.34),
  never lowered — while AHI/events stayed byte-identical, confirming the
  denominator correction behaves as intended on real data.

# v0.7.5 — 2026-07-22 — fix: RERA/RDI dropped on Cheyne-Stokes nights

> Stacks on **v0.7.4** (the output-preserving robustness/test PR); merge that first.

**Numerics-changing — but strictly additive: no AHI, event, or SpO2 number
changes.** On recordings where Cheyne-Stokes respiration is detected, the RERA
index, RDI, and REM/NREM AHI silently vanished from `summary`; they are now
retained.

## Fixed

- **RERA/RDI/REM-NREM AHI no longer wiped on CSR-positive nights**
  (`pipeline.py`). `_compute_rera_rdi` (step 8b) wrote `rera_index` / `rdi` /
  `n_rera` / `rem_ahi` / `nrem_ahi` into the respiratory summary, but the
  Cheyne-Stokes "Fix 3" step then replaced `output["respiratory"]["summary"]`
  wholesale with a fresh `_compute_summary()` — dropping every one of those keys
  whenever `csr_detected` was true. `_compute_rera_rdi` now runs **after** the
  CSR summary recompute, so the keys survive. Non-CSR nights are unaffected
  (the CSR block does not fire, and nothing between the old and new call
  position touches the summary → byte-identical); CSR nights now report RDI et
  al. instead of `None`.

## Validation

- The fix is a pure statement reorder. `_compute_rera_rdi` only ever **reads**
  `ahi_total` and **writes** the RERA-family keys, so it cannot move any AHI /
  OAHI / event / SpO2 value — RDI is derived as `ahi_total + rera_index`, never
  the reverse.
- **Golden harness (6/6):** the 3 CSR-positive synthetic cases show exactly one
  changed field — `resp.rdi: null → value` (= `ahi_total`, no arousals) — with
  `ahi_total`, every event, and all other summary fields byte-identical;
  re-blessed accordingly. The 3 non-CSR cases are unchanged.
- **PSG-IPA reproducibility:** byte-identical (clinical cohort, non-CSR).
- **New regression tests** (`test_rera_csr_ordering.py`, 4 cases) build a
  CSR-positive synthetic recording (periodic apneas) and assert the full RERA
  family — including non-trivial values with arousals (`rera_index > 0`) and REM
  (`rem_ahi` populated) — survives to the final output.
- **MESA empirical A/B — CONFIRMED (2026-07-22).** Ran `scripts/ab_rera_csr.py`
  on a 16-recording MESA sample (9 CSR-positive) with the validated `score_mesa`
  harness (validation-mode tuning, NSRR hypnogram + arousals): 0.7.3 vs 0.7.6.
  Result — **0 invariant fields moved** (`ahi_total` / `ahi_incl_uncertain` /
  `n_events` byte-identical on all 16), and the RERA family restored (None→value)
  on exactly the 9 CSR-positive recordings (45 keys). Confirms the fix is
  strictly additive on real clinical data.

# v0.7.4 — 2026-07-22 — robustness & test coverage (output-preserving)

Code-review follow-up. **No scoring changes** — golden harness + PSG-IPA
reproducibility byte-identical; full suite green (129 passed).

- **Graceful degradation for ancillary steps.** A failure in any single
  ancillary analysis (SpO2, position, heart rate, snore, PLM, arousal) now
  degrades to a `{"success": False, "error": …}` result via a new `_run_step`
  helper instead of aborting the whole `run_pneumo_analysis`, matching the
  pattern the interval/ML/CSR/hypoxic-burden/post-processing steps already used.
- **`apply_ml_reclassification` honours its docstring contract.** The whole
  featurise → predict → sort body is guarded, so a malformed candidate (missing
  `onset_s`), a booster shape/NaN mismatch, or the onset sort now falls back to
  the rule-based result rather than crashing.
- **`ecg_effort.detect_r_peaks` guards low sample rates.** The 5–30 Hz QRS
  bandpass would raise for `sf ≤ 60` (Nyquist ≤ 30); it now returns no peaks,
  which also stops `compute_tecg` crashing on low-rate ECG. Never fires on real
  PSG ECG (128–512 Hz), so output is unchanged.
- **Removed dead recomputation in `detect_respiratory_events`.** The initial
  hypopnea mask (peak + envelope + OR, plus a full breath-amplitude pass) was
  computed and then discarded — only the post-apnea-corrected mask is consumed.
  Deleting it removes redundant work per profile with byte-identical output.
- **New numeric unit tests** (`test_spo2_numeric`, `test_respiratory_core`,
  `test_postprocess_numeric`, `test_crash_safety`) so ODI / T90 / hypoxic
  burden / breath segmentation / dynamic baseline / CII and the crash-safety
  guards are checked on every `pytest tests/` — previously the only numeric
  coverage was the env-gated golden test.
- **Packaging:** ship `py.typed` (PEP 561) so downstream mypy/pyright see the
  inline annotations.
- Housekeeping: `signal_quality_channels.py` gets its own logger name (was
  colliding with `signal_quality.py`), a correct filename docstring, and the
  fictitious "v0.8.30" stamp removed; dropped a redundant inline
  `preprocess_flow` re-import in `pipeline.py`.

# v0.7.3 — 2026-06-09 — documentation / terminology

Docs and terminology only — **no code or scoring changes** (golden harness
byte-identical; full suite green).

- **Dropped the "AASM 2.6" version label throughout** (project description, README,
  DISCLAIMER, docstrings and code comments). The library supports AASM v1 / v2 / v3
  via scoring profiles (default `aasm_v3_rec` = AASM 2023), so pinning the prose to the
  2.6 manual was inaccurate. Text now reads "AASM-compliant" / "AASM" / the specific
  profile name.
- **Refreshed the developer handbook** (`docs/developer_handbook.md`) to the current
  state: v0.7.2 / YASAFlaskified v0.12.4, psgscoring now pip-installed (not bundled),
  the git → PR → CI → GitHub-Release/OIDC → rsync workflow, the v0.7.x profile set +
  3-profile AHI interval + v0.7.2 shared-preprocessing speed-up, and the paper-v37
  validation table. Authors section updated.

# v0.7.2 — 2026-06-07 — performance (shared preprocessing)

**No scoring changes — output is byte-identical.** Speeds up `run_pneumo_analysis`
by ~1.8–2.0× by removing redundant work in the 3-profile AHI confidence interval.

## Performance

- **Preprocessing computed once across the AHI-interval profiles** (`pipeline.py`,
  `respiratory.py`). The confidence interval ran `detect_respiratory_events` 4×
  per recording (primary + strict/standard/sensitive), each redoing the full
  preprocessing: Hilbert envelopes (~43 s), dynamic baseline / rolling percentile
  (~24 s), position-change detection (~10 s), MMSD, effort/SpO₂ baselines, breath
  detection. All of this depends only on the raw signals and on baseline params
  that are identical across the three interval profiles, so it is byte-identical
  across the reruns. A shared `_precomputed` cache now computes it once; only
  per-profile event qualification reruns. Baseline-dependent caches are keyed by
  `(BASELINE_WINDOW_S, BASELINE_PERCENTILE)` so a non-v3 primary stays correct.
- **Position changes computed once** (part of the cached baseline block).
- **Fixed an interval reuse bug:** the primary result was matched against the
  legacy alias (`standard`), so canonical names (`aasm_v3_rec`) never matched and
  the primary was needlessly re-scored a 4th time. Now matched on `_PROFILE_NAME`.

Validated byte-identical (every summary field, AHI interval, and individual
event matches the previous output exactly) against:
- the golden harness (6/6) and the full unit suite (104 passed, 11 skipped);
- the **MESA q7 holdout** — all 92 locally-available recordings, via the
  `score_one_mesa` validation path (arousal injection + artifact epochs +
  the `aasm_v2_rec` profile with the validation-mode tuning);
- the **PSG-IPA clinical cohort** (SN1–SN5, profile `standard` = `aasm_v3_rec`);
- the `cms_medicare` param-keyed cache path (MESA id 301).

Measured speedup on the analysis: MESA id 301 125.8 s → 62.5 s (2.0×),
id 33 90.5 s → 49.7 s (1.8×).

# v0.7.1 — 2026-06-03 — documentation

Docs-only release — **no code or scoring changes** (identical scoring to
v0.7.0). Corrects the README / PyPI project description:

- Quick Start uses the canonical profile name `aasm_v3_rec` (not the
  deprecated `standard` alias) and the correct `results["ahi_interval"]` key.
- Architecture section: drop the removed `pipeline_profiles` module, add
  `ml_classifier`; refresh module/test/line counts (17 submodules, 115 tests).
- PSG-IPA validation figure corrected to the validated mean |ΔAHI| = 1.8/h
  (Pearson r = 0.997).

# v0.7.0 — 2026-06-03 — Tier-1 scoring-accuracy fixes

These fixes correct over-detection on degraded signal and profile-specific
counting. They were **validated to leave both paper cohorts byte-identical**
(PSG-IPA clinical and the MESA q=7 holdout: 0 recordings changed vs v0.6.2)
and to have negligible real-data impact on the MESA q∈[2,4] POOR cohort
(1/94 recordings changed; agreement vs the scorer reference unchanged) — i.e.
they fix genuine edge-case bugs without moving any validation number. The
large effects on synthetic stress cases (a literal flat-line channel; the
`cms_medicare` profile) do not arise on real recordings. Clean-signal output
is unchanged.

## Fixed

- **Dead/flat signal regions no longer scored as apneas**
  (`respiratory.py` `_detect_signal_gaps`). A frozen/dead channel had
  `flow_norm ≈ 0` across the flat span and scored as back-to-back apneas;
  the span itself is now excluded from event detection (not just the
  post-gap recovery ramp). Large effect on POOR-quality / dropout
  recordings — golden `poor_quality`: 14→2 events, AHI 88.4→12.6;
  `flat_dropout`: 5→4, AHI 31.6→25.3.
- **CMS / AASM-v1 profiles no longer reinstate arousal-only hypopneas**
  (`pipeline.py` Rule 1B gated on `DESAT_OR_AROUSAL`). `cms_medicare` and
  `aasm_v1_rec` score hypopneas on desaturation only; arousal-coupled
  reinstatement is now skipped for them — golden `cms_arousal`:
  `rule1b_reinstated` 2→0, AHI 37.9→25.3. No effect on `aasm_v3_*` /
  `mesa_shhs` (`DESAT_OR_AROUSAL=True`).
- **Hypopnea baseline honors the profile window/percentile on the
  mixed-sample-rate / RIPsum-fallback path** (`respiratory.py`
  `_setup_hypop_channel`). Previously reverted to the 300 s / 95th
  defaults instead of e.g. `mesa_shhs`'s 120 s / 85th. Affects `mesa_shhs`
  on mixed-sample-rate / degraded-nasal recordings — needs real-data
  validation (single-sfreq synthetic cases do not exercise it).
- **SpO2 sensor-dropout gaps no longer manufacture desaturations**
  (`spo2.py` `detect_desaturations`). NaN gaps were filled with a constant
  95%, creating a fake plateau at gap edges that registered as a
  desaturation; gaps are now interpolated for smoothing and the dropout
  regions are excluded from detection.
- **Mixed-apnea decomposition is NaN-safe** (`postprocess.py`). A NaN in
  the effort segment made `np.max` → NaN, silently defeating central-portion
  detection; now uses `np.nanmax` with a finite guard. (Type label only;
  no AHI change.)

## Validation status

#1 and #3 are quantified by the golden harness (above). #4 and #6 are
implemented but need real `mesa_shhs` / MESA validation. Before any 0.7.0
release: run the q7 cohort on 0.7.x vs 0.6.2 (golden_snapshot / score_mesa)
and re-run the PSG-IPA clinical validation to confirm the clinical headline
is unchanged.

# v0.6.2 — 2026-06-03

Dual AHI reporting for unsubtyped apneas. No change to event detection or to
any existing field; adds two summary keys.

## Added

- **`summary["ahi_incl_uncertain"]`** — scorer-calibrated AHI that also counts
  apneas the effort-based classifier could not subtype (`type == "uncertain"`,
  typically when the RIP/effort signal is degraded). `ahi_total` continues to
  exclude these (conservative, flag-for-review). On the MESA q=7 holdout the
  inclusive index is essentially unbiased against the NSRR scorer reference
  (bias ≈ −0.3/h), whereas `ahi_total` runs ≈1.5/h lower; both are now
  reported so downstream consumers can choose.
- **`summary["n_uncertain_apnea"]`** — count of detected-but-unsubtyped apneas
  (the events that distinguish the two indices), for transparency / review.

## Notes

- An `uncertain` apnea is still a scored apnea; AHI by definition counts every
  apnea regardless of obstructive/central/mixed subtype. Clinical (non-ML)
  profiles produce no `uncertain` apneas, so `ahi_incl_uncertain == ahi_total`
  there and PSG-IPA reproducibility is unchanged.

# v0.6.1 — 2026-06-03

Robustness and reproducibility patch. Fixes two POOR-quality scoring
crashes, a Python 3.9 import regression, and a signal-quality grading
bug; adds a golden-output regression harness. **Clinical AASM profiles
are unchanged** — PSG-IPA reproducibility (10/10) is intact and the
fixes only recover recordings that previously crashed; they do not
alter the scoring of any recording that already produced a result.

## Fixed

- **Rule 1B reinstatement crash (`KeyError: 'stage'`).** The
  stable-breathing rejection filter stored rejected hypopnea
  candidates without `stage`/`epoch`, so `reinstate_rule1b_hypopneas`
  crashed whenever such a candidate coincided with an arousal — i.e.
  any recording scored with EEG arousal detection (the normal clinical
  path). The Rule 1B call had no `try/except`, so the whole analysis
  aborted.
- **ML re-classification crash (`KeyError: 'type'`).** The `mesa_shhs`
  LightGBM re-classifier promotes pooled candidates into the accepted
  list, but rejected hypopnea candidates carried no `type` key, so a
  promoted one was type-less and crashed `_compute_summary`. Recovered
  ~64% of the MESA q∈[2,4] graceful-degradation cohort (30/100 → 94/100
  scored). No effect on ML decisions (feature extraction already
  defaulted `type` to `"hypopnea"`).
- **Python 3.9 import crash.** `signal_quality_channels.py` used a
  PEP 604 `list | None` annotation without
  `from __future__ import annotations`, making the module unimportable
  on Python 3.9 — silently disabling per-channel quality grading
  (`channel_quality: "unknown"`) for all 3.9 users.
- **`_count_flat_samples` overcount.** The flat-sample count was
  inflated by ~the window length (a `/len*len` cancellation), so
  `flat_pct` could exceed 100% and clean channels were mis-graded
  `"poor"`. Metadata only — no effect on AHI.
- **Robustness guards.** Empty-flow input in `_detect_signal_gaps`;
  out-of-range event epoch in `analyze_position` (one bad epoch no
  longer drops the whole per-position summary).

## Added

- `tests/test_golden_output.py` + `tests/golden/` — golden-output
  regression harness (6 deterministic synthetic cases; gated behind
  the `PSGSCORING_GOLDEN` env var, out of the default CI matrix).
- `scripts/golden_snapshot.py` — bless/check pipeline output on real
  EDFs with a per-recording AHI-delta table.
- `tests/test_rejected_candidate_invariant.py` — guards the invariant
  that rejected candidates carry `type`/`stage`/`epoch`.

## Changed

- `_types.PLMSummary` field names aligned with `analyze_plm` output
  (`plm_index`, `n_resp_associated`, …); added the missing keys.
- Removed the unreferenced `pipeline_profiles.py` stub (stale
  duplicate `run_pneumo_analysis`).

# v0.6.0 — 2026-05-05

LightGBM candidate-level re-classifier as an optional post-detection
step. Trained on the q$\geq$5∖q=7 stratum of the MESA cohort
(n=653 recordings, ~210k labelled candidates) with 5-fold group-CV
by mesaid; held out q=7 entirely from training. Default in the
`mesa_shhs` profile; clinical profiles leave it None and skip the
step (PSG-IPA paper-v31 reproducibility 10/10 pass).

## Added

- `psgscoring/ml_classifier.py` — module providing `load_booster`,
  `apply_ml_reclassification`, and a runtime feature extractor
  (`_extract_candidate_features`) mirroring the training-time
  `build_lightgbm_dataset.extract_features` (32 features per
  candidate: event-intrinsic, sleep-stage, cluster context,
  recording-level, surrounding arousals).
- `PostProcessingRules.ml_classifier_path: str | None = None`
  — path (absolute or relative to package data dir) to a LightGBM
  booster file. None disables the step (default for clinical).
- `PostProcessingRules.ml_threshold: float = 0.65`
  — bias-near-zero operating point established on q=7 holdout.
- `psgscoring/data/lightgbm_v06_q7holdout.txt` — shipped 810 KB
  trained model. Top-5 features by gain: `desaturation_pct`,
  `n_arousals_per_h`, `stage_r`, `time_to_next_event_s`,
  `n_arousals_within_30s`.
- `pyproject.toml` — `ml` extra (`pip install psgscoring[ml]`)
  installs `lightgbm>=3.0`; package-data section ships
  `data/*.txt`.
- `pipeline.py` — Step 8a (between Rule 1B and RERA/RDI) calls
  `apply_ml_reclassification` when profile sets a path.

## Changed

- `mesa_shhs` profile sets
  `ml_classifier_path="data/lightgbm_v06_q7holdout.txt"` and
  `ml_threshold=0.65`. Clinical profiles unchanged.

## Validation

- **PSG-IPA reproducibility:** 10/10 tests pass in 11:59
  (clinical-profile defaults unchanged → paper v31 strict / standard
  / sensitive numerics bit-identical to v0.5.2 and earlier).
- **MESA q=7 holdout:** with the natively-loaded `mesa_shhs`
  profile and threshold 0.65,
  **bias $-0.02$/h, MAE 5.34/h, Pearson $r$ 0.872, weighted $\kappa$
  0.497, severity-match 63\%** vs the v0.5.2 rule-based baseline
  (bias $+1.10$, $r$ 0.804, $\kappa$ 0.481, sev 59\%). Threshold
  sweep (paper v35 §3.6.1) exposes the calibration curve from
  0.45 to 0.70.
- **Cross-validation:** 5-fold AUC $0.818\pm 0.006$;
  within-distribution test AUC $0.811$.

## Known limitations carried forward

- Severe-AHI under-detection on Compumedics (paper v34 §S5.7–S5.8:
  mesaid 6382 with 91% never-seen events) is upstream of this
  re-classifier — the LightGBM cannot recover events the rule-based
  detector never proposes. Closing that residual is targeted for
  v0.7 (multi-channel candidate fusion or deep-feature extraction).

# v0.5.2 — 2026-05-04

`mesa_shhs` profile: cohort-specific re-enablement of envelope smoothing
and a lower consecutive-breath threshold for peak-based detection. No
psgscoring source-code changes — purely two field overrides on the
`mesa_shhs` PostProcessingRules. Clinical profile defaults
(`aasm_v3_*`, `aasm_v2_rec`, `aasm_v1_rec`, `cms_medicare`,
`chicago_1999`) are unchanged and PSG-IPA reproducibility is preserved.

## Changed (`mesa_shhs` profile only)

- `mesa_shhs.post_processing.flow_smoothing_s = 3.0` (was 0.0).
  Re-enables the 3-second envelope smoothing that was the pre-v0.2.8
  default. v0.2.8 removed it from clinical defaults because it caused
  +54 false hypopneas on PSG-IPA SN1 (severity drift Mild → Moderate),
  but the MESA Compumedics signal characteristics differ from the
  PSG-IPA Vitalograph hardware — the smoothing bridges brief noise
  excursions in the Pres envelope and lets legitimate ≥10s
  flow-reduction periods form contiguous below-threshold runs in the
  candidate-formation step.
- `mesa_shhs.post_processing.peak_min_consecutive_breaths = 2`
  (was 3). Lowers the run-length requirement for peak-based detection
  to match the existing sensitive profile. Recovers brief
  high-amplitude-drop sequences typical of dense MESA event clusters
  that the 3-breath rule lets slip through.

## Validation

- **PSG-IPA reproducibility:** clinical profiles unchanged → expected
  10/10 pass; verified post-commit.
- **MESA q=7 n=99:** with the v0.5.2 `mesa_shhs` profile,
  bias **+1.10**/h (was −1.55 on v0.5.1, still within Anderer 2022
  stretch target ≤3),
  MAE 6.06 (≈ unchanged),
  SD 8.43 (down from 8.97),
  Pearson $r$ **0.804** (up from 0.775, +0.029),
  weighted $\kappa$ **0.481** (up from 0.400, +0.081 — into the
  DRIVEN range 0.55-0.65 lower bound),
  severity-match **59%** (up from 53%, +6 pct).
- **Mesaid 6382 specifically:** algo AHI 4.7 vs v0.5.1's 4.3 — the
  noise on this single recording is too extreme even for the
  smoothing-plus-min-breaths-2 combination; cohort-level lift comes
  from other recordings benefiting from the smoother envelope.

## Caveat

The smoothing re-enablement is justified because clinical profiles
keep `flow_smoothing_s=0.0` and so cannot regress on PSG-IPA. Any
future deployment of `mesa_shhs` for non-MESA-style cohorts should
re-evaluate whether 3.0s is still appropriate for the target sensor
characteristics.

# v0.5.1 — 2026-05-03

Profile-tunable dynamic baseline parameters and opt-in RIPsum fallback
for the hypopnea channel. Both adaptations were motivated by the
research review of best-performing tools on MESA/SHHS (Vaquerizo-Villar
DRIVEN, Nassi WaveNet, Lazazzera per-sample digitised, Koley & Dey
robust airflow envelope tracking). Defaults preserve paper v31
PSG-IPA reproducibility (10/10 pass on
`tests/test_psgipa_reproducibility.py`).

## Added

- `PostProcessingRules.baseline_window_s` (default `300`).
  Sliding-window length (seconds) for `signal.compute_dynamic_baseline`.
  Previously hard-coded.
- `PostProcessingRules.baseline_percentile` (default `95.0`).
  Envelope percentile used as the local baseline anchor in
  `compute_dynamic_baseline`. Previously hard-coded as 95.
- `PostProcessingRules.flow_fallback_strategy` (default `"none"`).
  When set to `"ripsum_on_nasal_failure"`, the pipeline checks
  the Pres-signal amplitude excursion; if median 30-second
  peak-to-trough excursion is below 0.5% of the 99th-percentile
  amplitude, OR the night-wide excursion-CV is below 0.10
  (signal flat / sensor disconnected), the thoracoabdominal RIP
  sum (thorax + abdomen) is substituted as the hypopnea
  detection input. The quality test runs in
  `pipeline._maybe_apply_ripsum_fallback`.

## Changed (`mesa_shhs` profile only)

- `mesa_shhs.post_processing.baseline_window_s = 120`
  (Lazazzera 2020-style 2-minute window; clinical profiles keep 300).
- `mesa_shhs.post_processing.baseline_percentile = 85.0`
  (Lazazzera/Koley range 80–90; clinical profiles keep 95).
- `mesa_shhs.post_processing.flow_fallback_strategy = "ripsum_on_nasal_failure"`.

## Validation

- **PSG-IPA reproducibility (regression):** 10/10 tests pass in 11:58.
  Paper v31 strict/standard/sensitive numerics on SN1–SN5 are
  bit-identical because clinical-profile defaults are unchanged.
- **MESA q=7 n=99:** with the v0.5.1 `mesa_shhs` profile,
  bias $-1.55$/h (was $-0.78$ on v0.5.0), MAE 6.05 (≈unchanged),
  Pearson $r$ 0.775 (up from 0.759, +0.016), weighted $\kappa$ 0.40
  (unchanged), severity-match 53% (≈unchanged).
- **Per-recording diagnostic on `mesaid` 6382:** the RIPsum
  fallback did NOT trigger (Pres signal `rel_exc=0.352`,
  `cv=2.808`; both well above the trigger thresholds), confirming
  that the under-detection on this recording is not driven by
  nasal-pressure quality but by something deeper in the
  airflow-envelope or breath-segmentation logic. The MESA cohort
  outliers thus remain a v0.6 (real algorithmic rework) target;
  v0.5.1 nevertheless ships these knobs because they (a) are useful
  for cohorts where the Pres signal does degrade, and (b) the
  baseline window/percentile tuning produced a small but real
  $r$ improvement on the highest-quality MESA stratum.

## Known limitations carried forward

- Severe under-detection on dense-cluster OSA recordings
  (\S~S5.7–S5.8 of paper v34) remains. The flow-detection-sensitivity
  gap on `mesaid` 6382 (91% never-seen candidates) is not closed by
  v0.5.1's profile-tunable baseline; the gap appears upstream of the
  baseline computation in the per-breath segmentation or Hilbert
  envelope construction. Identified as the v0.6 priority.

# v0.5.0 — 2026-05-03

Profile-tunable local-baseline validator and Rule 1B arousal-coupling
window. Three previously-hardcoded thresholds in `respiratory.py` are
now exposed as fields on `PostProcessingRules`, enabling per-profile
cohort-specific tuning. The MESA/SHHS dataset profile (`mesa_shhs`)
gets a corrected metadata block plus the cohort-tuned values
established in paper v34 §S5.6. Clinical profiles (`aasm_v3_*`,
`aasm_v2_rec`, `aasm_v1_rec`, `cms_medicare`, `chicago_1999`) keep the
released defaults; PSG-IPA reproducibility 10/10 pass — paper v31
numerics bit-identical.

## Added (profile-tunable; defaults preserve paper v31 reproducibility)

- `PostProcessingRules.local_baseline_min_reduction_pct`
  (default `20.0`). Floor for the local-baseline-validator's
  reduction requirement, previously the hard-coded
  `min_reduction_pct=20.0` parameter default in
  `respiratory._validate_local_reduction`.
- `PostProcessingRules.local_baseline_pre_win_s`
  (default `30.0`). Pre-event window (seconds) for the local baseline
  computation, previously hard-coded as `pre_win_s=30.0`.
- `PostProcessingRules.rule1b_arousal_window_s`
  (default `15.0`). Window within which a scored arousal must follow
  a rejected hypopnea candidate for AASM Rule 1B reinstatement,
  previously the module-level constant `RULE1B_AROUSAL_WINDOW_S=15.0`.

These three fields are wired through `constants.py:_profile_to_legacy_dict`
into the legacy SCORING_PROFILES dict, then read in
`respiratory.detect_respiratory_events` and passed down to
`_detect_hypopneas` → `_validate_local_reduction`. The Rule 1B
window is plumbed via a new `arousal_window_s` parameter on
`reinstate_rule1b_hypopneas` (kw-only; falls back to module-level
constant if `None`), and `pipeline.py` forwards
`profile["RULE1B_AROUSAL_WINDOW_S"]` from the legacy dict.

## Changed (`mesa_shhs` profile only)

- `mesa_shhs` metadata block corrected: `sensor` is now
  `"nasal_pressure_primary"` (was `"rip_bands_primary"`, which
  contradicted the MESA Sleep PSG Scoring Manual);
  `desat_threshold=0.03` (was `None` which fell back to the legacy
  default), `desat_or_arousal=True` and `desat_required=False`,
  matching the canonical NSRR `nsrr_ahi_hp3u` clinical AHI definition
  (≥30% airflow reduction with ≥3% desat OR coupled arousal).
- `mesa_shhs.post_processing` cohort-tuning per paper v34 §S5.6:
  `local_baseline_min_reduction_pct=15.0`,
  `local_baseline_pre_win_s=60.0`,
  `rule1b_arousal_window_s=5.0`.

## Validation

- **PSG-IPA reproducibility:** 10/10 tests pass (`tests/test_psgipa_reproducibility.py`).
  Paper v31 strict/standard/sensitive numerics on SN1–SN5 are
  bit-identical to v0.3.2 and v0.4.5 because clinical-profile
  `PostProcessingRules` defaults are unchanged.
- **MESA q=7 n=99 (full highest-quality stratum):** with the
  natively-tuned `mesa_shhs` profile (no monkey-patches),
  bias **$-0.78$/h**, MAE 6.04/h, SD 9.20/h, Pearson $r$ 0.76,
  weighted $\kappa$ 0.40, severity-match 54%. Within Anderer 2022
  stretch target ($|\text{bias}|<3$/h). Bit-identical to the
  monkey-patched v0.4.5 recipe documented in paper v34 §S5.6,
  confirming clean source-level integration.

## Known limitations carried forward

- The flow-detection-sensitivity gap on severe MESA recordings
  (\S~S5.7–S5.8 of paper v34: `mesaid` 6139 with 40% never-seen
  candidates, `mesaid` 6382 with 91%) is upstream of these
  profile-tunable knobs and remains a v0.5.x/v0.6 algorithmic
  rework target.

# v0.4.5 — 2026-05-02

External-arousal injection for dataset-faithful validation. Motivated
by MESA cohort smoke runs where comparison to NSRR ``ahi_hp3u``
(3%-desat OR arousal) was systematically biased low (−9.4/h on n=30,
q≥5 cohort) because the algorithm's Rule 1B reinstatement was firing
against EEG-detected arousals (or, on quiet recordings, no arousals
at all) instead of the scorer-determined arousals embedded in the
NSRR XML. Default behaviour unchanged; PSG-IPA reproducibility 10/10
pass.

## Added

- ``run_pneumo_analysis(arousal_events=...)`` — optional list of
  pre-scored arousals to use for Rule 1B reinstatement, bypassing
  internal EEG-based detection. Each item must be a dict with at
  least ``onset_s`` and ``duration_s``. Use this for dataset-faithful
  validation against scorer-derived AHI variants that include
  arousal-coupled hypopneas. When ``None`` (default), the existing
  EEG-based path runs as before.

## Fixed

- ``pipeline.py`` — removed a duplicate local import of
  ``_compute_summary`` inside ``run_pneumo_analysis`` that shadowed
  the module-level binding and would have raised
  ``UnboundLocalError`` whenever Rule 1B reinstatement actually fired
  on a recording. The bug was latent because PSG-IPA's EEG-detected
  arousal stream rarely produces reinstatable hypopneas; on MESA with
  injected arousals the path lights up and the shadowing surfaced.

## Validation context (informational, not part of the release)

- MESA n=30, q≥5, seed=42, profile ``aasm_v2_rec`` + injected NSRR
  arousals: bias **−2.52/h** (within Anderer 2022 reasonable target
  ≤5), MAE 7.46, weighted κ 0.241, severity-match 51.7%. Same cohort
  with no arousals: bias −9.43, κ 0.045, severity-match 44.8%.
- A diagnostic on mesaid 6139 (severe AHI, ref 42.4/h → algo 10.9/h
  default): 406/411 hypopnea candidates rejected by
  ``_validate_local_reduction``. Even with the profile-tunable
  stability branch fully relaxed, only ~16 events recoverable due to
  a hardcoded ``min_reduction_pct=20.0`` floor that is **not**
  profile-aware. Making it profile-tunable is queued for v0.5; the
  larger gap (~198 events the algorithm never sees as candidates) is
  a flow-detection-sensitivity issue specific to MESA Compumedics
  data and out of scope for v0.4.x.

# v0.4.4 — 2026-05-01

Algorithm-review release. An internal v0.4.x review flagged 8
behavioural concerns and 3 AASM-mapping documentation gaps. After
PSG-IPA cross-validation, the changes that materially affected the
paper v31 numerics were demoted from default-changing fixes to
**documented opt-in parameters** so default behaviour is unchanged
and paper v31 reproduction passes. The genuinely-defensive fixes
(B1, B5, B8) and the documentation gaps are applied as defaults.

## Fixed (default behaviour change)

- **B1** `respiratory.py:_validate_local_reduction()` — when <3 s of
  pre-event signal is available, the validator was returning
  ``(True, 100.0)`` and downstream consumers treated this sentinel as
  a real measurement. Now returns ``(True, float('nan'))`` so the
  "not measured" case is unambiguous. Same for flat-line baselines.
- **B5** `ecg_effort.ecg_effort_assessment()` — added an
  ``evidence_strength`` field (``'dual'`` vs ``'spectral_only'``) so
  the upstream confidence penalty in `classify.py:Rule 5b` is
  inspectable. The penalty itself was already correctly applied
  (Rule 5b uses 0.75 vs 0.85 based on ``ecg_effort_present``).
- **B8** `plm._detect_lm_channel()` — the ``unit='auto'`` heuristic
  was 1000× wrong for mV-scaled EDF data. Replaced with a three-band
  heuristic (V → ×1e6, mV → ×1e3, µV → no scaling) and added an
  explicit ``leg_unit`` parameter so callers can pass the EDF
  physical unit rather than relying on amplitude inference. The 8 µV
  AASM amplitude threshold is unit-sensitive.

## Added (opt-in parameters; defaults preserve paper v31 numerics)

- **B7** `spo2.get_desaturation()` gained
  ``global_baseline_min_local_pct`` (default ``None`` = paper v31
  always-override behaviour). Set to a value (e.g. 88) to gate the
  global-baseline override so it only fires when the local baseline
  is implausibly low. Helps avoid artificially inflating the baseline
  for chronic-desaturator patients (COPD, OHS). The
  ``early_nadir_min_drop_pct`` default stays at 5.0 (paper v31);
  pass 3.0 to align with the AASM ≥3% criterion.

## Documented (no behaviour change)

- **B2** `_validate_local_reduction()` — full docstring rewrite with
  explicit AASM-mapping note, paper reference, and instructions for
  disabling the stability-aware tightening per profile (set
  ``stability_strict_reduction == min_reduction_pct``).
- **B3** `classify.py:Rule 6` — comment added documenting the
  deliberate AASM-deviation (effort 0.30–0.40 defaults to central,
  not obstructive) introduced in v0.8.30 to handle cardiac-pulsation
  artefact. Default unchanged; lower the 0.40 threshold to 0.30 in a
  fork to revert to AASM-strict behaviour.
- **B4** `signal_quality.py:FALLBACK_OBSTRUCTIVE_RATIO` — comment
  added flagging that single-channel-fallback may misclassify
  cardiac-pulsation-only events as obstructive at the 0.50
  threshold. A 0.70 threshold would be more conservative; default
  unchanged for backward compatibility.
- **B6** `ancillary.detect_cheyne_stokes()` autocorrelation peak
  threshold — docstring NOTE added pointing out that the literature
  uses tighter thresholds (Trinder *Sleep* 1991: >0.4; He et al.
  *EHJ* 2023: >0.5). Default kept at 0.3 for paper v31 compatibility;
  v0.5 will expose the threshold as a parameter.
- **G1** `classify.py:Rule 5b` — added comment noting that
  pattern-level CSR reclassification (the AASM v3 ≥3-consecutive-
  central + crescendo-decrescendo + ≥40 s rule) is detected by
  `ancillary.detect_cheyne_stokes()` and applied downstream by
  `postprocess.reclassify_csr_events()`, not in classify.py.
- **G2** `postprocess.decompose_mixed_apneas()` — documented the
  AASM-conform "leading low-effort = central phase" assumption.
- **G3** `postprocess.reclassify_csr_events()` — documented that the
  preserved ``original_type`` field provides v0.4.4-interim audit-
  trail rollback for false-positive CSR reclassifications. Full
  append-only audit log is on the v0.5 roadmap.

## Reproducibility

PSG-IPA standard-profile aggregate (n=5) under v0.4.4 reproduces
paper v31 metrics bit-identically (default parameters). The
``test_psgipa_reproducibility.py`` integration tests pass with
``PSGIPA_DATA_DIR`` set.

---

# v0.4.3 — 2026-05-01

## Added

- **`tests/test_psgipa_reproducibility.py`** — pytest that asserts
  paper v31 metrics on PSG-IPA (bias, MAE, Pearson r, F1 SN3, mean Δt)
  within tolerance. Skipped when `PSGIPA_DATA_DIR` is not set, so CI
  without the dataset still passes; gates against silent algorithmic
  drift on full runs.
- **Robustness-grade output** in `validate_psgipa.py` (per-recording
  A/B/C grade computed across the three clinical profiles).

## Changed

- **`validate_psgipa.py` fully rewritten.** The previous v3 script
  read scorer-1 stages from `Sleep_stages/Annotations/manual/` and
  events from `Resp_events/Annotations/manual/` and applied a
  `meas_date` shift to align them; that cross-subtree alignment
  introduced small epoch-attribution errors and produced
  bias +3.6/h on PSG-IPA instead of paper v31's +1.8/h.
  The new harness is faithful to paper v31 supplement S3.2: scorer-1
  stages and events are both read from
  `Resp_events/Annotations/manual/{SN}_Respiration_manual_scorer1.edf`,
  which shares its time axis with the primary Respiration EDF.
  Reproduces paper v31 standard-profile metrics bit-identically and
  emits a robustness grade per recording.
- The harness now runs all three clinical profiles
  (`aasm_v3_strict`, `aasm_v3_rec`, `aasm_v3_sensitive`) per recording
  and emits a JSON payload consumed by `validation_report.py`.

## Reproducibility

- PSG-IPA standard-profile aggregate (n=5):
  bias +1.77/h, MAE 1.84/h, SD 2.00/h, LoA [-2.15, +5.68]/h,
  Pearson r 0.997, weighted κ 0.84, F1 SN3 0.886, mean Δt SN3 1.97 s.
  Standard-profile per-recording AHIs reproduce paper v31 Table 1
  bit-identically; strict and sensitive per-recording AHIs differ
  from v31 because v0.4.0 retuned those profile parameters (see
  the v0.4.0 entry below).

---

# v0.4.2 — 2026-04-29

## Fixed
- **Local baseline validation now profile-aware.** The hardcoded
  `local_cv < 0.30` stability check in `_validate_local_reduction`
  is now driven by two new profile parameters
  (`local_baseline_cv_threshold` and `local_baseline_strict_reduction`).
- **Scope bug fix:** `sp` dict was incorrectly referenced inside
  `_detect_hypopneas` (which doesn't receive `sp`); the new
  parameters are threaded through the call chain via function
  arguments.

## Added
- `PostProcessingRules.local_baseline_cv_threshold` (default 0.30)
- `PostProcessingRules.local_baseline_strict_reduction` (default 25.0)
- Per-profile values:
    - `aasm_v3_strict`: cv=0.30, strict_reduction=30.0
    - `aasm_v3_rec`: cv=0.30, strict_reduction=25.0
    - `aasm_v3_sensitive`: cv=0.20, strict_reduction=20.0
    - 5 other profiles: defaults

## Changed
- `_detect_hypopneas` and `_validate_local_reduction` signatures
  extended (backward compatible — defaults match legacy behaviour)
- `_profile_to_legacy_dict` exports new `LOCAL_BL_CV_THRESHOLD` and
  `LOCAL_BL_STRICT_RED` keys

## Validation
- PSG-IPA aggregate metrics improved: r=0.994, kappa=0.800,
  F1 SN3=0.860, mean Δt=1.39s
- Severity concordance 4/5 (paper v31 claim retained)
- Profile-sweep monotonie not yet fully restored on borderline
  cases (SN2, SN4); to be addressed in future release after
  deeper review of flow_smoothing × peak_detection × local_baseline
  interaction

---

# v0.4.1 — 2026-04-27

## Fixed
- **Profile parameter integration bug** (cause of monotonie violations
  in v0.4.0). The hardcoded stability filter threshold `0.45` in
  `respiratory.py` ignored the per-profile `stability_filter_cv` and
  `peak_min_consecutive_breaths` parameters. Now correctly read from
  scoring profile dict, restoring intended profile differentiation.
  PSG-IPA validation re-run shows expected monotonic ordering
  (strict ≤ standard ≤ sensitive for hypopnea-dominated recordings).

## Added
- **3-point clinically calibrated confidence sweep** in scoring summary:
    - `oahi_sweep`: `{lenient: c≥0.30, primary: c≥0.47, strict: c≥0.65}`
    - `oahi_sweep_width`: max−min spread in /h
    - `robustness_grade`: 'A' (<5/h), 'B' (5-10/h), 'C' (≥10/h)
  Calibrated to AASM inter-scorer variability (~10-20% AHI).
  Mean sweep width on PSG-IPA: 3.9/h. Replaces use of legacy 4-point
  `oahi_thresholds` for clinical interpretation; the latter is
  preserved for backward compatibility.
- New TypedDict `OAHISweep3pt` for IDE/mypy support.

## Changed
- `oahi_thresholds` (legacy 4-point sweep) is no longer used in
  clinical UI displays but kept in output for compatibility.
- Profile parameters `STABILITY_FILTER_CV` and
  `PEAK_MIN_CONSECUTIVE_BREATHS` are now properly threaded through
  to the detection pipeline.

## Notes
- The monotonie-fix is a behavioural change: profile differentiation
  now produces meaningfully different OAHI values. Re-validation
  on PSG-IPA recommended for users comparing to v0.4.0 results.
- All 8 historical profiles (aasm_v3_rec/strict/sensitive,
  aasm_v2_rec, aasm_v1_rec, cms_medicare, mesa_shhs, chicago_1999)
  remain available.

---

# v0.4.0 — 2026-04-26

Major refactor: introduces the unified scoring-profile framework.
Eight named profiles ship in `psgscoring.PROFILES`, exposed via the
`scoring_profile=` kwarg of `run_pneumo_analysis()`.

## Added

- **Three clinical profiles** (`PROFILE_GROUPS["clinical"]`):
  - `aasm_v3_strict`   — conservative variant of AASM v3 Rule 1A
  - `aasm_v3_rec`      — recommended (3%-or-arousal hypopnea)
  - `aasm_v3_sensitive` — UARS-oriented sensitive variant
- **Five historical / dataset profiles**:
  `aasm_v2_rec`, `aasm_v1_rec`, `cms_medicare`, `mesa_shhs`, `chicago_1999`.
- Profile dataclasses: `HypopneaRules`, `ApneaRules`, `SpO2Rules`,
  `PostProcessingRules`, aggregated by `Profile`.
- `PROFILE_GROUPS`: convenience aliases (`clinical`, `aasm_era`,
  `coverage`, `dataset`, `full_6`, `all`).
- Legacy aliases `strict` / `standard` / `sensitive` accepted for
  backward compatibility (deprecation-warning, removal planned in v0.5.0).

## Changed — strict vs. v0.3.x defaults

The strict profile is a **deliberate tightening** relative to the
single hardcoded default of v0.3.x:

- `stability_filter_cv` 0.30 (no longer hardcoded 0.45)
- `breath_level_detection` off  ·  `flow_smoothing_s` 0
- `spo2.nadir_search_s` 30 s  ·  `local_baseline_strict_reduction` 30

## Changed — sensitive vs. v0.3.x defaults

The sensitive profile is a **deliberate loosening** for UARS detection:

- `hypopnea.flow_reduction_threshold` 0.25 (vs. 0.30 in rec/strict)
- `hypopnea.max_duration_s` 90 s (vs. 60 s in rec/strict)
- `flow_smoothing_s` 5.0  ·  `breath_level_detection` on
- `peak_min_consecutive_breaths` 2 (vs. 3 in rec/strict)
- `stability_filter_cv` 0.50  ·  `local_baseline_cv_threshold` 0.20
- `local_baseline_strict_reduction` 20.0
- `artefact_flank_exclusion` off (deliberate; reduces false negatives
  on flow recovery slopes)
- `apnea.max_duration_s` 120 s (vs. 90 s in rec/strict)

## Reproducibility note vs. paper v31

Paper v31 (Rombaut et al. 2026) was generated against
**psgscoring v0.3.2**, where strict / standard / sensitive were
configurable presets sharing a single hardcoded stability-filter
threshold. The v0.4.0 profile system makes that threshold (and
several other rules) profile-specific, which causes the strict and
sensitive per-recording AHIs on PSG-IPA SN1-SN5 to diverge from the
v31 Table 1 values. The standard (`aasm_v3_rec`) profile remains
parameter-equivalent for the rules active on this dataset and
reproduces v31's standard-profile AHIs and aggregate metrics
(bias +1.8/h, MAE 1.8/h, r 0.997, F1 SN3 0.886) bit-identically.
Users wishing to reproduce v31 Table 1 verbatim should pin to v0.3.2:
`pip install psgscoring==0.3.2`.

## Known issue (fixed in v0.4.1)

The profile parameters introduced in v0.4.0 were not actually read
by `respiratory.py`, which kept a hardcoded 0.45 stability-filter
threshold. PSG-IPA monotonie was therefore not yet established in
v0.4.0; the v0.4.1 release wires the parameters through.

---

# v0.3.2 — 2026-04-21

Bugfix release.

## Fixed
- `signal_quality_channels._check_montage()`: numpy boolean-ambiguity error
  caused by `flow = _get("flow") or _get("flow_pressure")` when both channels
  returned ndarrays. Replaced with explicit `None`-check. This bug was
  silently caught by the pipeline exception handler, leaving
  `output["channel_quality"]` populated with `{"overall_grade": "unknown"}`,
  and was a no-op for classification (which reads the separate
  `output["signal_quality"]` from `compare_rip_pair`), but prevented
  channel-level quality metadata from reaching PDF reports and validation
  exports. Detected during PSG-IPA re-validation on 2026-04-21.

# Changelog

## v0.3.1 — 2026-04-21

### Added

- `classify_apnea_type()` accepts optional `signal_quality` parameter
  (output of `compare_rip_pair`). When gate reports
  `recommended_mode="single-channel"`, classification routes to
  `single_channel_fallback_classify()` on the working channel instead
  of the bilateral 7-rule chain. When `"unreliable"`, returns
  `"uncertain"`.

### Fixed

- **Bug 2: respiratory classifier now consumes the RIP-pair quality
  gate.** The v0.2.962 gate detected single-sensor failures correctly,
  but `classify_apnea_type` ignored the signal, defaulting to
  obstructive classification for dead-sensor events. This caused the
  Loos case (AZORG April 2026) to appear as 100% obstructive while the
  true underlying pathology was likely CSAS (CAI 45.1 on
  abdomen-only analysis).

### Changed

- Classifier decision chain has new Rule -1 (RIP pair quality gate)
  before Rule 0 (phase angle). With `signal_quality=None` or
  `recommended_mode="bilateral"`, behavior is unchanged from v0.2.963.
- Pipeline reorders signal_quality computation to run BEFORE respiratory
  event detection (previously was after, as a documentation-only step).

### Clinical impact

Patients with single RIP sensor failures now receive classification
based on the working channel instead of a mechanical obstructive
default. The clinically important shift is for central sleep apnea
syndrome (CSAS) detection in patients whose thorax or abdomen RIP
sensor fails during the recording.

### Test coverage

- `tests/test_bug2_classifier_quality_gate.py`: 9 passing tests
  covering fallback in isolation, end-to-end classifier routing,
  bilateral preservation, and unreliable-gate scenarios.

---



## v0.2.963 — 2026-04-20

### Fixed

- **assess_rip_channel 2D input regression** — MNE's `raw.get_data(picks=[ch])`
  returns shape `(1, N)` even for single-channel requests. The welch()
  PSD computation on a 2D input produced a 2D output, which then broke
  1D boolean masking in the breath-band energy calculation. The fix
  squeezes the input to 1D at the top of `assess_rip_channel()` and
  returns a defensive 'failed' status for genuinely higher-dimensional
  input.

### Clinical impact

  Without this fix, signal quality assessment silently failed in the
  real deployment pipeline, leaving the RIP pair quality gate
  ineffective at detecting single-sensor failures. The Loos case
  (AZORG April 2026, thorax RIP dead, ratio 6862×) was the motivating
  clinical scenario.

### Added

- `tests/test_signal_quality_2d.py` — 5 regression tests covering
  1D baseline, 2D MNE-shape input, higher-dimensional defensive
  rejection, and Loos-like single-sensor failure scenarios.

---



All notable changes to **psgscoring** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.951] 

### Added

- **Ensemble-averaged hypoxic burden** (`baseline_method="ensemble"`).
  Subject-specific search window derived from the ensemble average of all
  time-aligned SpO₂ curves, reproducing the original Azarbarzin et al.
  (Eur Heart J 2019) method.
- Pre-event baseline = SpO₂ at left peak of ensemble curve; area
  integrated within the ensemble-derived search window.
- Automatic fallback to percentile method when fewer than 3 events available.
- Helper function `_ensemble_search_window()` for ensemble curve computation.

### Changed

- `compute_hypoxic_burden()` now accepts a `baseline_method` parameter:
  `"percentile"` (default, backward compatible) or `"ensemble"`.
- Return dict now includes `baseline_method` and `ensemble_window_s` keys.
- `spo2.py`: 395 → 545 lines (+150).

### References

- Azarbarzin A et al. The hypoxic burden of sleep apnoea predicts
  cardiovascular disease-related mortality. *Eur Heart J*.
  2019;40(14):1149-1157.
- He S, Cistulli PA, de Chazal P. Comparison of oximetry event
  desaturation transient area-based methods. *IEEE EMBC*. 2024.

---

## [0.2.95]

### Changed

- Maintenance release consolidating v0.2.93 and v0.2.94 fixes prior to
  the ensemble-HB addition in v0.2.951.
- Internal cleanup of post-processing return dictionary keys for
  consistency across CSR, mixed decomposition, and CII outputs.

---

## [0.2.94] 

### Fixed

- Stale documentation references to v0.2.92 updated to v0.2.93 in
  `DISCLAIMER.md` and module-level docstrings.
- Minor correction to CSR cycle-matching tolerance (±2 s) in
  `postprocess.reclassify_csr_events()` to avoid over-reclassification
  on recordings with irregular CSR periodicity.

---

## [0.2.93]

### Added

- **iSLEEPS validation** (Maiti et al., *Sci Data* 2026, 39 ischemic
  stroke patients). Mean absolute error 3.3 /h for normal/mild severity;
  systematic under-scoring at moderate/severe consistent with the high
  prevalence of central and mixed apneas in stroke populations.
- **Event-level temporal validation** on PSG-IPA severe-OSA recording
  (SN3, 322 reference events): F1 = 0.890, IoU = 0.866, mean onset
  difference Δt = 2.3 s (IoU ≥ 0.20 matching).

### Changed

- CSR reclassification threshold tuned from v0.2.92 pilot runs on
  iSLEEPS (365 events reclassified across 36/96 studies).

---

## [0.2.92] 

### Added

- **Hypoxic burden** (`spo2.compute_hypoxic_burden`).
  Total area of SpO₂ desaturation associated with respiratory events,
  normalised per hour of sleep (%·min/h), following Azarbarzin et al.
  (AJRCCM 2019).
  - Per-event integration from onset to SpO₂ recovery.
  - Pre-event baseline (90th percentile, 120 s window) with global
    95th-percentile fallback.
  - Clinical thresholds: <20 low, 20–73 moderate, >73 high CV risk.
  - Automatically computed in `run_pneumo_analysis()` (Step 10).
  - Available at `output["hypoxic_burden"]` and
    `output["spo2"]["summary"]["hypoxic_burden"]`.
  - NumPy ≥2.0 compatible (`np.trapezoid` with `np.trapz` fallback).
- **Post-processing module** (`postprocess.py` — new).
  - `reclassify_csr_events()`: CSR-flagged obstructive/mixed events
    reclassified as central (addresses cardiac pulsation artefact in
    heart failure).
  - `decompose_mixed_apneas()`: analyses effort signal to measure
    central vs. obstructive portion; reclassifies to central if the
    central portion is ≥10 s.
  - `compute_central_instability_index()`: quantifies
    profile-dependent uncertainty in obstructive/central
    classification on a 0–1 scale.
  - `postprocess_respiratory_events()`: master function calling all
    three.
  - Automatically runs in `run_pneumo_analysis()` (Step 11).
  - Results at `output["postprocess"]`.

### Changed

- Pipeline now has 11 steps (was 9): Step 10 added hypoxic burden
  computation; Step 11 added post-processing.
- Public API: 42 exports (was 38). New: `compute_hypoxic_burden`,
  `postprocess_respiratory_events`, `reclassify_csr_events`,
  `decompose_mixed_apneas`, `compute_central_instability_index`.

### References

- Azarbarzin A et al. The hypoxic burden of sleep apnoea is an
  independent predictor of incident cardiovascular outcomes.
  *AJRCCM*. 2019;200(2):211-219.

---

## [0.2.91] 

### Added

- **External validation on PSG-IPA** (PhysioNet, Bakker et al.
  *Physiol Meas* 2021): 5 recordings, 59 scorer sessions from up to
  12 certified RPSGT/ESRS technologists.
  Mean AHI bias +1.6 /h, mean |ΔAHI| = 1.8 /h, Pearson r = 0.990,
  AASM severity concordance 4/5 (80 %).
- **Stability-aware threshold** (Fix 6 refinement): when local breath
  coefficient of variation is <0.30, the 70 % hypopnea threshold is
  tightened to 30 % reduction to avoid over-counting in stable
  breathing.
- **Consecutive breath requirement** (Fix 7 refinement): peak-based
  hypopnea detection now requires ≥3 consecutive sub-threshold
  breaths before flagging, reducing sensitivity to single aberrant
  breaths.

### Changed

- Documentation: added "AHI confidence interval" as a named feature
  in README and PyPI project description.
- DUA-ready project description updated for MESA/SHHS data-access
  applications (primary dataset MESA due to dual-sensor support;
  SHHS secondary for thermistor-only validation).

---

## [0.2.9]

### Added

- **Dual-sensor flow detection** per AASM.
  Apneas are now scored on the oronasal thermistor signal; hypopneas
  on the nasal pressure transducer, following AASM recommendations.
  - Channel auto-detection via transducer metadata and channel-name
    patterns.
  - Intelligent fallback: when only one flow channel is available, it
    is used for both event types (backward compatible).
  - Result metadata (`meta.flow_channels`) logs which sensor was used
    for which event type.

### Fixed

- Channel-name matching is now order-independent (earlier versions
  picked the first match, which could cause thermistor
  misclassification on devices that list both channels generically).

---

## [0.2.8] 

### Added

- **AHI confidence interval** with robustness grading.
  Every analysis now runs three scoring profiles simultaneously and
  reports the AHI as an interval rather than a point estimate.
  - **Profiles**: `strict`, `standard` (default), `sensitive`.
  - **Robustness grade**: `A` (all three profiles agree on severity
    — treatment decision unambiguous), `B` (two of three concordant
    — probable, clinical correlation recommended), `C` (all discordant
    — manual review recommended).
  - Output at `results["ahi_interval"]` with fields `strict`,
    `standard`, `sensitive`, and `robustness_grade`.
- **Breath-amplitude stability filter** (Fix 6).
  For each hypopnea candidate, the coefficient of variation of breath
  amplitudes in the surrounding four minutes is computed. Candidates
  with CV < 0.45 (stable, non-pathological breathing) are rejected
  as normal variability rather than true events.
  - Ablation on PSG-IPA SN4 (normal OSA): 56 false-positive hypopneas
    rejected, correcting Mild → Normal.
  - Ablation on PSG-IPA SN3 (severe OSA): 11 events rejected (−3 %),
    confirming the filter targets false positives rather than true
    pathology.

### Removed

- **3-second flow smoothing** removed from the standard profile.
  Ablation analysis on PSG-IPA SN1 identified 3-second smoothing as
  the dominant source of over-counting, bridging recovery breaths into
  continuous "reduced flow" segments and generating +54 false
  hypopneas on a single mild-OSA recording. The smoothing shifted
  severity classification from Mild to Moderate.
  - SN1 standard-profile AHI: 15.9 → 8.1 (−49 %).
  - SN2 standard-profile AHI: 21.9 → 9.3 (−57 %).
  - SN4 standard-profile AHI: 13.6 → 4.3 (−68 %).
  - SN3 (severe) standard-profile AHI: 55.6 → 53.8 (−3 %,
    true events preserved).
- `HYPOPNEA_SMOOTH_S` constant deprecated (now 0.0 for standard).

### Changed

- Scoring profiles module reorganised into a central
  `constants.SCORING_PROFILES` dictionary.
- `run_pneumo_analysis()` now accepts a `scoring_profile` parameter
  (default `"standard"`).
- Internal event-detection pipeline calls all three profiles in
  sequence and combines results into the interval/grade structure.

---

## [0.2.7] 

### Changed

- Stability release of the TECG and spectral-effort modules
  introduced in v0.2.4. Reference version cited in manuscript v25.
- Minor performance improvement in R-peak detection for noisy ECG:
  fallback to Pan-Tompkins when WFDB method fails.

### Fixed

- `ecg_effort.ecg_effort_assessment()` previously returned `None` when
  the ECG channel was short (<30 s); now returns a dict with
  `assessment="insufficient_data"` and no reclassification is applied.

---

## [0.2.6] 

### Fixed

- Edge case in `classify.classify_apnea_type()` where ECG-based Rule
  5b reclassification could conflict with Rule 0 (Hilbert
  phase-angle), producing inconsistent labels. Rule precedence now
  documented: 0 → 5 → 5b, with 5b only applied when Rule 5 produced
  an "uncertain" result.
- Minor numerical stability fix in Hilbert phase-angle computation
  for recordings with intermittent RIP channel dropouts.

---

## [0.2.4] 

### Added

- **ECG-derived effort classification** (new module
  `psgscoring/ecg_effort.py`).
  - **Transformed ECG (TECG)** method (Berry et al., *JCSM* 2019):
    QRS blanking + 30 Hz high-pass filtering to reveal inspiratory
    EMG bursts from intercostal muscles.
  - **Spectral effort classifier**: cardiac (0.8–2.5 Hz) vs.
    respiratory (0.1–0.5 Hz) power analysis on RIP bands during
    apnea events; flags cardiac dominance.
  - **Combined reclassification logic**: events reclassified as
    central when *both* TECG (no inspiratory bursts) *and* spectral
    analysis (cardiac spectral dominance) agree.
  - New output field `n_ecg_reclassified_central` in respiratory
    results.
- Public API: `ecg_effort_assessment`, `compute_tecg`,
  `detect_r_peaks`, `qrs_blanking`, `detect_inspiratory_bursts`,
  `spectral_effort_classifier`.

### Changed

- `pipeline.py`: ECG channel now extracted and passed to respiratory
  scoring when available; graceful degradation when absent.
- `respiratory.py`: TECG computed once per recording; the ECG
  assessment is passed to both apnea and hypopnea
  `classify_apnea_type()` calls.
- `classify.py`: ECG-based reclassification integrated as Rule 5b in
  the 7-rule decision tree.

### References

- Berry RB et al. Use of a transformed ECG signal to detect
  respiratory effort during apnea. *JCSM*. 2019;15(11):1653-1660.

---

## [0.2.3] 

### Notes

- Content-identical re-release of v0.2.2 for PyPI. The v0.2.2 package
  name was temporarily unavailable on PyPI at the time of publication;
  v0.2.3 was published with the same source to guarantee PyPI
  availability and avoid long-term ambiguity.

### Added

- `signal_quality.py` module: per-channel flat-line, clipping,
  disconnect, line-noise, and montage-plausibility checks.
- **Flattening-based RERA** detection (Hosselet et al. *AJRCCM*
  1998): sequences of ≥3 consecutive breaths with flattening index
  >0.30 spanning ≥10 s, terminated by an arousal.
- Dual-source RDI computation: `RDI = AHI + (FRI-RERA + flattening-RERA) / TST`.
- Hypopnea subtype counts in summary output: `n_hypopnea_obstr`,
  `n_hypopnea_central`, `n_hypopnea_mixed`.
- `assess_signal_quality()` added to public exports.

### Changed

- Pipeline now 11 steps (was 10): Step 1b added for signal-quality
  assessment, executed before event detection.

### References

- Hosselet J et al. Detection of flow limitation with a nasal
  cannula/pressure transducer system. *AJRCCM*. 1998;157(5):1461-1467.

---

## [0.2.2] 

### Added

- Initial implementation of signal-quality assessment and flattening
  RERA (see v0.2.3 notes for the content description; this version
  was tagged on GitHub but its PyPI publication was delayed to
  v0.2.3).

---

## [0.2.0] 

### Added

Initial public release of `psgscoring` as a standalone library,
extracted from the YASAFlaskified clinical platform.

**Core algorithms** (pure scipy / NumPy, no deep learning, no GPU):

- Square-root linearisation of nasal pressure (Bernoulli correction,
  Thurnheer et al. *AJRCCM* 2001).
- Dynamic 5-minute rolling baseline (95th percentile) with
  stage-specific blending.
- Hilbert envelope for instantaneous amplitude.
- MMSD artefact validation (Lee et al. 2008, κ = 0.78).
- Dual-sensor detection with exclusion masking.
- Temporally constrained SpO₂ coupling (Uddin et al. 2021).
- Two-pass Rule 1B hypopnea detection with breath-cycle validation.
- 7-rule apnea type classification (obstructive / central / mixed)
  with Hilbert phase-angle analysis.
- Two-phase arousal detection with spindle exclusion and
  cardiovascular reactivity (CVR) coupling.
- PLM scoring per AASM and WASM criteria (Zucconi 2006).
- Cheyne-Stokes detection via autocorrelation.

**Package**:

- 21 unit tests, example script, BSD-3 licence.
- GitHub Actions CI (Python 3.9 – 3.12).

### References

- Thurnheer R, Xie X, Bloch KE. Accuracy of nasal cannula pressure
  recordings. *AJRCCM*. 2001;164(10):1914-1919.
- Lee H et al. Detection of apneic events from single-channel nasal
  airflow. *Physiol Meas*. 2008;29:N37-N45.
- Uddin A et al. Automated detection of respiratory events during
  sleep from pulse oximetry and airflow signals. *Sleep Breath*.
  2021;25(1):127-138.
- Vallat R, Walker MP. An open-source, high-performance tool for
  automated sleep staging. *eLife*. 2021;10:e70092.

---

## Versions not on PyPI

The following version numbers appear in internal development history
but were never published as separate PyPI artefacts:

- **0.1.x** — pre-release development, internal only.
- **0.2.1** — skipped; improvements rolled into v0.2.2.
- **0.2.5** — skipped; improvements rolled into v0.2.6.
