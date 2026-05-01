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

- **Dual-sensor flow detection** per AASM 2.6.
  Apneas are now scored on the oronasal thermistor signal; hypopneas
  on the nasal pressure transducer, following AASM 2.6 recommendations.
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
- PLM scoring per AASM 2.6 and WASM criteria (Zucconi 2006).
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
