# Changelog

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
