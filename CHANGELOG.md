## v0.2.94 (April 2026)

### New features

**Ensemble-averaged hypoxic burden** (`baseline_method="ensemble"`)
- Subject-specific search window derived from ensemble average of all
  time-aligned SpO₂ curves (Azarbarzin et al., Eur Heart J 2019 original)
- Pre-event baseline = SpO₂ at left peak of ensemble curve
- Area integrated within the ensemble-derived search window
- Automatic fallback to percentile method when <3 events available
- Helper function `_ensemble_search_window()` for ensemble curve computation

### Changed

- `compute_hypoxic_burden()` now accepts `baseline_method` parameter:
  `"percentile"` (default, backward compatible) or `"ensemble"`
- Return dict now includes `baseline_method` and `ensemble_window_s` keys
- spo2.py: 395 → 545 lines (+150)

### References

- Azarbarzin A, et al. The hypoxic burden of sleep apnoea predicts
  cardiovascular disease-related mortality. Eur Heart J. 2019;40(14):
  1149-1157.
- He S, Cistulli PA, de Chazal P. Comparison of oximetry event
  desaturation transient area-based methods. IEEE EMBC. 2024.

---

## v0.2.92 (April 2026)

### New features

**Hypoxic burden** (`spo2.compute_hypoxic_burden`)
- Computes total area of SpO₂ desaturation associated with respiratory
  events, normalised per hour of sleep (%·min/h)
- Follows Azarbarzin et al. (AJRCCM 2019) methodology
- Per-event integration from onset to SpO₂ recovery
- Pre-event baseline (90th pct, 120s window) with global 95th pct fallback
- Clinical thresholds: <20 low, 20-73 moderate, >73 high CV risk
- Automatically computed in `run_pneumo_analysis()` (Step 10)
- Available at `output["hypoxic_burden"]` and `output["spo2"]["summary"]["hypoxic_burden"]`
- numpy ≥2.0 compatible (`np.trapezoid` with `np.trapz` fallback)

**Post-processing module** (`postprocess.py` — NEW)
- `reclassify_csr_events()`: CSR-flagged obstructive/mixed events →
  reclassified as central (addresses cardiac pulsation artifact in HF)
- `decompose_mixed_apneas()`: analyses effort signal to measure central
  vs obstructive portion; reclassifies to central if central ≥10s
- `compute_central_instability_index()`: quantifies profile-dependent
  uncertainty in O/C classification (0-1 scale with interpretation)
- `postprocess_respiratory_events()`: master function calling all three
- Automatically runs in `run_pneumo_analysis()` (Step 11)
- Results at `output["postprocess"]`

### Pipeline changes

- Step 10 added: Hypoxic burden computation
- Step 11 added: Post-processing (CSR reclassification + mixed decomposition)
- Pipeline now 11 steps (was 9)

### New exports (42 total, was 38)

- `compute_hypoxic_burden`
- `postprocess_respiratory_events`
- `reclassify_csr_events`
- `decompose_mixed_apneas`
- `compute_central_instability_index`

### References

- Azarbarzin A, et al. The hypoxic burden of sleep apnoea is an
  independent predictor of incident cardiovascular outcomes.
  AJRCCM. 2019;200(2):211-219.
