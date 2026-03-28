# Changelog — psgscoring

All notable changes documented per [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0]

### Added — v0.8.11 features

**Phase-angle effort classification (`classify.py`):**
- `_compute_phase_angle()`: Hilbert instantaneous phase difference thorax/abdomen
- New Rule 0: phase angle ≥45° → obstructive (confidence 0.75–0.97)
- Fires before all legacy rules; largely eliminates Rule-6 borderline defaults
- Minimum 5 s event required for reliable Hilbert estimate
- Field `phase_angle_deg` in event detail dict

**Baseline anchoring (`signal.py`):**
- `compute_anchor_baseline()`: event-free N2 median RMS as patient-specific reference
- `mouth_breathing_suspected: True` when signal RMS <60% of anchor
- Result exposed in pipeline output as `output["anchor_baseline"]`
- Requires ≥6 stable N2 epochs; `anchor_reliable: False` otherwise

**LightGBM confidence calibration (`classify.py`):**
- `_lgbm_confidence()`: optional 10-feature model via `PSGSCORING_LGBM_MODEL` env var
- Features: effort_ratio, raw_var_ratio, paradox_correlation, half/quarter efforts,
  phase_angle_deg, duration_s, rule_index
- Transparent fallback to rule-based confidence when model unavailable
- Field `lgbm_confidence` per event when model active

### Changed
- `classify_apnea_type()` returns `phase_angle_deg` in detail dict
- `compute_stage_baseline()` accepts optional `dynamic_baseline` parameter
  to avoid duplicate `compute_dynamic_baseline()` call (+5 s saved)
- Pipeline Step 1b: `compute_anchor_baseline()` called after respiratory scoring

---

## [0.1.0]

### Added — v0.8.10 features

**Five systematic over-counting corrections (`respiratory.py`):**

Fix 1 — Post-apnoea hyperpnoea baseline exclusion:
- `_build_postapnea_recovery_mask()`: 30-s recovery window after each apnoea
- `_recompute_baseline_with_recovery_excluded()`: sparse cumsum loop
  (only anchors where recovery mask >5% of 5-min window are recomputed)

Fix 2 — SpO₂ cross-contamination:
- `_spo2_cross_contaminated()`: suppresses SpO₂ coupling if preceding event's
  30-s window is still active at candidate onset
- Field `spo2_cross_contaminated` per event

Fix 3 — Cheyne-Stokes AHI inflation:
- `_flag_csr_events()`: retroactive CSR flagging via IEI matching (±12 s)
- Fields: `csr_flagged` per event, `n_csr_flagged`, `ahi_csr_corrected` in summary
- Applied in `pipeline.py` after CSR detection (Step 9)

Fix 4 — Borderline default confidence stratification:
- `n_low_conf_borderline` (confidence 0.40–0.59)
- `n_low_conf_noise` (confidence <0.40)
- `ahi_excl_noise` (AHI excluding confidence <0.40 events)
- `oahi_thresholds`: OAHI at ≥0.85 / ≥0.60 / ≥0.40 / 0.00

Fix 5 — Artefact-flank exclusion:
- `_detect_signal_gaps()`: flatline/frozen ≥10 s → 15-s post-gap exclusion mask
- Field `n_gap_excluded` in detection result

**Performance optimisations:**
- O(n×k) `np.where(labeled == i)` loops replaced by O(n) `scipy.ndimage.find_objects()`
  (benchmark: 820 s extrapolated → 0.8 s for 350,000 candidate regions)
- `_setup_hypop_channel()`: reuses apnoea-channel baseline when sf equal
- `compute_stage_baseline()`: vectorised via `np.repeat()` instead of Python loop
- `_pre_event_baseline()`: O(1) lookup into precomputed baseline array

---

## [0.0.1]

### Added — Initial release (extracted from YASAFlaskified v0.8.5)

Monolithic `pneumo_analysis.py` (2,439 lines) refactored into 10 domain-specific submodules:

| Module | Responsibility |
|--------|----------------|
| `constants.py` | AASM thresholds, band limits |
| `utils.py` | Sleep masks, helper functions |
| `signal.py` | Linearisation, baseline, MMSD |
| `breath.py` | Breath-by-breath analysis, flattening index |
| `classify.py` | Apnoea type classification (6-rule decision tree) |
| `spo2.py` | SpO₂ coupling, ODI, T90 |
| `plm.py` | PLM detection (AASM 2.6 + WASM) |
| `ancillary.py` | HR, snoring, position, Cheyne-Stokes |
| `respiratory.py` | Apnoea/hypopnoea pipeline orchestration |
| `pipeline.py` | MNE-facing master function |

- 112 unit tests across 6 test files (Python 3.9–3.12 CI matrix)
- Backward-compatible 81-line `pneumo_analysis.py` shim
- Public API: 33 exported symbols in `__init__.py`
- Strict one-directional dependency graph (no circular imports)
