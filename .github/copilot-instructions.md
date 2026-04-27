# psgscoring AI Coding Guidelines

## Project Overview
`psgscoring` is a Python library for automated AASM 2.6-compliant polysomnography (PSG) respiratory event scoring. It processes sleep study data to detect apneas, hypopneas, and RERAs, providing Apnea-Hypopnea Index (AHI) with confidence intervals and clinical auditability.

**Key Features:**
- Extends YASA (sleep staging) into respiratory scoring
- 12 systematic bias corrections for over/under-counting
- Three scoring profiles: strict/standard/sensitive
- Confidence-scored events with classification details
- MNE-Python integration for EDF processing

## Architecture
**10 submodules** organized by signal processing pipeline:
- `pipeline` - MNE-facing master function (`run_pneumo_analysis`)
- `respiratory` - Event detection orchestration
- `signal` - Preprocessing (bandpass, baselines, MMSD)
- `breath` - Breath segmentation and amplitude analysis
- `classify` - Apnea type classification (obstructive/central/mixed)
- `spo2` - Oxygen saturation coupling and hypoxic burden
- `ecg_effort` - ECG-derived respiratory effort
- `ancillary` - Position, heart rate, snore, Cheyne-Stokes
- `plm` - Periodic limb movement detection
- `postprocess` - CSR reclassification and corrections
- `utils` - Hypnogram helpers, channel detection, sleep masks
- `constants` - AASM thresholds, scoring profiles, channel patterns

**Data Flow:** EDF → MNE Raw → channel auto-detection → signal preprocessing → breath detection → event candidates → classification + bias corrections → summary statistics

## Key Patterns & Conventions

### Signal Processing
- **NumPy-first:** All processing uses numpy arrays; MNE only at pipeline entry
- **Sample-rate aware:** Functions take `sf` (Hz) parameter; convert seconds to samples as `int(seconds * sf)`
- **Baseline computation:** Dynamic 5-minute windows with position-change resets
- **Safe operations:** Use `safe_r(value, decimals)` for rounding that handles None/NaN

### Sleep Stage Handling
- **Hypnogram format:** List of strings `['W', 'N1', 'N2', 'N3', 'R', ...]`
- **Numeric conversion:** `hypno_to_numeric()` → `[0, 1, 2, 3, 4, ...]` (W=0, R=4)
- **Sleep masks:** `build_sleep_mask()` excludes wake and artifacts at sample level

### Event Structures
- **TypedDict outputs:** Use `RespiratoryEvent`, `ScoringSummary` for type safety
- **Confidence scoring:** Every event has 0.0-1.0 confidence + `classify_detail` dict
- **Bias corrections:** 12 fixes tracked per event (e.g., `spo2_cross_contaminated`, `csr_flagged`)

### Channel Detection
- **Pattern matching:** `CHANNEL_PATTERNS` dict maps signal types to name patterns
- **Auto-mapping:** `channel_map_from_user()` handles manual overrides
- **Multilingual support:** Patterns include EN/NL/DE/FR channel names

## Development Workflow

### Testing
- **Smoke tests only in CI:** `pytest` runs basic imports; algorithmic validation separate
- **Validation scripts:** `scripts/validate_ptt_sn3.py` for external datasets
- **CI matrix:** Python 3.9-3.12 on Ubuntu with MNE system deps

### Dependencies
- **Core:** `numpy>=1.21`, `scipy>=1.7`, `mne>=1.5`
- **Optional:** `yasa>=0.6`, `lightgbm>=3.0` for full features
- **No GPU required:** Pure CPU signal processing

### Code Style
- **Type hints:** Extensive use of `TypedDict`, union types, `Optional`
- **Logging:** `logger = logging.getLogger("psgscoring.module")`
- **Dutch comments:** Some internal comments in Dutch (heritage codebase)
- **Docstrings:** NumPy format with Parameters/Returns sections

## Common Tasks

### Adding New Signal Processing
1. Add preprocessing in `signal.py` (e.g., `preprocess_new_signal()`)
2. Integrate into `respiratory.detect_respiratory_events()` call graph
3. Update `pipeline.py` to pass new signal to detection function
4. Add channel patterns in `constants.py` if needed

### Modifying Scoring Thresholds
- Update `SCORING_PROFILES` in `constants.py`
- Profiles: `strict` (research), `standard` (AASM), `sensitive` (screening)
- Test impact on validation datasets

### Adding Bias Corrections
- Implement correction logic in `respiratory.py` (e.g., `_fix_new_bias()`)
- Track correction in event dict (e.g., `new_bias_flagged: bool`)
- Update summary statistics if correction affects counts

## Key Files to Reference
- `psgscoring/pipeline.py` - Main entry point and orchestration
- `psgscoring/constants.py` - All thresholds and profiles
- `psgscoring/_types.py` - Output data structures
- `psgscoring/respiratory.py` - Core event detection logic
- `README.md` - Usage examples and validation results