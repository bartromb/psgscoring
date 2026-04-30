# psgscoring

**Open-source AASM 2.6–compliant respiratory scoring for polysomnography.**

[![PyPI](https://img.shields.io/pypi/v/psgscoring)](https://pypi.org/project/psgscoring/)
[![Python](https://img.shields.io/pypi/pyversions/psgscoring)](https://pypi.org/project/psgscoring/)
[![License](https://img.shields.io/pypi/l/psgscoring)](https://github.com/bartromb/psgscoring/blob/main/LICENSE)
[![Tests](https://github.com/bartromb/psgscoring/actions/workflows/tests.yml/badge.svg)](https://github.com/bartromb/psgscoring/actions)

## Paper

> Rombaut B, Rombaut B, Rombaut C, et al. **Automated Polysomnography Scoring for Clinical Sleep Medicine: An Open-Source Platform Validated Against 59 Independent Scorer Sessions on PSG-IPA.** Manuscript in preparation, 2026.

Technical details (signal processing chain, classification logic, all twelve bias corrections): **[Technical Reference (Online Supplement)](https://github.com/bartromb/psgscoring/wiki/Technical-Reference)**

## What this library does

`psgscoring` detects and classifies respiratory events (apneas, hypopneas, RERAs) in polysomnography recordings following AASM 2.6 rules. It extends [YASA](https://github.com/raphaelvallat/yasa) (Vallat & Walker, *eLife* 2021) from sleep staging into a complete clinical respiratory scoring pipeline.

**Three contributions that distinguish this library:**

1. **Twelve bias corrections** — the first systematic identification and empirical quantification of six over-counting and six under-counting mechanisms in automated respiratory scoring
2. **AHI confidence interval** — every study is scored at three stringency levels (strict/standard/sensitive), yielding a per-study robustness grade (A/B/C) rather than a single AHI number
3. **Clinical auditability** — every event carries a confidence score, classification rule index, and per-correction counters, enabling the reviewing physician to verify individual scoring decisions

## Installation

```bash
pip install psgscoring
```

Requirements: Python ≥3.9, numpy, scipy, mne. **No GPU required.**

## Quick Start

```python
import mne
from psgscoring import run_pneumo_analysis

# Load EDF and provide a hypnogram (e.g., from YASA)
raw = mne.io.read_raw_edf("recording.edf", preload=True)
hypnogram = ["W", "N1", "N2", "N2", "N3", ...]  # per 30-s epoch

# Run the full pipeline
results = run_pneumo_analysis(raw, hypnogram, scoring_profile="standard")

# Access results
resp = results["respiratory"]["summary"]
print(f"AHI: {resp['ahi_total']}, Severity: {resp['severity']}")
print(f"Events: {resp['n_obstructive']} OA, {resp['n_hypopnea']} Hyp")

# AHI confidence interval
interval = results["interval"]
print(f"AHI interval: [{interval['strict']['ahi']}–{interval['sensitive']['ahi']}]")
print(f"Robustness: {interval['robustness_grade']}")
```

## Scoring Profiles

| Parameter | Strict | Standard | Sensitive |
|-----------|--------|----------|-----------|
| Hypopnea threshold | ≥30% | ≥30% | ≥25% |
| SpO₂ nadir window | 30 s | 45 s | 45 s |
| Peak-based detection | No | Yes | Yes |

## Validation

**PSG-IPA** (PhysioNet): 5 recordings, 59 independent scorer sessions. Mean |ΔAHI| = 2.0/h, severity concordance 4/5. See the [paper](#paper) for full results.

**PSG-Audio** (Sismanoglio Hospital, Athens): n=194, open access. External validation in progress.

## Twelve Bias Corrections

| # | Correction | Direction | Clinical impact |
|---|-----------|-----------|----------------|
| 1 | Post-apnea baseline inflation | Over-counting | Prevents false Mild→Moderate |
| 2 | SpO₂ cross-contamination | Over-counting | Flags uncertain coupling |
| 3 | Cheyne-Stokes trough scoring | Over-counting | Prevents HF misdiagnosis as OSA |
| 4 | Low-confidence defaults | Over-counting | Confidence stratification |
| 5 | Artefact-flank exclusion | Over-counting | Prevents post-disconnect events |
| 6 | Local baseline validation | Over-counting | Rejects inflated-baseline FPs |
| 7 | Peak-based amplitude detection | Under-counting | AASM-conformant breath-level |
| 8 | Extended SpO₂ nadir window | Under-counting | Catches delayed desaturations |
| 9 | Flow smoothing removal | Under-counting | Eliminated +54 FPs on PSG-IPA |
| 10 | Position signal auto-mapping | Under-counting | Handles raw ADC encoding |
| 11 | Configurable profiles | Under-counting | Sensitivity adjustment per study |
| 12 | Flattening-based RERA | Under-counting | Flow limitation without amplitude drop |

## Architecture

~8,000 lines across 18 submodules, 98 unit tests (CI: Python 3.9–3.12):

`constants` · `utils` · `signal` · `breath` · `classify` · `spo2` · `plm` · `ancillary` · `respiratory` · `pipeline` · `pipeline_profiles` · `profiles` · `postprocess` · `signal_quality` · `signal_quality_channels` · `ecg_effort` · `_types`

## Related

- **[YASAFlaskified](https://github.com/bartromb/YASAFlaskified)** — web platform integrating psgscoring with YASA staging, multilingual PDF reports, EDF+ export, and FHIR R4
- **[YASA](https://github.com/raphaelvallat/yasa)** — AI-based sleep staging (Vallat & Walker, *eLife* 2021)
- **[slaapkliniek.be](https://slaapkliniek.be)** — live instance (no installation required)

## Citation

```bibtex
@article{rombaut2026psgscoring,
  title     = {Automated Polysomnography Scoring for Clinical Sleep Medicine:
               An Open-Source Platform Validated Against 59 Independent
               Scorer Sessions on {PSG-IPA}},
  author    = {Rombaut, Bart and Rombaut, Briek and Rombaut, Cedric},
  year      = {2026},
  note      = {Manuscript in preparation}
}
```

## Disclaimer

**psgscoring is research software — not a medical device.** It is not CE-marked (MDR 2017/745) or FDA-cleared. All outputs are research-grade estimates that must be reviewed by a qualified clinician before any diagnostic or therapeutic decision. See **[DISCLAIMER.md](DISCLAIMER.md)** for the full text.

## License

BSD-3-Clause. See [LICENSE](LICENSE).

---

*Developed at Slaapkliniek AZORG, Aalst, Belgium.*
*Contact: bart.rombaut@azorg.be*
