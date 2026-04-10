# psgscoring

[![PyPI](https://img.shields.io/pypi/v/psgscoring.svg)](https://pypi.org/project/psgscoring/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-47%20passed-brightgreen.svg)](psgscoring/tests/)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](pyproject.toml)

**Open-source Python library for AASM 2.6-compliant automated polysomnography scoring.**

`psgscoring` extracts the core respiratory scoring algorithms from [YASAFlaskified](https://github.com/bartromb/YASAFlaskified) into a standalone, pip-installable library for the research community.

## Documentation

📖 **[Technical Handbook (PDF)](docs/handbook.pdf)** — 26-page guide covering:
- Clinical context and AASM 2.6 scoring rules
- Signal processing pipeline (linearisation, filtering, Hilbert envelope, dynamic baseline)
- 7-rule apnea type classification with Hilbert phase-angle analysis
- All 12 over-counting and under-counting corrections with pseudocode
- ECG-derived effort classification (spectral + TECG)
- Arousal detection, RERA/RDI, snoring, heart rate, body position, Cheyne-Stokes
- PLM detection, signal quality assessment, sleep cycle analysis
- Complete pipeline walkthrough with code examples
- Data structures, testing strategy, design decisions
- Glossary and exercises

Written for BSc-level computer science students. LaTeX source included in `docs/handbook.tex`.

## Features

- **AASM 2.6 respiratory scoring**: apnea/hypopnea detection with dual-sensor (thermistor + nasal pressure) support
- **7-rule apnea type classification**: obstructive / central / mixed with Hilbert phase-angle analysis
- **6 over-counting corrections**: post-apnea baseline inflation, SpO₂ cross-contamination, Cheyne-Stokes flagging, borderline classification, artefact-flank exclusion, local baseline validation
- **6 under-counting corrections**: peak-based hypopnea detection, SpO₂ de-blocking, extended nadir window, flow smoothing, position auto-mapping, configurable scoring profiles
- **ECG-derived effort classification**: adaptive cardiac band + TECG (Berry 2019) for central apnea detection
- **PLM scoring** per AASM 2.6 + WASM criteria
- **SpO₂ analysis**: ODI 3%/4%, baseline, T90, low-baseline warning
- **Signal quality assessment**: flat-line, clipping, disconnect, montage plausibility
- **RERA/RDI computation**: amplitude-reduction + flattening-based RERA detection

## Installation

```bash
pip install psgscoring
```

With optional dependencies (YASA staging + LightGBM):
```bash
pip install psgscoring[full]
```

## Quick start

```python
import numpy as np
from psgscoring import (
    bandpass_flow, linearise_nasal_pressure,
    compute_dynamic_baseline, classify_apnea_type,
    detect_breaths, compute_flattening_index,
)

# Load your flow signal (e.g., from MNE)
# flow = raw.get_data(picks='Flow')[0]

# Linearise nasal pressure (Bernoulli correction)
flow_lin = linearise_nasal_pressure(flow)

# Bandpass filter (0.05–3 Hz)
flow_filt = bandpass_flow(flow_lin, sf=256)

# Dynamic baseline (5-min sliding 95th percentile)
baseline = compute_dynamic_baseline(np.abs(flow_filt), sf=256)

# Detect breaths
breaths = detect_breaths(flow_filt, sf=256)
```

## Submodules

| Module | Responsibility |
|--------|---------------|
| `constants` | AASM thresholds, band limits |
| `utils` | Sleep mask, channel detection |
| `signal` | Linearisation, baseline, MMSD |
| `breath` | Breath segmentation, flattening index |
| `classify` | Apnea type (7-rule + Hilbert) |
| `spo2` | SpO₂ coupling, ODI |
| `plm` | PLM detection (AASM 2.6 + WASM) |
| `ecg_effort` | ECG-derived effort (TECG + spectral) |
| `ancillary` | HR, snore, position, CSR |
| `respiratory` | Apnea/hypopnea + 12 corrections |
| `pipeline` | Master `run_pneumo_analysis()` |
| `signal_quality` | Per-channel quality grading |

## Configurable scoring profiles

| Profile | Hypopnea threshold | Nadir window | Smoothing | Peak detection |
|---------|-------------------|-------------|-----------|---------------|
| Strict | ≥30% | 30 s | — | No |
| Standard | ≥30% | 45 s | 3 s | Yes |
| Sensitive | ≥25% | 45 s | 5 s | Yes |

## References

- Berry RB et al. *The AASM Manual for the Scoring of Sleep and Associated Events, Version 2.6.* AASM, 2020.
- Vallat R, Walker MP. An open-source, high-performance tool for automated sleep staging. *eLife*. 2021;10:e70092.
- Berry RB et al. Use of a transformed ECG signal to detect respiratory effort during apnea. *JCSM*. 2019;15(11):1653–1660.

## Citation

If you use `psgscoring` in your research, please cite:

> Rombaut B, Rombaut B, Rombaut C, Vallat R. psgscoring: An Open-Source Python Library for AASM 2.6-Compliant Automated Polysomnography Scoring. 2026. https://github.com/bartromb/psgscoring

## License

BSD-3-Clause. See [LICENSE](LICENSE).
