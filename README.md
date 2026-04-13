# psgscoring

[![PyPI](https://img.shields.io/pypi/v/psgscoring.svg)](https://pypi.org/project/psgscoring/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/bartromb/psgscoring/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](pyproject.toml)
[![Validated](https://img.shields.io/badge/validated-PSG--IPA%20(60%20sessions)-green.svg)](https://github.com/bartromb/psgscoring/wiki)

**Open-source Python library for AASM 2.6-compliant automated polysomnography scoring.**

`psgscoring` extracts the core respiratory scoring algorithms from [YASAFlaskified](https://github.com/bartromb/YASAFlaskified) into a standalone, pip-installable library for the research community.

## Validation (v0.2.92)

External validation on two public datasets:

### PSG-IPA (PhysioNet, 5 recordings, 60 scorer sessions from 12 RPSGT/ESRS)

| Metric | Result | Target |
|--------|--------|--------|
| AHI bias | **+1.6/h** | <±5/h ✓ |
| MAE | **2.5/h** | — |
| Pearson r | **0.990** | ≥0.85 ✓ |
| Severity concordance | **75%** | ≥70% ✓ |
| Event-level F1 (SN3) | **0.890** | — |

For 3/5 recordings, the algorithm's deviation from the scorer mean was **smaller than the inter-scorer variability**.

### iSLEEPS (39 ischemic stroke patients, SOMNOmedics)

| Severity | Bias | MAE |
|----------|------|-----|
| Normal/Mild (n=13) | −0.1/h | **3.3/h** |
| Moderate/Severe (n=26) | −16.6/h | 16.6/h |

Excellent for standard populations; systematic under-scoring in stroke patients (central apnea predominance).

## Features

- **AASM 2.6 respiratory scoring**: apnea/hypopnea detection with dual-sensor support
- **12 systematic bias corrections**: 6 over-counting + 6 under-counting
- **Breath-amplitude stability filter**: rejects false-positive hypopneas during normal breathing
- **AHI confidence interval**: strict/standard/sensitive profiles with robustness grade (A/B/C)
- **Hypoxic burden** (v0.2.92): total SpO₂ desaturation area per event, normalised per hour (Azarbarzin et al., AJRCCM 2019)
- **Post-processing** (v0.2.92): CSR-aware central reclassification, mixed apnea decomposition, central instability index
- **ECG-derived effort classification**: spectral + TECG for central apnea detection
- **Configurable scoring profiles**: strict (research), standard (AASM 2.6), sensitive (UARS)
- **PLM, SpO₂, RERA/RDI, signal quality assessment**

## Installation

```bash
pip install psgscoring
```

## Quick start

```python
from psgscoring import run_pneumo_analysis
import mne

raw = mne.io.read_raw_edf("recording.edf", preload=True)
hypno = ["W", "N1", "N2", "N3", "R", ...]  # 30-s epochs

results = run_pneumo_analysis(raw, hypno, scoring_profile="standard")

# AHI with confidence interval
iv = results["ahi_interval"]
print(f"AHI: {iv['standard']['ahi']} [{iv['strict']['ahi']}–{iv['sensitive']['ahi']}]")
print(f"Grade: {iv['robustness_grade']}")

# Hypoxic burden (v0.2.92)
hb = results["hypoxic_burden"]
print(f"Hypoxic burden: {hb['hypoxic_burden']} {hb['unit']}")

# Post-processing results (v0.2.92)
pp = results["postprocess"]
print(f"CSR reclassified: {pp['n_csr_reclassified']}")
print(f"Mixed decomposed: {pp['n_mixed_decomposed']}")
```

## What's new in v0.2.92

- **Hypoxic burden**: per-event SpO₂ desaturation area, normalised %·min/h
  - Clinical thresholds: <20 low, 20–73 moderate, >73 high CV risk
- **CSR-aware reclassification**: flagged obstructive/mixed events in CSR troughs → central
- **Mixed apnea decomposition**: central portion ≥10s → reclassified as central
- **Central instability index**: profile-dependent O/C uncertainty (0–1 scale)
- **iSLEEPS validation**: 39 stroke patients, MAE 3.3/h at normal/mild
- **Event-level validation**: F1=0.890, Δt=2.3s on severe-OSA recording

## Documentation

📖 [Online Supplement (Wiki)](https://github.com/bartromb/psgscoring/wiki)
📖 [Technical Handbook](https://github.com/bartromb/psgscoring/blob/main/docs/handbook.pdf)

## Live platform

**[slaapkliniek.be](https://slaapkliniek.be)** — upload EDF, receive complete PSG report.

## Citation

> Rombaut B, Rombaut B, Rombaut C, Vallat R. psgscoring: An Open-Source Python Library for AASM 2.6-Compliant Automated Polysomnography Scoring. 2026. https://github.com/bartromb/psgscoring

This library builds on YASA:

> Vallat R, Walker MP. An open-source, high-performance tool for automated sleep staging. *eLife*. 2021;10:e70092.

## License

BSD-3-Clause. See [LICENSE](LICENSE).

**Disclaimer**: Research use only. Not CE-marked or FDA-cleared. See [DISCLAIMER.md](DISCLAIMER.md).
