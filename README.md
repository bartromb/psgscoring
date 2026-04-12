# psgscoring

[![PyPI](https://img.shields.io/pypi/v/psgscoring.svg)](https://pypi.org/project/psgscoring/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/bartromb/psgscoring/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-47%20passed-brightgreen.svg)](https://github.com/bartromb/psgscoring/blob/main/psgscoring/tests/)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](pyproject.toml)
[![Validated](https://img.shields.io/badge/validated-PSG--IPA%20(47%20scorers)-green.svg)](https://github.com/bartromb/psgscoring/wiki)

**Open-source Python library for AASM 2.6-compliant automated polysomnography scoring.**

`psgscoring` extracts the core respiratory scoring algorithms from [YASAFlaskified](https://github.com/bartromb/YASAFlaskified) into a standalone, pip-installable library for the research community.

## Validation (v0.2.8)

External validation on the [PSG-IPA dataset](https://physionet.org/content/psg-ipa/) (PhysioNet, 5 recordings, 47 independent scorer sessions):

| PSG | Scorers | Median AHI | psgscoring AHI | ΔAHI | Severity |
|-----|---------|-----------|----------------|------|----------|
| SN1 | 11 | 6.0 | 8.1 | +2.1 | Mild ✓ |
| SN2 | 4 | 4.4 | 9.3 | +4.9 | Mild / Normal |
| SN3 | 12 | 54.0 | 53.8 | −0.2 | Severe ✓ |
| SN4 | 12 | 3.5 | 4.3 | +0.8 | Normal ✓ |
| SN5 | 12 | 9.9 | 11.4 | +1.5 | Mild ✓ |
| **Mean** | | | | **1.9** | **4/5 concordant** |

Details: [Online Supplement (Wiki)](https://github.com/bartromb/psgscoring/wiki)

## Features

- **AASM 2.6 respiratory scoring**: apnea/hypopnea detection with dual-sensor support
- **12 systematic bias corrections**: 6 over-counting + 6 under-counting ([details](https://github.com/bartromb/psgscoring/wiki))
- **Breath-amplitude stability filter** (v0.2.8): rejects false-positive hypopneas during normal breathing
- **AHI confidence interval**: strict/standard/sensitive profiles with robustness grade (A/B/C)
- **Configurable scoring profiles**: strict (research), standard (AASM 2.6), sensitive (UARS)
- **ECG-derived effort classification**: spectral + TECG for central apnea detection
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

# AHI
print(f"AHI: {results['respiratory']['summary']['ahi_total']}")

# AHI confidence interval (v0.2.8)
iv = results["ahi_interval"]
print(f"Interval: [{iv['interval'][0]} – {iv['interval'][1]}] Grade: {iv['robustness_grade']}")
```

## What's new in v0.2.8

- **Removed flow smoothing** from standard profile (root cause: +54 false hypopneas)
- **Stability filter**: rejects hypopneas during stable breathing (CV < 0.45)
- **AHI confidence interval**: 3-profile analysis with robustness A/B/C
- **PSG-IPA validation**: mean |ΔAHI| = 1.9/h, severity concordance 4/5

## Documentation

📖 [Online Supplement (Wiki)](https://github.com/bartromb/psgscoring/wiki) — signal processing, corrections, validation  
📖 [Technical Handbook (PDF)](https://github.com/bartromb/psgscoring/blob/main/docs/handbook.pdf) — 26-page guide

## Live platform

**[slaapkliniek.be](https://slaapkliniek.be)** — upload EDF, receive complete PSG report. No installation required.

## Citation

> Rombaut B, Rombaut B, Rombaut C. psgscoring: An Open-Source Python Library for AASM 2.6-Compliant Automated Polysomnography Scoring. 2026. https://github.com/bartromb/psgscoring

This library builds on YASA:

> Vallat R, Walker MP. An open-source, high-performance tool for automated sleep staging. *eLife*. 2021;10:e70092.

## License

BSD-3-Clause. See [LICENSE](LICENSE).

## Medical & Clinical Disclaimer

**psgscoring is research software — not a medical device.**

This software has **not** been evaluated, cleared, or approved by any regulatory authority, including the European Union Medical Device Regulation (EU MDR 2017/745) and the U.S. Food and Drug Administration (FDA). It does **not** carry a CE mark or FDA clearance.

All computed indices — including AHI, OAHI, ODI, PLMI, arousal index, and RDI — are **research-grade estimates**. They must be:

- Reviewed by a qualified, licensed clinician before any diagnostic or therapeutic decision
- Validated against manual scoring by a registered polysomnographic technologist (RPSGT)
- Interpreted in the context of the full clinical picture and patient history

**Intended use**: academic sleep research, algorithm benchmarking, clinical research under ethics committee approval, educational purposes. **Not intended for** standalone clinical diagnosis, automated treatment decisions, or unsupervised patient screening.

**Validation status**: External validation on PSG-IPA (PhysioNet, 5 recordings, 59 scorer sessions) demonstrated mean |ΔAHI| = 2.0/h and severity concordance 4/5. A formal validation study (AZORG-YASA-2026-001, n≥50) is in preparation. All outputs should be treated as preliminary and verified by a qualified clinician.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. See [DISCLAIMER.md](https://github.com/bartromb/psgscoring/blob/main/DISCLAIMER.md) for the full legal text.
