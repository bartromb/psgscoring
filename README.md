# psgscoring

[![PyPI](https://img.shields.io/pypi/v/psgscoring.svg)](https://pypi.org/project/psgscoring/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/bartromb/psgscoring/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-47%20passed-brightgreen.svg)](https://github.com/bartromb/psgscoring/blob/main/psgscoring/tests/)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](pyproject.toml)
[![Validated](https://img.shields.io/badge/validated-PSG--IPA%20(59%20scorers)-green.svg)](https://github.com/bartromb/psgscoring/wiki)

**Open-source Python library for AASM 2.6-compliant automated polysomnography scoring.**

`psgscoring` extracts the core respiratory scoring algorithms from [YASAFlaskified](https://github.com/bartromb/YASAFlaskified) into a standalone, pip-installable library for the research community.

## Validation on PSG-IPA (v0.2.91)

External validation on the [PSG-IPA dataset](https://physionet.org/content/psg-ipa/) ([Alvarez-Estevez & Rijsman, *PLOS ONE* 2022](https://doi.org/10.1371/journal.pone.0275530)), comprising 5 diagnostic PSG recordings each independently scored by up to 12 certified sleep technologists (59 scorer sessions total, SOMNOscreen™ Plus hardware):

| PSG | Scorers | Median AHI | psgscoring AHI | ΔAHI | Severity |
|-----|---------|-----------|----------------|------|----------|
| SN1 | 11 | 6.0 | 8.1 | +2.1 | Mild ✓ |
| SN2 | 12 | 3.7 | 9.3 | +5.6 | Mild / Normal* |
| SN3 | 12 | 54.0 | 53.8 | −0.2 | Severe ✓ |
| SN4 | 12 | 3.5 | 4.3 | +0.8 | Normal ✓ |
| SN5 | 12 | 9.9 | 11.4 | +1.5 | Mild ✓ |
| **Mean** | | | | **2.0** | **4/5 concordant** |

*\*SN2: scorers split 8/12 Normal, 4/12 Mild. The sensitive profile (AHI 5.4) falls within the scorer IQR [1.8–5.6] — see [AHI confidence interval](#ahi-confidence-interval).*

Details: [Online Supplement (Wiki)](https://github.com/bartromb/psgscoring/wiki) · [Ablation analysis](https://github.com/bartromb/psgscoring/wiki/Ablation-Analysis)

## Features

- **AASM 2.6 respiratory scoring**: dual-sensor detection (thermistor for apneas, nasal pressure for hypopneas) with square-root linearisation, Hilbert envelope, and dynamic P95 baseline
- **7-rule apnea type classification**: obstructive / central / mixed with Hilbert phase-angle analysis and ECG-derived effort assessment (spectral + TECG; Berry 2019)
- **12 systematic bias corrections**: 6 over-counting (baseline inflation, SpO₂ cross-contamination, Cheyne-Stokes, borderline classification, artefact-flank, local baseline validation) + 6 under-counting (peak-based detection, extended SpO₂ window, position auto-mapping, configurable profiles, flattening-RERA)
- **Breath-amplitude stability filter** (v0.2.91): rejects false-positive hypopneas during normal breathing by comparing per-breath amplitude variability (CV < 0.45) — reduces over-counting by 57–68% in normal patients while preserving accuracy in severe OSA (−3%)
- **AHI confidence interval**: automatic 3-profile analysis reporting AHI as [strict – standard – sensitive] interval with robustness grade (A/B/C)
- **Configurable scoring profiles**: strict (research), standard (AASM 2.6), sensitive (UARS/screening)
- **Signal quality assessment**: flat-line, clipping, disconnect, line-noise, montage plausibility (EEG↔EOG, thorax↔abdomen cross-correlation)
- **PLM scoring** per AASM 2.6 + WASM criteria
- **SpO₂ analysis**: ODI 3%/4%, baseline (P90), T90, low-baseline warning
- **RERA/RDI computation**: amplitude-reduction RERA + flattening-based RERA (Hosselet 1998)
- **Cheyne-Stokes detection** via autocorrelation with event flagging and CSR-corrected AHI

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

# AHI confidence interval
iv = results["ahi_interval"]
print(f"Interval: [{iv['interval'][0]} – {iv['interval'][1]}]")
print(f"Robustness: {iv['robustness_grade']}")
```

## AHI confidence interval

Every analysis automatically runs all three scoring profiles and reports the AHI as an interval rather than a point estimate:

| Grade | Criterion | Clinical interpretation |
|-------|-----------|----------------------|
| **A** | All 3 profiles same severity | Robust — treatment decision unambiguous |
| **B** | 2/3 profiles concordant | Probable — clinical correlation recommended |
| **C** | All discordant | Uncertain — manual review recommended |

On PSG-IPA SN2, where 12 technologists split 8/4 between Normal and Mild, the interval [5.4–9.3] immediately communicates that the AHI hovers near the 5/h diagnostic threshold — precisely the situation where a single AHI number would mislead.

## What's new in v0.2.91

- **Removed flow smoothing** from all profiles — ablation analysis on PSG-IPA showed 3s smoothing added +54 false hypopneas to a mild-OSA recording (the dominant source of over-counting)
- **Breath-amplitude stability filter**: rejects hypopneas during stable breathing (CV < 0.45), calibrated on 59 independent scorer sessions
- **AHI confidence interval**: automatic 3-profile analysis with robustness grading (A/B/C)
- **Consecutive breath requirement**: peak detection requires ≥3 consecutive reduced breaths
- **PSG-IPA validation**: mean |ΔAHI| = 2.0/h, severity concordance 4/5

## Documentation

📖 **[Technical Handbook (PDF, 29 pages)](https://github.com/bartromb/psgscoring/blob/main/docs/handbook.pdf)** — complete guide covering signal processing, classification, all 12 corrections, stability filter, AHI interval, PSG-IPA validation, and exercises. Written for BSc-level CS students.

📖 **[Online Supplement (Wiki)](https://github.com/bartromb/psgscoring/wiki)** — signal processing chain, bias corrections, validation details, ablation analysis

## Live platform

**[slaapkliniek.be](https://slaapkliniek.be)** — upload an anonymised EDF recording; receive a complete PSG report (PDF, Excel, EDF+, FHIR R4) within 5–10 minutes. No local installation, Python, or GPU required.

## Citation

If you use `psgscoring` in your research, please cite:

> Rombaut B, Rombaut B, Rombaut C. psgscoring: An Open-Source Python Library for AASM 2.6-Compliant Automated Polysomnography Scoring. 2026. https://github.com/bartromb/psgscoring

This library builds on YASA for sleep staging:

> Vallat R, Walker MP. An open-source, high-performance tool for automated sleep staging. *eLife*. 2021;10:e70092. [doi:10.7554/eLife.70092](https://doi.org/10.7554/eLife.70092)

Validation was performed on the PSG-IPA dataset:

> Alvarez-Estevez D, Rijsman RM. Computer-assisted analysis of polysomnographic recordings improves inter-scorer associated agreement and scoring times. *PLOS ONE*. 2022;17(9):e0275530. [doi:10.1371/journal.pone.0275530](https://doi.org/10.1371/journal.pone.0275530)

> Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*. 2000;101(23):e215–e220.

## License

BSD-3-Clause. See [LICENSE](LICENSE).

## Medical & Clinical Disclaimer

**psgscoring is research software — not a medical device.**

This software has **not** been evaluated, cleared, or approved by any regulatory authority, including the European Union Medical Device Regulation (EU MDR 2017/745) and the U.S. Food and Drug Administration (FDA). It does **not** carry a CE mark or FDA clearance.

All computed indices — including AHI, OAHI, ODI, PLMI, arousal index, and RDI — are **research-grade estimates**. They must be reviewed by a qualified, licensed clinician before any diagnostic or therapeutic decision, validated against manual scoring by a registered polysomnographic technologist (RPSGT), and interpreted in the context of the full clinical picture and patient history.

**Intended use**: academic sleep research, algorithm benchmarking, clinical research under ethics committee approval, educational purposes. **Not intended for** standalone clinical diagnosis, automated treatment decisions, or unsupervised patient screening.

**Validation status**: External validation on PSG-IPA (PhysioNet, 5 recordings, 59 scorer sessions) demonstrated mean |ΔAHI| = 2.0/h and severity concordance 4/5. A formal validation study (AZORG-YASA-2026-001, n≥50) is in preparation. All outputs should be treated as preliminary and verified by a qualified clinician.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. See [DISCLAIMER.md](https://github.com/bartromb/psgscoring/blob/main/DISCLAIMER.md) for the full legal text.
