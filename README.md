# psgscoring

**AASM 2.6-compliant respiratory event scoring for polysomnography — pure NumPy/SciPy.**

[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](CHANGES.md)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](.github/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-112%20passing-brightgreen.svg)](tests/)

`psgscoring` extracts the core respiratory scoring algorithms from
[YASAFlaskified](https://github.com/bartromb/YASAFlaskified) into a
standalone, pip-installable Python library for research use.

No deep learning, no GPU required. Pure NumPy + SciPy.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Scoring Algorithms](#scoring-algorithms)
- [Over-counting Corrections (v0.8.10)](#over-counting-corrections-v0810)
- [Signal Processing Improvements (v0.8.11)](#signal-processing-improvements-v0811)
- [Module Structure](#module-structure)
- [Citation](#citation)
- [License](#license)

---

## Features

- **Square-root nasal pressure linearisation** (Bernoulli correction, AASM 2.6)
- **MMSD apnea validation** — distinguishes apnoea from signal-dropout (κ=0.78)
- **Dual-sensor detection** — thermistor for apnoea, NPT for hypopnoea
- **Hypopnoea Rule 1A + 1B** — SpO₂ coupling + arousal reinstatement
- **7-rule apnoea type classification** — obstructive / central / mixed + confidence
- **Phase-angle classification** via Hilbert transform (v0.8.11)
- **Dynamic 5-min sliding baseline** with stage-specific correction
- **SpO₂ coupling** with temporal constraints (30-s post-event window)
- **Breath-by-breath analysis** — flattening index, zero-crossing rate
- **PLM scoring** (AASM 2.6 + WASM criteria)
- **Cheyne-Stokes detection** via autocorrelation
- **Five systematic over-counting corrections** (v0.8.10)
- **Patient-specific baseline anchoring** — mouth-breathing detection (v0.8.11)
- **Optional LightGBM confidence calibration** (v0.8.11)

---

## Installation

```bash
# From PyPI (when published)
pip install psgscoring

# From source
git clone https://github.com/bartromb/psgscoring.git
cd psgscoring
pip install -e .
```

**Dependencies:** `numpy>=1.22`, `scipy>=1.8`, `mne>=1.0`

**Optional:** `lightgbm` (confidence calibration only)

---

## Quick Start

### Full pipeline via MNE Raw object

```python
import mne
from psgscoring import run_pneumo_analysis

raw = mne.io.read_raw_edf("study.edf", preload=False, verbose=False)
hypno = [...]  # list of stage strings: "W", "N1", "N2", "N3", "R"

results = run_pneumo_analysis(raw=raw, hypno=hypno)

summary = results["respiratory"]["summary"]
print(f"AHI:  {summary['ahi_total']:.1f} /h")
print(f"OAHI: {summary['oahi']:.1f} /h")
print(f"ODI:  {summary['odi_3pct']:.1f} /h")

# Over-counting correction indices (v0.8.10)
print(f"CSR-flagged events: {summary['n_csr_flagged']}")
print(f"AHI excl. noise:    {summary['ahi_excl_noise']:.1f} /h")

# Baseline anchoring (v0.8.11)
anchor = results["anchor_baseline"]
print(f"Mouth breathing suspected: {anchor['mouth_breathing_suspected']}")
```

### Respiratory scoring only (NumPy arrays)

```python
import numpy as np
from psgscoring.respiratory import detect_respiratory_events

# flow_data, hypop_flow: NumPy arrays at sf_flow Hz
result = detect_respiratory_events(
    flow_data    = flow_array,      # thermistor (apnoea)
    hypop_flow   = npd_array,       # nasal pressure (hypopnoea)
    thorax_data  = thorax_array,    # RIP thorax
    abdomen_data = abdomen_array,   # RIP abdomen
    spo2_data    = spo2_array,
    sf_flow      = 256.0,
    sf_spo2      = 25.6,
    sf_hypop     = 256.0,
    hypno        = hypno_list,
)

events  = result["events"]          # list of dicts
summary = result["summary"]

for ev in events[:5]:
    print(f"{ev['type']:20s}  onset={ev['onset_s']:.1f}s  "
          f"conf={ev['confidence']:.2f}  phase={ev.get('phase_angle_deg','—')}°")
```

### Apnoea type classification only

```python
from psgscoring.classify import classify_apnea_type

ev_type, confidence, detail = classify_apnea_type(
    onset_idx    = 5120,
    end_idx      = 9216,
    thorax_env   = thorax_envelope,
    abdomen_env  = abdomen_envelope,
    thorax_raw   = thorax_raw,
    abdomen_raw  = abdomen_raw,
    effort_baseline = 1.0,
    sf           = 256.0,
)

print(f"Type: {ev_type}, Confidence: {confidence:.2f}")
print(f"Phase angle: {detail['phase_angle_deg']}°")
print(f"Decision reason: {detail['decision_reason']}")
```

### Baseline anchoring

```python
from psgscoring.signal import preprocess_flow, compute_anchor_baseline

flow_env = preprocess_flow(flow_data, sf=256.0, is_nasal_pressure=False)
anchor   = compute_anchor_baseline(
    flow_env, sf=256.0, hypno=hypno,
    events=scored_events,
)

print(f"Anchor value:  {anchor['anchor_value']:.4f}")
print(f"Anchor ratio:  {anchor['anchor_ratio']:.3f}")
print(f"Mouth breathing: {anchor['mouth_breathing_suspected']}")
```

---

## API Reference

### `run_pneumo_analysis(raw, hypno, **kwargs) → dict`

Master function. Accepts an MNE `Raw` object and hypnogram list.

Returns a dict with keys:
`respiratory`, `spo2`, `plm`, `arousal`, `cheyne_stokes`,
`anchor_baseline`, `heart_rate`, `snoring`, `position`,
`signal_quality`, `breath_analysis`.

### `detect_respiratory_events(...) → dict`

Core respiratory scoring. Returns:
- `events`: list of event dicts
- `rejected_hypopneas`: candidates that failed Rule 1A/1B
- `summary`: all respiratory indices
- `n_gap_excluded`: artefact gaps detected (Fix 5)

**Summary fields (selected):**

| Field | Description |
|-------|-------------|
| `ahi_total` | Total AHI (/h TST) |
| `oahi` | OAHI — all obstructive + hypopnoeas (AASM) |
| `oahi_thresholds` | OAHI at confidence ≥0.85 / ≥0.60 / ≥0.40 / 0.00 |
| `confidence_bands` | `{high, moderate, borderline, low}` event counts |
| `n_spo2_cross_contaminated` | Fix 2 events |
| `n_csr_flagged` | Fix 3 — CSR-related events |
| `ahi_csr_corrected` | AHI excluding CSR-flagged events |
| `n_low_conf_borderline` | Fix 4 — confidence 0.40–0.59 |
| `n_low_conf_noise` | Fix 4 — confidence <0.40 |
| `ahi_excl_noise` | AHI excluding confidence <0.40 events |
| `n_gap_excluded` | Fix 5 — artefact gaps |

### `classify_apnea_type(onset_idx, end_idx, ...) → (str, float, dict)`

Returns `(type, confidence, detail)`.
Type: `"obstructive"`, `"central"`, or `"mixed"`.

### `compute_anchor_baseline(flow_env, sf, hypno, ...) → dict`

Returns patient-specific N2-anchor baseline dict.

### `compute_dynamic_baseline(flow_env, sf) → np.ndarray`

5-min sliding 95th-percentile baseline, linearly interpolated.

---

## Scoring Algorithms

### Signal processing chain

```
Raw EDF signal
  │
  ├─ [Apnoea channel] Thermistor (no sqrt)
  │    → bandpass 0.05–3 Hz
  │    → Hilbert envelope
  │    → MMSD validation
  │    → dynamic baseline (5 min, P95)
  │    → stage-specific baseline blend
  │
  └─ [Hypopnoea channel] Nasal pressure transducer
       → sign(x)·√|x|  (Bernoulli linearisation)
       → bandpass 0.05–3 Hz
       → Hilbert envelope
       → dynamic baseline
```

### Apnoea type classification (7 rules)

```
Rule 0: Hilbert phase angle ≥45°  →  obstructive  (v0.8.11)
Rule 1: paradox correlation + raw variability  →  obstructive
Rule 2: high raw variability, low envelope  →  obstructive
Rule 3: first half absent + second half present  →  mixed
Rule 4: effort ratio > EFFORT_PRESENT_RATIO  →  obstructive
Rule 5: fully flat (no effort signs)  →  central
Rule 6: borderline default  →  obstructive (confidence 0.40)
```

### Performance (8-hour PSG at 256 Hz)

| Step | Time |
|------|------|
| `compute_dynamic_baseline` | ~3.5 s |
| `compute_stage_baseline` | ~2 s |
| Event detection (`find_objects`) | <1 s |
| All 5 over-counting fixes | <1 s |
| **Total `detect_respiratory_events`** | **~9–12 s** |

---

## Over-counting Corrections (v0.8.10)

See [YASAFlaskified CHANGES.md](https://github.com/bartromb/YASAFlaskified/blob/main/CHANGES.md)
for detailed descriptions. Summary:

| Fix | Field in summary | Bias addressed |
|-----|-----------------|----------------|
| 1 | (implicit — baseline) | Post-apnoea hyperpnoea baseline inflation |
| 2 | `n_spo2_cross_contaminated` | SpO₂ nadir cross-contamination at AHI >60/h |
| 3 | `n_csr_flagged`, `ahi_csr_corrected` | Cheyne-Stokes decrescendo scored as hypopnoea |
| 4 | `n_low_conf_borderline`, `n_low_conf_noise`, `ahi_excl_noise` | Borderline Rule-6 defaults at poor RIP quality |
| 5 | `n_gap_excluded` | Post-gap recovery ramp scored as event |

---

## Signal Processing Improvements (v0.8.11)

| Feature | Function | Notes |
|---------|----------|-------|
| Phase-angle classification | `_compute_phase_angle()` in `classify.py` | Rule 0 fires before 6 legacy rules |
| Baseline anchoring | `compute_anchor_baseline()` in `signal.py` | N2 event-free median RMS |
| LightGBM calibration | `_lgbm_confidence()` in `classify.py` | `PSGSCORING_LGBM_MODEL` env var |

---

## Module Structure

```
psgscoring/
├── __init__.py       # Public API (33 symbols)
├── constants.py      # AASM thresholds, band limits
├── utils.py          # Sleep masks, build_sleep_mask()
├── signal.py         # Preprocessing, baseline, MMSD, anchoring
├── breath.py         # Breath-by-breath, flattening index
├── classify.py       # Apnoea type classification (7 rules + Hilbert)
├── spo2.py           # SpO₂ coupling, ODI
├── plm.py            # PLM detection
├── ancillary.py      # HR, snore, position, CSR
├── respiratory.py    # Apnoea/hypopnoea + 5 corrections
├── pipeline.py       # run_pneumo_analysis() master function
└── tests/
    ├── test_signal.py
    ├── test_classify.py
    ├── test_respiratory.py
    ├── test_spo2.py
    ├── test_plm.py
    └── test_pipeline.py
```

---

## Citation

```bibtex
@software{rombaut2026psgscoring,
  author    = {Rombaut, Bart},
  title     = {{psgscoring}: AASM 2.6-compliant respiratory event scoring
               for polysomnography},
  year      = {2026},
  version   = {0.2.0},
  publisher = {GitHub},
  url       = {https://github.com/bartromb/psgscoring}
}
```

Also cite:
- YASA: Vallat & Walker (2021), *eLife* 10:e70092
- AASM 2.6: Berry et al. (2020), American Academy of Sleep Medicine
- Nasal pressure linearisation: Thurnheer et al. (2001), *AJRCCM* 164:1914

---

## License

BSD 3-Clause — Copyright (c) 2024–2026 Bart Rombaut / Slaapkliniek AZORG.
