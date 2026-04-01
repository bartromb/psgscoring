# psgscoring

**AASM 2.6-compliant respiratory event scoring for polysomnography.**

[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](CHANGES.md)
[![Python](https://img.shields.io/badge/python-≥3.9-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen.svg)](tests/)
[![AASM](https://img.shields.io/badge/AASM-2.6-orange.svg)](https://aasm.org)
[![No GPU](https://img.shields.io/badge/GPU-not%20required-lightgrey.svg)]()

A Python library implementing validated signal processing algorithms for automated detection of apneas, hypopneas, arousals, periodic limb movements, and SpO2 desaturations from standard PSG recordings.

No deep learning. No GPU. Pure signal processing with `scipy` and `numpy`.

## Installation

```bash
pip install psgscoring
```

Or from source:

```bash
git clone https://github.com/bartromb/psgscoring.git
cd psgscoring
pip install -e .
```

**Dependencies:** `numpy>=1.22`, `scipy>=1.8`, `mne>=1.0`

## Quick Start

```python
import mne
from psgscoring import run_full_analysis

# Load PSG recording
raw = mne.io.read_raw_edf("recording.edf", preload=True)

# Sleep stages from YASA or manual scoring (one label per 30s epoch)
hypno = ["W", "N1", "N2", "N2", "N3", "N3", "N2", "R", ...]

# Run full analysis
results = run_full_analysis(raw, hypno)

# Access results
ahi = results["respiratory"]["summary"]["ahi_total"]
oahi = results["respiratory"]["summary"]["oahi"]
odi = results["spo2"]["summary"]["odi"]
plmi = results["plm"]["summary"]["plm_index"]

print(f"AHI: {ahi:.1f}, OAHI: {oahi:.1f}, ODI: {odi:.1f}, PLMI: {plmi:.1f}")
```

## Individual Functions

Each algorithm can be used independently:

```python
import numpy as np
from psgscoring import (
    linearize_nasal_pressure,
    compute_mmsd,
    preprocess_flow,
    compute_dynamic_baseline,
    detect_breaths,
    compute_flattening_index,
    bandpass_flow,
)

# Load nasal pressure channel
nasal = raw.get_data(picks=["NasalPressure"])[0]
sf = raw.info["sfreq"]  # e.g. 256 Hz

# 1. Linearize nasal pressure (Monserrat/Thurnheer)
nasal_lin = linearize_nasal_pressure(nasal)

# 2. Compute flow envelope
envelope = preprocess_flow(nasal, sf, is_nasal_pressure=True)

# 3. Dynamic baseline
baseline = compute_dynamic_baseline(envelope, sf)

# 4. Normalized flow (1.0 = normal, 0.0 = apnea)
flow_norm = np.clip(envelope / baseline, 0, 2)

# 5. MMSD for drift-independent validation
filtered = bandpass_flow(nasal_lin, sf)
mmsd = compute_mmsd(filtered, sf)

# 6. Breath-by-breath analysis
breaths = detect_breaths(filtered, sf)
for b in breaths[:5]:
    fi = compute_flattening_index(b["insp_segment"])
    print(f"  Breath at {b['onset_s']:.1f}s, amp={b['amplitude']:.3f}, flat={fi:.2f}")
```

---

## Algorithms

### A. Square-Root Linearization of Nasal Pressure

**Problem:** Nasal pressure transducers produce a signal proportional to flow² (Bernoulli's principle). A 50% flow reduction appears as a 75% amplitude reduction in the raw signal → systematic overestimation of hypopneas.

**Solution:**

```
x_lin(t) = sign(x(t)) × √|x(t)|
```

Applied before bandpass filtering, exclusively on the nasal pressure channel (hypopnea detection), not on the thermistor (apnea detection). This preserves the AASM dual-sensor paradigm.

**Function:** `linearize_nasal_pressure(data) → ndarray`

**References:**
- Thurnheer R, Xie X, Bloch KE. *Accuracy of nasal cannula pressure recordings for assessment of ventilation during sleep.* Am J Respir Crit Care Med. 2001;164(10):1914-1919. — Confirmed r²=0.88–0.96 vs pneumotachography.
- Montserrat JM, et al. *Effectiveness of CPAP treatment in daytime function in sleep apnea syndrome.* Am J Respir Crit Care Med. 2001;164(4):608-613.
- AASM Scoring Manual v2.6 Rule 3: *"nasal pressure transducer (with or without square root transformation of the signal)"*

### B. MMSD Apnea Validation

**Problem:** During long recordings, baseline drift from sensor displacement or mouth breathing creates false amplitude drops that the envelope method misinterprets as apneas.

**Solution:** The Mean Magnitude of Second Derivative (MMSD) measures the "sharpness" of the flow waveform — independent of absolute amplitude and drift:

```
MMSD(t) = (1/N) × Σ |x''(i)|   over 1-second window
```

Active breathing has high MMSD (sharp wave transitions). True apnea has near-zero MMSD. If normalized MMSD > 40% of baseline during a candidate apnea, respiratory activity is still present → false positive rejected.

**Function:** `compute_mmsd(flow_data, sf, window_s=1.0) → ndarray`

**Reference:**
- Lee H, Park J, Kim H, Lee K-J. *Detection of apneic events from single channel nasal airflow using 2nd derivative method.* Physiol Meas. 2008;29:N37-N45. — 92% agreement with manual scoring (κ=0.78) on 24 PSG recordings.

### C. Dual-Sensor Detection (AASM 2.6)

The AASM recommends different sensors for different event types:

| Sensor | Role | Threshold |
|--------|------|-----------|
| Oronasal thermistor | Apnea (cessation) | < 10% baseline, ≥10s |
| Nasal pressure (√-linearized) | Hypopnea (partial) | 10–70% baseline, ≥10s |

**Apnea–Hypopnea Exclusion Mask:** After apnea detection, a ±5s margin around each apnea is masked out before hypopnea labeling, preventing double-counting of apnea flanks.

**Apnea Type Classification** uses four-step effort analysis on thoracic/abdominal RIP:
1. Amplitude ratio vs baseline (>40% = obstructive, <20% = central)
2. Coefficient of variation (paradoxical breathing)
3. Cross-correlation thorax–abdomen (out-of-phase = obstructive)
4. First/second half comparison (mixed event detection)

**Function:** `detect_respiratory_events(flow_data, thorax_data, abdomen_data, spo2_data, sf_flow, sf_spo2, hypno, ...) → dict`

**Reference:**
- Berry RB, et al. *The AASM Manual for the Scoring of Sleep and Associated Events.* Version 2.6. AASM; 2020.

### D. Temporally Constrained SpO2 Coupling

A hypopnea requires ≥3% SpO2 desaturation (Rule 1A) or an arousal (Rule 1B).

**Improvements over naive matching:**
- **Baseline:** 90th percentile of 120s pre-event SpO2 (or global sleep baseline if local is depressed during cluster apneas)
- **Nadir window:** Event onset → 30s post-event (reduced from 45s)
- **Temporal validation:** Nadir must fall ≥3s after event onset (circulatory delay). Early nadirs with small desaturation (<5%) are rejected as coincidental.

**Reference:**
- Uddin MB, Chow CM, Ling SH, Su SW. *A novel algorithm for automatic diagnosis of sleep apnea from airflow and oximetry signals.* Physiol Meas. 2021;42:015001.

### E. Two-Pass Rule 1B (Arousal Criterion)

**Novel approach:** Hypopnea candidates without ≥3% desaturation are stored as *rejected candidates* (not discarded). After arousal detection completes in a later pipeline stage, candidates are re-evaluated. An arousal within 15s of event termination → event reinstated as Rule 1B hypopnea with AHI recalculation.

**Function:** `reinstate_rule1b_hypopneas(rejected, arousal_events, resp_events, hypno) → (reinstated, updated_events)`

### F. Breath-by-Breath Analysis

Zero-crossing segmentation of bandpass-filtered flow (1–15s per breath cycle). Per breath:
- **Amplitude:** peak-to-trough distance (AASM definition)
- **Local baseline:** median of preceding 10 breaths
- **Flattening index:** fraction of inspiratory segment >80% of peak flow. Values >0.3 indicate flow limitation (relevant for RERA detection).

**Functions:** `detect_breaths()`, `compute_breath_amplitudes()`, `compute_flattening_index()`

### G. Cheyne-Stokes Respiration

Autocorrelation of very-low-frequency flow envelope (0.005–0.05 Hz, 20–200s periodicities). Peak correlation >0.3 in 40–120s lag range → periodic crescendo-decrescendo pattern. Clinical flag: association with heart failure (NYHA III–IV).

**Function:** `detect_cheyne_stokes(flow_env, sf, hypno) → dict`

### H. Two-Phase Arousal Detection

AASM-compliant spectral arousal detection with sigma-band spindle exclusion:

- **Phase 1:** Identify regions where combined arousal power (α_narrow 8–11Hz + θ 4–8Hz + β 16–30Hz) exceeds 2.0× per-stage baseline. Label contiguous segments ≥3s.
- **Phase 2:** Validate each candidate event:
  - Pre-sleep: ≥60% of 10s pre-window is sleep
  - Onset abruptness: first 1s power / 3s pre-power > 1.5×
  - Spindle exclusion: reject if sigma >2× baseline AND arousal power < sigma
  - REM EMG: EMG rise >2× baseline for ≥1s

**Function:** `detect_arousals(eeg_data, sf, hypno, emg_data=None) → dict`

### I. PLM Scoring (AASM 2.6)

- EMG ≥8µV above resting, 0.5–10s duration (auto V→µV conversion)
- Bilateral integration (±0.5s)
- Wake excluded; respiratory-associated (±0.5s of event end) excluded
- Series: ≥4 consecutive with 5–90s intervals

**Function:** `analyze_plm(leg_l, leg_r, sf, hypno, resp_events=None) → dict`

**Reference:**
- Zucconi M, et al. *WASM standards for recording and scoring PLM.* Sleep Med. 2006;7(2):175-183.

---

## Output Structure

`run_full_analysis()` returns a dict:

```python
{
    "respiratory": {
        "success": True,
        "events": [
            {"type": "obstructive", "onset_s": 1234.5, "duration_s": 15.2,
             "stage": "N2", "desaturation_pct": 4.1, "min_spo2": 88.3, ...},
            ...
        ],
        "summary": {
            "ahi_total": 23.4,
            "oahi": 20.1,
            "central_index": 3.3,
            "n_obstructive": 85,
            "n_central": 14,
            "n_mixed": 3,
            "n_hypopnea": 65,
            "severity": "moderate",
            ...
        },
    },
    "spo2": {"summary": {"odi": 18.2, "mean_spo2": 93.1, "min_spo2": 71, ...}},
    "plm": {"summary": {"plm_index": 8.3, "n_plm": 42, ...}},
    "arousal": {"summary": {"arousal_index": 28.1, "n_respiratory_arousals": 45, ...}},
    "cheyne_stokes": {"csr_detected": False, ...},
    "position": {...},
    "heart_rate": {...},
    "snore": {...},
}
```

## Disclaimer

> **psgscoring is research software — not a medical device.**
>
> It is intended exclusively for use by qualified professionals (physicians,
> researchers, registered polysomnographic technologists) in a research or
> clinical research context. It has **not** been evaluated, cleared, or approved
> by the EU (MDR 2017/745), the U.S. FDA, or any equivalent regulatory authority.
> It does **not** carry a CE mark or FDA clearance.
>
> All computed indices (AHI, OAHI, ODI, PLMI, arousal index, RDI) are
> **research-grade estimates** that must be reviewed by a qualified clinician
> before any diagnostic or therapeutic decision. Automated sleep staging
> (~85% epoch agreement) does not meet the standard for unsupervised clinical use.
>
> A pilot validation study (target n=50) is in preparation. Until published,
> all results should be treated as provisional.
>
> See [DISCLAIMER.md](DISCLAIMER.md) for the full medical and clinical disclaimer.

## License

BSD-3-Clause — see [LICENSE](LICENSE).

## References

### Scoring guidelines

1. Berry RB, Quan SF, Abreu AR, et al. **The AASM Manual for the Scoring of Sleep and Associated Events: Rules, Terminology and Technical Specifications.** Version 2.6. American Academy of Sleep Medicine; 2020.

### Nasal pressure linearization

2. Thurnheer R, Xie X, Bloch KE. **Accuracy of nasal cannula pressure recordings for assessment of ventilation during sleep.** *Am J Respir Crit Care Med.* 2001;164(10):1914–1919. doi:10.1164/ajrccm.164.10.2101072
3. Montserrat JM, Farré R, Ballester E, et al. **Evaluation of nasal prongs for estimating nasal flow.** *Am J Respir Crit Care Med.* 1997;155(1):211–215. doi:10.1164/ajrccm.155.1.9001314

### MMSD apnea validation

4. Lee H, Park J, Kim H, Lee K-J. **New method for the detection of apneic events from nasal cannula pressure recordings using the second derivative.** *Comput Biol Med.* 2008;38:1105–1112. doi:10.1016/j.compbiomed.2008.08.007

### SpO2 desaturation coupling

5. Uddin MB, Chow CM, Ling SH, Su SW. **A novel algorithm for automatic diagnosis of sleep apnea from airflow and oximetry signals.** *Physiol Meas.* 2021;42:015001. doi:10.1088/1361-6579/abd47a

### PLM scoring

6. Zucconi M, Ferri R, Allen R, et al. **The official World Association of Sleep Medicine (WASM) standards for recording and scoring periodic leg movements in sleep (PLMS) and wakefulness (PLMW).** *Sleep Med.* 2006;7(2):175–183. doi:10.1016/j.sleep.2006.01.001

### Arousal scoring

7. Bonnet MH, Doghramji K, Roehrs T, et al. **The scoring of arousal in sleep: reliability, validity and alternatives.** *J Clin Sleep Med.* 2007;3(2):133–145. doi:10.5664/jcsm.26815

### Flattening index / flow limitation

8. Aittokallio T, Saaresranta T, Polo-Kantola P, et al. **Analysis of inspiratory flow shapes in patients with partial upper-airway obstruction during sleep.** *Chest.* 2001;119(1):37–44. doi:10.1378/chest.119.1.37

### Sleep staging (upstream)

9. Vallat R, Walker MP. **An open-source, high-performance tool for automated sleep staging.** *eLife.* 2021;10:e70092. doi:10.7554/eLife.70092

### Cheyne-Stokes respiration

10. Leung RST, Bradley TD. **Sleep apnea and cardiovascular disease.** *Am J Respir Crit Care Med.* 2001;164(12):2147–2165. doi:10.1164/ajrccm.164.12.2107045

## Citation

If you use psgscoring in published research, please cite:

```bibtex
@software{rombaut2026psgscoring,
  author = {Rombaut, Bart},
  title = {psgscoring: AASM 2.6-compliant respiratory event scoring for polysomnography},
  year = {2026},
  url = {https://github.com/bartromb/psgscoring},
}
```

## Acknowledgments

Sleep staging relies on [YASA](https://github.com/raphaelvallat/yasa) by Raphaël Vallat and Matthew P. Walker (*eLife*, 2021). Signal processing builds on [MNE-Python](https://mne.tools), [SciPy](https://scipy.org), and [NumPy](https://numpy.org).

Developed at Slaapkliniek AZORG, Aalst, Belgium.
