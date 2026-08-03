# psgscoring

**Open-source AASM-compliant respiratory scoring for polysomnography.**

[![PyPI](https://img.shields.io/pypi/v/psgscoring)](https://pypi.org/project/psgscoring/)
[![Python](https://img.shields.io/pypi/pyversions/psgscoring)](https://pypi.org/project/psgscoring/)
[![License](https://img.shields.io/pypi/l/psgscoring)](https://github.com/bartromb/psgscoring/blob/main/LICENSE)
[![Tests](https://github.com/bartromb/psgscoring/actions/workflows/tests.yml/badge.svg)](https://github.com/bartromb/psgscoring/actions)

## Paper

> Rombaut B, Rombaut B, Rombaut C, et al. **Automated Polysomnography Scoring for Clinical Sleep Medicine: An Open-Source Platform Validated Against 59 Independent Scorer Sessions on PSG-IPA.** Manuscript in preparation, 2026.

Technical details (signal processing chain, classification logic, all twelve bias corrections): **[Technical Reference (Online Supplement)](https://github.com/bartromb/psgscoring/wiki/Technical-Reference)**

## What this library does

`psgscoring` detects and classifies respiratory events (apneas, hypopneas, RERAs) in polysomnography recordings following AASM rules. It extends [YASA](https://github.com/raphaelvallat/yasa) (Vallat & Walker, *eLife* 2021) from sleep staging into a complete clinical respiratory scoring pipeline.

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
results = run_pneumo_analysis(raw, hypnogram, scoring_profile="aasm_v3_rec")

# Access results
resp = results["respiratory"]["summary"]
print(f"AHI: {resp['ahi_total']}, Severity: {resp['severity']}")
print(f"Events: {resp['n_obstructive']} OA, {resp['n_hypopnea']} Hyp")

# AHI confidence interval
interval = results["ahi_interval"]
print(f"AHI interval: [{interval['strict']['ahi']}–{interval['sensitive']['ahi']}]")
print(f"Robustness: {interval['robustness_grade']}")
```

## Scoring Profiles

| Parameter | Strict | Standard | Sensitive |
|-----------|--------|----------|-----------|
| Hypopnea threshold | ≥30% | ≥30% | ≥25% |
| SpO₂ nadir window | 30 s | 45 s | 45 s |
| Peak-based detection | No | Yes | Yes |

`list_profiles()` enumerates the full registry (AASM v1/v2/v3, CMS/Medicare,
Chicago 1999, and the NSRR dataset profile).

### `aasm_v3_breath` — breath-graded hypopnea scoring (v0.13.0, opt-in)

```python
run_pneumo_analysis(raw, hypnogram, scoring_profile="aasm_v3_breath")
```

A second hypopnea detector that takes **the breath, not the sample, as the
unit**. The AASM speaks of *peak signal excursions*; this scores them
directly. Five differences from the default detector:

1. **Breath-level events.** An event is a run of consecutive breaths below
   the threshold, so boundaries land on breath transitions and a recovery
   breath ends the run by itself — no smoothing, and the merge/split
   problem that depresses event-F1 largely disappears.
2. **Two-pass patient calibration.** Pass 1 finds the incontestable events;
   pass 2 measures the baseline from *only* the breathing that is not an
   event, and derives that patient's **own SpO₂ delay** by cross-correlation
   instead of assuming a fixed 30–45 s window.
3. **Graded AASM predicates.** The rule structure stays literally the Rule 1A
   conjunction — ≥30% reduction AND ≥10 s AND (≥3% desaturation OR arousal) —
   but each threshold gets a tolerance rather than being infinitely sharp.
4. **Every event carries `p_scored`** plus a `criteria` dict giving each
   predicate's contribution, so the audit trail says *why*, not just *what*.

   `p_scored` **ranks** events by how well they satisfy the AASM conjunction.
   It is **not** the probability that a scorer would mark the event: measured
   against the 12-scorer fractions on PSG-IPA (163 events) the correlation is
   only r = 0.194 and the level is +0.33 too high. Use it to order events,
   not as a likelihood.
5. **One strictness axis** (`hypopnea_strictness`, default 0.50) instead of
   three parameter combinations.

Scope is hypopneas only; apneas keep the existing detector.

**Status — opt-in, not the default.** On PSG-IPA (5 recordings, 12 scorers)
the median event-F1 rises 0.343 → 0.434, the percentile within the
inter-scorer distribution p6 → p17, and mean |ΔAHI| against the scorer median
falls **1.84 → 0.29**. On MESA the paired advantage replicates on two
disjoint held-out samples of 50 (p = 0.0069 and p = 0.0016). But the two
datasets disagree on the *absolute* level by ~16 AHI points, and until that
is explained the existing clinical output stays the default. See
[`docs/interim_conclusie_klinisch_gebruik.md`](docs/interim_conclusie_klinisch_gebruik.md).

### `aasm_v3_dual` — apneas on both flow sensors (v0.14.0, opt-in)

```python
run_pneumo_analysis(raw, hypnogram, scoring_profile="aasm_v3_dual")
```

The AASM assigns apneas to the oronasal thermistor and hypopneas to nasal
pressure. Choosing one sensor means inheriting its blind spot: nasal pressure
misses mouth breathing, the thermistor is too slow for short events. This
profile detects apneas on **both** and merges them, so neither channel can
veto the other.

Each merged apnea carries `corroboration` — `both`, `thermistor_only` or
`pressure_only`. **Nothing is discarded on that basis.** Across 1785 apnea
detections on MESA only 19% are seen by both sensors, so requiring
corroboration would remove four events in five. Discarding can be licensed
explicitly (`corroborate_apnea_events(..., corroboration_licensed=True)`); no
profile does so.

`meta["flow_channels"]["thermistor_check"]` reports the envelope correlation
between the two channels either way. Under this profile it is *informational* —
the second sensor is additive and can only add events. It gates only the
substitutive path, where a dead thermistor would delete them.

**Why this exists.** v0.13.2 made the thermistor the apnea sensor outright. On
a montage where that channel carried no usable breathing signal, apneas went
from 93 to 0 and the conclusion moved from moderate CSAS — human-confirmed —
to mild SAS. That release was rolled back. `aasm_v3_rec` remains the default
and is byte-identical.

## Validation

**PSG-IPA** (PhysioNet): 5 recordings, 59 independent scorer sessions. Mean |ΔAHI| = 1.8/h, Pearson r = 0.997, severity concordance 4/5 (standard profile). See the [paper](#paper) for full results.

**MESA** (NSRR, external cohort): q=7 high-quality holdout, n=92 (held out from the optional LightGBM re-classifier's training). LightGBM-augmented AHI: bias −0.02/h, MAE 5.3/h, Pearson r = 0.87 against the NSRR `nsrr_ahi_hp3u` reference. SHHS-1 validation in progress.

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

~8,900 lines across 17 submodules, 115 unit tests (CI: Python 3.9–3.12):

`constants` · `utils` · `signal` · `breath` · `classify` · `spo2` · `plm` · `ancillary` · `respiratory` · `pipeline` · `ml_classifier` · `profiles` · `postprocess` · `signal_quality` · `signal_quality_channels` · `ecg_effort` · `_types`

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

**psgscoring is research software — not a medical device.** It is not CE-marked (MDR 2017/745) or FDA-cleared. All outputs are research-grade estimates that must be reviewed by a qualified clinician before any diagnostic or therapeutic decision. See **[DISCLAIMER.md](https://github.com/bartromb/psgscoring/blob/main/DISCLAIMER.md)** for the full text.

## License

BSD-3-Clause. See [LICENSE](LICENSE).

---

*Contact: bart.rombaut@gmail.com*
