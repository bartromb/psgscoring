# psgscoring

**Open-source AASM-compliant respiratory scoring for polysomnography.**

[![PyPI](https://img.shields.io/pypi/v/psgscoring)](https://pypi.org/project/psgscoring/)
[![Python](https://img.shields.io/pypi/pyversions/psgscoring)](https://pypi.org/project/psgscoring/)
[![License](https://img.shields.io/pypi/l/psgscoring)](https://github.com/bartromb/psgscoring/blob/main/LICENSE)
[![Tests](https://github.com/bartromb/psgscoring/actions/workflows/tests.yml/badge.svg)](https://github.com/bartromb/psgscoring/actions)

## Paper

> Rombaut B, Rombaut B, Rombaut C, et al. **Graded evidence in place of thresholds: an open-source, AASM-compliant method for respiratory event detection in polysomnography.** Manuscript in preparation, 2026.

Technical details (signal processing chain, classification logic, bias corrections): **[Technical Reference (Online Supplement)](https://github.com/bartromb/psgscoring/wiki/Technical-Reference)**

## What this library does

`psgscoring` detects and classifies respiratory events (apneas, hypopneas, RERAs) in polysomnography recordings following AASM rules. It extends [YASA](https://github.com/raphaelvallat/yasa) (Vallat & Walker, *eLife* 2021) from sleep staging into a complete clinical respiratory scoring pipeline.

**Five things that distinguish this library:**

1. **Graded evidence instead of a threshold cascade** — the AASM Rule 1A
   conjunction is evaluated as a product of graded terms (flow reduction,
   duration, desaturation-or-arousal, breath template) rather than a chain of
   yes/no cuts. One operating point replaces three parameter combinations.
   On MESA (n=150, held out) this raises event-level agreement over the rule
   cascade by a median ΔF1 of +0.029 (p = 7·10⁻⁸) without costing anything on
   the AHI itself.
2. **The AHI as a range, not a number** — every study is scored under several
   profiles, so a severity class that depends on where the hypopnea threshold
   sits is visible instead of implied. On one split-night recording the
   diagnostic AHI ran from 83.5 to 127.1/h across two accepted rules; the
   clinical picture held, the number did not.

   The library still computes an `ahi_interval` with an A/B/C
   `robustness_grade`, but **treat that grade with care**: it assumes
   strict ≤ standard ≤ sensitive, and on PSG-IPA with a manual hypnogram
   `sensitive` gave *fewer* events than `standard` on 5 of 5 recordings. The
   names describe the intent, not the behaviour. The reference application
   stopped displaying the grade in v0.15.0 for that reason. The interval itself
   is min/max of three numbers and assumes no ordering.
3. **Clinical auditability** — every event carries a confidence score, the
   rule that admitted it, the graded terms behind that decision, and the
   counters of every correction that touched it. Rejected candidates are kept
   with their rejection reason rather than discarded, so a reviewing physician
   can see what was *not* scored and why.
4. **Split-night studies are indexed in halves** — the transition to titration
   is detected from a step in flow amplitude *and* a recovery of the SpO₂
   baseline (both required: on ordinary nights the flow ratio alone ranged 2 to
   202). Each half gets its own AHI, sleep time, reliability flag and untyped
   fraction. A single average across both halves is the one number that cannot
   answer either question the study was ordered for.
5. **Measured, not asserted** — every behavioural change ships with the
   measurement that motivated it and a decision rule fixed *before* the sweep.
   `CHANGELOG.md` carries the numbers, including the ones that argued against
   the change. Validation runs on PSG-IPA (5 recordings, 12 scorers each) and
   MESA/NSRR (held out, nothing tuned on it).

## Release policy

Releases are **measurement points, not milestones**. Several may be cut on the
same day: every behavioural change is validated against the golden harness and
at least one cohort before it ships, and the version is what makes that
validation citable. A high version number reflects how often the work is
measured, not instability.

The stability guarantee lives in the **profiles**, not in the version number:

- `mesa_shhs` and `chicago_1999` are **frozen**. They reproduce published
  results (paper v31/v37 and the 1999 Chicago Criteria respectively) and are
  pinned against behavioural changes, including repairs that every other
  profile receives.
- All other profiles track the current best understanding of the AASM rules
  and may move between releases when a measurement justifies it. Each such
  change is behind a profile field whose default preserves the previous
  behaviour, is recorded in `CHANGELOG.md` with the measurement that motivated
  it, and is pinned by a test.

If you need scored values to stay identical across time — for a study, a
regulatory submission, or a paper — **pin the version** (`psgscoring==0.19.1`)
and record which profile you used. Do not rely on a profile name alone.

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

# How much does the AHI move with the profile?
interval = results["ahi_interval"]
lo, hi = interval["interval"]        # min/max over the profiles, no ordering assumed
print(f"AHI interval: [{lo:.1f}-{hi:.1f}]")

# Split-night: index the halves separately, or one average hides the diagnosis
sn = results.get("split_night") or {}
if sn.get("detected"):
    d = sn["segments"]["diagnostic"]
    t = sn["segments"]["therapeutic"]
    print(f"before therapy: {d['ahi']}/h   on therapy: {t['ahi']}/h")
```

## Scoring Profiles

| Parameter | Strict | Standard | Sensitive |
|-----------|--------|----------|-----------|
| Hypopnea threshold | ≥30% | ≥30% | ≥25% |
| SpO₂ nadir window | 30 s | 45 s | 45 s |
| Peak-based detection | No | Yes | Yes |

`list_profiles()` enumerates the full registry — 19 profiles: AASM v1/v2/v3,
CMS/Medicare, Chicago 1999, the NSRR dataset profile, and the exploratory arms
(breath-graded, dual-sensor, and the four on the envelope axis below).

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
[`docs/interim_conclusie_klinisch_gebruik.md`](https://github.com/bartromb/psgscoring/blob/main/docs/interim_conclusie_klinisch_gebruik.md).

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

### `aasm_v3_pressure` — nasal-pressure reference (v0.14.1, opt-in)

```python
run_pneumo_analysis(raw, hypnogram, scoring_profile="aasm_v3_pressure")
```

Apnea detection is not the only consumer of a flow signal. The AHI robustness
sweep, baseline anchoring, the arousal/RERA coupling, Cheyne-Stokes detection
and the ventilatory burden each take one — and each took the *apnea* channel.
This profile scores apneas exactly as `aasm_v3_rec` does and points those five
at the nasal pressure instead.

| profile | apneas | derived analyses |
|---|---|---|
| `aasm_v3_rec` | thermistor¹ | thermistor¹ |
| `aasm_v3_pressure` | thermistor¹ | **nasal pressure** |
| `aasm_v3_dual` | **both, merged** | nasal pressure |

¹ where it passes the quality check; nasal pressure otherwise.

The AASM assigns *quantitative* flow assessment to nasal pressure and treats
the oronasal thermistor as qualitative — able to show that flow stopped, not to
grade how much of it is left. Reading a thermistor for the ventilatory burden
therefore measures something the sensor does not claim to report: on one
recording it read 20.4% instead of 42.6%, below the ≤25% reference, for a
patient spending 94.7% of the night under 90% saturation.

Identical to `aasm_v3_rec` on any montage without a usable thermistor. Set
`flow_reference` on any profile to get the same behaviour;
`meta["flow_channels"]["reference_sensor"]` reports which channel was used.

### The envelope axis — four exploratory arms (v0.19.0, all off)

Every threshold in the library is applied to an amplitude envelope, and until
now there was only one way to build it: the analytic signal over the whole
recording. That transform is also the library's largest allocation. On a 12 h
recording at 256 Hz, one `preprocess_flow` call costs **592 MB of peak RSS and
34 s**, which is what makes a parallel sweep swap.

`envelope_method` and `envelope_fs` make the choice explicit. Measured on
`mesa-sleep-0001.edf` with `/usr/bin/time -v`, peak above a 701 MB load-only
floor:

| profile | `envelope_method` | peak | time | what changes |
|---|---|---|---|---|
| `aasm_v3_rec` | `hilbert` | 592 MB | 33.7 s | reference — every published result |
| `aasm_v3_env_chunked` | `hilbert_chunked` | 116 MB | 0.8 s | same transform, 30-min blocks |
| `aasm_v3_env_rectify` | `rectify_lowpass` | 255 MB | 0.9 s | AM demodulation |
| `aasm_v3_env_breath` | `breath_amplitude` | 92 MB | 1.1 s | per-breath amplitude, interpolated |
| `aasm_v3_env_decimated` | `hilbert`, `envelope_fs=10` | 152 MB | 1.1 s | decimate, transform, interpolate back |

**None of these is a default, and the blockwise one is not free.** It was
planned as an implementation detail — same numbers, no profile field — on the
reasoning that a generous overlap reproduces the full transform up to
numerical noise. It does not: the 1/(πt) kernel has no compact support, so the
interior residual floors around 1e-4 of the p95 envelope and does *not*
converge as the overlap grows, while the first and last samples differ by ~30 %
because an FFT wraps circularly. Far below any clinical threshold, but the
envelope is compared to a baseline sample by sample, so it moves boundaries.

The golden harness cannot referee this axis — its cases are shorter than one
block, so chunking never engages and every case passes regardless. The tests
that do use 8 h signals; see `tests/test_envelope_methods.py`.

Measured against human scoring on PSG-IPA (n = 5, twelve scorers each, anchor
in the same run — bias +1.69/h, MAE 1.76/h, F1 0.462):

| arm | F1 | bias | MAE | verdict |
|---|---|---|---|---|
| `aasm_v3_env_chunked` | 0.462 | +1.69 | 1.76 | identical to the anchor; 116 of 8350 boundaries move, each by one 0.1 s step |
| `aasm_v3_env_rectify` | 0.487 | +0.01 | 1.72 | better on nearly every axis — and still off, see below |
| `aasm_v3_env_breath` | 0.414 | +3.19 | 4.11 | fails; events lost to the IoU threshold go 63 → 205 |
| `aasm_v3_env_decimated` | 0.463 | +1.71 | 1.78 | within noise, one extra event |

And on MESA (n = 150, held out, paired per recording, anchor F1 0.438 / bias
−5.30 in the same job):

| arm | F1 med | bias | paired ΔF1 | p | verdict |
|---|---|---|---|---|---|
| `aasm_v3_env_chunked` | 0.436 | −5.29 | +0.0000 | 0.70 | equal on 129/150 recordings |
| `aasm_v3_env_rectify` | 0.416 | −5.67 | −0.0015 | 0.10 | the PSG-IPA gain did not replicate |
| `aasm_v3_env_breath` | 0.413 | −5.78 | −0.0206 | 2.4·10⁻⁶ | rejected on both cohorts |
| `aasm_v3_env_decimated` | 0.438 | −5.30 | +0.0000 | 0.91 | equal on 111/150 recordings |

`rectify_lowpass` is why the decision rule was fixed *before* the measurement.
It was the best arm on PSG-IPA on nearly every axis; on 150 held-out recordings
the sign flips and the advantage is gone. Five recordings had described those
five recordings. Had the rule been written afterwards, it would have shipped.

`hilbert_chunked` now meets its promotion criterion on both cohorts — same
bias, same mean F1, identical severity confusion matrix — at 592 MB → 116 MB
and 33.7 s → 0.8 s. It is still not promoted: that is a decision about the
published numbers (the MESA F1 median moves 0.438 → 0.436), not about the code.
The rule and the full numbers, including boundary offsets against the
human-versus-human distribution, are in the CHANGELOG.

## Validation

**PSG-IPA** (PhysioNet): 5 recordings, 59 independent scorer sessions.
Against the scorer median, `aasm_v3_rec` gives bias +1.69/h, MAE 1.76/h,
Pearson r = 0.997, weighted κ = 0.839, severity concordance 4/5. Note the
sample size: five recordings is enough to expose a defect, not enough to
estimate a population.

**MESA** (NSRR, external cohort, n = 150): nothing is tuned on it. Against a
reconstructed AASM-2015 reference, per profile:

| profile | median F1 | precision | recall | bias (/h) | MAE (/h) |
|---|---|---|---|---|---|
| `aasm_v3_rec` | 0.438 | 0.519 | 0.408 | −5.30 | 10.12 |
| `aasm_v3_breath` | 0.510 | 0.628 | 0.472 | −5.18 | 9.51 |
| `aasm_v3_prob` | 0.513 | 0.670 | 0.462 | −6.41 | 9.97 |
| `aasm_v3_dual` | 0.446 | 0.498 | 0.490 | −1.17 | 9.24 |
| `aasm_v3_breath_dual` | 0.504 | 0.577 | 0.511 | −2.34 | 9.37 |

Paired against `aasm_v3_rec`, breath-graded scoring raises event agreement by
a median ΔF1 of **+0.029** (95/150 recordings, p = 6.8·10⁻⁸); the probabilistic
variant by +0.036 (p = 1.4·10⁻⁹).

**These numbers moved substantially in 0.17.0**, and the reason is worth
stating plainly. The earlier figures on this cohort reported a bias of −11 to
−15/h and were read as under-detection. Most of it was not: a signal-quality
gate was rejecting effort channels by how the EDF declared its unit rather
than by the signal in them, and it failed 52 of 52 MESA recordings. Repairing
it halved the bias at **identical** F1, precision and recall on three of the
five profiles — the events had been found all along and were being discarded
in the accounting. See `CHANGELOG.md` 0.16.0 and `docs/`.

**And then the same mistake happened twice more**, which is why it is worth
recording rather than quietly fixing. The scale-free replacement counted
*consecutive identical samples* as evidence of a detached belt. At 250 Hz a
slow breathing signal moves less than one quantiser step between neighbours,
so a belt with 88% of its power in the breathing band scored 0.686 against a
failure threshold of 0.50 (0.29.0). Three times a gate meant to measure a
sensor measured the file instead.

The fix is a test, not a threshold: one synthetic breathing signal is rendered
at five sampling rates × three quantiser steps × two amplitudes, and every gate
statistic must return the same value across all thirty renderings. All seven
pass (largest spread 0.054); the unrepaired flatness statistic does not
(0.30.0). *Scale-free* and *file-invariant* turn out not to be the same
property.

## Bias corrections

The twelve below were the original systematic pass: six over-counting and six
under-counting mechanisms, each identified and quantified separately. The list
has grown since and is no longer the whole story — `CHANGELOG.md` carries the
current state, with the measurement behind every entry. The most consequential
recent one: a sensor-quality gate that judged how the EDF declared its unit
rather than the sensor itself.


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

**Since then** (see `CHANGELOG.md` for the measurements):

| Correction | Direction | What it was |
|---|---|---|
| Square-root linearisation applied per profile, not per montage | Over-counting | The AASM Rule 3 linearisation ran only when the hypopnoea and apnoea channels differed, so the same profile pre-processed one-channel and two-channel montages differently |
| Scale-free RIP quality gate | Under-counting | An absolute amplitude threshold rejected effort channels by how the EDF declared its unit; on MESA it failed 52 of 52 recordings, halving the reported AHI bias once repaired |
| Stability filter across all hypopnoea subtypes | Over-counting | The filter compared the event type exactly against `"hypopnea"`, so every subtyped hypopnoea escaped it |
| Desaturation re-use limit | Over-counting | One desaturation could confirm any number of adjacent events (opt-in, off by default) |
| Breath-granular event boundaries | Neither | The rule cascade starts hypopnoeas 2–5 s before human scorers; snapping to breath edges removes that offset (opt-in, off by default) |
| Coherence-based thermistor gate | Neither | The gate deciding which sensor carries apnoeas correlated amplitude *envelopes*, while the question is whether both sensors see the same breaths — a timing question (opt-in, off by default) |
| Long-event splitting | Neither | Splits events beyond a duration ceiling (opt-in, off by default) |
| Low-baseline desaturation relaxation | Neither | Relaxes the 3% desaturation requirement when the local baseline is already hypoxic (opt-in, off by default) |

**Four of these are off, and that is a result, not an omission.** Each was
implemented, measured, and left disabled because the measurement did not
support enabling it: the desaturation limit removed 8 events in 4303; breath
boundaries bought recall no precision; long-event splitting duplicated an
existing splitter and found no hypopnoea candidates; the coherence gate is a
real repair of a real defect but costs 2.95/h of AHI bias on MESA when
switched on. They ship as switches so the finding stays reproducible and the
next cohort can overturn it. `docs/third_party_comparison.md` records one row
per idea evaluated, including the ones rejected.

## Architecture

~15,000 lines across 21 submodules, 868 unit tests (CI: Python 3.9–3.12):

`constants` · `utils` · `signal` · `breath` · `breath_scoring` · `classify` · `spo2` · `plm` · `ancillary` · `arousal` · `respiratory` · `indices` · `ventilation` · `pipeline` · `ml_classifier` · `profiles` · `postprocess` · `signal_quality` · `signal_quality_channels` · `ecg_effort` · `_types`

Behaviour is pinned by a golden-output harness (`PSGSCORING_GOLDEN=1`) that
scores fixed synthetic recordings and compares a digest. Every release runs it
before publishing; a difference is a stop, not a re-bless.

## Related

- **[YASAFlaskified](https://github.com/bartromb/YASAFlaskified)** — web platform integrating psgscoring with YASA staging, multilingual PDF reports, EDF+ export, and FHIR R4
- **[YASA](https://github.com/raphaelvallat/yasa)** — AI-based sleep staging (Vallat & Walker, *eLife* 2021)
- **[slaapkliniek.be](https://slaapkliniek.be)** — live instance (no installation required)

## Citation

```bibtex
@article{rombaut2026psgscoring,
  title     = {Graded evidence in place of thresholds: an open-source,
               {AASM}-compliant method for respiratory event detection in
               polysomnography},
  author    = {Rombaut, Bart and Rombaut, Briek and Rombaut, Cedric},
  year      = {2026},
  note      = {Manuscript in preparation}
}
```

## Disclaimer

**psgscoring is research software — not a medical device.** It is not CE-marked (MDR 2017/745) or FDA-cleared. All outputs are research-grade estimates that must be reviewed by a qualified clinician before any diagnostic or therapeutic decision. See **[DISCLAIMER.md](https://github.com/bartromb/psgscoring/blob/main/DISCLAIMER.md)** for the full text.

## License

BSD-3-Clause. See [LICENSE](https://github.com/bartromb/psgscoring/blob/main/LICENSE).

---

*Contact: bart.rombaut@gmail.com*
