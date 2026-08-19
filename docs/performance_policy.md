# Performance policy — measurements first, rewrites last

Purpose: generic performance advice — Rust, Polars, GPU, columnar storage —
resurfaces every few months from reviewers, collaborators and LLMs. This
document records what has been measured, what was rejected because of those
measurements, and the specific condition under which each decision gets
revisited. It ends the recurring conversation.

The governing rule is the same as everywhere else in this project: **first the
number, then the conclusion.** No rewrite without a profile that demands it.

That rule applies to this document too. An earlier draft named staging
inference as the dominant term and put the EDF load at "10–30 s, ~700 MB".
Measuring it moved the dominant term by two orders of magnitude and inverted
one of the rejections below. The provenance column exists so that the next
reader can tell which rows have been through that and which have not.

---

## 1. What the profile actually says

Measured on `mesa-sleep-0001.edf` — 12.00 h, 27 signals, `/usr/bin/time -v`,
single-threaded (`OMP_NUM_THREADS=1`), Z6 G4, v0.19.x.

| stage | cost | peak RSS | runs | provenance |
|---|---|---|---|---|
| EDF header only (`preload=False`) | 0.19 s | — | once | measured 18-08-2026 |
| **EDF load, all 27 channels (`preload=True`)** | **175 s** | **5.09 GiB** | once | measured 18-08-2026 |
| EDF load, 3 channels (`exclude=` + `preload=True`) | 0.5 s | — | once | measured 18-08-2026 |
| YASA feature extraction (1439 epochs × 149 features) | 2.8 s | — | once | measured 18-08-2026 |
| YASA `clf.predict` (LightGBM, joblib) | 0.20 s | — | once | measured 18-08-2026 |
| envelope, full-night Hilbert (historical default) | 33.7 s | 592 MB above floor | per envelope method | CHANGELOG v0.19.0 |
| envelope, `hilbert_chunked` | 0.8 s | 116 MB above floor | per envelope method | CHANGELOG v0.19.0 |
| respiratory chain per profile | 30–125 s | — | per profile | **unverified — no recorded timing** |
| full `run_pneumo_analysis` | — | 6.07 GiB | once | working notes 16-08-2026 |
| AASM rule logic / corrections | milliseconds | — | per profile | pure Python, ~10²–10³ candidate events |

Four conclusions that every proposal below must survive:

1. **The dominant term is the EDF load, not staging.** 175 s against 3.0 s for
   the entire staging step — a factor of 58. Whether it also dominates the
   respiratory chain cannot be stated yet: that row is the unverified one. What
   can be stated is that the load is the single largest measured term in the
   run, and that no proposal below touches it.
2. **Staging is not a neural network and there is no inference term to
   offload.** YASA 0.7.0 loads a pre-trained `LGBMClassifier` from joblib;
   `torch` is not installed in any venv on this machine. Of the 3.0 s, 2.8 s is
   numpy/scipy/antropy feature extraction and 0.20 s is the classifier —
   6.8 % of staging, ~1.7 % of the run.
3. **The load is slow for a fixable reason.** The recording carries three
   native rates (10 channels at 1 Hz, 8 at 32 Hz, 9 at 256 Hz — 111 M samples,
   0.83 GiB as float64). MNE harmonises everything to the highest rate on
   load: 298.6 M samples, 2.22 GiB, a 2.7× inflation, with the peak at
   5.09 GiB for the intermediate copies. The 175 s is that resampling, not
   disk I/O (`File system inputs: 19168` blocks — the file was in page cache).
   Restricting to the three channels staging needs takes the load from 175 s
   to 0.5 s, because excluded channels are never read *or* resampled.
4. **The rule logic iterates over hundreds of candidate events, not millions
   of samples.** There is no hot Python loop at sample level; the sample-level
   work already lives in scipy/numpy C code.

The one large win to date came from an **algorithm choice**, not a language or
a device: chunked Hilbert took the envelope from 33.7 s / 592 MB to 0.8 s /
116 MB (v0.19.0) — and even that turned out not to be free (non-converging
kernel tail), which is why it shipped as a profile axis rather than a silent
swap.

### The freezes of 16-08-2026 were thermal, not memory

Recorded here because the memory explanation is the one that keeps coming
back, and a performance document is where it will be looked up.

At the second freeze, 57.8 GiB was in use with 62.1 GiB free and swap
untouched; `sensors` reported the package at 83 °C against `crit = 84 °C`.
Neither freeze left a single kernel log line, which fits a thermal hardlock —
there is no time to flush. The narrative is in the CHANGELOG under v0.19.0; the
fix is operational: 6 workers instead of 12, `CPUQuota=600%`, single-threaded
BLAS, and `scripts/thermal_guard.sh`, which writes every sample synced to disk
so the next hardlock leaves evidence.

Note also that the load-only peak measured above (5.09 GiB) is 84 % of the
6.07 GiB peak of a full analysis. The envelope axis was chosen partly to
relieve swap pressure; most of that pressure was never the envelope.

Two corrections to the folklore:

- **`earlyoom` was never installed and was not the fix.** Verified three
  times: no binary, no dpkg entry, unit `inactive`. Nor would it have fired,
  with 62.1 GiB free. For a genuine memory ceiling on a bounded job, a cgroup
  limit (`systemd-run --user -p MemoryMax=`) is the better instrument: it
  kills the job rather than guessing at the machine.
- The 15.3 GiB swap is a pre-existing NVMe partition, not a remediation.

Two notes on the "28 workers × ~3 GiB" figure that circulates with the memory
explanation. The workers were the sweep harness's `--workers`, not pytest:
`pytest-xdist` is not installed in this repo's venv and `-n auto` appears in no
config or workflow, so there is no auto-scaled test run to blame. And the
arithmetic does not reach a ceiling anyway — ~84 GiB against 125 GiB of RAM
does not exhaust memory, and indeed it did not.

---

## 2. Accepted

| tool | status | why |
|---|---|---|
| `/usr/bin/time -v` | **in use** | max RSS per run is the standard evidence line in CHANGELOG entries (v0.19.0, and every row above) |
| py-spy | **declared** (`[dev]` extra) | sampling profiler, ~zero overhead, safe to attach to a production RQ worker (`py-spy dump --pid`, `py-spy record -o flame.svg`) |
| Scalene | **declared** (`[dev]` extra) | line-level memory + Python/C split; the interesting outcome would be *unexpected* pure-Python time, since the profile says there should be almost none |
| worker cap (sweep harness `--workers`) | **in use, 6 workers** | a thermal cap, not a memory cap, and not 8: 10–12 workers drive the package to 83 °C. At 6 with `CPUQuota=600%`: mean 63.3 °C, peak 78, 6 of 2670 samples above 74 |
| selective EDF load (header first, then `exclude=`) | **in use in YF only** | `YASAFlaskified myproject/tasks.py:88–101` reads the header, computes `to_exclude`, then preloads. 175 s → 0.5 s on the staging load. **Not applied in the validation scripts** — see §4 |
| GPU | **no role** | superseded. The earlier draft permitted the GPU "for staging inference"; the measurement above shows inference is 0.20 s, ~0.1 % of the run. There is no term to accelerate. See §3-CuPy for the respiratory chain |

---

## 3. Rejected — with the measurement and the revisit condition

### Rust / PyO3 ("rewrite the heaviest AASM rules")

**Rejected because:** the premise is false for this codebase. Rule logic runs
per candidate event (~10²–10³/night) and costs milliseconds; the heavy paths
(Hilbert, filtering, baseline) are already C via scipy. A Rust extension would
optimise the cheapest part of the pipeline while adding a second language, a
build matrix, per-platform wheels, and a weaker audit story ("the whole chain
is readable Python a reviewer can walk through") to a bus-factor-one project.

**Revisit if:** a profiler shows a pure-Python function consuming >10 % of a
scoring run — and then the first answer is `@numba.njit` on that one function
(same file, same language, no build step, removable), not PyO3.

### Polars ("replace pandas")

**Rejected because:** the scoring path barely uses pandas — it is numpy arrays
plus event objects. Lazy evaluation and column pruning optimise tabular query
pipelines; a 1-D signal through a filter chain is not a query. The advice
addresses CAISR's architecture (per-sample state columns in a DataFrame), not
ours.

**Revisit if:** the phase-2 sweep candidate table reaches millions of rows with
repeated filter/aggregate passes. First answer even then: vectorised numpy over
one table (the design already in the sweep brief).

### CuPy ("swap numpy for the GPU")

**Rejected because**, in ascending order of weight:

1. The chain runs through `scipy.signal` and MNE, which do not accept CuPy
   arrays; `cupyx.scipy.signal` is not covering, and PCIe transfers eat the
   gains on chains of short ops.
2. The available card is a **Quadro P2000 (Pascal GP106, CC 6.1, 5 GiB,
   driver 555.58.02)** — an earlier draft named a Tesla M10, which is not in
   this machine. The conclusion survives the correction: GP106 does float64 at
   1/32 of float32 throughput, and the entire chain computes in float64.
   Moving to float32 is a behaviour change on every scored value. The 5 GiB of
   VRAM is a second, independent ceiling against a 5.09 GiB host-side peak.
3. **Decisive:** GPU floating point is not guaranteed bit-identical to CPU
   results, and can vary across driver versions. The golden harness, the
   byte-identity tests and the pinned reproduction profiles are incompatible
   with a compute path whose output is hardware-dependent. Determinism is the
   product; it is not negotiable for speed.

**Revisit:** never for the respiratory chain, and — after the measurement in
§1 — nowhere else either.

### Parquet / Zarr ("convert incoming EDFs internally")

**Rejected because:** columnar storage pays when you read few columns of many;
a study reads essentially **all** channels, each exactly once, so there is
nothing to prune — while the conversion itself costs one full parse-and-rewrite
per recording up front. Worse, the provenance chain (channel names, **unit
declarations** — the v0.17.0 lesson was found *because* the EDF declaration was
visible unfiltered — sampling rates, EDF+ export, FHIR) hangs off the source
file; an internal transform layer is exactly where that metadata gets lost. And
EDF being "archaic" is a feature here: it is the interchange format of the
entire field (NSRR, CAISR, every vendor, every collaborator). Running
internally on the clinical standard format is an audit argument.

One caveat the §1 measurement adds: the "read all channels" premise holds for a
*study*, but not for the *staging load*, which needs three of 27. That is an
argument for `exclude=`, not for a storage format.

**Revisit if:** a batch workflow repeatedly re-reads the same large cohort
(e.g. HSP sweeps). Then a one-time Zarr **cache next to the source** is
acceptable — never a replacement for it.

### mmap ("prevents OOM on the workers")

**Rejected — but the earlier draft's premise was wrong and is worth stating,
because it pointed the search away from the dominant term.** That draft held
that "the raw data (~1 GB) was never the problem" and that the peaks were
derived data. The ~1 GiB figure is the *native* size of the recording
(0.83 GiB); it is not what lands in RAM. Measured: the full-channel load alone
peaks at **5.09 GiB and takes 175 s**, against 592 MB for the historical
Hilbert envelope. The raw load is the largest single allocation in the run, not
a floor beneath it.

The rejection of mmap itself stands, for reasons the measurement strengthens
rather than weakens:

1. What costs the 175 s and most of the 5.09 GiB is MNE's **resampling** of
   1 Hz and 32 Hz channels up to 256 Hz. That array is computed, not stored;
   there is no file to map it from.
2. The access pattern defeats lazy loading regardless: filtering and Hilbert
   read every channel integrally and repeatedly, and `filtfilt` needs the full
   channel as an array.

The right answer to this term is not mmap but **selective loading** (§2, §4) —
same one-line class of change, aimed at the term that actually dominates.

**Revisit if:** the sweep harness shares one large read-only feature table
across many workers — then `np.load(..., mmap_mode='r')` gives shared pages
instead of N copies. One line, when the measurement asks for it.

---

## 4. Open — measured, not yet applied

### Selective loading in the validation scripts

`scripts/validate_mesa.py:323` (and the other sweep scripts) do an unfiltered
`mne.io.read_raw_edf(..., preload=True)`: 27 channels, 175 s, 5.09 GiB. The
scripts use a handful of those channels. YF already does it the other way
(`YASAFlaskified myproject/tasks.py:88–101`) and gets 0.5 s. At 6 workers this is the
difference between ~31 GiB and a few hundred MB of resident set, and it would
have removed the swap pressure that the envelope axis was partly chosen to
relieve.

**Not applied, deliberately.** The harness is a study artifact, and channel
auto-detection scans `raw.ch_names`; narrowing the load can change which
channel a profile picks, which is precisely the class of silent divergence that
`mesa_shhs` byte-identity exists to catch.

**The measurement that licenses it:** run one recording both ways and diff the
full result dict — all indices to full precision, not the 1-decimal golden
digest, which rounds and would hide exactly this. Identical on a POOR-quality
and a mixed-rate recording, then apply and re-run golden.

### Per-profile wall clock

The 30–125 s figure in §1 has no recorded provenance and is the one row in the
table that has not been through the rule at the top of this document. The
instrumentation already exists on the YF side — `_meta.wall_clock_s` in
`run_profile_comparison` — and one `time.monotonic()` per iteration in the
sweep scripts would close it. Until then, treat that row as folklore.

---

## 5. How to use this document

A performance proposal that reaches this repository gets three questions:

1. **Which line of the §1 profile does it change, and by how much?** If it
   cannot name the line, it has not read the profile.
2. **What does it cost the determinism guarantees?** Anything that makes a
   scored value hardware-, thread-, or seed-dependent is rejected regardless of
   speedup.
3. **Is there a measurement?** Proposals without numbers get one standard
   answer: run py-spy or Scalene, put the result in the CHANGELOG, and come
   back with the line item.

The fastest code change in this project's history was an algorithm swap guarded
by a profile axis and a declared decision rule. That is the template. Languages,
DataFrame engines and storage formats are answers to questions this codebase
has, so far, never asked — and the largest win still on the table is three
lines of channel selection that need a byte-identity check first.
