# YASAFlaskified & psgscoring — Developer Handbook

**Last updated:** June 2026 · **Versions:** psgscoring **v0.7.2** (PyPI + GitHub), YASAFlaskified **v0.12.4** (GitHub + Hetzner production + test VM)

---

## 1. Project Overview

### What is what

| Component | Description | License |
|---|---|---|
| **psgscoring** | Standalone Python library for AASM-compliant respiratory event scoring | BSD-3 |
| **YASAFlaskified** | Docker-based web platform for complete PSG analysis (wraps psgscoring + YASA) | BSD-3 |
| **YASA** | Sleep-staging library by Vallat & Walker (transitive dependency, not ours) | BSD-3 |

`psgscoring` is published on PyPI and **installed from PyPI by YASAFlaskified** via
`requirements.txt` (`psgscoring[ml]==0.7.2`) — it is **no longer bundled** as a copied
source tree (that was the pre-v0.10 model). The `[ml]` extra adds `lightgbm` for the
optional candidate-classifier used by the `mesa_shhs` profile.

### Authors & roles

| Person | Role |
|---|---|
| Bart Rombaut, MD | Principal developer, pulmonologist, Slaapkliniek AZORG Aalst |
| Briek Rombaut | Co-developer (MSc Computer Science Eng., UGent — graduated) |
| Cedric Rombaut | Co-developer (BSc Electrical Engineering, UGent — graduated) |
| Raphaël Vallat, PhD | Scientific advisor + **co-author**, YASA creator (UC Berkeley) |
| Remington Mallett, PhD | Scientific advisor + **co-author**, YASA co-maintainer |

Vallat reviewed the manuscript and accepted co-authorship; Mallett likewise. Paper
target: *Physiological Measurement* (the earlier JCSM/JSR targets were retired).

---

## 2. Directory Structure

### Working tree (this workstation — `/home/bart/CODE/`)

```
/home/bart/CODE/
├── psgscoring/                  # psgscoring library repo (git, GitHub bartromb/psgscoring)
│   ├── psgscoring/              # 18 submodules
│   │   ├── __init__.py          # __version__ = "0.7.2"
│   │   ├── respiratory.py       # Main scoring engine (~1,700 lines)
│   │   ├── pipeline.py          # run_pneumo_analysis() master function (MNE-facing)
│   │   ├── signal.py            # linearisation, baseline, MMSD, Hilbert envelope
│   │   ├── breath.py            # breath segmentation, flattening
│   │   ├── classify.py          # apnea-type classification (obstructive/central/mixed)
│   │   ├── spo2.py              # SpO₂ coupling, ODI, hypoxic burden
│   │   ├── plm.py               # PLM detection
│   │   ├── ancillary.py         # HR, snore, position, Cheyne-Stokes
│   │   ├── ecg_effort.py        # ECG-derived effort (TECG, spectral classifier)
│   │   ├── postprocess.py       # CSR reclassification, mixed decomp, CII
│   │   ├── ml_classifier.py     # LightGBM candidate re-classifier (mesa_shhs)
│   │   ├── profiles.py          # profile registry + legacy-alias resolution
│   │   ├── constants.py         # AASM thresholds + SCORING_PROFILES
│   │   └── utils.py             # sleep mask, channel detection
│   ├── tests/                   # 115 tests (incl. golden harness, gated by PSGSCORING_GOLDEN)
│   ├── scripts/golden_snapshot.py
│   ├── docs/developer_handbook.md   # ← this file
│   ├── docs/performance_policy.md   # measured profile; why Rust/Polars/GPU/mmap are out
│   ├── scripts/thermal_guard.sh     # stops a job before the chassis does (see policy §1)
│   ├── .github/workflows/       # tests.yml (3.9–3.12 + golden job), publish.yml (OIDC)
│   ├── pyproject.toml · README.md · CHANGELOG.md · DISCLAIMER.md
│
├── YASAFlaskified/              # web platform repo (git, GitHub bartromb/YASAFlaskified)
│   ├── myproject/
│   │   ├── app.py               # Flask routes
│   │   ├── tasks.py             # RQ async workers (calls psgscoring.run_pneumo_analysis)
│   │   ├── pneumo_analysis.py   # thin bridge → psgscoring (pip-installed)
│   │   ├── generate_pdf_report.py · generate_psg_report.py · generate_excel_report.py
│   │   ├── i18n.py              # translations (NL/FR/EN/DE)
│   │   ├── version.py           # __version__ + PSGSCORING_VERSION
│   │   └── templates/ · static/
│   ├── .github/workflows/ci.yml # lint (ruff) + pytest + docker build
│   ├── deploy.sh · redeploy.sh  # bootstrap + purge/reinstall scripts
│   ├── DEPLOY_RUNBOOK.md        # ← authoritative deploy procedure (read this first)
│   ├── docker-compose.yml · Dockerfile · requirements.txt
│   └── README.md · CHANGES.md · HETZNER_CURRENT_STATE.md
│
└── docs/                        # paper / supplement / cover letter / draaiboeken
    ├── YASAFlaskified_Paper_v37_PhysiolMeas.tex
    ├── YASAFlaskified_Supplement_v37_PhysiolMeas.tex
    ├── cover_letter_v37_PhysiolMeas.tex
    ├── draaiboek_multicenter_aasm3_validatie.md
    └── psgscoring_pitch_v0.12.4*.pptx

External (not in the repos): MESA harness `/home/bart/MESA-ab-test/`,
PSG-IPA validators `/home/bart/psgscoring-ab-test/`, datasets `/home/bart/MESA`,
`/home/bart/PSG-IPA`, `/home/bart/SHHS`. See the `reference_paths` memory.
```

### Hetzner production server

`/data/slaapkliniek/` — YASAFlaskified deployment (app + 8 RQ workers + Redis,
host nginx terminates TLS → app on `127.0.0.1:8071`). **It is NOT a git checkout**
(rsync-deployed). SSH `root@65.108.230.243` (alias `dedodedodo.be`). Public:
https://slaapkliniek.be.

### Test VM (VirtualBox on the workstation)

`bart@192.168.1.253` (`us01`), Ubuntu Server. Here `/data/slaapkliniek` **IS** a git
checkout owned by `bart` (in the docker group) → sudo-free redeploy. Disposable test
instance. See the `reference_paths` memory.

---

## 3. Version Management

### Version strings — single source of truth

**psgscoring** — both must match (PyPI uses `pyproject.toml`):
```
pyproject.toml          → version = "0.7.2"
psgscoring/__init__.py  → __version__ = "0.7.2"
```
CI (`tests.yml`) enforces that these agree.

**YASAFlaskified:**
```
myproject/version.py    → __version__         = "0.12.4"
                        → PSGSCORING_VERSION   = "0.7.2"   # the PyPI pin in requirements.txt
requirements.txt        → psgscoring[ml]==0.7.2
```
`deploy.sh` syncs `APP_VERSION` in `.env` to `version.py` on every run, so an
out-of-date image tag no longer slips through (fixed 2026-06-07).

### Version bump checklist

When bumping **psgscoring**: `pyproject.toml` + `psgscoring/__init__.py` (must match),
`CHANGELOG.md`, `README.md` if needed → branch → PR → CI green → merge → GitHub Release
(triggers OIDC PyPI publish). The README's static badges may need bumping.

When bumping **YASAFlaskified**: `myproject/version.py` (`__version__` and, if the pin
changed, `PSGSCORING_VERSION` + `requirements.txt`), `CHANGES.md` → branch → PR → CI
green → merge → **GitHub Release + bump the static release badge in `README.md`**
(the badge is static because the dynamic `github/v/release` shields.io endpoint
intermittently fails with "Unable to select next GitHub token from pool").

### Common versioning pitfalls

- **PyPI is immutable:** a version (even a failed partial upload) can never be
  re-uploaded. Bump the patch and retry.
- **Docker cache:** `docker compose restart` does NOT apply Python changes — Python
  files are `COPY`'d at build time. Always `docker compose build` (+ clear `__pycache__`).
- **`APP_VERSION` drift:** must equal `version.py` or compose builds/starts the OLD
  image tag (handled by `deploy.sh`; manual deploys must `sed` it — see the runbook).

---

## 4. PyPI Publishing (psgscoring) — OIDC trusted publishing

Publishing is **automatic via a GitHub Release** — no API tokens, no manual `twine`.

```bash
# on main, after the version-bump PR is merged and CI is green:
gh release create v0.7.2 --target main --title "..." --notes "..."   # notes from CHANGELOG.md
```
The `release: published` event triggers `.github/workflows/publish.yml`, which builds
the sdist+wheel and uploads to PyPI via OIDC trusted publishing (the `pypi`
environment is pre-registered on PyPI). Verify: `pip index versions psgscoring`.

PyPI's rendered README/description only updates on a **new release** (it lags GitHub).

### Known issues
- `build-backend = "setuptools.build_meta"` (never the non-existent
  `setuptools.backends._legacy:_Backend`).
- A packaging venv must live **outside** the source tree, or Python's stdlib `signal`
  gets shadowed by `psgscoring/signal.py` during the build.

---

## 5. GitHub Workflow (both repos are normal git repos)

Standard git flow — **no more ZIP-upload dance**:

```bash
cd /home/bart/CODE/psgscoring          # or YASAFlaskified
git checkout -b my/branch
# ... edits ...
git commit -m "..."                    # NO Claude/AI attribution in messages or PRs
git push -u origin my/branch
gh pr create --base main --title "..." --body "..."
# wait for CI green, then:
gh pr merge <n> --merge
```

CI:
- **psgscoring** `tests.yml`: matrix Python 3.9–3.12 (pytest + version-consistency
  check) + a dedicated **`golden`** job (`PSGSCORING_GOLDEN=1`, freezes the
  paper-relevant scoring digest; `FLOAT_ATOL=0.25`). `publish.yml` on release.
- **YASAFlaskified** `ci.yml`: ruff lint + pytest + docker build smoke-test.
- GitHub Actions were bumped to Node 24 (`checkout@v5`, `setup-python@v6`) in 2026-06.

The GitHub **wiki** (`psgscoring.wiki`) is a separate repo on branch `master`.

---

## 6. Deployment

**Read `YASAFlaskified/DEPLOY_RUNBOOK.md` — it is the authoritative procedure.**
Summary of the three paths:

| Target | App dir | Method | Auth |
|---|---|---|---|
| **Production** (Hetzner, slaapkliniek.be) | `/data/slaapkliniek` (NOT a git checkout) | **rsync** code → bump `.env APP_VERSION` → clear `__pycache__` → `docker compose build` → `up -d` → md5-verify | **explicit per-command** (the auto-mode classifier enforces this) |
| **Test VM** (192.168.1.253) | `/data/slaapkliniek` (git checkout, owned by bart) | sudo-free `git fetch` + checkout/reset → bump `.env` → rebuild → `up -d` | none (throwaway) |
| **Fresh server** | created by script | `deploy.sh` (bootstrap; version-agnostic) / `redeploy.sh` (purge+reinstall, test only) | — |

Key rules: rebuild after any Python change (not `restart`); `--checksum` + exclude all
data dirs (`instance/`, `uploads/`, `processed/`, `logs/`) + `.env` on every rsync;
if the `psgscoring` pin changed it must be **live on PyPI before** the Docker build.
Infra: AMD Ryzen 9 5950X, 128 GB RAM, Docker Compose (app + 8 workers + Redis).

---

## 7. psgscoring Architecture

### Pipeline (`run_pneumo_analysis`)

Signal chain: nasal-pressure linearisation (√ Bernoulli) → bandpass (0.05–3 Hz) →
Hilbert envelope → dynamic rolling baseline (P95) → MMSD artefact validation → event
detection (apnea/hypopnea) with the bias corrections → apnea-type classification →
CSR detection → SpO₂ / hypoxic burden → post-processing (CSR reclass, mixed decomp,
CII) → PLM → ancillary (HR/snore/position). Optional LightGBM candidate re-classifier
runs last on the `mesa_shhs` profile.

### Scoring profiles (v0.7.x — canonical names)

- `aasm_v3_rec` — **default**, AASM 2023 Rule 1A (3%-or-arousal)
- `aasm_v3_strict`, `aasm_v3_sensitive` — the interval bracket
- `aasm_v2_rec`, `aasm_v1_rec` — AASM v2 / v1 conventions
- `cms_medicare` — CMS 4%-desat, no-arousal (`DESAT_OR_AROUSAL=False`)
- `mesa_shhs` — NSRR hp3u convention; ships the LightGBM re-classifier
- `chicago_1999` — SHHS-1 pre-AASM convention
- legacy aliases `strict`/`standard`/`sensitive` still resolve (deprecated)

**3-profile AHI confidence interval:** every run also scores strict/standard/sensitive
and reports an `ahi_interval` + an A/B/C robustness grade.

**v0.7.2 performance:** the interval previously re-ran the full detection 4× per
recording; a shared `_precomputed` cache now computes the profile-independent
preprocessing once → **~1.8–2.0× faster, byte-identical output** (validated golden +
MESA q7 + PSG-IPA).

**12 bias corrections:** 6 over-counting (post-apnea baseline inflation, SpO₂
cross-contamination, Cheyne-Stokes trough, borderline classification, artefact-flank,
local-baseline validation) + 6 under-counting (peak-based breath detection, SpO₂
de-blocking, extended nadir window, consecutive-breath requirement, position
auto-mapping, configurable profiles). **Always "six", never "five".**

**Dual-AHI (v0.6.2+):** `summary["ahi_total"]` (conservative) vs
`summary["ahi_incl_uncertain"]` (counts unsubtyped apneas; ≈unbiased vs scorers on
MESA). Clinical profiles produce no `uncertain` apneas, so the two coincide there.

**Optional LightGBM re-classifier:** ~32 candidate-level features, trained on MESA
q∈{5,6}, q=7 fully held out, threshold 0.65. MESA q=7 honest holdout (n=92): bias
−0.02/h, r 0.87, κ 0.50.

---

## 8. YASAFlaskified Architecture

| Layer | Technology |
|---|---|
| Web | Flask + Gunicorn (Python 3.11) |
| Async | Redis 7 + RQ (8 workers) |
| EDF I/O | MNE-Python |
| Staging | YASA 0.7 (LightGBM) |
| Respiratory | psgscoring (pip, `[ml]`) |
| PDF | ReportLab + matplotlib |
| EDF+ export | pyedflib (primary), edfio (fallback) |
| Containers | Docker Compose |

Data flow: EDF upload → `app.py` → Redis queue → `tasks.py` worker → YASA staging +
`psgscoring.run_pneumo_analysis` + arousal/PLM/SpO₂ → `generate_pdf_report.py`
(PDF + Excel + EDF+ + FHIR R4).

**PDF epoch-example panels are currently DISABLED.** `_build_epoch_examples()` /
`_plot_epoch_example()` exist in `generate_pdf_report.py` but the call site is
commented out (since v0.8.22, originally over alignment issues). The functions were
made efficient in v0.12.4 (load the EDF once instead of per example event,
byte-identical) — but because the call site is still commented out, **that change does
not affect actual report-generation time**; it only matters if the panels are
re-enabled.

---

## 9. Key Lessons Learned (bug patterns / gotchas)

| Pattern | Fix |
|---|---|
| Local vars don't reach nested functions (`sp` profile dict) | thread profile params through signatures |
| `docker compose restart` keeps old Python | always `build` + clear `__pycache__` |
| edfio 0.4.x silently drops EDF+ annotations | use pyedflib `FILETYPE_EDFPLUS` |
| `t = date.today()` shadows the `t()` i18n function → PDF crash | never name a local `t` |
| `Redis(decode_responses=True)` breaks RQ silently | leave default (bytes) |
| sklearn pinning: LightGBM/YASA models fail to load on newer sklearn | pin in `requirements.txt` |
| Paradoxical-motion check must precede effort-absent | else obstructive → central misclass |
| bcrypt `$` mangled by shell | parameterised `sqlite3`, never string-interpolate |
| PyPI packaging venv must be **outside** the project tree | else `signal.py` shadows stdlib |
| MESA worker OOM (~9 GB each on POOR recs) | cap ~12 workers on 128 GB |
| MESA missing EDFs (q7 = 99 ids, 92 on disk) | report n transparently, don't fail |
| **Mixed-sample-rate EDF partial reads ≠ full `load_data()`** | mne upsamples differently for windowed/cropped reads (MESA `Pres` diverged up to 2.6); only a full channel load reproduces the values |
| **PSG-IPA AHI is harness-sensitive** | the paper's per-PSG AHIs are not reproduced by re-running `validate_psgipa_v3.py` (different hypnogram/alignment); the scorer-median reference IS stable |
| `APP_VERSION` drift on re-deploy | `deploy.sh` now syncs it to `version.py` |

---

## 10. Validation

| Dataset | n | Role | Status |
|---|---|---|---|
| PSG-IPA | 5 rec × 12 scorers | External validation | ✅ paper v37 §3 |
| iSLEEPS (stroke) | 96 | Cross-population | ✅ |
| MESA q=7 (NSRR) | 99 (92 on disk) | LightGBM honest holdout + e2e §S7 | ✅ paper v37 §4 + §S7 |
| MESA q∈{2,3,4} | 100 | Graceful-degradation §S6 | ✅ |
| AZORG-YASA-2026-001 | ≥50 | Single-centre prospective | ⏳ protocol v7.0 dept-head approved; EC pending |
| Multicenter AASM-v3 (3 ziekenhuizen) | TBD | Possible external validation | 📝 draaiboek written; awaiting §10 decisions |
| SHHS-1 | ~500–1000 | `chicago_1999` profile | ⏳ POOR-quality robustness fixed; full run pending |

### Paper v37 — headline results

| Metric | PSG-IPA (std profile) | MESA q=7 rule-based | MESA q=7 LightGBM @0.65 |
|---|---|---|---|
| AHI bias | **+1.8/h** | +1.10/h | **−0.02/h** |
| MAE | 1.8/h | 6.06/h | 5.34/h |
| Pearson r | **0.997** | 0.80 | 0.87 |
| Weighted κ | **0.91** | 0.48 | 0.50 |
| Event-level F1 (SN3) | 0.886 @ IoU 0.20 | — | — |

The **per-PSG numbers in the paper come from the paper's specific harness** and are
not bit-reproduced by the generic `validate_psgipa_*` scripts (harness-sensitive; see
§9). MESA/LightGBM `ahi_total` vs the external script differs by a reporting
convention (the library excludes `uncertain` apneas).

---

## 11. Status & pending items (June 2026)

| Status | Item |
|---|---|
| ✅ DONE | psgscoring **v0.7.2** on PyPI + GitHub (0.6.1 crash/3.9 fixes + golden harness; 0.6.2 dual-AHI; 0.7.0 five Tier-1 fixes; 0.7.1 docs; 0.7.2 ~2× perf, byte-identical) |
| 🔁 IN PR | **v0.7.4** — output-preserving robustness/test cleanup (crash-safety for ancillary steps + ML/ECG guards, dead-code removal, +25 numeric unit tests, `py.typed`); byte-identical (PR #7) |
| 🔁 IN PR | **v0.7.5** — fix: RERA/RDI/REM-NREM AHI were wiped on Cheyne-Stokes-positive nights (`_compute_rera_rdi` now runs after the CSR summary recompute). Strictly additive: no AHI/event/SpO2 number changes (golden 6/6 with only `resp.rdi` restored on the 3 CSR cases; PSG-IPA byte-identical). MESA/SHHS empirical A/B pending data (`scripts/ab_rera_csr.py`, 56-core) |
| ✅ DONE | YASAFlaskified **v0.12.4** on GitHub, **deployed to Hetzner production + test VM** |
| ✅ DONE | MESA q=7 LightGBM holdout (n=92): bias −0.02/h, κ 0.50; cms_medicare validated vs `ahi_a0h4` |
| ✅ DONE | golden harness now runs in CI; both repos on Node 24 actions |
| ✅ DONE | Vallat + Mallett confirmed co-authors; paper retargeted to *Physiological Measurement* |
| ⏳ OPEN | Paper **v37** pre-submission review with Vallat/Remy → submit |
| ⏳ OPEN | EC submission AZORG-YASA-2026-001 (protocol v7.0 approved) |
| ⏳ OPEN | Multicenter AASM-v3 validation — §10 decisions (draaiboek ready) |
| 📌 FUTURE | improve 4%-desat hypopnea sensitivity (prerequisite for a CMS result) |
| 📌 FUTURE | perf #2: downsample flow/effort before the Hilbert envelope (changes numerics → needs re-validation) |
| 📌 FUTURE | re-enable / fix PDF epoch-example panels (functions optimised, call site still commented) |
| 📌 FUTURE | SHHS-1 full validation; arousal-detector improvement (§S7 e2e variance locus); hybrid CNN+GBM |

---

## 12. Quick reference

```bash
# versions
grep 'version = ' /home/bart/CODE/psgscoring/pyproject.toml
grep '__version__' /home/bart/CODE/psgscoring/psgscoring/__init__.py
grep -E '__version__|PSGSCORING' /home/bart/CODE/YASAFlaskified/myproject/version.py
pip index versions psgscoring                      # expect 0.7.2

# psgscoring tests + golden
cd /home/bart/CODE/psgscoring && .venv/bin/python -m pytest -q
PSGSCORING_GOLDEN=1 .venv/bin/python -m pytest tests/test_golden_output.py -q

# Hetzner production version (read-only)
ssh root@65.108.230.243 "cd /data/slaapkliniek && docker compose exec -T app \
  python -c 'import version,psgscoring; print(version.__version__, psgscoring.__version__)'"
```

Release cycle: see §3 (bump → PR → CI), §4 (psgscoring PyPI via GitHub Release/OIDC),
§5 (git/PR), §6 + `DEPLOY_RUNBOOK.md` (rsync to Hetzner / git on the test VM).

---

## 13. Starting a new chat session

Provide this document + the current versions (psgscoring v0.7.2, YASAFlaskified
v0.12.4) + what you're working on. Key standing context:

- Paper target: *Physiological Measurement* (v37, pre-submission review).
- **Vallat and Mallett are confirmed co-authors.**
- AASM version is **not** pinned to "2.6" anywhere — the library supports v1/v2/v3 via
  profiles (default `aasm_v3_rec`); say "AASM" or the specific profile, not "AASM 2.6".
- Six over-counting corrections (never "five").
- psgscoring is **pip-installed** in YASAFlaskified (not bundled).
- Production = rsync (NOT a git checkout); requires explicit per-command authorization;
  PHI never in git/PyPI/logs/chat.
- Docker: always `build`, never `restart`, for Python changes.
- No Claude/AI attribution in commit messages or PR bodies.
- Memory location: `/home/bart/.claude/projects/-home-bart-CODE/memory/`.
