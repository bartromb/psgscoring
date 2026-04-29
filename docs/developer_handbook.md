# YASAFlaskified & psgscoring — Developer Handbook

**Last updated:** April 2026 (v0.4.2 release) · **Versions:** YASAFlaskified v0.9.1, psgscoring v0.4.2

---

## 1. Project Overview

### What is what

| Component | Description | License |
|---|---|---|
| **YASAFlaskified** | Docker-based web platform for complete PSG analysis | BSD-3 |
| **psgscoring** | Standalone Python library for AASM 2.6 respiratory scoring | BSD-3 |
| **YASA** | Sleep staging library by Vallat & Walker (dependency, not ours) | BSD-3 |

psgscoring is **bundled inside** YASAFlaskified (not installed from PyPI at runtime). It is also published independently on PyPI for the research community.

### Authors & roles

| Person | Role |
|---|---|
| Bart Rombaut, MD | Principal developer, pulmonologist, Slaapkliniek AZORG Aalst |
| Briek Rombaut | Co-developer (MSc candidate Computer Science Eng., UGent) |
| Cedric Rombaut | Co-developer (BSc Electrical Engineering, UGent) |
| Raphaël Vallat, PhD | Scientific advisor, YASA creator. **NOT co-author on papers** |
| Remington Mallett, PhD | Second scientific advisor, YASA co-maintainer |

**CRITICAL:** Vallat is NOT a co-author on psgscoring NOR on the YASAFlaskified paper. Only reference him via YASA citations (Vallat & Walker, eLife 2021). No JCSM references (paper not yet submitted).

---

## 2. Directory Structure

### Bart's local machine

```
~/Desktop/GITHUB/
├── github_psgscoring/          # psgscoring library repo
│   ├── psgscoring/
│   │   ├── __init__.py         # __version__ = "0.2.951"
│   │   ├── respiratory.py      # Main scoring engine (~1,061 lines)
│   │   ├── pipeline.py         # run_pneumo_analysis() master function
│   │   ├── signal.py           # Linearisation, baseline, MMSD
│   │   ├── breath.py           # Breath segmentation, flattening
│   │   ├── classify.py         # 7-rule apnea type classification
│   │   ├── spo2.py             # SpO₂ coupling, ODI, hypoxic burden
│   │   ├── plm.py              # PLM detection
│   │   ├── ancillary.py        # HR, snore, position, CSR
│   │   ├── postprocess.py      # CSR reclassification, mixed decomp, CII
│   │   ├── constants.py        # AASM thresholds
│   │   └── utils.py            # Sleep mask, channel detection
│   ├── tests/
│   ├── docs/
│   │   └── handbook.pdf        # Technical handbook
│   ├── pyproject.toml          # Version, dependencies, build config
│   ├── README.md
│   ├── LICENSE
│   ├── CHANGELOG.md
│   └── DISCLAIMER.md
│
├── github_YASAFlaskified/      # Web platform repo
│   ├── myproject/
│   │   ├── app.py              # Flask routes (~2,400 lines)
│   │   ├── tasks.py            # RQ async workers
│   │   ├── generate_pdf_report.py  # ReportLab PDF (~1,800 lines)
│   │   ├── pdf_report_additions.py # v0.8.36 Medatec-parity sections
│   │   ├── i18n.py             # Translations (449+ keys, NL/FR/EN/DE)
│   │   ├── version.py          # __version__ + PSGSCORING_VERSION
│   │   ├── pneumo_analysis.py  # Bridge to psgscoring
│   │   ├── psgscoring/         # ← BUNDLED copy of the library
│   │   │   ├── __init__.py
│   │   │   ├── respiratory.py
│   │   │   └── ... (same files as standalone)
│   │   ├── templates/
│   │   │   ├── channel_select.html
│   │   │   ├── report_editor.html
│   │   │   └── ...
│   │   └── static/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── README.md
│   ├── CHANGES.md
│   └── ROADMAP.md
│
├── psgscoring.wiki/            # GitHub wiki (separate git repo!)
│   ├── Home.md
│   ├── Technical-Reference.md
│   ├── Hypoxic-Burden.md
│   ├── Post-Processing.md
│   └── _Sidebar.md
│
└── psgscoring_lgbm/            # LightGBM training (future)
    ├── config.py
    ├── feature_extraction.py
    ├── prepare_mesa.py
    ├── train_lgbm.py
    └── hybrid.py
```

### Hetzner production server (65.108.230.243)

```
/data/slaapkliniek/             # YASAFlaskified deployment
├── myproject/
│   ├── app.py
│   ├── tasks.py
│   ├── generate_pdf_report.py
│   ├── psgscoring/             # Bundled library
│   └── ...
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

Access: `ssh root@dedodedodo.be` or `ssh root@65.108.230.243`
Nginx Proxy Manager: `panel.dedodedodo.be`
Portainer: via panel
DNS: Gandi (slaapkliniek.be, longziekten.eu, dedodedodo.be, sleepai.be/eu)

---

## 3. Version Management

### Version strings — single source of truth

**psgscoring:**
```
pyproject.toml          → version = "0.4.2"
psgscoring/__init__.py  → __version__ = "0.4.2"
```
Both MUST match. PyPI uses pyproject.toml.

**YASAFlaskified:**
```
myproject/version.py    → __version__ = "0.9.1"
                        → PSGSCORING_VERSION = "0.4.2"
```
The bundled `myproject/psgscoring/__init__.py` must also match.

### Version bump checklist

When bumping psgscoring:
1. `pyproject.toml` → version
2. `psgscoring/__init__.py` → `__version__`
3. `README.md` → version badge/text
4. `CHANGELOG.md` → new entry
5. `DISCLAIMER.md` → version ref (if present)
6. **YASAFlaskified** `myproject/version.py` → `PSGSCORING_VERSION`
7. **YASAFlaskified** `myproject/psgscoring/__init__.py` → `__version__`
8. **Wiki** `Home.md` + `_Sidebar.md` → version refs

When bumping YASAFlaskified:
1. `myproject/version.py` → `__version__`
2. `README.md`
3. `CHANGES.md`
4. Header in `generate_pdf_report.py` (fancyhead equivalent)

### Common versioning pitfalls

- **UI footer mismatch:** The PDF report header and the web footer both pull from `version.py`. If you edit version in one place but not the other, the footer shows the wrong version.
- **PyPI won't accept re-uploads:** If v0.2.95 is already on PyPI, you CANNOT re-upload it. Bump to v0.2.951 or v0.2.96.
- **Docker cache:** `docker compose restart` does NOT apply Python file changes. Always use `docker compose build --no-cache`.

---

## 4. PyPI Publishing Workflow

### One-time setup

```bash
python3 -m venv ~/venv-pypi
source ~/venv-pypi/bin/activate
pip install build twine
```

### Publishing a new version

```bash
source ~/venv-pypi/bin/activate
cd /home/bart/Desktop/GITHUB/github_psgscoring

# Verify versions match
grep 'version = ' pyproject.toml
grep '__version__' psgscoring/__init__.py

# Verify build-backend is correct
grep 'build-backend' pyproject.toml
# MUST say: build-backend = "setuptools.build_meta"
# NOT: setuptools.backends._legacy:_Backend  (this will fail!)

# Clean + build + upload
rm -rf dist/ build/ *.egg-info
python3 -m build
python3 -m twine upload dist/psgscoring-*

# Verify on PyPI
pip install psgscoring==0.2.951 --dry-run
```

### Known issues

- **`setuptools.backends._legacy:_Backend`**: This build-backend does NOT exist. Always use `setuptools.build_meta`.
- **Externally managed environment error**: Use the venv (`source ~/venv-pypi/bin/activate`), do NOT use `--break-system-packages`.
- **v0.2.95 already on PyPI**: This version was published. Had to bump to v0.2.951 for citation fix.

---

## 5. GitHub Workflows

### psgscoring → GitHub

```bash
cd /home/bart/Desktop/GITHUB/github_psgscoring
git add -A
git commit -m "v0.2.951: description of changes"
git tag v0.2.951
git push origin main --tags
```

### psgscoring Wiki → GitHub

The wiki is a **separate git repository**:

```bash
cd /home/bart/Desktop/GITHUB/psgscoring.wiki
# Wiki must be initialised first via GitHub web UI (Wiki tab → Create first page)
rm -f *.md
unzip -o ~/Downloads/psgscoring_wiki_vXXX.zip
git add -A
git commit -m "vX.X.X: description"
git push origin master     # NOTE: master, not main!
```

Wiki pages: `Home.md`, `Technical-Reference.md`, `Hypoxic-Burden.md`, `Post-Processing.md`, `_Sidebar.md`

### YASAFlaskified → GitHub

```bash
cd /home/bart/Desktop/GITHUB/github_YASAFlaskified
find . -maxdepth 1 -not -name '.git' -not -name '.' -exec rm -rf {} +
unzip -o ~/Downloads/YASAFlaskified_vXXX.zip
git add -A
git commit -m "v0.8.36: description"
git tag -f v0.8.36
git push origin main --tags -f
```

---

## 6. Hetzner Deployment

### Deploy YASAFlaskified to production

**One-liner (scp + build + restart):**
```bash
scp ~/Downloads/YASAFlaskified_v0836.zip root@dedodedodo.be:/data/slaapkliniek/ && \
ssh root@dedodedodo.be "cd /data/slaapkliniek && \
  unzip -qo YASAFlaskified_v0836.zip -d . && \
  rm YASAFlaskified_v0836.zip && \
  docker compose build --no-cache app worker1 worker2 worker3 worker4 worker5 worker6 worker7 worker8 && \
  docker compose up -d && \
  docker compose exec app python -c \"from version import __version__; print(f'v{__version__} deployed')\""
```

### Key deployment rules

| Rule | Why |
|---|---|
| Always `docker compose build --no-cache` | Python files are COPY'd at build time. `restart` uses old code. |
| Never `docker cp` for Python changes | Files get overwritten on next build. |
| `docker compose up -d` after build | Restarts containers with new image. |
| Check version after deploy | `docker compose exec app python -c "from version import __version__; print(__version__)"` |

### Infrastructure

| Service | Details |
|---|---|
| Server | Hetzner, AMD Ryzen 9 5950X, 128 GB RAM |
| IP | 65.108.230.243 |
| SSH | `root@dedodedodo.be` |
| Nginx Proxy Manager | `panel.dedodedodo.be` |
| Portainer | via panel |
| Docker Compose | app + 8 RQ workers + Redis |
| Code location | `/data/slaapkliniek/` |
| SMTP | Brevo for longziekten.eu contact form |

---

## 7. psgscoring Architecture

### Pipeline stages (run_pneumo_analysis)

| Step | Module | Description |
|---|---|---|
| 1 | signal.py | Nasal pressure linearisation (√ Bernoulli) |
| 2 | signal.py | Bandpass filtering (0.05–3 Hz, Butterworth) |
| 3 | signal.py | Hilbert envelope (instantaneous amplitude) |
| 4 | signal.py | Dynamic 5-min rolling baseline (P95) |
| 5 | signal.py | MMSD artefact validation |
| 6 | respiratory.py | Event detection (apnea/hypopnea) + 6 over-counting fixes |
| 7 | respiratory.py | 6 under-counting corrections |
| 8 | classify.py | 7-rule apnea type classification + Hilbert phase |
| 9 | ancillary.py | CSR detection (autocorrelation) |
| 10 | spo2.py | Hypoxic burden (percentile + ensemble) |
| 11 | postprocess.py | CSR reclassification, mixed decomp, CII |
| 12 | plm.py | PLM detection (AASM 2.6 + WASM) |
| 13 | ancillary.py | HR, snore, position |

### Key algorithms

**12 bias corrections:**
- 6 over-counting: post-apnea baseline inflation, SpO₂ cross-contamination, Cheyne-Stokes trough, borderline classification, artefact-flank, local baseline validation
- 6 under-counting: peak-based breath detection, SpO₂ de-blocking, extended nadir window (45s), flow smoothing, position auto-mapping, configurable profiles

**3 scoring profiles:**
- Strict: 70% threshold, 30s nadir, no smoothing, no peak detection
- Standard (default): 70%, 45s nadir, 3s smoothing, peak detection ON
- Sensitive: 75% (≥25% reduction), 45s nadir, 5s smoothing, peak detection ON

**Effort classification chain:** Hilbert phase angle (Rule 0) → 6-rule decision tree → ECG-derived TECG (Berry 2019) → spectral effort classifier

---

## 8. YASAFlaskified Architecture

### Tech stack

| Layer | Technology |
|---|---|
| Web framework | Flask + Gunicorn (Python 3.11) |
| Async jobs | Redis 7 + RQ (8 parallel workers) |
| EDF I/O | MNE-Python |
| Sleep staging | YASA 0.7 + LightGBM |
| PDF generation | ReportLab |
| EDF+ export | pyedflib (primary), edfio fallback |
| Containerisation | Docker Compose |
| Translations | Custom i18n.py (449+ keys, NL/FR/EN/DE) |

### Data flow

```
User uploads EDF → app.py → Redis queue → tasks.py worker
    │
    ├── YASA sleep staging → hypnogram
    ├── psgscoring respiratory analysis → events, AHI, HB
    ├── Arousal detection → arousal index
    ├── PLM detection
    ├── SpO₂ analysis
    ├── Signal quality assessment
    │
    └── generate_pdf_report.py → PDF + Excel + EDF+ + FHIR R4
```

### PDF report sections (v0.8.36)

1. Patient info + header (ESS, indication, referring physician)
2. Sleep staging (YASA) + AASM statistics
3. Sleep architecture (cycles, latencies)
4. Respiratory indices (AHI, OAHI, position × stage crosstable)
5. SpO₂ analysis (ODI, baseline, saturation bands, hypoxic burden)
6. Arousal index
7. PLM
8. Signal quality & confidence review
9. OSAS severity score (O-S-A-S, 0–12)
10. Snoring crosstable
11. Conclusion / clinical summary

**Note:** Section 8e (epoch signal examples) is temporarily disabled (v0.8.36) due to alignment issues. Functions `_plot_epoch_example()` and `_build_epoch_examples()` remain in code (lines 494-725), commented out at the call site.

---

## 9. Key Lessons Learned (Bug Patterns)

| Pattern | Description | Fix |
|---|---|---|
| **Variable scoping** | Local vars in one function silently fail in nested functions | Pass as explicit parameters |
| **Docker file updates** | `docker cp` unreliable with Dockerfile COPY | Always `docker compose build --no-cache` |
| **bcrypt shell escaping** | `$` in hashes mangled by bash | Use Python sqlite3 with parameterised queries |
| **EDF+ export** | edfio 0.4.13 silently writes plain EDF, drops annotations | Use pyedflib with FILETYPE_EDFPLUS |
| **SpO₂ baseline** | 60s window undershoots during cluster apneas | 120s + global P95 fallback |
| **Classification order** | Paradoxical breathing must check before effort-absent | Otherwise obstructive → central misclassification |
| **Translation shadowing** | `t=date.today()` shadows `t()` translation function | Total PDF failure, rename variable |
| **Redis decode_responses** | `decode_responses=True` breaks RQ silently | RQ expects bytes |
| **Build-backend** | `setuptools.backends._legacy:_Backend` doesn't exist | Use `setuptools.build_meta` |
| **Six corrections** | "Five" was persistent error in early manuscripts | Always say "six" over-counting corrections |
| **Profile dict scope** (v0.4.2) | `sp` only exists in `detect_respiratory_events`, not in nested `_detect_hypopneas` | Thread profile params through function signatures, not via `sp.get()` in nested scope |
| **Hardcoded fallbacks** | v0.4.1 had hardcoded 0.30 in `_validate_local_reduction` that ignored profile config | Always use profile dict; check for hidden hardcoded values when refactoring |
| **Arousal index location** | `arousal_index` is in `arousal.summary`, not `respiratory.summary` | Check correct dict path |
| **ESS data flow** | 3 separate breaks: form read, whitelist, task extraction | All three must be patched together |

---

## 10. Validation Datasets

| Dataset | n | Role | Status |
|---|---|---|---|
| PSG-IPA (PhysioNet) | 5 rec, 60 sessions | External validation + benchmarking | ✅ Complete |
| iSLEEPS (stroke) | 96 patients | Cross-population validation | ✅ Complete |
| MESA (NSRR) | ~2,056 PSGs | Training (LightGBM) + large-scale validation | ⏳ DUA pending |
| AZORG (prospective) | ≥50 PSGs | Independent clinical validation | ⏳ EC approval pending |

### Key results

| Metric | PSG-IPA | iSLEEPS |
|---|---|---|
| AHI bias | +1.6/h | — |
| Correlation | r = 0.990 | — |
| Event-level F1 (severe) | 0.890 | — |
| Event-level Δt (severe) | 2.3 s | — |
| MAE (normal/mild) | — | 3.3/h |
| Severity concordance | 75% (4/5) | — |


## 10b. Validation Results — v0.4.2 (April 2026)

After the architectural refactor (profile-aware local baseline validation),
re-validation on PSG-IPA shows:

| Metric | v0.2.951 (paper v31) | v0.4.2 (current) | Note |
|---|---|---|---|
| AHI bias | +1.8/h | +2.9/h | Slight increase in over-counting |
| LoA | [-2.1, +5.7] | [-1.54, +7.35] | Wider but symmetric |
| Pearson r | 0.997 | 0.994 | Negligible change |
| Weighted κ | 0.91 | 0.800 | Above 0.7 protocol target |
| F1 SN3 | 0.886 | 0.860 | Comparable |
| Mean Δt SN3 | 2.0 s | **1.39 s** | **Improved** |
| Severity concordance | 5/5 | 4/5 | SN2 now Mild (was Normal) |

Note: SN2 is a borderline recording where the 12 PSG-IPA scorers
themselves are split (8 Normal, 4 Mild). The v0.4.2 classification
of Mild is within the scorer range.

The slight κ decrease reflects the architectural shift from a
hardcoded heuristic to profile-driven parameters. The architecture
is now consistent and generalizable; future calibration on MESA
should improve performance on diverse populations.

---

## 11. Pending Items

| Priority | Item | Status |
|---|---|---|
| ✅ DONE | Deploy v0.9.1 to Hetzner | Done 2026-04-29 |
| ✅ DONE | psgscoring v0.4.2 architectural refactor | Done 2026-04-29 |
| ✅ DONE | YASAFlaskified PDF blank page fix | Done 2026-04-29 |
| HIGH | psgscoring v0.4.2 on PyPI | Run scripts/05_pypi_psgscoring.sh |
| HIGH | YASAFlaskified v0.9.1 on GitHub | Run scripts/04_github_yasaflaskified.sh |
| HIGH | psgscoring v0.4.2 on GitHub | Run scripts/03_github_psgscoring.sh |
| HIGH | Wiki v0.4.2 push | Apply wiki diff after GitHub deploy |
| MEDIUM | MESA DUA approval | Waiting on NSRR |
| MEDIUM | MESA download + validation | DUA |
| MEDIUM | EC submission AZORG | Piet Vercauter approved |
| LOW | Mail Didier Pevernagie | Draft ready |
| LOW | LightGBM training on MESA | After validation |
| LOW | Re-enable epoch signal examples | Fix alignment first |

---

## 12. Quick Reference Commands

### Check versions everywhere

```bash
# psgscoring (local)
grep 'version = ' ~/Desktop/GITHUB/github_psgscoring/pyproject.toml
grep '__version__' ~/Desktop/GITHUB/github_psgscoring/psgscoring/__init__.py

# YASAFlaskified (local)
cat ~/Desktop/GITHUB/github_YASAFlaskified/myproject/version.py

# Hetzner (production)
ssh root@dedodedodo.be "docker compose -f /data/slaapkliniek/docker-compose.yml exec app python -c 'from version import __version__; print(__version__)'"

# PyPI
pip index versions psgscoring
```

### Full deploy cycle (all components)

```bash
# ① psgscoring → GitHub + PyPI
cd ~/Desktop/GITHUB/github_psgscoring
git add -A && git commit -m "vX.X.X: changes" && git tag vX.X.X && git push origin main --tags
source ~/venv-pypi/bin/activate
rm -rf dist/ build/ *.egg-info && python3 -m build && python3 -m twine upload dist/*

# ② Wiki
cd ~/Desktop/GITHUB/psgscoring.wiki
rm -f *.md && unzip -o ~/Downloads/psgscoring_wiki_*.zip
git add -A && git commit -m "vX.X.X" && git push origin master

# ③ YASAFlaskified → GitHub
cd ~/Desktop/GITHUB/github_YASAFlaskified
find . -maxdepth 1 -not -name '.git' -not -name '.' -exec rm -rf {} +
unzip -o ~/Downloads/YASAFlaskified_*.zip
git add -A && git commit -m "vX.X.X: changes" && git tag -f vX.X.X && git push origin main --tags -f

# ④ YASAFlaskified → Hetzner
scp ~/Downloads/YASAFlaskified_*.zip root@dedodedodo.be:/data/slaapkliniek/ && \
ssh root@dedodedodo.be "cd /data/slaapkliniek && unzip -qo YASAFlaskified_*.zip -d . && rm YASAFlaskified_*.zip && docker compose build --no-cache app worker1 worker2 worker3 worker4 worker5 worker6 worker7 worker8 && docker compose up -d"
```

---

## 13. Starting a New Chat Session

When starting a new Claude conversation about this project, provide:

1. **This document** (as upload or project file)
2. **Current versions**: psgscoring vX.X.X, YASAFlaskified vX.X.X
3. **What you're working on** (code, paper, protocol, deployment)
4. **Any files needed** (ZIP of current code, specific .py files, .tex files)

Key context to mention:
- Vallat is NOT a co-author (only YASA references)
- No JCSM reference (paper not submitted)
- Six over-counting corrections (not five)
- Briek → MSc candidate (not BSc)
- pyproject.toml build-backend = "setuptools.build_meta"
- Docker: always build --no-cache, never restart for Python changes
- PyPI venv: `source ~/venv-pypi/bin/activate`
- v0.4.2: local baseline validation is profile-aware via 2 new fields
  in PostProcessingRules (cv_threshold, strict_reduction)
