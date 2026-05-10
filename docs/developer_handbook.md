# YASAFlaskified & psgscoring — Developer Handbook

**Last updated:** May 2026 (v0.6.0 release) · **Versions:** YASAFlaskified v0.9.7, psgscoring v0.6.0

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
| Briek Rombaut | Co-developer (MSc Computer Science Eng., UGent — graduated) |
| Cedric Rombaut | Co-developer (BSc Electrical Engineering, UGent — graduated) |
| Raphaël Vallat, PhD | Scientific advisor, YASA creator (UC Berkeley) — **co-author candidate on paper v36** |
| Remington Mallett, PhD | Second scientific advisor, YASA co-maintainer — **co-author candidate on paper v36** |
| Didier Pevernagie, MD PhD | Planned senior outreach (UGent emeritus, sleep medicine) |

**Status update (May 2026):** The earlier rule "Vallat NOT co-author" has been retired. Vallat reviewed paper v36 in early May 2026, his feedback is being incorporated, and he has been formally invited to join as co-author (middle or senior position open). Same offer extended to Remy Mallett. Paper target: *Physiological Measurement* (not JCSM).

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
├── psgscoring_lgbm/            # LightGBM training (now integrated in v0.6.0)
│   ├── config.py
│   ├── feature_extraction.py   # 32 hand-crafted candidate-level features
│   ├── prepare_mesa.py
│   ├── train_lgbm.py           # Trained on q∈{5,6}, q=7 fully held out
│   └── hybrid.py               # Re-classifier @ 0.65 threshold (default)
│
├── /home/bart/MESA-ab-test/    # MESA validation harness (NOT a git repo)
│   ├── score_mesa.py           # Native v0.5.0 mesa_shhs profile run
│   ├── score_mesa_yasa_e2e.py  # End-to-end YASA + psgscoring pipeline (§S7)
│   └── score_mesa_q*_n*.json   # Per-cohort run artefacts
│
└── /home/bart/CODE/docs/       # Paper / supplement / cover letter
    ├── YASAFlaskified_Paper_v36_PhysiolMeas.tex          # 20p, paper v36
    ├── YASAFlaskified_Supplement_v36_PhysiolMeas.tex     # 18p
    ├── cover_letter_v36_PhysiolMeas.tex
    ├── email_vallat_reply_v36_v4.md
    └── email_hertegonne_reply.md
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
pyproject.toml          → version = "0.6.0"
psgscoring/__init__.py  → __version__ = "0.6.0"
```
Both MUST match. PyPI uses pyproject.toml.

**YASAFlaskified:**
```
myproject/version.py    → __version__ = "0.9.7"
                        → PSGSCORING_VERSION = "0.6.0"
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
| 14 (opt) | hybrid.py | LightGBM candidate-level re-classification (@ 0.65) |

### Key algorithms

**12 bias corrections:**
- 6 over-counting: post-apnea baseline inflation, SpO₂ cross-contamination, Cheyne-Stokes trough, borderline classification, artefact-flank, local baseline validation
- 6 under-counting: peak-based breath detection, SpO₂ de-blocking, extended nadir window (45s), flow smoothing, position auto-mapping, configurable profiles

**Scoring profiles (v0.6.0):**
- `strict`: 70% threshold, 30s nadir, no smoothing, no peak detection
- `standard` (default): 70%, 45s nadir, 3s smoothing, peak detection ON
- `sensitive`: 75% (≥25% reduction), 45s nadir, 5s smoothing, peak detection ON
- `aasm_v2_rec`: PSG-IPA / clinical AASM v2 baseline
- `mesa_shhs`: native NSRR `mesa_shhs` hp3u convention (v0.5.0+, bit-identical to the v0.4.5 monkey-patched recipe)
- `chicago_1999`: SHHS-1 pre-AASM convention (in development)
- `cms`: CMS reimbursement profile

**Optional LightGBM re-classifier (v0.6.0):** 32 hand-crafted candidate-level features → trained on MESA q∈{5,6} stratum, q=7 fully held out. Threshold 0.65 by default. Runs after rule-based detection; preserves event-level structure but reclassifies candidates as keep/drop. On MESA q=7 honest holdout (n=92): bias −0.02/h, κ 0.50 (vs scorer-faithful baseline bias −0.78 to +1.10/h, κ 0.40-0.48).

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
| **YASA Hypnogram API** | `Hypnogram.labels` returns 7 categorical level names; `.hypno` returns per-epoch series | Use `list(hypno.hypno)` for YASA 0.7, fallback `tolist()` for 0.6 |
| **MESA worker OOM** | 28 workers × ~6.5GB anon-rss > 128GB RAM → BrokenProcessPool | Cap at `--workers 12` for end-to-end MESA runs |
| **MESA missing EDFs** | mesaids 93, 255, 1662, 2544, 2565, 3605, 5013 not on local disk | Report n=92/99 transparently; do not fail the run |
| **Cross-document `\ref`** | `\ref{sec:limitations}` from supplement to main paper is undefined | Use hard text "Section~5.3 (Limitations) of the main paper" |
| **PyPI venv outside project** | Project venv pollutes build artefacts and resolves wrong deps | Use dedicated `~/venv-pypi` outside the project tree |
| **edfio drops annotations** | edfio 0.4.13 silently writes plain EDF when EDF+ requested | Use pyedflib with `FILETYPE_EDFPLUS` |
| **i18n t-shadowing** | `t = date.today()` shadows the `t()` translation function → total PDF failure | Never name local vars `t` |
| **sklearn pinning** | LightGBM trained on sklearn 1.4 fails to load with sklearn 1.5+ | Pin sklearn version in pyproject.toml |
| **Classification ordering** | Paradoxical breathing must be checked before effort-absent | Otherwise obstructive → central misclassification |

---

## 10. Validation Datasets

| Dataset | n | Role | Status |
|---|---|---|---|
| PSG-IPA (PhysioNet) | 5 rec × 12 scorers | External validation + benchmarking | ✅ Complete (paper v36 §3) |
| iSLEEPS (stroke) | 96 patients | Cross-population validation | ✅ Complete |
| MESA q=7 (NSRR) | 99 (92 EDFs on disk) | Honest LightGBM holdout + e2e §S7 | ✅ Complete (paper v36 §4 + §S7) |
| MESA q∈{2,3,4} (NSRR) | 100 random | Graceful-degradation §S6 | ✅ Complete |
| MESA q∈{5,6} (NSRR) | ~training stratum | LightGBM training | ✅ Complete |
| SHHS-1 (NSRR) | smoke-test pending | Chicago 1999 profile validation | ⏳ Blocked on POOR-quality robustness bug |
| AZORG-YASA-2026-001 | ≥50 PSGs | Single-centre prospective | ⏳ Protocol v7.0 dept-head approved; EC approval expected this week |
| UZ Gent (UGent) | TBD | Possible second clinical site | ⏳ Hertegonne contact ongoing (May 2026) |

### Paper v36 — headline results (PSG-IPA + MESA q=7)

| Metric | PSG-IPA | MESA q=7 (rule-based) | MESA q=7 (LightGBM @0.65) |
|---|---|---|---|
| AHI bias | +1.6/h | −0.78 to +1.10/h (per profile) | −0.02/h |
| Pearson r | 0.997 | 0.84 | 0.87 |
| Weighted κ | 0.91 | 0.40–0.48 | 0.50 |
| Event-level F1 (severe rec.) | 0.886 | — | — |
| Mean Δt (severe rec.) | < 2 s | — | — |
| ROC AUC @ AHI≥5/15/30 | — | 0.88–0.93 | 0.88–0.93 |

### End-to-end pipeline (§S7) — YASA staging + arousal detector + psgscoring

Same q=7 cohort, n=92, NSRR upstream replaced by YASA + internal EEG-arousal detector:

| Metric | Manual upstream | End-to-end | Δ |
|---|---|---|---|
| AHI bias | within ±3.5/h | within ±3.5/h | preserved |
| Pearson r | 0.80 | 0.66 | −0.14 |
| Severity match | 59% | 51% | −8 pp |
| |ΔTST| | — | 0.29 h | YASA staging is fine |
| |Δarousals| | — | 71 | Variance locus |

**Interpretation:** The locus of e2e degradation is the EEG-arousal detector, not YASA staging. This drives the Future Work item on arousal-detector improvement (parameter tuning / candidate-level re-classifier / 1D-CNN; see §5.5 of paper v36).

---

## 11. Pending Items (May 2026)

| Priority | Item | Status |
|---|---|---|
| ✅ DONE | psgscoring v0.5.0 native `mesa_shhs` profile | Bit-identical to v0.4.5 monkey-patched recipe |
| ✅ DONE | psgscoring v0.6.0 on PyPI + GitHub | Optional LightGBM re-classifier shipped |
| ✅ DONE | YASAFlaskified v0.9.7 on Hetzner | Production deployment current |
| ✅ DONE | MESA q=7 LightGBM honest holdout (n=92) | bias −0.02/h, κ 0.50 |
| ✅ DONE | End-to-end §S7 (YASA + psgscoring on q=7) | bias ±3.5/h, r 0.66 |
| ✅ DONE | Paper v36 + supplement v36 + cover letter | 20p + 18p + 2p, clean compile |
| ✅ DONE | Vallat paper v36 review round 1 | Feedback received early May 2026 |
| ✅ DONE | UGent (Hertegonne) initial outreach | Reply received 2026-05-09 |
| HIGH | Send paper v36 reply to Vallat (v4) | `email_vallat_reply_v36_v4.md` ready |
| HIGH | Send reply to Hertegonne (UGent) | `email_hertegonne_reply.md` ready |
| HIGH | Humanisation pass on paper v36 (weekend) | Per Vallat feedback point 1 |
| HIGH | EC submission AZORG-YASA-2026-001 | Approval expected this week (May 2026); inclusion start June |
| MEDIUM | Confirm Vallat / Mallett co-author position | Awaiting reply |
| MEDIUM | Pevernagie senior-author outreach | Draft ready |
| MEDIUM | SHHS-1 smoke test | Blocked on POOR-quality robustness bug |
| MEDIUM | Arousal-detector improvement (Future Work §5.5) | Optie A: parameter tuning vs NSRR labels |
| LOW | Hybrid CNN + GBM (next-paper scope) | Sketched in §5.5 |
| LOW | Re-enable epoch signal examples in PDF | Fix alignment first |

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
pip index versions psgscoring   # expect 0.6.0 latest
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
2. **Current versions**: psgscoring v0.6.0, YASAFlaskified v0.9.7
3. **What you're working on** (code, paper, protocol, deployment)
4. **Any files needed** (ZIP of current code, specific .py files, .tex files)

Key context to mention:
- Paper v36 target: *Physiological Measurement* (NOT JCSM — that target was retired)
- **Vallat IS a co-author candidate** on paper v36 (rule changed May 2026); Mallett same offer
- Pevernagie is the planned senior-author outreach
- Six over-counting corrections (not five)
- Briek → MSc graduated; Cedric → BSc graduated
- pyproject.toml build-backend = "setuptools.build_meta"
- Docker: always build --no-cache, never restart for Python changes
- PyPI venv: `source ~/venv-pypi/bin/activate` — outside the project tree
- v0.5.0: native `mesa_shhs` profile (bit-identical to v0.4.5 monkey-patched recipe)
- v0.6.0: optional LightGBM candidate-level re-classifier (32 features, threshold 0.65)
- MESA q=7 honest holdout n=92 (7 EDFs missing); LightGBM trained on q∈{5,6}
- §S7 end-to-end variance locates in EEG-arousal detector, not YASA staging
- AZORG-YASA-2026-001 protocol v7.0 (dept-head approved); EC approval expected week of 2026-05-09
- UGent / Hertegonne: possible second clinical site, contact ongoing
- No commit/PR attribution to Claude/AI in messages or PR bodies
- Memory location: `/home/bart/.claude/projects/-home-bart-CODE/memory/`
