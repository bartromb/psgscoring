# Changelog — psgscoring

All notable changes documented per [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.2] — 2026-04-02

### Added
- `signal_quality.py`: per-channel flat-line, clipping, disconnect, line-noise assessment
- Montage plausibility checks (cross-correlation EEG↔EOG, thorax↔abdomen, flow↔effort)
- Flattening-based RERA detection (Hosselet et al., AJRCCM 1998)
- Dual-source RERA: FRI-RERA (amplitude) + Flattening-RERA (shape)
- `_find_flattening_sequences()` in pipeline
- Flattening data included in `_breaths` output
- Hypopnea subtypes: `n_hypopnea_obstr`, `n_hypopnea_central`, `n_hypopnea_mixed`
- RERA index and RDI computation (AHI + RERA index)
- REM/NREM AHI in respiratory summary
- SpO2 samplerate check (AASM max 3s averaging)
- EDF patient info extraction (`_parse_edf_patient_info`)
- `assess_signal_quality` exported from package

### Changed
- `_compute_rera_rdi()`: two RERA sources (FRI-RERA + Flattening-RERA)
- Pipeline: 11 steps (added Step 1b: signal quality, Step 8b: RERA/RDI)
- `pyproject.toml` build-backend fixed to `setuptools.build_meta`

---

## [0.2.1] — 2026-03-31

### Changed
- Nadir window 30 → 45 s (finger oximetry delay 20–40 s)
- 3 s flow smoothing (`uniform_filter1d`) on normalised flow before thresholding

---

## [0.2.0] — 2026-03-31

### Added
- 51 unit tests with conftest.py + importlib pytest config
- DISCLAIMER.md
- README.md with 6 badges, 10 numbered references with DOIs
- `pyproject.toml` build-backend fixed

---

## [0.1.0] — 2026-03-20

### Added
- Initial release: respiratory event detection, apnea classification,
  SpO2 coupling, PLM, position, heart rate, snore, Cheyne-Stokes
- Rule 1A + Rule 1B hypopnea scoring
- Five over-counting corrections
- BSD-3 license
