# v0.4.2 Release Scripts

These scripts were used to apply the v0.4.2 architectural refactor
(profile-aware local baseline validation).

## What these scripts did

1. `00_apply_v042_refactor.sh` — Added `local_baseline_cv_threshold` and
   `local_baseline_strict_reduction` fields to `PostProcessingRules`,
   updated profile values, made `_validate_local_reduction` profile-aware.

2. `00b_fix_sp_scope.sh` — Fixed scope bug where `sp` dict was referenced
   inside `_detect_hypopneas` (which doesn't have access to `sp`). Threaded
   the two new params through the call chain via function parameters.

3. `test_stability_filter_sn2.py` — Diagnostic test script comparing
   profile-sweep AHI before/after the refactor on PSG-IPA SN2.

These are kept for reproducibility/audit. They are NOT part of the
psgscoring API and should not be re-run on a fresh install — the
changes have already been incorporated.
