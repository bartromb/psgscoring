"""
psgscoring.ml_classifier
=========================
Optional LightGBM-based candidate-level re-classifier (v0.6.0+).

When a profile sets `post_processing.ml_classifier_path` to a valid
LightGBM booster file, the pipeline calls
`apply_ml_reclassification` after Rule 1B reinstatement: each
candidate (currently-accepted event + currently-rejected hypopnea
candidate) is featurized, scored by the booster, and kept iff the
score >= `ml_threshold`. The events list and respiratory summary
are then recomputed from the kept set.

The classifier ships with one pre-trained model:
  `data/lightgbm_v06_q7holdout.txt`
trained on the q≥5∖q=7 stratum of the MESA cohort (n=653 recordings,
~210k labeled candidates) with 5-fold group-CV. It is consumed by
the `mesa_shhs` profile by default.

Per paper v34 / v35 §S5.6, on the q=7 holdout (n=92 successful) the
classifier at threshold 0.65 yields:
  bias    -0.02 /h    MAE 5.34/h    SD 7.30/h
  Pearson r 0.872     weighted κ 0.497     severity-match 63%
vs the v0.5.2 rule-based baseline:
  bias    +1.10 /h    MAE 6.06/h
  Pearson r 0.804     weighted κ 0.481     severity-match 59%

The model is interpretable via feature importances; top-5 features
by gain are: desaturation_pct, n_arousals_per_h, stage_r,
time_to_next_event_s, n_arousals_within_30s.

Clinical profiles (`aasm_v3_*`, `aasm_v2_rec`, `aasm_v1_rec`,
`cms_medicare`, `chicago_1999`) leave `ml_classifier_path=None`
and skip this step entirely; PSG-IPA paper-v31 reproducibility is
preserved.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("psgscoring.ml_classifier")

# Feature column order — must match build_lightgbm_dataset.py FEATURE_COLUMNS
# plus the trailing `is_accepted` flag.
FEATURE_COLUMNS: tuple[str, ...] = (
    "duration_s", "flow_reduction_pct", "desaturation_pct", "min_spo2",
    "confidence", "is_apnea", "is_hypopnea_obstructive",
    "is_hypopnea_central", "is_hypopnea_mixed", "is_rule1b",
    "stage_w", "stage_n1", "stage_n2", "stage_n3", "stage_r",
    "n_events_within_5min", "time_since_prior_event_s", "time_to_next_event_s",
    "onset_min", "fraction_of_recording",
    "tst_h", "n_arousals_per_h", "overall_qual5", "median_spo2",
    "thermistor_type",
    "n_arousals_within_30s", "has_arousal_within_5s",
    "envelope_pre_30s_mean", "envelope_post_30s_mean", "envelope_event_mean",
    "envelope_local_reduction_pct",
    "is_accepted",
)


_BOOSTER_CACHE: dict[str, Any] = {}


def _resolve_path(path: str) -> Path:
    """Resolve `path` against the package data dir if relative."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    pkg_data = Path(__file__).parent / path
    if pkg_data.exists():
        return pkg_data
    return p  # let load_booster raise if missing


def load_booster(path: str):
    """Lazy-load a LightGBM booster, cached by path."""
    if path in _BOOSTER_CACHE:
        return _BOOSTER_CACHE[path]
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "lightgbm is required for ML re-classification but not installed. "
            "Install with `pip install psgscoring[ml]` or `pip install lightgbm`."
        ) from e
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"LightGBM model not found: {path} (resolved to {resolved})")
    booster = lgb.Booster(model_file=str(resolved))
    _BOOSTER_CACHE[path] = booster
    logger.info("[ml] Loaded booster %s (%d features expected)",
                resolved.name, len(FEATURE_COLUMNS))
    return booster


def _extract_candidate_features(
    candidate: dict,
    is_accepted: bool,
    all_candidates: list[dict],
    arousals: list[dict],
    hypno: list[str],
    sig_dur_s: float,
    tst_h: float,
    overall_qual5: int,
    median_spo2: float,
    thermistor_type: int,
) -> dict:
    """Compute the same ~32 features used to train the classifier.

    See `build_lightgbm_dataset.extract_features` (in MESA-ab-test/) for
    the canonical implementation; this is the runtime mirror that lives
    inside the package.
    """
    onset = float(candidate.get("onset_s", 0))
    dur   = float(candidate.get("duration_s", 0))
    end   = onset + dur
    stage = candidate.get("stage", "W")
    ev_type = candidate.get("type", "hypopnea")

    n_events_within_5min = sum(
        1 for c in all_candidates
        if abs(float(c.get("onset_s", 0)) - onset) <= 300
        and c is not candidate
    )
    prior_t = max(
        (float(c.get("onset_s", 0)) for c in all_candidates
         if float(c.get("onset_s", 0)) < onset),
        default=onset - 3600,
    )
    next_t = min(
        (float(c.get("onset_s", 0)) for c in all_candidates
         if float(c.get("onset_s", 0)) > onset),
        default=onset + 3600,
    )

    n_arousals_within_30s = sum(
        1 for a in arousals
        if abs(float(a["onset_s"]) - onset) <= 30
    )
    has_arousal_within_5s = any(
        abs(float(a["onset_s"]) - end) <= 5 for a in arousals
    )
    arousals_per_h = (len(arousals) / tst_h) if tst_h > 0 else 0.0

    return {
        "duration_s":               dur,
        "flow_reduction_pct":       float(candidate.get("flow_reduction_pct") or 0.0),
        "desaturation_pct":         float(candidate.get("desaturation_pct")
                                          or candidate.get("desat") or 0.0),
        "min_spo2":                 float(candidate.get("min_spo2") or 95.0),
        "confidence":               float(candidate.get("confidence") or 0.7),
        "is_apnea":                 1 if "apnea" in ev_type and "hypopnea" not in ev_type else 0,
        "is_hypopnea_obstructive":  1 if ev_type in ("hypopnea", "hypopnea_obstructive") else 0,
        "is_hypopnea_central":      1 if ev_type == "hypopnea_central" else 0,
        "is_hypopnea_mixed":        1 if ev_type == "hypopnea_mixed" else 0,
        "is_rule1b":                1 if candidate.get("rule1b", False) else 0,
        "stage_w":                  1 if stage == "W"  else 0,
        "stage_n1":                 1 if stage == "N1" else 0,
        "stage_n2":                 1 if stage == "N2" else 0,
        "stage_n3":                 1 if stage == "N3" else 0,
        "stage_r":                  1 if stage == "R"  else 0,
        "n_events_within_5min":     n_events_within_5min,
        "time_since_prior_event_s": min(onset - prior_t, 3600),
        "time_to_next_event_s":     min(next_t - onset,  3600),
        "onset_min":                onset / 60.0,
        "fraction_of_recording":    onset / sig_dur_s if sig_dur_s > 0 else 0.0,
        "tst_h":                    tst_h,
        "n_arousals_per_h":         arousals_per_h,
        "overall_qual5":            overall_qual5,
        "median_spo2":              median_spo2,
        "thermistor_type":          thermistor_type,
        "n_arousals_within_30s":    n_arousals_within_30s,
        "has_arousal_within_5s":    1 if has_arousal_within_5s else 0,
        # Reserved for future signal-context features (currently 0)
        "envelope_pre_30s_mean":    0.0,
        "envelope_post_30s_mean":   0.0,
        "envelope_event_mean":      0.0,
        "envelope_local_reduction_pct": 0.0,
        "is_accepted":              1 if is_accepted else 0,
    }


def apply_ml_reclassification(
    accepted: list[dict],
    rejected: list[dict],
    arousals: list[dict],
    hypno: list[str],
    sig_dur_s: float,
    tst_h: float,
    overall_qual5: int,
    median_spo2: float,
    thermistor_type: int,
    booster_path: str,
    threshold: float = 0.65,
) -> tuple[list[dict], list[dict], dict]:
    """Re-classify candidates with LightGBM and return new (accepted, rejected, meta).

    The returned `accepted` list is a (possibly re-ordered) subset of the
    union (accepted ∪ rejected) where the classifier score >= threshold.
    `rejected` contains the rest. `meta` reports counts and threshold.

    On any error the original (accepted, rejected) pair is returned with
    `meta["status"] = "skipped: <reason>"`.
    """
    try:
        booster = load_booster(booster_path)
    except Exception as e:
        logger.warning("[ml] Failed to load booster %s: %s", booster_path, e)
        return accepted, rejected, {"status": f"load_error: {e}"}

    all_candidates = list(accepted) + list(rejected)
    is_acc_flags   = [True] * len(accepted) + [False] * len(rejected)
    if not all_candidates:
        return accepted, rejected, {"status": "no_candidates", "threshold": threshold}

    feat_rows = [
        _extract_candidate_features(
            cand, is_acc, all_candidates, arousals, hypno,
            sig_dur_s, tst_h, overall_qual5, median_spo2, thermistor_type,
        )
        for cand, is_acc in zip(all_candidates, is_acc_flags)
    ]
    X = np.array([[r[c] for c in FEATURE_COLUMNS] for r in feat_rows], dtype=float)
    scores = booster.predict(X)
    keep_mask = scores >= threshold

    new_accepted = [c for c, k in zip(all_candidates, keep_mask) if k]
    new_rejected = [c for c, k in zip(all_candidates, keep_mask) if not k]

    # Sort by onset for downstream consistency
    new_accepted.sort(key=lambda x: float(x["onset_s"]))
    new_rejected.sort(key=lambda x: float(x["onset_s"]))

    type_counts = Counter(c.get("type", "?") for c in new_accepted)
    meta = {
        "status":             "ok",
        "threshold":          threshold,
        "n_candidates":       len(all_candidates),
        "n_accepted_input":   len(accepted),
        "n_rejected_input":   len(rejected),
        "n_accepted_output":  len(new_accepted),
        "n_rejected_output":  len(new_rejected),
        "score_min":          float(scores.min()),
        "score_max":          float(scores.max()),
        "score_mean":         float(scores.mean()),
        "score_p50":          float(np.percentile(scores, 50)),
        "type_counts_output": dict(type_counts),
    }
    logger.info(
        "[ml] Re-classified %d candidates → %d accepted (threshold %.2f); "
        "delta vs rule-based = %+d events",
        len(all_candidates), len(new_accepted), threshold,
        len(new_accepted) - len(accepted),
    )
    return new_accepted, new_rejected, meta
