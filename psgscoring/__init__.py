"""
psgscoring
==========
Open-source Python library for AASM-compliant PSG respiratory scoring.

Quick start
-----------
>>> from psgscoring import run_pneumo_analysis
>>> results = run_pneumo_analysis(raw, hypno)
>>> ahi = results["respiratory"]["summary"]["ahi_total"]

Module layout
-------------
constants    – AASM thresholds and channel-name patterns
utils        – safe_r, hypno helpers, sleep mask, channel detection
signal       – preprocessing: linearize, MMSD, bandpass, envelope, baselines
breath       – breath segmentation, amplitude ratios, flattening index
classify     – apnea-type classification (obstructive / central / mixed)
spo2         – SpO2 coupling (Rule 1A), full SpO2 analysis, hypoxic burden
plm          – PLM detection (AASM)
ancillary    – position, heart rate, snore, Cheyne-Stokes
respiratory  – apnea/hypopnea detection, Rule 1B, summary statistics
ecg_effort   – ECG-derived effort (TECG, spectral classifier) for central/obstructive differentiation
arousal      – EEG arousal & RERA detection + respiratory-arousal coupling (multi-derivation-ready)
postprocess  – CSR reclassification, mixed apnea decomposition, central instability index
pipeline     – MNE-facing master function (run_pneumo_analysis)
"""

# Public API
from .pipeline import run_pneumo_analysis

# Type definitions (for IDE autocomplete and mypy)
from ._types import (
    RespiratoryEvent, ClassifyDetail, ScoringSummary,
    SpO2Summary, PLMSummary, PositionSummary, PneumoResults,
    OAHIThresholds, OAHISweep3pt, ConfidenceBands,
)

from .respiratory import (
    detect_respiratory_events,
    reinstate_rule1a_arousal_hypopneas,
    reinstate_rule1b_hypopneas,      # deprecated alias
)
from .signal import (
    ENVELOPE_METHODS,
    compute_envelope,
    denoise_flow_wavelet,
    linearize_nasal_pressure,
    compute_mmsd,
    preprocess_flow,
    preprocess_effort,
    bandpass_flow,
    compute_dynamic_baseline,
    compute_pre_event_baseline,
    compute_stage_baseline,
    compute_anchor_baseline,
    detect_position_changes,
    reset_baseline_at_position_changes,
)
from .breath import (
    detect_breaths,
    compute_breath_amplitudes,
    compute_flattening_index,
    detect_breath_events,
)
from .classify import classify_apnea_type
from .ecg_effort import ecg_effort_assessment, compute_tecg, compute_adaptive_cardiac_band
from .arousal import (
    detect_arousals,
    detect_arousals_multi,
    detect_reras,
    run_arousal_respiratory_analysis,
)
from .ventilation import compute_ventilatory_burden
from .spo2 import analyze_spo2, compute_hypoxic_burden, detect_desaturations, get_desaturation
from .plm import analyze_plm
from .ancillary import (
    analyze_position,
    analyze_heart_rate,
    analyze_snore,
    detect_cheyne_stokes,
)
from .postprocess import (
    postprocess_respiratory_events,
    reclassify_csr_events,
    decompose_mixed_apneas,
    compute_central_instability_index,
)
from .utils import (
    detect_channels,
    channel_map_from_user,
    build_sleep_mask,
    hypno_to_numeric,
    is_nrem, is_rem, is_sleep,
    safe_r,
)

__version__ = "0.31.3"
__all__ = [
    # Master
    "run_pneumo_analysis",
    # Respiratory
    "detect_respiratory_events",
    "reinstate_rule1a_arousal_hypopneas",
    "reinstate_rule1b_hypopneas",    # deprecated alias
    # Signal
    "ENVELOPE_METHODS",
    "compute_envelope",
    "denoise_flow_wavelet",
    "linearize_nasal_pressure",
    "compute_mmsd",
    "preprocess_flow",
    "preprocess_effort",
    "bandpass_flow",
    "compute_dynamic_baseline",
    "compute_pre_event_baseline",
    "compute_stage_baseline",
    "compute_anchor_baseline",
    "detect_position_changes",
    "reset_baseline_at_position_changes",
    # Breath
    "detect_breaths",
    "compute_breath_amplitudes",
    "compute_flattening_index",
    "detect_breath_events",
    # Classify
    "classify_apnea_type",
    # ECG effort
    "ecg_effort_assessment",
    "compute_adaptive_cardiac_band",
    "compute_tecg",
    # Arousal & RERA
    "detect_arousals",
    "detect_arousals_multi",
    "detect_reras",
    "run_arousal_respiratory_analysis",
    # Ventilatory burden
    "compute_ventilatory_burden",
    # SpO2
    "analyze_spo2",
    "compute_hypoxic_burden",
    "detect_desaturations",
    "get_desaturation",
    # PLM
    "analyze_plm",
    # Ancillary
    "analyze_position",
    "analyze_heart_rate",
    "analyze_snore",
    "detect_cheyne_stokes",
    # Post-processing
    "postprocess_respiratory_events",
    "reclassify_csr_events",
    "decompose_mixed_apneas",
    "compute_central_instability_index",
    # Utils
    "detect_channels",
    "channel_map_from_user",
    "build_sleep_mask",
    "hypno_to_numeric",
    "is_nrem", "is_rem", "is_sleep",
    "safe_r",
]

# v0.4.0: profile-based scoring
from psgscoring.profiles import (
    Profile, get_profile, list_profiles, list_profile_groups,
    PROFILES, PROFILE_GROUPS,
)
