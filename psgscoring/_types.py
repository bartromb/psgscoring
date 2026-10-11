"""
psgscoring._types
=================
TypedDict definitions for the main output structures.

These are for documentation and type-checking only (no runtime overhead).
The module is private (``_types``); the names are re-exported from the
package root, so import them from there:

    from psgscoring import RespiratoryEvent, ScoringSummary, PneumoResults

    def process(event: RespiratoryEvent) -> None:
        print(event["onset_s"])  # IDE autocomplete + type check

Every TypedDict here is ``total=False``: a key may be absent when the step
that writes it did not run (no channel, step failed, option not requested).
``tests/test_types_match_output.py`` runs the synthetic golden case and checks
that every documented top-level key and every documented summary key is
actually produced, so a key documented here is one the pipeline writes.
"""

from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any


class ClassifyDetail(TypedDict, total=False):
    """Detail dict from apnea type classification (classify.py)."""
    rule_index: int
    decision_reason: str
    effort_ratio: float
    raw_var_ratio: float
    paradox_correlation: float
    phase_angle_deg: float
    first_ratio: float
    second_ratio: float
    quarter_efforts: List[float]
    flattening_index: Optional[float]
    ecg_effort_present: Optional[bool]
    spectral_cardiac_pct: Optional[float]
    spectral_respiratory_pct: Optional[float]
    tecg_bursts_detected: Optional[bool]
    reclassify_as_central: Optional[bool]


class RespiratoryEvent(TypedDict, total=False):
    """A single scored respiratory event (apnea or hypopnea).

    Returned in results["respiratory"]["events"].
    """
    type: str                    # "obstructive", "central", "mixed", "hypopnea", ...
    onset_s: float               # seconds from recording start
    duration_s: float            # event duration in seconds
    stage: str                   # sleep stage at onset: "W","N1","N2","N3","R"
    desaturation_pct: Optional[float]  # SpO2 drop (Rule 1A), None if no desat
    min_spo2: Optional[float]    # SpO2 nadir during post-event window
    flow_nadir: float            # minimum normalised flow during event (apneas)
    flow_reduction: Optional[float]     # flow reduction ratio (hypopneas)
    flow_reduction_pct: Optional[float] # flow reduction as percentage
    pre_baseline: float          # baseline amplitude at event onset
    confidence: float            # classification confidence 0.0–1.0
    classify_detail: ClassifyDetail
    epoch: int                   # 30-s epoch index at onset
    spo2_cross_contaminated: bool  # Fix 2: SpO2 possibly from prior event
    csr_flagged: bool            # Fix 3: part of Cheyne-Stokes cycle (postprocess.py)
    # `local_baseline_rejected` and `_label` were documented here until
    # 2026-10-11 but no module ever writes them: rejected candidates live in
    # results["respiratory"]["rejected_hypopneas"], not as a flag on events.


class OAHIThresholds(TypedDict):
    """OAHI at different confidence thresholds (Fix 4, legacy 4-point sweep)."""
    high: float      # ≥0.85
    moderate: float   # ≥0.60
    borderline: float # ≥0.40
    all: float        # all events (official)


class OAHISweep3pt(TypedDict):
    """v0.4.1: Clinically calibrated 3-point OAHI confidence sweep.

    Calibrated on PSG-IPA validation:
      - lenient (c≥0.30): mean OAHI 8.7/h
      - primary (c≥0.47): empirical best-fit (matches scorer mediaan)
      - strict  (c≥0.65): mean OAHI 4.7/h
    Mean sweep width on PSG-IPA: 3.9/h (matches AASM inter-scorer
    variability ~10-20%). Replaces the previous 4-point sweep
    (oahi_thresholds) which produced ~9.3/h widths considered
    too wide for clinical interpretation.
    """
    lenient: float    # c ≥ 0.30 — inclusief
    primary: float    # c ≥ 0.47 — best-fit
    strict:  float    # c ≥ 0.65 — conservatief


class ConfidenceBands(TypedDict):
    """Event counts per confidence tier."""
    high: int        # ≥0.85
    moderate: int    # 0.60–0.84
    borderline: int  # 0.40–0.59
    low: int         # <0.40


class ScoringSummary(TypedDict, total=False):
    """Summary statistics from respiratory scoring.

    Returned in results["respiratory"]["summary"].
    """
    # ── Primary indices ──
    n_ah_total: int
    ahi_total: float              # conservative: excludes unsubtyped `uncertain` apneas
    ahi_incl_uncertain: float     # v0.6.2 dual-AHI: counts them (≈unbiased vs scorers on MESA)
    ahi_dual: Dict                # both numbers side by side, for the report
    ahi_rule: Dict                # which AASM rule carries the AHI (ahi_rule_conformity)
    oahi: float
    oahi_all: float               # alias for backward compat
    oahi_conf60: float
    oahi_thresholds: OAHIThresholds  # legacy 4-point sweep (deprecated)
    severity: str                 # "normal", "mild", "moderate", "severe"
    oahi_severity: str

    # ── v0.4.1: 3-point clinical sweep + robustness grade ──
    oahi_sweep: OAHISweep3pt      # clinically calibrated 3-point sweep
    oahi_sweep_width: float       # max - min, in /h
    robustness_grade: str         # "A" (robust), "B" (probable), "C" (uncertain)

    # ── Counts per type ──
    n_apnea_total: int
    n_obstructive: int
    n_central: int
    n_mixed: int
    n_hypopnea: int
    n_hypopnea_obstr: int         # AASM v3 VIII.D.2 hypopnea subtyping (optional rule)
    n_hypopnea_central: int
    n_hypopnea_mixed: int
    n_uncertain_apnea: int        # unsubtyped apneas (outside ahi_total)
    uncertain_fraction: float

    # ── Per-type indices ──
    obstructive_index: float
    central_index: float
    mixed_index: float
    hypopnea_index: float

    # ── RERA / RDI / flow-limitation ──
    n_rera: int
    n_rera_flattening: int
    n_rera_fri: int
    rera_index: float
    rdi: float
    n_fri: int
    fri_index: float

    # ── Stage-specific ──
    ahi_rem: float
    ahi_nrem: float
    ahi_rem_incl_uncertain: float
    ahi_nrem_incl_uncertain: float
    rem_ahi: float                # aliases kept for report readers
    nrem_ahi: float
    rem_min: float
    nrem_min: float
    ahi_rem_reliable: bool        # enough REM to report an index
    ahi_rem_caveat: Optional[str]
    obstructive_rem: int
    obstructive_nrem: int
    central_rem: int
    central_nrem: int
    mixed_rem: int
    mixed_nrem: int
    hypopnea_rem: int
    hypopnea_nrem: int

    # ── Duration stats ──
    max_apnea_dur_s: float
    avg_apnea_dur_s: float
    max_hypopnea_dur_s: float
    avg_hypopnea_dur_s: float
    apnea_max_dur_s: float        # the profile's APNEA_MAX_DUR_S split limit
    n_apneas_at_max_dur: int      # events that hit it (v0.8.22 splitting)

    # ── Quality metrics ──
    avg_desaturation: float
    avg_classification_confidence: float
    n_low_confidence: int
    confidence_bands: ConfidenceBands
    scorer_agreement_expectation: Dict

    # ── Sleep time / denominators ──
    tst_hours: float
    tst_minutes: float
    n_artifact_epochs_excluded: int
    indices_computable: bool      # False when the denominator is unusable
    index_denominator_h: Optional[float]
    index_unavailable_reason: Optional[str]

    # ── Correction counters (Table 4 in paper) ──
    # `n_gap_excluded` and `n_local_baseline_rejected` are NOT summary keys:
    # they sit on results["respiratory"] itself.
    n_csr_flagged: int
    ahi_csr_corrected: float
    oahi_csr_corrected: float
    n_spo2_cross_contaminated: int
    n_low_conf_borderline: int
    n_low_conf_noise: int
    ahi_excl_noise: float

    # ── Derived blocks ──
    phenotypes: Dict              # positional / REM / arousal phenotypes
    ventilatory_burden: Dict
    hypoventilation: Dict
    warnings: List[str]


class SpO2Summary(TypedDict, total=False):
    """SpO2 analysis summary.

    Returned in results["spo2"]["summary"]. Time-in-band keys come in a
    formatted-string form (``time_below_90``, via ``fmt_time``) and a
    minutes form (``time_below_90_min`` ...).
    """
    mean_spo2: float
    avg_spo2: float               # alias of mean_spo2
    baseline_spo2: float          # 90th percentile
    min_spo2: float
    n_desaturations: int
    desat_index: float            # alias of odi_3pct
    odi_3pct: float               # desaturations ≥3% per hour
    odi_4pct: float
    n_desat_3pct: int
    n_desat_4pct: int
    pct_below_90: float           # % of TST (was documented as time_below_90_pct)
    time_below_90: str            # formatted duration (fmt_time); no *_min twin for 90/80
    pct_below_80: float
    time_below_80: str
    pct_below_70: float
    time_below_70: str
    time_below_70_min: float
    pct_95_100: float
    time_95_100_min: float
    pct_90_95: float
    time_90_95_min: float
    pct_80_90: float
    time_80_90_min: float
    pct_70_80: float
    time_70_80_min: float
    hypoxic_burden: Optional[float]
    hypoxic_burden_unit: str
    hypoxic_burden_method: str
    low_baseline_warning: bool
    low_baseline_note: Optional[str]
    rem_min_spo2: Optional[float]
    rem_pct_below_90: Optional[float]
    rem_time_below_90: Optional[str]
    nrem_min_spo2: Optional[float]
    nrem_pct_below_90: Optional[float]
    nrem_time_below_90: Optional[str]
    total_sleep_s: float


class SpO2Block(TypedDict, total=False):
    """results["spo2"] — the block around the summary.

    ``spo2_low_samplerate`` lives HERE (pipeline.py writes
    ``output["spo2"]["spo2_low_samplerate"] = True`` when the oximeter
    averages over >3 s), not in the summary where it was documented before
    2026-10-11.
    """
    success: bool
    error: Optional[str]
    summary: SpO2Summary
    desaturations: List[Dict]
    timeseries: Dict
    spo2_samplerate: float
    spo2_low_samplerate: bool     # warning: >3 s averaging (only present when True)


class PLMSummary(TypedDict, total=False):
    """PLM analysis summary.

    Returned in results["plm"]["summary"]. Empty dict when no leg-EMG channel.
    """
    n_events_truncated: int       # eligible movements not in events[] (plm_event_list_cap)
    n_lm_total: int
    n_lm_sleep: int
    n_lm_wake: int
    n_resp_associated: int
    n_plm_eligible: int
    n_plm: int                    # PLMs in series
    n_plm_series: int
    lm_index: float               # leg-movement index (/h sleep)
    plm_index: float              # PLM index (/h sleep)
    plm_severity: str
    total_sleep_h: float


class PositionSummary(TypedDict, total=False):
    """Position analysis summary (ancillary.analyze_position).

    Returned in results["position"]["summary"]; empty dict without a position
    channel. Per-position values are dicts keyed by position name ("Supine",
    "Left", ...), not flat ``ahi_supine``/``pct_supine`` keys as documented
    before 2026-10-11.
    """
    sleep_time_min: Dict[str, float]
    sleep_pct: Dict[str, float]
    ahi_per_pos: Dict[str, Optional[float]]  # None below POSITION_MIN_MINUTES
    n_events_per_pos: Dict[str, int]
    min_minutes_for_index: float
    position_mapping_method: str  # "coded" | "levels" | "percentile"


class AHIIntervalArm(TypedDict):
    ahi: float
    oahi: float
    severity: str


class AHIInterval(TypedDict, total=False):
    """results["ahi_interval"] — the strict/standard/sensitive bracket."""
    strict: AHIIntervalArm
    standard: AHIIntervalArm
    sensitive: AHIIntervalArm
    interval: List[float]         # [min, max] AHI over the three arms
    robustness_grade: str         # "A" | "B" | "C"
    robustness_label: str
    primary_profile: str          # the name the caller passed
    error: str                    # only when the computation failed


class ChannelQualityBlock(TypedDict, total=False):
    """results["channel_quality"] (signal_quality_channels.assess_signal_quality)."""
    overall_grade: str            # "good" | "fair" | "poor" | "unknown"
    channels: Dict[str, Dict]
    montage_warnings: List[str]
    error: str


class AnchorBaselineBlock(TypedDict, total=False):
    """results["anchor_baseline"] — mouth-breathing / baseline-anchor check."""
    anchor_reliable: bool
    anchor_value: Optional[float]
    anchor_ratio: Optional[float]
    anchor_epochs_used: int
    mouth_breathing_suspected: bool


class CheyneStokesBlock(TypedDict, total=False):
    """results["cheyne_stokes"] (ancillary.detect_cheyne_stokes)."""
    success: bool
    csr_detected: bool
    csr_minutes: float
    central_events: int
    central_events_per_h: float
    criteria_met: Dict
    density_criterion_met: bool
    monitoring_hours: float
    periodicity_s: Optional[float]
    error: str


class HypoxicBurdenBlock(TypedDict, total=False):
    """results["hypoxic_burden"] (spo2.compute_hypoxic_burden)."""
    hypoxic_burden: Optional[float]
    unit: str                     # "%min/h"
    baseline_method: str
    mean_event_burden: Optional[float]
    n_events_with_burden: int
    total_area_pct_s: float
    error: str


class PostprocessBlock(TypedDict, total=False):
    """results["postprocess"] (postprocess.postprocess_respiratory_events)."""
    n_csr_reclassified: int
    n_mixed_decomposed: int
    n_mixed_to_central: int
    central_instability: Dict
    cai_change: Optional[float]
    error: str


class MetaBlock(TypedDict, total=False):
    """results["meta"] — provenance for the report. Built in pipeline.py from
    the rendered profile dict (not from profiles.profile_metadata)."""
    psgscoring_version: str
    channels_used: Dict[str, str]
    all_channels: List[str]
    sfreq: float
    duration_min: float
    scoring_profile: str          # resolved canonical profile name
    scoring_label: str
    scoring_input_name: str       # what the caller passed (may be a legacy alias)
    env_overrides: Dict[str, Any]
    hypopnea_criterion: Optional[str]
    patient_info: Dict
    flow_channels: Dict           # apnea/hypopnea/reference sensor + thermistor gate
    flow_fallback: Dict
    plm_channels: Dict
    arousal_emg_channel: Optional[str]
    # Conditional: `dual_sensor_fallback` (dual-sensor profiles only).


class PneumoResults(TypedDict, total=False):
    """Top-level results from run_pneumo_analysis().

    Usage:
        results = run_pneumo_analysis(raw, hypno, channel_map)
        ahi = results["respiratory"]["summary"]["ahi_total"]
    """
    meta: MetaBlock
    channel_availability: Dict[str, bool]   # channel role -> present in raw
    respiratory: Dict               # {"success", "events": List[RespiratoryEvent],
                                    #  "summary": ScoringSummary, "rejected_hypopneas",
                                    #  "n_gap_excluded", "n_local_baseline_rejected",
                                    #  "rule1a_arousal_stats", "scoring_thresholds", ...}
    spo2: SpO2Block
    plm: Dict                       # {"success", "error", "summary": PLMSummary, "events", "series"}
    position: Dict                  # {"success", "error", "summary": PositionSummary, ...}
    snore: Dict
    heart_rate: Dict
    signal_quality: Optional[Dict]  # RIP pair comparison; None when no effort pair
    arousal: Dict                   # {"success", "events", "summary", "source", ...}
    ahi_interval: AHIInterval
    channel_quality: ChannelQualityBlock
    anchor_baseline: AnchorBaselineBlock
    cheyne_stokes: CheyneStokesBlock
    hypoxic_burden: HypoxicBurdenBlock
    postprocess: PostprocessBlock
    split_night: Dict               # only when run_pneumo_analysis(..., split_night=...) is requested
