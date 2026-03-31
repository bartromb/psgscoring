"""
psgscoring — AASM 2.6-compliant respiratory event scoring for polysomnography.

No deep learning, no GPU — pure signal processing with scipy and numpy.

Example
-------
>>> import mne
>>> from psgscoring import run_full_analysis
>>> raw = mne.io.read_raw_edf("recording.edf", preload=True)
>>> hypno = [...]  # stage labels from YASA or manual scoring
>>> results = run_full_analysis(raw, hypno)
>>> print(f"AHI: {results['respiratory']['summary']['ahi_total']}")
"""

__version__ = "0.1.0"
__author__ = "Bart Rombaut"

from ._core import (
    linearize_nasal_pressure, compute_mmsd, preprocess_flow,
    compute_dynamic_baseline, compute_stage_baseline, bandpass_flow,
    detect_breaths, compute_breath_amplitudes, compute_flattening_index,
    detect_respiratory_events, classify_apnea_type,
    analyze_spo2, analyze_plm, analyze_position,
    analyze_heart_rate, analyze_snore, detect_cheyne_stokes,
    reinstate_rule1b_hypopneas,
    run_pneumo_analysis as run_full_analysis,
    build_sleep_mask, is_nrem, is_rem, is_sleep, detect_channels,
)
from ._arousal import detect_arousals, run_arousal_respiratory_analysis
