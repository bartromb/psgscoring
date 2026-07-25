"""Guards the v0.8.0 move of arousal & RERA detection into psgscoring.

These do NOT re-validate the numeric detector (that is covered by the golden
harness + the real-data PSG-IPA byte-identical check); they guard the structural
port: the API is in-package, the pipeline uses it in-package, and the detector
runs end-to-end on a synthetic signal without raising.
"""
import numpy as np


def test_arousal_api_exposed_in_package():
    import psgscoring
    for name in ("detect_arousals", "detect_reras", "run_arousal_respiratory_analysis"):
        assert hasattr(psgscoring, name), f"psgscoring.{name} missing after in-package move"


def test_pipeline_uses_in_package_arousal():
    import psgscoring.pipeline as pl
    # arousal is now always available in-package (no fragile external import)
    assert pl._AROUSAL_AVAILABLE is True
    assert pl.run_arousal_respiratory_analysis.__module__ == "psgscoring.arousal"


def test_detect_arousals_runs_on_synthetic_eeg():
    from psgscoring import detect_arousals
    sf = 100.0
    dur_s = 300
    rng = np.random.default_rng(0)
    eeg = rng.normal(0.0, 20.0, int(sf * dur_s))          # ~µV-scale noise
    hypno = ["N2"] * (dur_s // 30)                          # 30-s epochs
    res = detect_arousals(eeg, sf, hypno)
    assert isinstance(res, dict)
    assert res.get("success") is True
    assert isinstance(res.get("events"), list)
    assert isinstance(res.get("summary"), dict)


def test_lgbm_disabled_by_default():
    # Default (no env) must be the pure rule-based path — the reproducibility guarantee.
    from psgscoring.arousal import _is_arousal_lgbm_enabled
    assert _is_arousal_lgbm_enabled() is False


# ── multi-derivation ────────────────────────────────────────────────────────

def _synthetic_eeg():
    sf = 100.0
    dur_s = 600
    rng = np.random.default_rng(1)
    t = np.arange(int(sf * dur_s)) / sf
    eeg = rng.normal(0.0, 15.0, t.size)
    for c in (120, 250, 380, 470):                      # inject arousal-like bursts
        m = (t > c) & (t < c + 4)
        eeg[m] += 25 * np.sin(2 * np.pi * 10 * t[m]) + 15 * np.sin(2 * np.pi * 20 * t[m])
    hypno = ["N2"] * (dur_s // 30)
    return eeg, sf, hypno


def test_multi_one_derivation_is_byte_identical():
    """The hard invariant: multi with a single derivation == single-channel detect."""
    from psgscoring.arousal import detect_arousals, detect_arousals_multi
    eeg, sf, hypno = _synthetic_eeg()
    base = detect_arousals(eeg, sf, hypno)
    one = detect_arousals_multi([("C4-M1", eeg)], sf, hypno)
    onsets = lambda r: [round(e["onset_s"], 4) for e in r["events"]]
    assert one["events"] == base["events"]     # exact same objects/values
    assert onsets(one) == onsets(base)


def test_multi_union_dedups_and_tags_provenance():
    from psgscoring.arousal import detect_arousals, detect_arousals_multi
    eeg, sf, hypno = _synthetic_eeg()
    base = detect_arousals(eeg, sf, hypno)
    two = detect_arousals_multi([("C4-M1", eeg), ("O2-M1", eeg)], sf, hypno)
    # identical channels → the union must collapse to the same event count
    assert len(two["events"]) == len(base["events"])
    # every merged event carries provenance from both derivations
    assert all(set(e["derivations"]) == {"C4-M1", "O2-M1"} for e in two["events"])
    assert two["summary"]["n_derivations"] == 2


# ── Fase 2: thresholds-as-params + EOG reject ───────────────────────────────

def test_threshold_params_default_is_identical():
    """Explicit default thresholds == implicit (module) defaults — byte-identical."""
    from psgscoring.arousal import detect_arousals
    eeg, sf, hypno = _synthetic_eeg()
    a = detect_arousals(eeg, sf, hypno)
    b = detect_arousals(eeg, sf, hypno, ratio_thresh=2.0, abrupt_thresh=1.5)
    assert a["events"] == b["events"]


def test_higher_ratio_threshold_is_stricter():
    from psgscoring.arousal import detect_arousals
    eeg, sf, hypno = _synthetic_eeg()
    lo = detect_arousals(eeg, sf, hypno, ratio_thresh=2.0)
    hi = detect_arousals(eeg, sf, hypno, ratio_thresh=5.0)
    assert len(hi["events"]) <= len(lo["events"])


def test_eog_reject_drops_occipital_only_on_eye_movement():
    from psgscoring.arousal import _eog_reject_occipital
    sf = 100.0
    n = int(sf * 300)
    rng = np.random.default_rng(3)
    eog = rng.normal(0.0, 10.0, n)                 # quiet baseline
    eog[int(100 * sf):int(102 * sf)] += 300.0      # large eye movement at 100–102 s
    events = [
        {"onset_s": 100.5, "duration_s": 3.0, "derivations": ["EEG O2-M1"]},               # occ-only ON eye mvmt → drop
        {"onset_s": 150.0, "duration_s": 3.0, "derivations": ["EEG O2-M1"]},               # occ-only, quiet → keep
        {"onset_s": 100.5, "duration_s": 3.0, "derivations": ["EEG C4-M1", "EEG O2-M1"]},  # cross-confirmed → keep
    ]
    kept, dropped = _eog_reject_occipital(events, eog, sf, factor=3.0)
    assert dropped == 1
    keptset = {(e["onset_s"], tuple(e["derivations"])) for e in kept}
    assert (150.0, ("EEG O2-M1",)) in keptset               # quiet occipital survives
    assert (100.5, ("EEG C4-M1", "EEG O2-M1")) in keptset   # cross-confirmed survives


def test_profile_default_arousal_mode():
    """v0.9.0: clinical profiles default to multi; dataset (NSRR repro) stays single."""
    from psgscoring.constants import SCORING_PROFILES
    assert SCORING_PROFILES["aasm_v3_rec"]["AROUSAL_DERIVATION_MODE"] == "multi"
    assert SCORING_PROFILES["aasm_v3_strict"]["AROUSAL_DERIVATION_MODE"] == "multi"
    assert SCORING_PROFILES["mesa_shhs"]["AROUSAL_DERIVATION_MODE"] == "single"
