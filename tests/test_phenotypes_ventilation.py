"""v0.10.0: POSA / REM-predominant phenotype flags + ventilatory burden."""
import numpy as np
from psgscoring.ventilation import compute_ventilatory_burden
from psgscoring.pipeline import _compute_phenotypes


def test_vb_zero_when_at_baseline():
    # VB = % of time with amplitude < 50% of eupneic baseline.
    sf = 10.0
    fn = np.ones(int(sf * 600))                       # flow at eupneic baseline
    assert compute_ventilatory_burden(fn, sf) == 0.0


def test_vb_is_percentage_of_small_breaths():
    sf = 10.0
    fn = np.ones(int(sf * 600))
    fn[int(0 * sf):int(120 * sf)] = 0.3               # 120 s of 600 s = 20% "small breaths"
    vb = compute_ventilatory_burden(fn, sf)
    assert abs(vb - 20.0) < 0.1                        # bounded 0–100 %


def test_vb_threshold_is_strict_below_50pct():
    sf = 10.0
    fn = np.full(int(sf * 100), 0.5)                   # exactly 50% → NOT counted (<0.5)
    assert compute_ventilatory_burden(fn, sf) == 0.0


def test_vb_restricts_to_sleep():
    sf = 1.0
    # 4 epochs: W, N2, N2, W. Small breaths only in the two N2 (sleep) epochs.
    fn = np.ones(4 * 30)
    fn[30:90] = 0.2                                    # epochs 1–2 fully reduced
    hypno = ["W", "N2", "N2", "W"]
    vb = compute_ventilatory_burden(fn, sf, hypno=hypno)
    assert abs(vb - 100.0) < 0.1                       # all sleep time is "small"


def test_vb_guards():
    assert compute_ventilatory_burden(None, 10) is None
    assert compute_ventilatory_burden(np.array([]), 10) is None


def test_phenotypes_posa_and_rem():
    out = {
        "respiratory": {"summary": {"ahi_total": 20, "rem_ahi": 40, "nrem_ahi": 10}},
        "position": {"summary": {
            "ahi_per_pos": {"Supine": 30, "Left": 5, "Right": 6, "Prone": 0},
            "sleep_time_min": {"Supine": 120, "Left": 90, "Right": 60, "Prone": 0},
            "sleep_pct": {"Supine": 44},
        }},
    }
    hypno = ["R"] * 80 + ["N2"] * 400                  # 40 min REM
    _compute_phenotypes(out, hypno)
    ph = out["respiratory"]["summary"]["phenotypes"]
    assert ph["positional_osa"]["flag"] is True        # supine 30 ≥ 2× non-supine (~5.4)
    assert ph["positional_osa"]["ahi_non_supine"] == 5.4
    assert ph["rem_predominant"]["flag"] is True        # REM 40 ≥ 2× NREM 10


def test_phenotypes_absent_when_no_osa():
    out = {"respiratory": {"summary": {"ahi_total": 2, "rem_ahi": 3, "nrem_ahi": 2}},
           "position": {"summary": {}}}
    _compute_phenotypes(out, ["N2"] * 400)
    assert out["respiratory"]["summary"]["phenotypes"] == {}
