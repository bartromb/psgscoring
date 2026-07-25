"""v0.10.0: POSA / REM-predominant phenotype flags + ventilatory burden."""
import numpy as np
from psgscoring.ventilation import compute_ventilatory_burden
from psgscoring.pipeline import _compute_phenotypes


def test_vb_zero_when_at_baseline():
    sf = 10.0
    fn = np.ones(int(sf * 600))                       # flow at eupneic baseline
    ev = [{"onset_s": 100, "duration_s": 20}]
    assert compute_ventilatory_burden(fn, sf, ev, tst_h=1.0) == 0.0


def test_vb_integrates_event_deficit():
    sf = 10.0
    fn = np.ones(int(sf * 600))
    fn[int(100 * sf):int(120 * sf)] = 0.5             # 50% deficit for 20 s
    vb = compute_ventilatory_burden(fn, sf, [{"onset_s": 100, "duration_s": 20}], tst_h=1.0)
    assert abs(vb - 16.67) < 0.1                       # 0.5 * 20 s = 16.67 %·min/h


def test_vb_guards():
    assert compute_ventilatory_burden(None, 10, [], 1.0) is None
    assert compute_ventilatory_burden(np.ones(10), 10, [{"onset_s": 0, "duration_s": 1}], 0) is None


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
