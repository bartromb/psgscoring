"""v0.10.0: POSA / REM-predominant phenotype flags + ventilatory burden."""
import numpy as np
from psgscoring.ventilation import compute_ventilatory_burden
from psgscoring.pipeline import _compute_phenotypes


def _breaths(sf, total_s, period_s=4.0):
    return [{"onset_s": t, "duration_s": period_s}
            for t in np.arange(0, total_s, period_s)]


def test_vb_zero_when_all_breaths_normal():
    # VB = % of breaths whose PEAK is < 50% of the eupneic baseline.
    sf = 10.0
    fn = np.ones(int(sf * 400))                        # every breath peaks at 1.0
    assert compute_ventilatory_burden(fn, sf, _breaths(sf, 400)) == 0.0


def test_vb_counts_reduced_breaths():
    sf = 10.0
    fn = np.ones(int(sf * 400))
    fn[:int(80 * sf)] = 0.3                            # first 20 of 100 breaths reduced
    vb = compute_ventilatory_burden(fn, sf, _breaths(sf, 400))
    assert abs(vb - 20.0) < 0.1                        # 20 / 100 breaths


def test_vb_ignores_inter_breath_troughs():
    # A normal signal whose envelope dips to 0.2 *between* breaths must NOT inflate VB:
    # each breath's PEAK is 1.0, so VB = 0 even though many samples are < 0.5.
    sf = 10.0
    fn = np.full(int(sf * 400), 0.2)
    for t in np.arange(0, 400, 4.0):                   # a 1 s peak (=1.0) inside each breath
        fn[int((t + 1) * sf):int((t + 2) * sf)] = 1.0
    assert compute_ventilatory_burden(fn, sf, _breaths(sf, 400)) == 0.0


def test_vb_threshold_is_strict_below_50pct():
    sf = 10.0
    fn = np.full(int(sf * 400), 0.5)                   # peak exactly 0.5 → NOT counted (<0.5)
    assert compute_ventilatory_burden(fn, sf, _breaths(sf, 400)) == 0.0


def test_vb_restricts_to_sleep():
    sf = 1.0
    fn = np.ones(4 * 30)
    fn[30:90] = 0.2                                    # epochs 1–2 (sleep) reduced
    hypno = ["W", "N2", "N2", "W"]
    br = [{"onset_s": float(t), "duration_s": 1.0} for t in range(120)]
    vb = compute_ventilatory_burden(fn, sf, br, hypno=hypno)
    assert abs(vb - 100.0) < 0.1                       # all sleep breaths are small


def test_vb_guards():
    assert compute_ventilatory_burden(None, 10, [{"onset_s": 0}]) is None
    assert compute_ventilatory_burden(np.ones(10), 10, []) is None


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
