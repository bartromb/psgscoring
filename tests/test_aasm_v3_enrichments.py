"""v0.11.0: AASM v3 clinical enrichments — dual AHI (A1), CSR density (A2),
arousal aetiology (A3), apnea-cap flag (A4), hypopnea criterion (A5),
hypoventilation stub (A6). All output-additive; none touch the AHI/OSAS grade."""
import numpy as np
from psgscoring.pipeline import (
    _ahi_severity,
    _hypopnea_criterion_str,
    _compute_dual_ahi,
    _annotate_csr_density,
    _compute_arousal_etiology,
    _flag_apneas_at_cap,
    _mark_hypoventilation_not_assessed,
)

EPOCHS_2H = 240  # 240 × 30 s = 2 h


# ── A1: dual AHI (Rule 1A vs 1B/4%) ────────────────────────────────────────
#
# Rule 1B comes from the final event list, not from the `strict` arm of the
# robustness sweep. aasm_v3_strict keeps desat_or_arousal with a 3% threshold,
# so its AHI is a conservative Rule *1A* number; printing it under a "≥4%
# desat" heading stated a criterion that had never been applied.


def _ev(etype, desat=None):
    e = {"type": etype}
    if desat is not None:
        e["desaturation_pct"] = desat
    return e


def _out(events, ahi_total=22.0, tst_h=2.0, spo2_ok=True):
    return {
        "ahi_interval": {"standard": {"ahi": 22.0, "severity": "moderate"},
                         "strict":   {"ahi": 14.0, "severity": "mild"}},
        "spo2": {"success": spo2_ok},
        "respiratory": {"summary": {"ahi_total": ahi_total, "tst_hours": tst_h},
                        "events": events},
    }


def test_rule_1b_keeps_apneas_and_only_the_hypopneas_reaching_four_percent():
    events = [_ev("obstructive"), _ev("central"), _ev("mixed"),
              _ev("hypopnea", 4.5), _ev("hypopnea", 4.0),   # kept
              _ev("hypopnea", 3.2), _ev("hypopnea", None)]  # dropped (3–4%, arousal-only)
    out = _out(events)
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": True})
    d = out["respiratory"]["summary"]["ahi_dual"]
    assert d["rule_1a"]["ahi"] == 22.0                    # headline (3%-or-arousal)
    assert d["rule_1b_4pct"]["ahi"] == 2.5                # (3 apneas + 2 hyp) / 2 h


def test_rule_1b_never_exceeds_rule_1a():
    """The rules differ only in the hypopnea qualifier, so 1B ⊆ 1A."""
    events = [_ev("obstructive")] * 4 + [_ev("hypopnea", 6.0)] * 4
    out = _out(events, ahi_total=4.0)                     # 8 events / 2 h
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": True})
    d = out["respiratory"]["summary"]["ahi_dual"]
    assert d["rule_1b_4pct"]["ahi"] <= d["rule_1a"]["ahi"]


def test_rule_1b_is_never_below_the_apnea_index():
    """Apneas count under both rules — the failure the strict arm produced.

    A dead thermistor collapsed the sweep's strict arm to 10.3/h on a night
    with 78 apneas, i.e. below the apnea index alone. Deriving 1B from the
    event list makes that arithmetically impossible.
    """
    events = [_ev("central")] * 20 + [_ev("hypopnea", 3.0)] * 30
    out = _out(events, ahi_total=25.0)
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": True})
    d = out["respiratory"]["summary"]["ahi_dual"]
    assert d["rule_1b_4pct"]["ahi"] == 10.0               # 20 apneas / 2 h, no hypopneas
    assert d["rule_1b_4pct"]["ahi"] >= 20 / 2.0


def test_uncertain_apneas_are_excluded_exactly_as_ahi_total_excludes_them():
    """Counting them on one side only would let 1B exceed 1A."""
    events = [_ev("central"), _ev("uncertain"), _ev("uncertain")]
    out = _out(events)
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": True})
    assert out["respiratory"]["summary"]["ahi_dual"]["rule_1b_4pct"]["ahi"] == 0.5


def test_severity_is_derived_from_the_1b_number_itself():
    events = [_ev("central")] * 12                        # 6/h → mild
    out = _out(events)
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": True})
    d = out["respiratory"]["summary"]["ahi_dual"]
    assert d["rule_1b_4pct"]["ahi"] == 6.0
    assert d["rule_1b_4pct"]["severity"] == "mild"        # not the strict arm's label


def test_no_spo2_reports_nothing_rather_than_apneas_only():
    """Every desaturation is None then, and "apneas only" looks like a result."""
    events = [_ev("central")] * 10 + [_ev("hypopnea", None)] * 10
    out = _out(events, spo2_ok=False)
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": True})
    assert "ahi_dual" not in out["respiratory"]["summary"]


def test_a_four_percent_profile_reports_its_own_headline_as_rule_1b():
    """For CMS/AASM-v1 the headline already IS Rule 1B; 1A comes from the sweep."""
    out = _out([_ev("obstructive")], ahi_total=14.0)
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": False})
    d = out["respiratory"]["summary"]["ahi_dual"]
    assert d["rule_1b_4pct"]["ahi"] == 14.0               # headline, not the strict arm
    assert d["rule_1a"]["ahi"] == 22.0                    # standard (1A) arm


def test_dual_ahi_absent_on_interval_error_for_a_four_percent_profile():
    out = {"ahi_interval": {"error": "boom"},
           "respiratory": {"summary": {"ahi_total": 5, "tst_hours": 2.0}, "events": []}}
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": False})
    assert "ahi_dual" not in out["respiratory"]["summary"]


def test_dual_ahi_survives_a_missing_sweep_for_a_three_percent_profile():
    """The 1A path no longer needs the sweep at all."""
    out = {"spo2": {"success": True},
           "respiratory": {"summary": {"ahi_total": 10.0, "tst_hours": 2.0},
                           "events": [_ev("central")] * 8}}
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": True})
    assert out["respiratory"]["summary"]["ahi_dual"]["rule_1b_4pct"]["ahi"] == 4.0


def test_missing_tst_reports_nothing():
    out = _out([_ev("central")], tst_h=None)
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": True})
    assert "ahi_dual" not in out["respiratory"]["summary"]


# ── A2: Cheyne-Stokes density criterion G.1(b) ─────────────────────────────

def test_csr_density_met_requires_periodicity_and_density():
    events = [{"type": "central", "duration_s": 20}] * 12          # 12 central events
    out = {
        "cheyne_stokes": {"csr_detected": True},
        "respiratory": {"summary": {}, "events": events},
    }
    _annotate_csr_density(out, ["N2"] * EPOCHS_2H)                 # 2 h sleep → 6/h
    csr = out["cheyne_stokes"]
    assert csr["central_events"] == 12
    assert csr["central_events_per_h"] == 6.0
    assert csr["density_criterion_met"] is True
    assert csr["criteria_met"] is True                            # periodicity AND density


def test_csr_criteria_not_met_when_periodicity_absent():
    events = [{"type": "central", "duration_s": 20}] * 12
    out = {"cheyne_stokes": {"csr_detected": False},
           "respiratory": {"summary": {}, "events": events}}
    _annotate_csr_density(out, ["N2"] * EPOCHS_2H)
    assert out["cheyne_stokes"]["density_criterion_met"] is True
    assert out["cheyne_stokes"]["criteria_met"] is False          # no periodicity


def test_csr_density_not_met_below_5_per_hour():
    events = [{"type": "central", "duration_s": 20}] * 4          # 4 events over 2 h = 2/h
    out = {"cheyne_stokes": {"csr_detected": True},
           "respiratory": {"summary": {}, "events": events}}
    _annotate_csr_density(out, ["N2"] * EPOCHS_2H)
    assert out["cheyne_stokes"]["density_criterion_met"] is False


# ── A3: arousal aetiology indices ──────────────────────────────────────────

def test_arousal_etiology_indices_sum_to_arousal_index():
    # arousal_index (25.0) is split by aetiology fraction; resp + spont == AI exactly,
    # regardless of the (artifact-corrected) TST used to compute the index.
    out = {
        "arousal": {"summary": {"arousal_index": 25.0,
                                "n_respiratory_arousals": 40,
                                "n_spontaneous_arousals": 10},
                    "events": [{"onset_s": 100.0}]},
        "plm": {"events": [{"onset_s": 99.8, "duration_s": 1.0},   # arousal @100 within window
                           {"onset_s": 500.0, "duration_s": 1.0}]},  # no arousal near
    }
    _compute_arousal_etiology(out, ["N2"] * EPOCHS_2H)
    s = out["arousal"]["summary"]
    assert s["respiratory_arousal_index"] == 20.0                 # 25 × 40/50
    assert s["spontaneous_arousal_index"] == 5.0                  # 25 × 10/50
    # the whole point of the fix: sub-indices reconstitute the total
    assert round(s["respiratory_arousal_index"] + s["spontaneous_arousal_index"], 1) == 25.0
    assert s["n_plm_arousals"] == 1
    assert s["plm_arousal_index"] == 0.5                          # 25 × 1/50 (subset of spont)


# ── A4: apnea-at-cap flag ──────────────────────────────────────────────────

def test_apnea_cap_flag_counts_events_at_cap():
    events = [
        {"type": "central", "duration_s": 90.0},      # at cap
        {"type": "obstructive", "duration_s": 89.5},  # within 1 s of cap
        {"type": "obstructive", "duration_s": 30.0},  # normal
        {"type": "hypopnea", "duration_s": 90.0},     # hypopnea, not counted
    ]
    out = {"respiratory": {"summary": {}, "events": events}}
    _flag_apneas_at_cap(out, {"APNEA_MAX_DUR_S": 90.0})
    assert out["respiratory"]["summary"]["n_apneas_at_max_dur"] == 2
    assert out["respiratory"]["summary"]["apnea_max_dur_s"] == 90.0


# ── A5: hypopnea criterion string ──────────────────────────────────────────

def test_hypopnea_criterion_rule_1a():
    prof = {"HYPOPNEA_THRESHOLD": 0.70, "DESATURATION_DROP_PCT": 3.0,
            "DESAT_OR_AROUSAL": True, "_AASM_RULE": "1A"}
    s = _hypopnea_criterion_str(prof)
    assert "30%" in s and "3%" in s and "arousal" in s and s.startswith("1A")


def test_hypopnea_criterion_rule_1b_desat_only():
    prof = {"HYPOPNEA_THRESHOLD": 0.70, "DESATURATION_DROP_PCT": 4.0,
            "DESAT_REQUIRED": True, "DESAT_OR_AROUSAL": False}
    s = _hypopnea_criterion_str(prof)
    assert "4%" in s and "arousal" not in s


# ── A6: hypoventilation scope statement ────────────────────────────────────

def test_hypoventilation_marked_not_assessed():
    out = {"respiratory": {"summary": {}}}
    _mark_hypoventilation_not_assessed(out)
    hv = out["respiratory"]["summary"]["hypoventilation"]
    assert hv["assessed"] is False and "PCO2" in hv["reason"]


# ── severity helper ────────────────────────────────────────────────────────

def test_ahi_severity_bands():
    assert _ahi_severity(3) == "normal"
    assert _ahi_severity(10) == "mild"
    assert _ahi_severity(20) == "moderate"
    assert _ahi_severity(40) == "severe"
    assert _ahi_severity(None) is None
