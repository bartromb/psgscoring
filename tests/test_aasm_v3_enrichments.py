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

def test_dual_ahi_uses_headline_for_1a_and_strict_for_1b():
    out = {
        "ahi_interval": {
            "standard": {"ahi": 22.0, "severity": "moderate"},
            "strict":   {"ahi": 14.0, "severity": "mild"},
        },
        "respiratory": {"summary": {"ahi_total": 22.0}},
    }
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": True})
    d = out["respiratory"]["summary"]["ahi_dual"]
    assert d["rule_1a"]["ahi"] == 22.0            # headline (3%/arousal)
    assert d["rule_1a"]["severity"] == "moderate"
    assert d["rule_1b_4pct"]["ahi"] == 14.0       # strict 4% pass
    assert d["rule_1b_4pct"]["severity"] == "mild"


def test_dual_ahi_falls_back_to_standard_pass_for_desat_only_profile():
    out = {
        "ahi_interval": {
            "standard": {"ahi": 22.0, "severity": "moderate"},
            "strict":   {"ahi": 14.0, "severity": "mild"},
        },
        "respiratory": {"summary": {"ahi_total": 14.0}},  # primary is a 4%/CMS profile
    }
    _compute_dual_ahi(out, {"DESAT_OR_AROUSAL": False})
    d = out["respiratory"]["summary"]["ahi_dual"]
    assert d["rule_1a"]["ahi"] == 22.0            # taken from the standard(1A) pass


def test_dual_ahi_absent_on_interval_error():
    out = {"ahi_interval": {"error": "boom"}, "respiratory": {"summary": {"ahi_total": 5}}}
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
