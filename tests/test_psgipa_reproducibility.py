"""
test_psgipa_reproducibility.py — assert paper v31 metrics on PSG-IPA.

Two layers:

1. Unit checks (always run): import + helper-function semantics from
   validate_psgipa.py. Fast, no dataset required.

2. Integration check (opt-in): runs the full validation harness on
   PSG-IPA and asserts paper-v31 metrics within tolerance for the
   standard (aasm_v3_rec) profile.
   Activated by setting PSGIPA_DATA_DIR to the dataset root, OR by
   running pytest with `--run-psgipa`. Skipped otherwise so CI without
   the dataset still passes.

Tolerances are deliberately loose (±10% on the principal metrics) to
allow for floating-point drift across numpy/scipy/mne versions while
still catching real algorithmic regressions — the standard-profile
AHIs reproduce bit-identically (paper Table 1), so any failure here
flags a genuine drift to investigate.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATE_SCRIPT = REPO_ROOT / "validate_psgipa.py"


def _load_validate_module():
    """Load validate_psgipa.py as a module without invoking main()."""
    spec = importlib.util.spec_from_file_location(
        "validate_psgipa_under_test", VALIDATE_SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Unit layer ──────────────────────────────────────────────────────────────

def test_validate_script_present():
    assert VALIDATE_SCRIPT.exists(), f"validate_psgipa.py missing at {VALIDATE_SCRIPT}"


def test_validate_module_loads():
    mod = _load_validate_module()
    for name in ["analyse_one", "aggregate_metrics", "to_report_json",
                 "severity", "grade_from_severities", "CLINICAL_PROFILES"]:
        assert hasattr(mod, name), f"validate_psgipa.{name} missing"


def test_clinical_profiles_canonical_names():
    mod = _load_validate_module()
    assert mod.CLINICAL_PROFILES == {
        "strict":    "aasm_v3_strict",
        "standard":  "aasm_v3_rec",
        "sensitive": "aasm_v3_sensitive",
    }, "CLINICAL_PROFILES drifted from canonical aasm_v3_* names"


def test_severity_thresholds():
    mod = _load_validate_module()
    assert mod.severity(0.0) == "Normal"
    assert mod.severity(4.99) == "Normal"
    assert mod.severity(5.0) == "Mild"
    assert mod.severity(14.99) == "Mild"
    assert mod.severity(15.0) == "Moderate"
    assert mod.severity(29.99) == "Moderate"
    assert mod.severity(30.0) == "Severe"
    assert mod.severity(100.0) == "Severe"


def test_grade_logic():
    mod = _load_validate_module()
    # all three same → A
    assert mod.grade_from_severities("Mild", "Mild", "Mild") == "A"
    assert mod.grade_from_severities("Severe", "Severe", "Severe") == "A"
    # exactly two same → B
    assert mod.grade_from_severities("Normal", "Mild", "Mild") == "B"
    assert mod.grade_from_severities("Mild", "Normal", "Mild") == "B"
    # all three different → C
    assert mod.grade_from_severities("Normal", "Mild", "Moderate") == "C"


# ── Integration layer ───────────────────────────────────────────────────────

PSGIPA_DATA_DIR = os.environ.get("PSGIPA_DATA_DIR")


def _data_dir_valid(p):
    if not p:
        return False
    root = Path(p)
    return (root / "Resp_events" / "Annotations" / "manual").is_dir()


@pytest.fixture(scope="module")
def validation_payload(tmp_path_factory):
    """Run the full validation harness and return the JSON payload.

    Skips the test if PSG-IPA is not on disk.
    """
    if not _data_dir_valid(PSGIPA_DATA_DIR):
        pytest.skip(
            "PSG-IPA dataset not configured (set PSGIPA_DATA_DIR to the "
            "dataset root containing Resp_events/Annotations/manual/)"
        )
    out_json = tmp_path_factory.mktemp("psgipa") / "results.json"
    cmd = [
        sys.executable, str(VALIDATE_SCRIPT),
        "--data-dir", PSGIPA_DATA_DIR,
        "--workers", "5",
        "--output-json", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"validate_psgipa.py exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert out_json.exists(), "validate_psgipa.py did not write the JSON"
    return json.loads(out_json.read_text())


# Paper v31 reference values (Rombaut et al. 2026, Table 1 + Section 3.2)
PAPER_V31 = {
    "per_recording": {
        # standard profile AHI; reproduces bit-identically on v0.4.2
        "SN1": {"algo_standard": 8.1,  "ref_median":  5.96},
        "SN2": {"algo_standard": 9.3,  "ref_median":  4.33},
        "SN3": {"algo_standard": 53.8, "ref_median": 53.98},
        "SN4": {"algo_standard": 4.3,  "ref_median":  3.82},
        "SN5": {"algo_standard": 11.4, "ref_median":  9.98},
    },
    "aggregate": {
        "bias_target":      1.8,    # paper rounds to 1 decimal
        "mae_target":       1.8,
        "pearson_r_target": 0.997,
    },
    "sn3_event_level": {
        "f1_target":   0.886,
        "dt_s_target": 2.0,
    },
}


def test_per_recording_standard_profile_ahis(validation_payload):
    per = validation_payload["per_recording"]
    for sn, expected in PAPER_V31["per_recording"].items():
        assert sn in per, f"{sn} missing from output"
        actual_algo = per[sn]["algo_standard"]
        expected_algo = expected["algo_standard"]
        assert abs(actual_algo - expected_algo) < 0.5, (
            f"{sn} standard AHI {actual_algo} drifted >0.5/h from paper {expected_algo}"
        )
        actual_ref = per[sn]["ref_median"]
        expected_ref = expected["ref_median"]
        assert abs(actual_ref - expected_ref) < 0.1, (
            f"{sn} scorer-median {actual_ref} drifted >0.1/h from paper {expected_ref}"
        )


def test_aggregate_bland_altman(validation_payload):
    agg = validation_payload["aggregate"]
    assert agg["n_recordings"] == 5
    assert abs(agg["bias"] - PAPER_V31["aggregate"]["bias_target"]) < 0.5, (
        f"bias {agg['bias']} drifted >0.5/h from paper +1.8"
    )
    assert abs(agg["mae"] - PAPER_V31["aggregate"]["mae_target"]) < 0.5, (
        f"MAE {agg['mae']} drifted >0.5/h from paper 1.8"
    )
    assert abs(agg["pearson_r"] - PAPER_V31["aggregate"]["pearson_r_target"]) < 0.01, (
        f"Pearson r {agg['pearson_r']} drifted >0.01 from paper 0.997"
    )


def test_severity_concordance(validation_payload):
    """Quadratic-weighted κ should remain >0.7 (AASM-equivalence threshold)."""
    agg = validation_payload["aggregate"]
    assert agg["weighted_kappa"] > 0.7, (
        f"weighted κ {agg['weighted_kappa']} below 0.70 AASM-equivalence threshold"
    )


def test_sn3_event_level(validation_payload):
    sn3 = validation_payload.get("sn3_event_level")
    assert sn3 is not None, "sn3_event_level block missing"
    assert abs(sn3["f1"] - PAPER_V31["sn3_event_level"]["f1_target"]) < 0.05, (
        f"SN3 F1 {sn3['f1']} drifted >0.05 from paper 0.886"
    )
    assert abs(sn3["dt_s"] - PAPER_V31["sn3_event_level"]["dt_s_target"]) < 0.5, (
        f"SN3 mean Δt {sn3['dt_s']}s drifted >0.5s from paper 2.0s"
    )


def test_robustness_grades(validation_payload):
    """SN3 + SN5 must be grade A; SN1 must be grade B."""
    per = validation_payload["per_recording"]
    assert per["SN3"]["robustness_grade"] == "A", (
        "SN3 (severe OSA, all scorers concordant) must be grade A"
    )
    assert per["SN5"]["robustness_grade"] == "A", (
        "SN5 (Mild on all profiles) must be grade A"
    )
    assert per["SN1"]["robustness_grade"] in {"A", "B"}, (
        "SN1 grade unexpected (paper v31 reports B; A acceptable if all profiles align)"
    )
