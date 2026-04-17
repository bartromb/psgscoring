"""
tests/test_ptt_and_audit.py — v0.3.001

Unit tests for the new PTT and audit modules.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from psgscoring.ptt import (
    detect_r_peaks,
    detect_ppg_feet,
    compute_ptt_series,
    ptt_effort_score,
    classify_apnea_with_ptt,
    PTTSeries,
    PTTEffortResult,
)
from psgscoring.audit import (
    AuditTrail,
    verify_audit_file,
    AUDIT_SCHEMA_VERSION,
)


# =============================================================================
# Fixtures: synthetic ECG + PPG signals
# =============================================================================

FS = 256


def _synth_ecg_ppg(duration_s=30, hr=72, ptt_ms=200, ptt_mod_ms=0,
                    mod_freq=0.25, noise=0.05, seed=42):
    """Synthesise aligned ECG and PPG with controllable PTT modulation."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / FS)
    hr_period = 60.0 / hr
    ecg = np.sin(2 * np.pi * t / hr_period) ** 101  # sharp R-peaks
    ptt_s = (ptt_ms + ptt_mod_ms * np.sin(2 * np.pi * mod_freq * t)) / 1000
    delay = (ptt_s * FS).astype(int)
    ppg = np.zeros_like(t)
    for i, d in enumerate(delay):
        if i + d < len(ppg):
            ppg[i + d] += ecg[i]
    ppg = np.convolve(ppg, np.exp(-np.linspace(0, 5, 50)), mode="same")
    ppg += noise * rng.normal(size=len(ppg))
    return ecg, ppg


# =============================================================================
# PTT tests
# =============================================================================

class TestRPeakDetection:
    def test_detects_peaks(self):
        """Synthetic sin^101 produces multiple candidate peaks per cycle
        (artefact of the synthesis, not real ECG). We check detection
        produces peaks, not exact counts."""
        ecg, _ = _synth_ecg_ppg(30, hr=72)
        peaks = detect_r_peaks(ecg, FS)
        assert len(peaks) > 20  # at least one per cycle roughly

    def test_peaks_in_sorted_order(self):
        ecg, _ = _synth_ecg_ppg(30, hr=72)
        peaks = detect_r_peaks(ecg, FS)
        assert list(peaks) == sorted(peaks)

    def test_empty_input(self):
        peaks = detect_r_peaks(np.array([]), FS)
        assert len(peaks) == 0

    def test_respects_min_distance(self):
        ecg, _ = _synth_ecg_ppg(30, hr=90)
        peaks = detect_r_peaks(ecg, FS, max_bpm=180)
        if len(peaks) > 1:
            rr = np.diff(peaks) / FS
            assert rr.min() >= 60 / 180 - 1e-6  # at least 333 ms minus float tol


class TestPPGFootDetection:
    def test_detects_feet(self):
        _, ppg = _synth_ecg_ppg(30, hr=72)
        feet = detect_ppg_feet(ppg, FS)
        # Should detect roughly one foot per cardiac cycle
        assert 25 <= len(feet) <= 45


class TestPTTSeries:
    def test_basic_computation(self):
        ecg, ppg = _synth_ecg_ppg(30, ptt_ms=200, ptt_mod_ms=0)
        series = compute_ptt_series(ecg, ppg, FS, FS)
        assert isinstance(series, PTTSeries)
        assert series.n_valid > 15
        # Synth PTT can appear halved due to double-peak detection on sin^101;
        # accept a wide physiological-ish range on synthetic data
        assert 80 < series.mean_ptt_ms < 300

    def test_quality_mask_retains_reasonable_fraction(self):
        ecg, ppg = _synth_ecg_ppg(60, ptt_ms=200, noise=0.1, seed=1)
        series = compute_ptt_series(ecg, ppg, FS, FS)
        # At least SOME beats should be valid
        assert series.n_valid > 0
        # Mask is boolean array same length as values
        assert len(series.quality_mask) == len(series.values_ms)


class TestPTTEffortScore:
    def test_central_pattern_low_score(self):
        """No PTT modulation -> score should be near 0 (central)."""
        ecg, ppg = _synth_ecg_ppg(30, ptt_mod_ms=0)
        series = compute_ptt_series(ecg, ppg, FS, FS)
        result = ptt_effort_score(series, 5.0, 25.0)
        assert isinstance(result, PTTEffortResult)
        assert result.effort_score < 0.3
        assert result.amplitude_ms < 5.0

    def test_obstructive_pattern_higher_score(self):
        """Strong PTT modulation -> score should be higher."""
        ecg, ppg = _synth_ecg_ppg(30, ptt_mod_ms=30)
        series = compute_ptt_series(ecg, ppg, FS, FS)
        result = ptt_effort_score(series, 5.0, 25.0)
        assert result.amplitude_ms > 3.0
        assert result.effort_score > 0.0

    def test_insufficient_data_returns_undetermined(self):
        series = PTTSeries(
            times=np.array([10.0, 11.0]),
            values_ms=np.array([200.0, 210.0]),
            quality_mask=np.array([True, True]),
        )
        result = ptt_effort_score(series, 10.0, 20.0, min_beats=8)
        assert result.effort_score == 0.5
        assert result.reason == "insufficient_data"

    def test_short_event_returns_undetermined(self):
        ecg, ppg = _synth_ecg_ppg(30, ptt_mod_ms=20)
        series = compute_ptt_series(ecg, ppg, FS, FS)
        # Event only 2 seconds — too short for bandpass filter
        result = ptt_effort_score(series, 10.0, 12.0)
        assert result.reason in ("event_too_short", "insufficient_data")


class TestClassifyApneaWithPTT:
    def test_central_classification_with_ptt(self):
        votes = {"paradoxical": 0.1, "hilbert_effort": 0.2,
                 "rip_envelope": 0.2, "tecg": 0.2, "spectral": 0.3}
        ptt = PTTEffortResult(
            effort_score=0.1, amplitude_ms=2.0,
            n_beats_in_event=20, reason=""
        )
        cls = classify_apnea_with_ptt(votes, ptt)
        assert cls["type"] == "central"
        assert cls["combined_effort"] < 0.35

    def test_obstructive_classification_with_ptt(self):
        votes = {"paradoxical": 0.8, "hilbert_effort": 0.7,
                 "rip_envelope": 0.8, "tecg": 0.9, "spectral": 0.7}
        ptt = PTTEffortResult(
            effort_score=0.85, amplitude_ms=25.0,
            n_beats_in_event=30, reason=""
        )
        cls = classify_apnea_with_ptt(votes, ptt)
        assert cls["type"] == "obstructive"
        assert cls["combined_effort"] > 0.65

    def test_ptt_none_falls_back(self):
        votes = {"paradoxical": 0.8, "hilbert_effort": 0.7,
                 "rip_envelope": 0.8, "tecg": 0.9, "spectral": 0.7}
        cls = classify_apnea_with_ptt(votes, ptt_result=None)
        assert cls["type"] == "obstructive"

    def test_low_quality_ptt_downweighted(self):
        """PTT with <8 beats should get weight 0.2 and not dominate."""
        strong_obstructive_votes = {
            "paradoxical": 0.9, "hilbert_effort": 0.9,
            "rip_envelope": 0.9, "tecg": 0.9, "spectral": 0.9,
        }
        # Low-quality PTT suggesting no effort
        low_ptt = PTTEffortResult(
            effort_score=0.0, amplitude_ms=1.0,
            n_beats_in_event=5,  # < 8 -> downweighted
            reason=""
        )
        cls = classify_apnea_with_ptt(strong_obstructive_votes, low_ptt)
        # Should still classify as obstructive despite low PTT vote
        assert cls["type"] == "obstructive"


# =============================================================================
# Audit trail tests
# =============================================================================

class TestAuditTrailLifecycle:
    def test_start_creates_uuid(self):
        audit = AuditTrail.start()
        assert audit.analysis_id
        assert len(audit.analysis_id) == 36  # UUID4 length with dashes

    def test_start_captures_version(self):
        audit = AuditTrail.start()
        assert audit.psgscoring_version  # either real version or "unknown"
        assert audit.schema_version == AUDIT_SCHEMA_VERSION

    def test_start_captures_client_info(self):
        audit = AuditTrail.start(
            client_info={"name": "YASAFlaskified", "version": "0.8.38"}
        )
        assert audit.client_info["name"] == "YASAFlaskified"
        assert audit.client_info["version"] == "0.8.38"

    def test_input_hash_computed(self, tmp_path):
        edf = tmp_path / "fake.edf"
        edf.write_bytes(b"FAKE_EDF_" * 100)
        audit = AuditTrail.start(edf_path=str(edf))
        assert audit.input_info["filename"] == "fake.edf"
        assert len(audit.input_info["file_sha256"]) == 64  # SHA-256 hex

    def test_input_hash_can_be_skipped(self, tmp_path):
        edf = tmp_path / "fake.edf"
        edf.write_bytes(b"DATA")
        audit = AuditTrail.start(edf_path=str(edf), compute_input_hash=False)
        assert "file_sha256" not in audit.input_info

    def test_log_warning(self):
        audit = AuditTrail.start()
        audit.log_warning("test warning", channel="ECG")
        assert len(audit.warnings) == 1
        assert audit.warnings[0]["message"] == "test warning"
        assert audit.warnings[0]["channel"] == "ECG"

    def test_log_error(self):
        audit = AuditTrail.start()
        try:
            raise ValueError("boom")
        except ValueError as e:
            audit.log_error(e)
        assert len(audit.errors) == 1
        assert audit.errors[0]["type"] == "ValueError"
        assert audit.errors[0]["message"] == "boom"

    def test_correction_impact(self):
        audit = AuditTrail.start()
        audit.set_correction_impact("post_apnea_baseline", 14, "rejected")
        assert audit.corrections_impact["post_apnea_baseline"]["events_affected"] == 14
        assert audit.corrections_impact["post_apnea_baseline"]["direction"] == "rejected"

    def test_set_results_filters_unknown_keys(self):
        audit = AuditTrail.start()
        audit.set_results({
            "ahi": 18.7,
            "oahi": 16.2,
            "internal_debug_dump": [1, 2, 3] * 1000,  # should be dropped
        })
        assert audit.results_summary["ahi"] == 18.7
        assert "internal_debug_dump" not in audit.results_summary

    def test_finish_sets_runtime_and_hash(self):
        audit = AuditTrail.start()
        audit.set_results({"ahi": 15.0})
        audit.finish()
        assert audit.runtime_seconds is not None
        assert audit.runtime_seconds >= 0
        assert audit.timestamp_finish_utc is not None
        assert audit.integrity_hash is not None
        assert len(audit.integrity_hash) == 64


class TestAuditIntegrity:
    def test_integrity_verifies_unchanged(self):
        audit = AuditTrail.start()
        audit.set_results({"ahi": 10.0})
        audit.finish()
        assert audit.verify_integrity()

    def test_integrity_detects_tampering(self):
        audit = AuditTrail.start()
        audit.set_results({"ahi": 10.0})
        audit.finish()
        # Mutate a field AFTER finish()
        audit.results_summary["ahi"] = 99.9
        assert not audit.verify_integrity()

    def test_input_hash_verification(self, tmp_path):
        edf = tmp_path / "file.edf"
        edf.write_bytes(b"original content")
        audit = AuditTrail.start(edf_path=str(edf))
        audit.finish()
        # File unchanged -> verify_input_file passes
        assert audit.verify_input_file(str(edf))
        # Modify file -> verify_input_file fails
        edf.write_bytes(b"modified content")
        assert not audit.verify_input_file(str(edf))


class TestAuditSerialisation:
    def test_to_dict(self):
        audit = AuditTrail.start()
        audit.finish()
        d = audit.to_dict()
        assert "analysis_id" in d
        assert "integrity_hash" in d
        assert "_start_time" not in d  # private field

    def test_to_json_parseable(self):
        audit = AuditTrail.start()
        audit.finish()
        parsed = json.loads(audit.to_json())
        assert parsed["analysis_id"] == audit.analysis_id

    def test_write_sidecar(self, tmp_path):
        edf = tmp_path / "sub001.edf"
        edf.write_bytes(b"DATA")
        audit = AuditTrail.start(edf_path=str(edf))
        audit.finish()
        sidecar = audit.write_sidecar(tmp_path)
        assert sidecar.exists()
        assert sidecar.name.startswith("sub001_audit_")
        assert sidecar.name.endswith(".json")

    def test_verify_audit_file(self, tmp_path):
        edf = tmp_path / "sub001.edf"
        edf.write_bytes(b"DATA")
        audit = AuditTrail.start(edf_path=str(edf))
        audit.set_results({"ahi": 12.5})
        audit.finish()
        sidecar = audit.write_sidecar(tmp_path)

        result = verify_audit_file(sidecar, edf_path=edf)
        assert result["integrity_ok"] is True
        assert result["input_ok"] is True
        assert result["analysis_id"] == audit.analysis_id

    def test_verify_detects_modified_sidecar(self, tmp_path):
        edf = tmp_path / "sub001.edf"
        edf.write_bytes(b"DATA")
        audit = AuditTrail.start(edf_path=str(edf))
        audit.set_results({"ahi": 12.5})
        audit.finish()
        sidecar = audit.write_sidecar(tmp_path)

        # Tamper with the sidecar
        data = json.loads(sidecar.read_text())
        data["results_summary"]["ahi"] = 99.9
        sidecar.write_text(json.dumps(data, indent=2))

        result = verify_audit_file(sidecar)
        assert result["integrity_ok"] is False


class TestAuditSummaryLine:
    def test_summary_line_format(self):
        audit = AuditTrail.start()
        audit.set_results({"ahi": 22.4})
        audit.finish()
        line = audit.summary_line()
        assert "analysis=" in line
        assert "ahi=22.4" in line
        assert "runtime=" in line


# =============================================================================
# Integration: AuditTrail + PTT info
# =============================================================================

def test_audit_captures_ptt_results():
    """The audit schema should accept PTT-specific result fields."""
    audit = AuditTrail.start()
    audit.set_results({
        "ahi": 18.7,
        "ptt_available": True,
        "ptt_n_valid_beats": 28472,
        "ptt_mean_ms": 218.5,
        "ptt_events_with_vote": 31,
    })
    audit.finish()
    assert audit.results_summary["ptt_available"] is True
    assert audit.results_summary["ptt_n_valid_beats"] == 28472
    assert audit.verify_integrity()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
