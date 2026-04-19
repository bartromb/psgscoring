"""
tests/test_signal_quality.py — Unit tests voor psgscoring v0.2.96 signal_quality

Run standalone:
    cd psgscoring
    python3 tests/test_signal_quality.py

Or via pytest:
    python3 -m pytest tests/test_signal_quality.py -v
"""
import sys
import os

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_TEST_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np
from psgscoring.signal_quality import (
    assess_rip_channel,
    compare_rip_pair,
    single_channel_fallback_classify,
    quality_warning_text,
    quality_badge_summary,
)


def make_breathing_signal(duration_s=600, sf=256, breath_freq=0.25, amplitude=0.5, noise=0.05):
    n = int(duration_s * sf)
    t = np.arange(n) / sf
    signal = amplitude * np.sin(2 * np.pi * breath_freq * t)
    signal += noise * np.random.randn(n)
    return signal


def make_dead_signal(duration_s=600, sf=256, noise=0.002):
    n = int(duration_s * sf)
    return noise * np.random.randn(n)


def make_weak_signal(duration_s=600, sf=256, breath_freq=0.25, amplitude=0.02):
    return make_breathing_signal(duration_s, sf, breath_freq, amplitude, noise=0.01)


def test_assess_normal_channel():
    np.random.seed(42)
    sig = make_breathing_signal()
    q = assess_rip_channel(sig, sf=256)
    assert q["status"] == "ok", f"Expected 'ok', got {q['status']}: {q['reason']}"
    assert 0.20 < q["peak_freq"] < 0.30, f"Peak freq {q['peak_freq']} not near 0.25 Hz"


def test_assess_dead_channel():
    np.random.seed(42)
    sig = make_dead_signal()
    q = assess_rip_channel(sig, sf=256)
    assert q["status"] == "failed", f"Expected 'failed', got {q['status']}: {q['reason']}"


def test_assess_weak_channel():
    np.random.seed(42)
    sig = make_weak_signal()
    q = assess_rip_channel(sig, sf=256)
    assert q["status"] in ("weak", "ok"), f"Expected 'weak' or 'ok', got {q['status']}"


def test_assess_empty_signal():
    q = assess_rip_channel(np.array([]), sf=256)
    assert q["status"] == "failed"


def test_compare_both_ok():
    np.random.seed(42)
    thor = make_breathing_signal()
    abd = make_breathing_signal(breath_freq=0.24, amplitude=0.48)
    q = compare_rip_pair(thor, abd, sf=256)
    assert q["classification_reliable"] is True
    assert q["recommended_mode"] == "bilateral"
    assert q["working_channel"] is None


def test_compare_dead_thorax():
    np.random.seed(42)
    thor = make_dead_signal()
    abd = make_breathing_signal()
    q = compare_rip_pair(thor, abd, sf=256)
    assert q["classification_reliable"] is False
    assert q["recommended_mode"] == "single-channel"
    assert q["working_channel"] == "abdomen"
    assert q["thorax"]["status"] == "failed"
    assert q["abdomen"]["status"] == "ok"


def test_compare_dead_abdomen():
    np.random.seed(42)
    thor = make_breathing_signal()
    abd = make_dead_signal()
    q = compare_rip_pair(thor, abd, sf=256)
    assert q["working_channel"] == "thorax"


def test_compare_both_dead():
    np.random.seed(42)
    thor = make_dead_signal()
    abd = make_dead_signal()
    q = compare_rip_pair(thor, abd, sf=256)
    assert q["recommended_mode"] == "unreliable"
    assert q["working_channel"] == "none"


def test_compare_extreme_asymmetry():
    np.random.seed(42)
    thor = make_breathing_signal(amplitude=0.5)
    abd = make_breathing_signal(amplitude=0.005)
    q = compare_rip_pair(thor, abd, sf=256)
    assert q["energy_ratio"] > 100, f"Expected >100x ratio, got {q['energy_ratio']:.1f}"
    assert q["recommended_mode"] in ("single-channel", "unreliable")


def test_fallback_central():
    np.random.seed(42)
    sf = 256
    baseline = make_breathing_signal(300, sf, amplitude=0.5)
    silence = 0.001 * np.random.randn(int(30 * sf))
    full = np.concatenate([baseline, silence])

    result = single_channel_fallback_classify(
        apnea_start_s=300, apnea_end_s=330,
        effort_signal=full, sf=sf,
    )
    assert result == "central", f"Expected 'central', got '{result}'"


def test_fallback_obstructive():
    np.random.seed(42)
    sf = 256
    baseline = make_breathing_signal(300, sf, amplitude=0.5)
    continued_effort = make_breathing_signal(30, sf, amplitude=0.5)
    full = np.concatenate([baseline, continued_effort])

    result = single_channel_fallback_classify(
        apnea_start_s=300, apnea_end_s=330,
        effort_signal=full, sf=sf,
    )
    assert result == "obstructive", f"Expected 'obstructive', got '{result}'"


def test_fallback_uncertain():
    np.random.seed(42)
    sf = 256
    baseline = make_breathing_signal(300, sf, amplitude=0.5)
    reduced = make_breathing_signal(30, sf, amplitude=0.17)
    full = np.concatenate([baseline, reduced])

    result = single_channel_fallback_classify(
        apnea_start_s=300, apnea_end_s=330,
        effort_signal=full, sf=sf,
    )
    assert result == "uncertain", f"Expected 'uncertain', got '{result}'"


def test_fallback_too_short():
    sf = 256
    signal = make_breathing_signal(100, sf)
    result = single_channel_fallback_classify(
        apnea_start_s=50, apnea_end_s=50.5,
        effort_signal=signal, sf=sf,
    )
    assert result == "uncertain"


def test_warning_none_when_reliable():
    np.random.seed(42)
    q = compare_rip_pair(make_breathing_signal(), make_breathing_signal(), sf=256)
    assert quality_warning_text(q) is None


def test_warning_generated_when_failed():
    np.random.seed(42)
    q = compare_rip_pair(make_dead_signal(), make_breathing_signal(), sf=256)
    w = quality_warning_text(q, lang="en")
    assert w is not None
    assert "WARNING" in w.upper()
    assert "failed" in w.lower() or "thorax" in w.lower()


def test_warning_multilingual():
    np.random.seed(42)
    q = compare_rip_pair(make_dead_signal(), make_breathing_signal(), sf=256)
    for lang in ("nl", "fr", "de"):
        w = quality_warning_text(q, lang=lang)
        assert w is not None
        assert len(w) > 50, f"Lang {lang}: warning too short"


def test_badge_ok():
    np.random.seed(42)
    q = compare_rip_pair(make_breathing_signal(), make_breathing_signal(), sf=256)
    b = quality_badge_summary(q)
    assert b["level"] == "ok"
    assert b["label"] == "OK"


def test_badge_danger_when_both_failed():
    np.random.seed(42)
    q = compare_rip_pair(make_dead_signal(), make_dead_signal(), sf=256)
    b = quality_badge_summary(q)
    assert b["level"] == "danger"
    assert "Failed" in b["label"]


def test_badge_warning_single_channel():
    np.random.seed(42)
    q = compare_rip_pair(make_dead_signal(), make_breathing_signal(), sf=256)
    b = quality_badge_summary(q)
    assert b["level"] == "warning"
    assert "abdomen" in b["label"].lower()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    tests = [
        ("test_assess_normal_channel", test_assess_normal_channel),
        ("test_assess_dead_channel", test_assess_dead_channel),
        ("test_assess_weak_channel", test_assess_weak_channel),
        ("test_assess_empty_signal", test_assess_empty_signal),
        ("test_compare_both_ok", test_compare_both_ok),
        ("test_compare_dead_thorax", test_compare_dead_thorax),
        ("test_compare_dead_abdomen", test_compare_dead_abdomen),
        ("test_compare_both_dead", test_compare_both_dead),
        ("test_compare_extreme_asymmetry", test_compare_extreme_asymmetry),
        ("test_fallback_central", test_fallback_central),
        ("test_fallback_obstructive", test_fallback_obstructive),
        ("test_fallback_uncertain", test_fallback_uncertain),
        ("test_fallback_too_short", test_fallback_too_short),
        ("test_warning_none_when_reliable", test_warning_none_when_reliable),
        ("test_warning_generated_when_failed", test_warning_generated_when_failed),
        ("test_warning_multilingual", test_warning_multilingual),
        ("test_badge_ok", test_badge_ok),
        ("test_badge_danger_when_both_failed", test_badge_danger_when_both_failed),
        ("test_badge_warning_single_channel", test_badge_warning_single_channel),
    ]

    print("=" * 60)
    print("  psgscoring v0.2.96 - signal_quality tests")
    print("=" * 60)

    passed = failed = 0
    for name, test in tests:
        try:
            test()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERR ] {name}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"  {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
