"""
tests/test_rule1a_arousal_naming.py — 1B -> 1A hernoeming met aliassen.

De AASM plaatst het arousal-criterium in Rule **1A** (">=30% flowreductie EN
(>=3% desaturatie OF arousal)"). Rule **1B** is de variant met >=4%
desaturatie die arousals juist UITSLUIT. De code had het omgekeerd.

Deze test bewaakt beide kanten van de hernoeming:
  * de nieuwe namen bestaan
  * de oude namen blijven werken — YASAFlaskified leest ze op drie plekken
    (pneumo_analysis.py importeert de functie; generate_pdf_report.py en
    generate_psg_report.py lezen `rule1b_reinstated`), en
    ml_classifier.py gebruikt de kandidaat-vlag `rule1b` als FEATURE van het
    getrainde LightGBM-model. Een "opruiming" die de alias weghaalt zou het
    model stil een kenmerk ontnemen.
"""
import numpy as np
import pytest

import psgscoring
from psgscoring.respiratory import (
    reinstate_rule1a_arousal_hypopneas,
    reinstate_rule1b_hypopneas,
)


# ══════════════════════════════════════════════════════════════
#  Publieke API
# ══════════════════════════════════════════════════════════════

def test_new_name_is_exported():
    assert hasattr(psgscoring, "reinstate_rule1a_arousal_hypopneas")
    assert "reinstate_rule1a_arousal_hypopneas" in psgscoring.__all__


def test_old_name_still_exported_as_alias():
    assert hasattr(psgscoring, "reinstate_rule1b_hypopneas")
    assert "reinstate_rule1b_hypopneas" in psgscoring.__all__
    assert (psgscoring.reinstate_rule1b_hypopneas
            is psgscoring.reinstate_rule1a_arousal_hypopneas)


def test_alias_is_the_same_function_object():
    assert reinstate_rule1b_hypopneas is reinstate_rule1a_arousal_hypopneas


# ══════════════════════════════════════════════════════════════
#  Event-velden op een gereinstateerde kandidaat
# ══════════════════════════════════════════════════════════════

def _candidate(onset=100.0, dur=20.0):
    return {
        "onset_s": onset, "duration_s": dur, "stage": "N2", "epoch": int(onset // 30),
        "desat": None, "min_spo2": None,
        "reject_reason": "no_desaturation",
    }


def _reinstate_one():
    """Eén kandidaat met een arousal 5 s na het einde -> moet kwalificeren."""
    cand = _candidate()
    end = cand["onset_s"] + cand["duration_s"]
    arousals = [{"onset_s": end + 5.0, "duration_s": 3.0}]
    hypno = ["N2"] * 40
    reinstated, _all = reinstate_rule1a_arousal_hypopneas(
        rejected=[cand], arousal_events=arousals, resp_events=[], hypno=hypno,
        breaths=[],
    )
    return reinstated


def test_reinstated_event_carries_both_flags():
    r = _reinstate_one()
    if not r:
        pytest.skip("kandidaat kwalificeerde niet in deze configuratie")
    ev = r[0]
    assert ev.get("rule1a_arousal") is True, "nieuwe vlag ontbreekt"
    assert ev.get("rule1b") is True, (
        "alias 'rule1b' weg — ml_classifier.py gebruikt hem als modelfeature")


def test_reinstated_event_reports_rule_1a():
    r = _reinstate_one()
    if not r:
        pytest.skip("kandidaat kwalificeerde niet in deze configuratie")
    detail = r[0].get("classify_detail") or {}
    assert detail.get("rule") == "1A_arousal"
    assert detail.get("rule_legacy") == "1B_arousal"


def test_ml_feature_still_reads_the_alias():
    """ml_classifier leest candidate['rule1b'] — pin dat contract vast."""
    from psgscoring.ml_classifier import FEATURE_COLUMNS
    assert "is_rule1b" in FEATURE_COLUMNS, (
        "modelfeature hernoemd? dan moet ml_classifier.py mee, en het "
        "getrainde model opnieuw")
    r = _reinstate_one()
    if not r:
        pytest.skip("kandidaat kwalificeerde niet in deze configuratie")
    assert bool(r[0].get("rule1b", False)) is True


# ══════════════════════════════════════════════════════════════
#  Summary-sleutels
# ══════════════════════════════════════════════════════════════

def test_pipeline_emits_both_summary_keys():
    """Beide sleutels moeten bestaan en gelijk zijn."""
    mne = pytest.importorskip("mne")
    sf = 32.0
    n = int(sf * 60 * 6)
    t = np.arange(n) / sf
    rng = np.random.default_rng(3)
    flow = np.sin(2 * np.pi * 0.25 * t)
    info = mne.create_info(["Resp nasal", "SaO2"], sf, ["misc", "misc"])
    raw = mne.io.RawArray(np.vstack([flow, np.full(n, 97.0)]), info, verbose=False)
    hypno = ["N2"] * int(np.ceil(raw.times[-1] / 30.0))

    out = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile="aasm_v3_rec")
    resp = out["respiratory"]
    assert "rule1a_arousal_reinstated" in resp, "nieuwe sleutel ontbreekt"
    assert "rule1b_reinstated" in resp, (
        "alias weg — YASAFlaskified leest deze in twee rapportgeneratoren")
    assert resp["rule1a_arousal_reinstated"] == resp["rule1b_reinstated"]
