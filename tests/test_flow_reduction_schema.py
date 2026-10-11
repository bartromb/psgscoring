"""`flow_reduction` is overal een VERHOUDING; `flow_reduction_pct` het percentage.

`_types.py` zegt het al: ``flow_reduction: Optional[float]  # flow reduction
ratio``. Drie producenten schreven het veld, en twee van de drie schreven er
een percentage in:

  * de cascade (`_detect_hypopneas`)           -> 0,475   (verhouding)
  * de Rule-1A-herstelpas (`reinstate_...`)     -> 47,5    (percentage)
  * de ademteugdetector (`breath_scoring`)      -> 47,5    (percentage)

Eén sleutel, twee eenheden, binnen één eventlijst. Wie `flow_reduction`
over alle hypopneeën middelt, krijgt een getal dat niets betekent. Nu is het
overal de verhouding en draagt elk event daarnaast `flow_reduction_pct`.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from psgscoring.breath_scoring import score_hypopneas_breathwise
from psgscoring.respiratory import (
    detect_respiratory_events,
    reinstate_rule1a_arousal_hypopneas,
)


def _check(e, tol=0.2):
    r, p = e.get("flow_reduction"), e.get("flow_reduction_pct")
    assert r is not None and p is not None, e
    assert 0.0 <= r <= 1.0, f"flow_reduction {r} is geen verhouding"
    assert p > 1.0, f"flow_reduction_pct {p} is geen percentage"
    assert p == pytest.approx(100.0 * r, abs=tol), (r, p)


def test_ademteugdetector():
    from test_breath_scoring import SF_SPO2, make_night, make_spo2

    breaths, hypno, windows = make_night([300.0, 700.0, 1100.0])
    events, _ = score_hypopneas_breathwise(
        breaths, hypno, spo2=make_spo2(windows, lag_s=20.0),
        sf_spo2=SF_SPO2, strictness=0.50)
    assert events
    for e in events:
        _check(e)


def test_herstelpas():
    cand = {"onset_s": 100.0, "duration_s": 20.0, "stage": "N2", "epoch": 3,
            "desat": None, "min_spo2": None, "reject_reason": "no_desaturation",
            "flow_reduction_pct": 47.5, "local_reduction_pct": 33.0}
    rein, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=[cand],
        arousal_events=[{"onset_s": 122.0, "duration_s": 3.0}],
        resp_events=[], hypno=["N2"] * 10, breaths=[], stats={})
    assert len(rein) == 1
    _check(rein[0])
    assert rein[0]["flow_reduction"] == pytest.approx(0.475)
    assert rein[0]["flow_reduction_pct"] == 47.5


def test_herstelpas_zonder_debietveld_blijft_none():
    cand = {"onset_s": 100.0, "duration_s": 20.0, "stage": "N2", "epoch": 3,
            "desat": None, "min_spo2": None, "reject_reason": "no_desaturation"}
    rein, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=[cand],
        arousal_events=[{"onset_s": 122.0, "duration_s": 3.0}],
        resp_events=[], hypno=["N2"] * 10, breaths=[], stats={})
    assert rein[0]["flow_reduction"] is None
    assert rein[0]["flow_reduction_pct"] is None


def test_cascade():
    """Korte opname met variabele ademhaling (anders wijst het
    stabiliteitsfilter elke kandidaat af) en één hypopneu met desaturatie."""
    sf, dur = 32.0, 900.0
    n = int(sf * dur)
    t = np.arange(n) / sf
    rng = np.random.default_rng(5)
    # Variabiliteit 0,90 zoals de golden hypopneu-cases: bij 0,5 haalt de
    # ademhaling CV 0,36 en wijst het stabiliteitsfilter (0,45) alles af.
    amp = np.clip(1.0 + rng.normal(0, 0.9, int(dur / 4) + 2), 0.35, 1.9)
    amp = amp[np.clip((t / 4.0).astype(int), 0, amp.size - 1)]
    flow = amp * np.sin(2 * np.pi * 0.25 * t)
    a, b = int(300 * sf), int(320 * sf)
    flow[a:b] *= 0.4
    spo2 = np.full(n, 96.0)
    spo2[int(324 * sf):int(336 * sf)] = 91.0
    hypno = ["N2"] * int(dur // 30)
    from psgscoring.constants import SCORING_PROFILES

    # Stabiliteitsfilter uit (cv=0): de synthetische ademhaling zit met CV
    # 0,45 precies op de drempel, en dit is een schematest, geen detectietest.
    profiel = dict(SCORING_PROFILES["aasm_v3_rec"], STABILITY_FILTER_CV=0.0)
    out = detect_respiratory_events(
        flow_data=flow, thorax_data=None, abdomen_data=None, spo2_data=spo2,
        sf_flow=sf, sf_spo2=sf, hypno=hypno, scoring_profile=profiel)
    assert out["success"], out.get("error")
    hyp = [e for e in out["events"] if "hypopnea" in e["type"]]
    assert hyp, "de fixture levert geen hypopneu; de test meet niets"
    for e in hyp:
        # De cascade leidt de twee velden van verschillende basislijnen af
        # (per-sample genormaliseerde envelope tegenover de pre-event-
        # basislijn op de onset), dus ze schelen een paar tienden procent.
        # Dat is bestaand gedrag en geen eenheidsfout; de toets hier is dat
        # het DEZELFDE grootheid in twee eenheden is.
        _check(e, tol=5.0)
