"""
tests/test_diagnose_hypopnea.py — helpers van de hypopneediagnose.

De analyse zelf heeft PSG-IPA nodig; deze tests dekken de pure functies waar
de conclusies op rusten. Twee daarvan hebben me in dit onderzoek daadwerkelijk
misleid voordat ze getest waren:

  * cluster_events — zonder clustering telt één event twaalf keer mee (één per
    scoorder) en lijken percentages veel steviger dan ze zijn. Een eerdere
    conclusie ("53% van de consensus-events gemist") bleek na clustering op
    3 van 6 events te rusten.
  * rolling_p2p / flow_reduction_pct — een maat zonder controlegroep zegt
    niets; deze tests pinnen dat vlak signaal 0% geeft en gehalveerde
    ademhaling ~50%.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diagnose_hypopnea import (  # noqa: E402
    cluster_events,
    flow_reduction_pct,
    median_in,
    overlaps_any,
    pick_flow_channel,
    rolling_p2p,
    summarise,
)


# ══════════════════════════════════════════════════════════════
#  cluster_events
# ══════════════════════════════════════════════════════════════

def test_cluster_merges_overlapping_scorer_instances():
    """Twaalf scoorders die hetzelfde event zien = één distinct event."""
    inst = [(100.0 + i * 0.5, 130.0 + i * 0.5, i) for i in range(12)]
    out = cluster_events(inst)
    assert len(out) == 1
    assert len(out[0]["members"]) == 12


def test_cluster_keeps_separate_events_apart():
    inst = [(100.0, 130.0, 0), (500.0, 530.0, 1), (900.0, 930.0, 2)]
    assert len(cluster_events(inst)) == 3


def test_cluster_counts_distinct_scorers_not_instances():
    """Twee markeringen van dezelfde scoorder tellen als één scoorder."""
    inst = [(100.0, 130.0, 0), (105.0, 135.0, 0), (110.0, 140.0, 1)]
    out = cluster_events(inst)
    assert len(out) == 1
    assert len(set(out[0]["members"])) == 2


def test_cluster_empty_input():
    assert cluster_events([]) == []


def test_cluster_touching_but_not_overlapping_stays_split():
    assert len(cluster_events([(0.0, 30.0, 0), (30.0, 60.0, 1)])) == 2


# ══════════════════════════════════════════════════════════════
#  rolling_p2p / flow_reduction_pct
# ══════════════════════════════════════════════════════════════

def _breathing(sf, seconds, amplitude, rate_hz=0.25):
    t = np.arange(0, seconds, 1.0 / sf)
    return amplitude * np.sin(2 * np.pi * rate_hz * t)


def test_rolling_p2p_of_flat_signal_is_zero():
    assert np.allclose(rolling_p2p(np.zeros(1000), 100), 0.0)


def test_rolling_p2p_tracks_amplitude():
    sf = 32.0
    env = rolling_p2p(_breathing(sf, 120, amplitude=1.0), int(3 * sf))
    mid = env[int(20 * sf):int(100 * sf)]
    assert 1.5 < float(np.median(mid)) <= 2.0     # p2p van een sinus = 2A


def test_rolling_p2p_handles_empty_and_short_input():
    assert rolling_p2p([], 10).size == 0
    assert rolling_p2p(np.ones(3), 100).size == 3


def test_flow_reduction_zero_when_breathing_unchanged():
    """De nulhypothese: ongestoord ademen geeft ~0% reductie."""
    sf = 32.0
    env = rolling_p2p(_breathing(sf, 400, 1.0), int(3 * sf))
    red = flow_reduction_pct(env, sf, onset=200.0, offset=220.0, duration_s=400.0)
    assert abs(red) < 5.0


def test_flow_reduction_detects_halved_amplitude():
    sf = 32.0
    sig = _breathing(sf, 400, 1.0)
    lo, hi = int(200 * sf), int(220 * sf)
    sig[lo:hi] *= 0.5
    env = rolling_p2p(sig, int(3 * sf))
    red = flow_reduction_pct(env, sf, onset=201.0, offset=219.0, duration_s=400.0)
    assert 40.0 < red < 60.0


def test_flow_reduction_is_nan_without_usable_baseline():
    env = rolling_p2p(np.zeros(1000), 32)
    assert np.isnan(flow_reduction_pct(env, 32.0, onset=5.0, offset=10.0,
                                       duration_s=30.0))


def test_median_in_empty_window_is_nan():
    env = np.ones(100)
    assert np.isnan(median_in(env, 1.0, 50.0, 50.0))


# ══════════════════════════════════════════════════════════════
#  overige helpers
# ══════════════════════════════════════════════════════════════

def test_overlaps_any_uses_iou_threshold():
    assert overlaps_any(0.0, 12.0, [(8.0, 20.0)])          # IoU = 0.20
    assert not overlaps_any(0.0, 12.0, [(9.0, 21.0)])      # IoU < 0.20
    assert not overlaps_any(0.0, 12.0, [])


def test_pick_flow_channel_prefers_nasal():
    assert pick_flow_channel(["EEG C4-M1", "Resp nasal", "Flow"]) == "Resp nasal"
    assert pick_flow_channel(["EEG C4-M1", "Airflow"]) == "Airflow"
    assert pick_flow_channel(["EEG C4-M1", "ECG"]) is None


def test_summarise_reports_quartiles_and_share_above_30():
    s = summarise([10.0, 20.0, 40.0, 60.0])
    assert s["n"] == 4
    assert s["median"] == pytest.approx(30.0)
    assert s["pct_ge_30"] == pytest.approx(50.0)


def test_summarise_ignores_nan_and_empty():
    assert summarise([]) is None
    assert summarise([float("nan")]) is None
    assert summarise([float("nan"), 50.0])["n"] == 1
