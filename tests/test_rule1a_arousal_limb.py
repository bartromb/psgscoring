"""
tests/test_rule1a_arousal_limb.py — de AASM Rule 1A arousal-tak.

Fase 1 stelde vast dat de tak dood was door een doorgiftefout: de detectie zet
arousals in ``output["arousal"]["arousals"]["events"]``, terwijl stap 8
``output["arousal"]["events"]`` leest — een sleutel die het auto-detectiepad
nooit vulde (issue #16). Op SN3: 232 arousals gedetecteerd, 0 gezien.

Deze module dekt drie dingen:
  1. de doorgifte is hersteld en blijft hersteld
  2. de drie casussen uit de fasebeschrijving kwalificeren zoals verwacht
  3. de tellers maken 'nul' altijd zichtbaar mét reden — dat een tak stil op
     nul staat mag niet nog eens jaren onopgemerkt blijven
"""
import numpy as np
import pytest

from psgscoring.pipeline import _normalise_arousal_block
from psgscoring.respiratory import reinstate_rule1a_arousal_hypopneas


# ══════════════════════════════════════════════════════════════
#  1. De doorgifte
# ══════════════════════════════════════════════════════════════

def test_nested_events_become_visible_flat():
    block = {"success": True,
             "arousals": {"events": [{"onset_s": 10.0}, {"onset_s": 40.0}]}}
    assert len(_normalise_arousal_block(block)["events"]) == 2


def test_external_flat_events_are_preserved():
    block = {"success": True, "events": [{"onset_s": 1.0}],
             "arousals": {"events": []}}
    assert len(_normalise_arousal_block(block)["events"]) == 1


def test_absent_arousals_give_empty_list_not_keyerror():
    for b in ({"success": False}, {"arousals": None}, {"arousals": {}}):
        assert _normalise_arousal_block(b)["events"] == []


# ══════════════════════════════════════════════════════════════
#  2. De drie casussen uit de fasebeschrijving
# ══════════════════════════════════════════════════════════════

def _candidate(onset=100.0, dur=20.0):
    return {"onset_s": onset, "duration_s": dur, "stage": "N2",
            "epoch": int(onset // 30), "desat": None, "min_spo2": None,
            "reject_reason": "no_desaturation"}


def test_flow_reduction_with_arousal_qualifies():
    """Eén flow-reductie zonder desaturatie + arousal 5 s na het einde."""
    cand = _candidate()
    end = cand["onset_s"] + cand["duration_s"]
    stats = {}
    reinstated, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=[cand], arousal_events=[{"onset_s": end + 5.0, "duration_s": 3.0}],
        resp_events=[], hypno=["N2"] * 40, breaths=[], stats=stats,
    )
    assert len(reinstated) == 1
    assert reinstated[0]["rule1a_arousal"] is True
    assert stats["n_candidates_tested"] == 1
    assert stats["n_arousal_coupled"] == 1
    assert stats["n_qualified"] == 1


def test_same_case_without_arousal_does_not_qualify():
    cand = _candidate()
    stats = {}
    reinstated, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=[cand], arousal_events=[], resp_events=[], hypno=["N2"] * 40,
        breaths=[], stats=stats,
    )
    assert reinstated == []


def test_arousal_far_outside_the_window_does_not_qualify():
    cand = _candidate()
    end = cand["onset_s"] + cand["duration_s"]
    reinstated, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=[cand],
        arousal_events=[{"onset_s": end + 120.0, "duration_s": 3.0}],
        resp_events=[], hypno=["N2"] * 40, breaths=[],
    )
    assert reinstated == []


def test_desaturation_only_profile_never_reaches_the_limb():
    """cms_medicare heeft DESAT_OR_AROUSAL=False -> de tak mag niet draaien."""
    from psgscoring.constants import SCORING_PROFILES
    assert SCORING_PROFILES["cms_medicare"]["DESAT_OR_AROUSAL"] is False


# ══════════════════════════════════════════════════════════════
#  3. Gap-check, nu profielinstelbaar
# ══════════════════════════════════════════════════════════════

def test_breath_gap_blocks_a_late_arousal():
    """Meer ademteugen tussen event en arousal dan toegestaan -> geen koppeling."""
    cand = _candidate()
    end = cand["onset_s"] + cand["duration_s"]
    breaths = [{"onset_s": end + d, "amplitude": 1.0} for d in (1.0, 3.0, 5.0, 7.0)]
    reinstated, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=[cand], arousal_events=[{"onset_s": end + 9.0, "duration_s": 3.0}],
        resp_events=[], hypno=["N2"] * 40, breaths=breaths, gap_max_breaths=1,
    )
    assert reinstated == []


def test_gap_tolerance_is_configurable():
    cand = _candidate()
    end = cand["onset_s"] + cand["duration_s"]
    breaths = [{"onset_s": end + d, "amplitude": 1.0} for d in (1.0, 3.0, 5.0, 7.0)]
    reinstated, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=[cand], arousal_events=[{"onset_s": end + 9.0, "duration_s": 3.0}],
        resp_events=[], hypno=["N2"] * 40, breaths=breaths, gap_max_breaths=10,
    )
    assert len(reinstated) == 1


# ══════════════════════════════════════════════════════════════
#  4. Nul is nooit meer stil
# ══════════════════════════════════════════════════════════════

def _synthetic_raw():
    mne = pytest.importorskip("mne")
    sf, minutes = 32.0, 8
    n = int(sf * 60 * minutes)
    t = np.arange(n) / sf
    rng = np.random.default_rng(11)
    flow = np.sin(2 * np.pi * 0.25 * t)
    for start in range(60, 60 * minutes - 60, 90):
        flow[int(start * sf):int((start + 20) * sf)] *= 0.3
    eeg = rng.normal(0, 20e-6, n)
    for start in range(82, 60 * minutes - 60, 90):
        a, b = int(start * sf), int((start + 4) * sf)
        eeg[a:b] += 40e-6 * np.sin(2 * np.pi * 10.0 * t[a:b])
    info = mne.create_info(["Resp nasal", "SaO2", "EEG C4-M1", "EMG chin"],
                           sf, ["misc", "misc", "eeg", "emg"])
    return mne.io.RawArray(
        np.vstack([flow, np.full(n, 97.0), eeg, rng.normal(0, 5e-6, n)]),
        info, verbose=False)


def test_stats_always_present_with_a_reason():
    import psgscoring
    raw = _synthetic_raw()
    hypno = ["N2"] * int(np.ceil(raw.times[-1] / 30.0))
    out = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile="aasm_v3_rec")
    st = out["respiratory"].get("rule1a_arousal_stats")
    assert st is not None, "tellers ontbreken"
    for k in ("enabled", "n_arousals_available", "n_candidates_tested",
              "n_arousal_coupled", "n_qualified", "skipped_reason"):
        assert k in st, f"teller '{k}' ontbreekt"
    # default staat de tak uit -> dat moet expliciet vermeld staan
    assert st["enabled"] is False
    assert st["skipped_reason"] == "disabled_by_profile"


def test_plumbing_is_repaired_end_to_end():
    """De platte sleutel moet nu vullen wat de detectie vond."""
    import psgscoring
    raw = _synthetic_raw()
    hypno = ["N2"] * int(np.ceil(raw.times[-1] / 30.0))
    out = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile="aasm_v3_rec")
    ar = out["arousal"]
    nested = ((ar.get("arousals") or {}).get("events") or [])
    assert len(ar.get("events", [])) == len(nested), (
        "platte sleutel loopt weer uit de pas met de detectie (issue #16)")
