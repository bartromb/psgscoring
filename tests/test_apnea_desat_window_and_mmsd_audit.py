"""Twee gaten in de apneulus van `_detect_apneas`.

1. HET POST-EVENT-VENSTER VAN HET PROFIEL BEREIKTE DE APNEUS NIET.
   De hypopneelus geeft `post_win_s=post_event_win_s` door aan
   `get_desaturation`; de apneulus liet het weg en draaide dus op de default
   van 45 s, wat het profiel ook zei. Op de 45 s-profielen is dat
   byte-identiek; `aasm_v3_strict` (30 s) kreeg voor zijn apneus stilletjes
   het venster van `aasm_v3_rec`.

2. EEN DOOR MMSD AFGEWEZEN APNEU VERDWEEN ZONDER SPOOR.
   `continue` zonder boekhouding. Elke andere afwijzing in dit bestand
   schrijft een kandidaat met `reject_reason` weg; deze niet, dus in een
   review was niet te zien dat er een kandidaat van >= 10 s onder 10 %
   had gestaan. De lijst heet `rejected_apneas`, NIET `rejected_hypopneas`:
   die laatste voedt REINSTATABLE_REJECTIONS en de ML-promotor, en een
   apneu hoort daar niet in terecht te komen.
"""
import numpy as np
import pytest

from psgscoring.respiratory import _detect_apneas, detect_respiratory_events

SF = 32.0
DUR_S = 600.0
AP_ON, AP_OFF = 100.0, 115.0        # één apneu van 15 s


def _arrays():
    n = int(SF * DUR_S)
    flow_env = np.ones(n)
    flow_env[int(AP_ON * SF):int(AP_OFF * SF)] = 0.02
    baseline = np.ones(n)
    flow_norm = np.clip(flow_env / baseline, 0, 2)
    apnea_raw = flow_norm < 0.10
    sleep = np.ones(n, dtype=bool)
    hypno = ["N2"] * int(DUR_S // 30)
    return apnea_raw, sleep, flow_env, flow_norm, baseline, hypno


def _spo2(dip_at_s, drop=4.0, sf_spo2=1.0):
    """SpO2 op 1 Hz: 96 %, met een dip van `drop` % rond `dip_at_s`."""
    s = np.full(int(DUR_S * sf_spo2), 96.0)
    c = int(dip_at_s * sf_spo2)
    s[c - 2:c + 3] = 96.0 - drop
    return s


def _run(post_win_s=None, mmsd_norm=None, rejected=None, spo2=None):
    apnea_raw, sleep, env, norm, bl, hypno = _arrays()
    kw = {}
    if post_win_s is not None:
        kw["post_event_win_s"] = post_win_s
    if rejected is not None:
        kw["rejected"] = rejected
    return _detect_apneas(
        apnea_raw, sleep, env, norm, bl, SF, 1.0, hypno,
        None, None, None, None, 1.0,
        spo2, None, mmsd_norm, **kw)


# ── 1. post-event-venster ─────────────────────────────────────────────────

def test_het_venster_van_het_profiel_bepaalt_de_apneudesaturatie():
    """Dip 38 s na het einde: binnen 45 s, buiten 30 s."""
    spo2 = _spo2(AP_OFF + 38.0)
    ev45 = _run(post_win_s=45.0, spo2=spo2)
    ev30 = _run(post_win_s=30.0, spo2=spo2)
    assert len(ev45) == len(ev30) == 1
    assert ev45[0]["desaturation_pct"] == pytest.approx(4.0, abs=0.1)
    assert ev30[0]["desaturation_pct"] is None, ev30[0]


def test_zonder_argument_blijft_45_s_de_default():
    """Byte-identiek voor elke aanroeper die het venster niet meegeeft."""
    spo2 = _spo2(AP_OFF + 38.0)
    assert _run(spo2=spo2)[0]["desaturation_pct"] == pytest.approx(4.0, abs=0.1)


# ── 2. MMSD-audit ─────────────────────────────────────────────────────────

def test_een_mmsd_afwijzing_laat_een_kandidaat_achter():
    n = int(SF * DUR_S)
    hoog = np.full(n, 1.0)                     # > MMSD_APNEA_THRESH overal
    rejected = []
    events = _run(mmsd_norm=hoog, rejected=rejected)
    assert events == []
    assert len(rejected) == 1, rejected
    r = rejected[0]
    assert r["reject_reason"].startswith("mmsd_"), r
    assert r["type"] == "apnea"
    assert r["onset_s"] == pytest.approx(AP_ON, abs=0.1)
    assert r["duration_s"] == pytest.approx(AP_OFF - AP_ON, abs=0.1)
    for sleutel in ("stage", "epoch", "mmsd_mean"):
        assert sleutel in r, sorted(r)


def test_zonder_mmsd_afwijzing_blijft_de_lijst_leeg():
    n = int(SF * DUR_S)
    rejected = []
    events = _run(mmsd_norm=np.zeros(n), rejected=rejected)
    assert len(events) == 1 and rejected == []


def test_de_telling_staat_in_het_resultaat():
    """Leveringsoppervlak: `rejected_apneas` en de telling, los van
    `rejected_hypopneas`."""
    n = int(SF * DUR_S)
    t = np.arange(n) / SF
    flow = np.sin(2 * np.pi * 0.25 * t)
    flow[int(AP_ON * SF):int(AP_OFF * SF)] *= 0.02
    hypno = ["N2"] * int(DUR_S // 30)
    out = detect_respiratory_events(
        flow_data=flow, thorax_data=None, abdomen_data=None,
        spo2_data=None, sf_flow=SF, sf_spo2=1.0, hypno=hypno)
    assert out["success"] is True, out.get("error")
    assert "rejected_apneas" in out
    assert isinstance(out["rejected_apneas"], list)
    assert out["n_mmsd_rejected_apneas"] == len(out["rejected_apneas"])
    assert all(r.get("type") != "apnea" for r in out["rejected_hypopneas"])
