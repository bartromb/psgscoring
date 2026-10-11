"""Positie en PLM rekenen op de DEFINITIEVE eventlijst, één keer.

`analyze_position` (stap 3) en `analyze_plm` (stap 6) draaiden op de
eventlijst van stap 1. Daarna vervangt of verandert een rij stappen die
lijst: de voorwaardelijke vereniging, stap 7b (ademteug-gegradeerde
hypopneeën), Rule 1B-herstel, de LightGBM-herclassificatie, het splitsen van
lange events en de desaturatiebegrenzing. Alleen 7b rekende de positie
opnieuw; de PLM-stap -- waarvan de respiratoire exclusie (LM binnen 0,5 s
van een eventeinde) rechtstreeks van die lijst afhangt -- nooit.

De reparatie: beide draaien NA de laatste mutatie van de lijst, één keer, en
de ad-hoc herberekening in 7b is weg. De arousalstap, die de events als
INVOER nodig heeft vóór 7b, krijgt een expliciete momentopname.
"""
import pathlib

import numpy as np
import pytest

mne = pytest.importorskip("mne")

SF = 64.0
FIRST_S, GAP_S, N_MOVES = 60.0, 20.0, 40
DUR_S = FIRST_S + N_MOVES * GAP_S + 120.0
N = int(DUR_S * SF)
HYPNO = ["N2"] * int(np.ceil(DUR_S / 30.0))
PROFIEL = "aasm_v3_breath"


def _leg():
    rng = np.random.default_rng(23)
    leg = rng.normal(0.0, 1e-6, N)
    for k in range(N_MOVES):
        t0 = FIRST_S + k * GAP_S
        a, b = int(t0 * SF), int((t0 + 1.5) * SF)
        leg[a:b] += rng.normal(0.0, 60e-6, b - a)
    return leg


def _raw():
    t = np.arange(N) / SF
    leg = _leg()
    pos = np.zeros(N)
    info = mne.create_info(
        ["Resp nasal", "SaO2", "EMG LAT", "EMG RAT", "Pos."],
        SF, ["misc", "misc", "emg", "emg", "misc"])
    data = np.vstack([np.sin(2 * np.pi * 0.25 * t), np.full(N, 97.0),
                      leg, leg, pos])
    return mne.io.RawArray(data, info, verbose="ERROR"), leg, pos


def _graded_hypopneas():
    """Tien hypopneeën die precies op een beenbeweging eindigen, zodat de
    respiratoire exclusie van de PLM-stap ze móét zien."""
    hyps = []
    for k in range(10):
        einde = FIRST_S + k * GAP_S
        onset = einde - 15.0
        hyps.append({"type": "hypopnea", "onset_s": onset, "duration_s": 15.0,
                     "stage": "N2", "epoch": int(onset // 30),
                     "confidence": 0.8, "desaturation_pct": 3.5})
    return hyps


@pytest.fixture(scope="module")
def run():
    import psgscoring
    import psgscoring.breath_scoring as bs
    mp = pytest.MonkeyPatch()
    hyps = _graded_hypopneas()
    mp.setattr(bs, "score_hypopneas_breathwise",
               lambda **kw: (list(hyps), {"n_scored": len(hyps),
                                          "n_candidates": len(hyps),
                                          "spo2_lag_s": 0.0}))
    raw, leg, pos = _raw()
    try:
        out = psgscoring.run_pneumo_analysis(
            raw, hypno=HYPNO, scoring_profile=PROFIEL,
            channel_map={"flow_pressure": "Resp nasal", "spo2": "SaO2",
                         "leg_l": "EMG LAT", "leg_r": "EMG RAT",
                         "position": "Pos."})
    finally:
        mp.undo()
    return out, leg, pos


def test_stap_7b_heeft_de_lijst_echt_veranderd(run):
    """Anders toetst de rest niets."""
    out, _leg, _pos = run
    ev = out["respiratory"]["events"]
    assert len(ev) == 10
    assert all(e["type"] == "hypopnea" for e in ev)


def test_de_plm_exclusie_is_op_de_definitieve_lijst_gerekend(run):
    """De kern: dezelfde uitkomst als een directe aanroep op de eindlijst."""
    from psgscoring import pipeline as pl
    from psgscoring.constants import SCORING_PROFILES
    from psgscoring.plm import analyze_plm

    out, leg, _pos = run
    p = SCORING_PROFILES[PROFIEL]
    ref = analyze_plm(leg, leg, SF, HYPNO,
                      resp_events=out["respiratory"]["events"],
                      artifact_epochs=None,
                      time_base_fix=bool(p["PLM_TIME_BASE"]),
                      event_list_cap=None,
                      offset_aasm=pl._plm_offset_aasm(p),
                      bilateral_window_s=pl._plm_bilateral_window_s(p))
    assert ref["success"], ref.get("error")
    assert ref["summary"]["n_resp_associated"] >= 5, (
        "fixture: de gegradeerde hypopneeën raken geen beenbeweging")
    ps = out["plm"]["summary"]
    for k in ("n_resp_associated", "n_plm_eligible", "n_plm", "plm_index",
              "lm_index"):
        assert ps[k] == ref["summary"][k], (k, ps[k], ref["summary"][k])


def test_de_positieanalyse_telt_dezelfde_events_als_het_rapport(run):
    from psgscoring.ancillary import analyze_position

    out, _leg, pos = run
    ev = out["respiratory"]["events"]
    ref = analyze_position(pos, SF, HYPNO, ev)
    assert out["position"]["summary"] == ref["summary"]
    assert sum(out["position"]["summary"]["n_events_per_pos"].values()) == len(ev)


def test_de_arousalstap_kreeg_een_expliciete_momentopname():
    """De events zijn INVOER voor de arousalkoppeling en bestaan vóór 7b; dat
    is de enige plek die de lijst vóór de laatste mutatie mag lezen."""
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "psgscoring" / "pipeline.py").read_text()
    assert "_resp_events_for_arousal" in src


def test_de_ad_hoc_herberekening_in_7b_is_weg():
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "psgscoring" / "pipeline.py").read_text()
    assert "positieanalyse herrekend" not in src
    # Beide indices staan ná de laatste lijstmutatie (desaturatiebegrenzing,
    # stap 8b) en vóór de CSR-stap die de samenvatting herrekent.
    i_8b = src.index('output["respiratory"]["desat_reuse_limit"] = _dstats')
    i_csr = src.index("Cheyne-Stokes detection...")
    i_pos = src.index("analyze_position(")
    i_plm = src.index("analyze_plm(")
    assert i_8b < i_pos < i_csr, "positieanalyse staat niet na stap 8b"
    assert i_8b < i_plm < i_csr, "PLM-analyse staat niet na stap 8b"
