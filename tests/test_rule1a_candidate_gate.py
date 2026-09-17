"""
tests/test_rule1a_candidate_gate.py — kandidaatpoort voor arousal-kwalificatie.

Orakel-decompositie 17-09-2026 (docs/orakel_rule1a_20260917.md): met
PERFECTE arousals matcht 44 % van wat de Rule-1A-arousaltak herstelt een
referentie-hypopneu, terwijl elke koppeling een echte arousal heeft (venster
en gap getrouw). De bijvangst zit in de kandidaat-eligibility: elke
`no_desaturation`-afwijzing van >=30 %/>=10 s mag terugkomen, ook lange,
zwakke debietdalingen (mediaan 25 s, 90e percentiel 50 s).

De poort hier laat een kandidaat alleen via een arousal kwalificeren als de
debietdaling zelf overtuigend is (`min_flow_reduction_pct`, lokaal
`min_local_reduction_pct`) en het event niet te lang is (`max_duration_s`).
Alle drie default None = uit = byte-identiek aan het huidige gedrag.
"""
import pytest

from psgscoring.respiratory import reinstate_rule1a_arousal_hypopneas


def _cand(onset=100.0, dur=20.0, red=45.0, local=35.0):
    return {"onset_s": onset, "duration_s": dur, "stage": "N2",
            "epoch": int(onset // 30), "desat": None, "min_spo2": None,
            "reject_reason": "no_desaturation",
            "flow_reduction_pct": red, "local_reduction_pct": local}


def _run(cands, **gate):
    end = max(c["onset_s"] + c["duration_s"] for c in cands)
    stats = {}
    rein, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=cands,
        arousal_events=[{"onset_s": c["onset_s"] + c["duration_s"] + 2.0,
                         "duration_s": 3.0} for c in cands],
        resp_events=[], hypno=["N2"] * int(end // 30 + 5), breaths=[],
        stats=stats, **gate)
    return rein, stats


def test_poort_uit_is_bestaand_gedrag():
    rein, st = _run([_cand(red=31.0, dur=80.0, local=5.0)])
    assert len(rein) == 1
    assert st["n_gate_rejected"] == 0


def test_min_flow_reduction_weert_zwakke_kandidaat():
    rein, st = _run([_cand(red=35.0)], min_flow_reduction_pct=50.0)
    assert rein == []
    assert st["n_gate_rejected"] == 1
    assert st["gate_rejected_by_reason"] == {"flow_reduction": 1}


def test_min_flow_reduction_laat_sterke_kandidaat_door():
    rein, _ = _run([_cand(red=55.0)], min_flow_reduction_pct=50.0)
    assert len(rein) == 1


def test_max_duration_weert_lange_kandidaat():
    rein, st = _run([_cand(dur=61.0)], max_duration_s=60.0)
    assert rein == []
    assert st["gate_rejected_by_reason"] == {"duration": 1}


def test_min_local_reduction_weert_kandidaat_zonder_lokale_daling():
    rein, st = _run([_cand(local=12.0)], min_local_reduction_pct=30.0)
    assert rein == []
    assert st["gate_rejected_by_reason"] == {"local_reduction": 1}


def test_ontbrekend_veld_wordt_niet_stil_doorgelaten():
    """Een kandidaat zonder debietveld kan de poort niet halen: een poort
    die bij ontbrekende invoer openstaat is geen poort."""
    c = _cand(); c.pop("flow_reduction_pct")
    rein, st = _run([c], min_flow_reduction_pct=40.0)
    assert rein == []
    assert st["gate_rejected_by_reason"] == {"flow_reduction_missing": 1}


def test_herstelde_event_draagt_zijn_debietdaling():
    rein, _ = _run([_cand(red=47.5, local=33.0)])
    assert rein[0]["flow_reduction"] == 47.5
    assert rein[0]["local_reduction_pct"] == 33.0


def test_poort_telt_voor_de_koppeling_niet_erna():
    """Een door de poort geweerde kandidaat telt niet als 'getest' in de
    koppelstatistiek — anders lijkt de tak zwakker te koppelen dan hij doet."""
    rein, st = _run([_cand(red=35.0), _cand(onset=400.0, red=60.0)],
                    min_flow_reduction_pct=50.0)
    assert len(rein) == 1
    assert st["n_candidates_tested"] == 1
    assert st["n_gate_rejected"] == 1


def test_profielvelden_bestaan_en_staan_uit():
    from psgscoring.profiles import PostProcessingRules, get_profile
    pp = PostProcessingRules()
    assert pp.rule1a_arousal_min_flow_reduction_pct is None
    assert pp.rule1a_arousal_max_duration_s is None
    assert pp.rule1a_arousal_min_local_reduction_pct is None
    from psgscoring.constants import _profile_to_legacy_dict
    d = _profile_to_legacy_dict(get_profile("aasm_v3_rec"))
    for k in ("RULE1A_AROUSAL_MIN_FLOW_RED_PCT", "RULE1A_AROUSAL_MAX_DUR_S",
              "RULE1A_AROUSAL_MIN_LOCAL_RED_PCT"):
        assert k in d and d[k] is None


def test_de_detector_levert_het_debietbewijs_op_de_kandidaat():
    """Een poort op een veld dat de producent nooit zet, is een filter op
    niets. Zelfde fixture als de eligibility-test."""
    import importlib.util
    from pathlib import Path
    from psgscoring.respiratory import detect_respiratory_events
    _spec = importlib.util.spec_from_file_location(
        "eligibility_fixture",
        Path(__file__).with_name("test_rule1a_reinstatement_eligibility.py"))
    _mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    _opname_met_hypopneus_zonder_desaturatie = _mod._opname_met_hypopneus_zonder_desaturatie
    flow, spo2, hypno, sf = _opname_met_hypopneus_zonder_desaturatie()
    r = detect_respiratory_events(
        flow_data=flow, thorax_data=None, abdomen_data=None, spo2_data=spo2,
        sf_flow=sf, sf_spo2=1.0, hypno=hypno,
        scoring_profile={"STABILITY_FILTER_CV": 0.0,
                         "LOCAL_BL_MIN_REDUCTION_PCT": 10.0,
                         "LOCAL_BL_STRICT_RED": 10.0,
                         "HYPOPNEA_THRESHOLD": 0.70,
                         "DESATURATION_DROP_PCT": 3.0})
    afgewezen = [x for x in r.get("rejected_hypopneas", [])
                 if x.get("reject_reason") == "no_desaturation"]
    assert afgewezen, "fixture levert geen no_desaturation-kandidaten -- meet niets"
    for x in afgewezen:
        assert "flow_reduction_pct" in x and x["flow_reduction_pct"] is not None
        assert x["flow_reduction_pct"] >= 30.0
        assert "local_reduction_pct" in x
