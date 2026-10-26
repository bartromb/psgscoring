"""Elke detectorweg levert dezelfde eventsleutels, en de samenvatting leest ze.

De autonome re-ranker bouwde zijn events uit vier velden (onset_s, duration_s,
autonomic_score, lgbm_proba). `_recompute_arousal_summary` leest daarna
`a.get("stage", "W")` en `a.get("dominant_band")`: de NREM/REM-indices werden
0,0 en de bandtellingen 0, naast een correcte totaalindex. Dezelfde val
dreigt voor elke weg die een eventlijst zonder stadium aflevert, dus
`finalize_arousal_events` vult stage/epoch/end_s aan vóór elke samenvatting.

Daarnaast drie kleinere lekken op dezelfde paden:

  * de LGBM-terugval riep `detect_arousals` opnieuw aan zonder
    rem_alpha_baseline, score_wake_arousals, alpha_band_wide, lgbm_threshold,
    resp_event_ends en event_locked_threshold -- een vlag die aan stond, was
    na een mislukt classifierfilter stil uit;
  * het U-Net-pad kreeg `min_interval_s` ruw door, zonder de env-override die
    de LGBM-paden wel lezen;
  * `detect_arousals_multi` negeerde een per afleiding opgegeven
    samplefrequentie, en het re-rankerrecept herkende REM alleen als "R".
"""
import os

import numpy as np
import pytest

from psgscoring import arousal as ar
from psgscoring import arousal_unet as au
from psgscoring.arousal import (
    autonomic_rerank_selection,
    detect_arousals,
    detect_arousals_multi,
    finalize_arousal_events,
    run_arousal_respiratory_analysis,
)

SF = 128.0
N_S = 1200
VEREIST = {"onset_s", "end_s", "duration_s", "stage", "epoch"}


def _eeg(n_s=N_S, sf=SF, f_burst=11.0, amp=60.0, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_s * sf)) / sf
    x = rng.normal(0, 8.0, t.size) + 30.0 * np.sin(2 * np.pi * 1.5 * t)
    for b0 in range(120, n_s - 60, 180):
        m = (t >= b0) & (t < b0 + 6)
        x[m] += amp * np.sin(2 * np.pi * f_burst * t[m])
    return x


def _emg(n_s=N_S, sf=SF, seed=11):
    return np.random.default_rng(seed).normal(0, 5.0, int(n_s * sf))


def _hypno(n_s=N_S):
    return ["N2"] * int(n_s // 30)


def _pleth(n_s=N_S, sf=SF):
    t = np.arange(int(n_s * sf)) / sf
    return np.sin(2 * np.pi * 1.1 * t)


def _hr(n_s=N_S, sf=SF):
    return np.full(int(n_s * sf), 60.0)


def _reranked_multi(hypno=None):
    """Het multi-pad met classifier én re-ranker; None als de omgeving het
    niet kan (geen lightgbm, geen kandidaten boven de drempel)."""
    pytest.importorskip("lightgbm")
    hypno = hypno or _hypno()
    der = [("C4-M1", _eeg(seed=7)), ("O2-M1", _eeg(seed=8))]
    uit = detect_arousals_multi(
        der, SF, hypno, emg_data=_emg(), lgbm=True, lgbm_threshold=0.05,
        hr_data=_hr(), sf_hr=SF, pleth_data=_pleth(), sf_pleth=SF,
        autonomic_rerank=True, min_interval_s=10.0)
    prov = (uit.get("summary") or {}).get("autonomic_rerank") or {}
    if not prov.get("active") or not uit.get("events"):
        pytest.skip(f"re-ranker niet actief in deze omgeving: {prov}")
    return uit


# ── finalize_arousal_events ─────────────────────────────────────────────

def test_finalize_vult_stage_epoch_en_end_in():
    hypno = ["W", "N2", "N2", "R"]
    ev = [{"onset_s": 40.0, "duration_s": 5.0},
          {"onset_s": 95.0, "duration_s": 4.0, "stage": "N3"}]
    uit = finalize_arousal_events(ev, hypno)
    assert uit[0]["stage"] == "N2" and uit[0]["epoch"] == 1 and uit[0]["end_s"] == 45.0
    # Wat er al staat, blijft staan.
    assert uit[1]["stage"] == "N3" and uit[1]["epoch"] == 3


def test_finalize_laat_een_geheel_hypnogram_heel():
    uit = finalize_arousal_events([{"onset_s": 100.0, "duration_s": 3.0}], [0, 2, 2, 4])
    assert uit[0]["stage"] == 4 and uit[0]["epoch"] == 3


def test_finalize_buiten_het_hypnogram_is_wake():
    uit = finalize_arousal_events([{"onset_s": 500.0, "duration_s": 3.0}], ["N2"] * 4)
    assert uit[0]["stage"] == "W" and uit[0]["epoch"] == 16


# ── de re-ranker en zijn samenvatting ───────────────────────────────────

def test_re_ranker_events_dragen_stadium_band_en_einde():
    uit = _reranked_multi()
    for e in uit["events"]:
        assert VEREIST <= set(e), sorted(e)
        assert "dominant_band" in e and "autonomic_score" in e


def test_re_ranker_samenvatting_telt_per_stadium_en_per_band():
    """All-N2-nacht: de NREM-index IS de arousal-index, en elke arousal heeft
    een dominante band. Zonder stadium op de events stond hier 0,0 naast een
    correcte totaalindex."""
    uit = _reranked_multi()
    s = uit["summary"]
    assert s["n_arousals"] > 0
    assert s["nrem_arousal_index"] == s["arousal_index"]
    assert s["rem_arousal_index"] is None          # geen REM in dit hypnogram
    n_band = s["n_theta_dominant"] + s["n_alpha_dominant"] + s["n_beta_dominant"]
    assert n_band == s["n_arousals"]


def test_re_ranker_ziet_rem_ook_als_vier():
    """Het recept kent een REM-feature; een numeriek hypnogram gaf daar
    altijd 0 voor."""
    kand = [{"onset_s": 100.0, "duration_s": 6.0, "proba": 0.90},
            {"onset_s": 400.0, "duration_s": 5.0, "proba": 0.70}]
    n_ep = N_S // 30
    tekst = ["N2"] * n_ep
    getal = [2] * n_ep
    tekst[13] = "R"
    getal[13] = 4
    kw = dict(threshold=0.5, pleth=_pleth(), sf_pleth=SF, hr=_hr(), sf_hr=SF,
              min_interval_s=10.0)
    ev_t, p_t = autonomic_rerank_selection(kand, hypno=tekst, **kw)
    ev_g, p_g = autonomic_rerank_selection(kand, hypno=getal, **kw)
    assert p_t["active"] and p_g["active"]
    assert [e["autonomic_score"] for e in ev_t] == [e["autonomic_score"] for e in ev_g]
    assert [e["stage"] for e in ev_g][-1] == 4


# ── dezelfde sleutels op elke weg ───────────────────────────────────────

def test_elke_detectorweg_levert_dezelfde_vereiste_sleutels():
    hypno = _hypno()
    wegen = {}
    regels = detect_arousals(_eeg(), SF, hypno, emg_data=_emg(), lgbm=False)
    assert regels["events"], "de fixture hoort regelgebaseerde arousals te geven"
    wegen["regels"] = regels["events"]
    multi = detect_arousals_multi([("C4-M1", _eeg(seed=7)), ("O2-M1", _eeg(seed=8))],
                                  SF, hypno, emg_data=_emg(), lgbm=False)
    wegen["multi"] = multi["events"]
    try:
        wegen["re-ranker"] = _reranked_multi()["events"]
    except pytest.skip.Exception:
        pass
    try:
        pytest.importorskip("onnxruntime")
        ok, _ = au.available()
        if ok:
            u = au.detect_arousals_unet(_eeg(n_s=600), SF, _eeg(n_s=600, seed=9), SF,
                                        _emg(n_s=600), SF, _hypno(600), threshold=0.05)
            if u["success"] and u["events"]:
                wegen["unet"] = u["events"]
    except pytest.skip.Exception:
        pass
    for naam, events in wegen.items():
        for e in events:
            assert VEREIST <= set(e), f"{naam}: {sorted(e)}"


# ── de LGBM-terugval houdt zijn vlaggen ─────────────────────────────────

def test_lgbm_terugval_houdt_alpha_band_wide(monkeypatch):
    """Burst op 11,75 Hz: binnen de brede alfaband (8-13), buiten de smalle
    (8-11). Mislukt het classifierfilter NA het laden, dan hoort de terugval
    hetzelfde te geven als het regelgebaseerde pad MET de vlag."""
    pytest.importorskip("lightgbm")
    monkeypatch.delenv("PSGSCORING_AROUSAL_LGBM", raising=False)
    monkeypatch.delenv("YASAFLASKIFIED_AROUSAL_LGBM", raising=False)
    eeg, emg, hypno = _eeg(n_s=900, f_burst=11.75, amp=30.0, seed=3), _emg(n_s=900), _hypno(900)
    smal = detect_arousals(eeg, SF, hypno, emg_data=emg, lgbm=False, alpha_band_wide=False)
    breed = detect_arousals(eeg, SF, hypno, emg_data=emg, lgbm=False, alpha_band_wide=True)
    assert len(smal["events"]) == 0 and len(breed["events"]) > 0, (
        len(smal["events"]), len(breed["events"]))

    def _kapot(*a, **k):
        raise RuntimeError("filter stuk")

    monkeypatch.setattr(ar, "_filter_candidates_with_lgbm", _kapot)
    terug = detect_arousals(eeg, SF, hypno, emg_data=emg, lgbm=True, alpha_band_wide=True)
    assert terug["success"] and "lgbm_error" in terug
    assert terug["events"] == breed["events"], (
        f"terugval gaf {len(terug['events'])} events, met de vlag horen het er "
        f"{len(breed['events'])} te zijn")


# ── het U-Net-pad ───────────────────────────────────────────────────────

def _stub_unet(monkeypatch, gevangen: dict):
    monkeypatch.setattr(au, "available", lambda *a, **k: (True, None))

    def _stub(eeg, sf_eeg, eog, sf_eog, emg, sf_emg, hypno, **kw):
        gevangen.update(kw, n_eeg=len(eeg), n_eog=len(eog), n_emg=len(emg),
                        sf_eeg=sf_eeg, sf_eog=sf_eog, sf_emg=sf_emg)
        return {"success": True, "events": [], "summary": {"detector": "unet_v1"},
                "error": None}

    monkeypatch.setattr(au, "detect_arousals_unet", _stub)


def test_tien_seconden_regel_uit_env_bereikt_het_unet_pad(monkeypatch):
    gevangen: dict = {}
    _stub_unet(monkeypatch, gevangen)
    monkeypatch.setenv("PSGSCORING_AROUSAL_MIN_INTERVAL_S", "10")
    n_s = 300
    uit = run_arousal_respiratory_analysis(
        eeg_data=_eeg(n_s), sf_eeg=SF, flow_data=None, flow_norm=None, sf_flow=None,
        resp_events=[], hypno=_hypno(n_s), emg_data=_emg(n_s), eog_data=_eeg(n_s, seed=2),
        arousal_detector="unet_v1", min_interval_s=0.0)
    assert uit["success"]
    assert gevangen["min_interval_s"] == 10.0


def test_run_analysis_brengt_eog_en_emg_op_de_eeg_frequentie(monkeypatch):
    """Een EMG op 64 Hz naast EEG op 128 Hz: of de aanroeper geeft sf_emg mee
    en het kanaal wordt herbemonsterd, of het U-Net kreeg stil een kanaal dat
    half zo lang duurde."""
    gevangen: dict = {}
    _stub_unet(monkeypatch, gevangen)
    n_s = 300
    emg_64 = _emg(n_s, sf=64.0)
    uit = run_arousal_respiratory_analysis(
        eeg_data=_eeg(n_s), sf_eeg=SF, flow_data=None, flow_norm=None, sf_flow=None,
        resp_events=[], hypno=_hypno(n_s), emg_data=emg_64, eog_data=_eeg(n_s, seed=2),
        sf_emg=64.0, arousal_detector="unet_v1")
    assert uit["success"]
    assert gevangen["sf_emg"] == SF
    assert gevangen["n_emg"] == gevangen["n_eeg"]


def test_prepare_input_weigert_een_onverklaard_lengteverschil():
    """Een kanaal dat na herbemonstering 150 s korter is dan het EEG, is een
    verkeerd opgegeven samplefrequentie -- niet iets om stil af te knippen."""
    eeg = np.zeros(int(300 * 128))
    emg_half = np.zeros(int(300 * 64))      # 300 s op 64 Hz, opgegeven als 128
    with pytest.raises(ValueError, match="sf"):
        au.prepare_input(eeg, 128.0, eeg, 128.0, emg_half, 128.0)
    X = au.prepare_input(eeg, 128.0, eeg, 128.0, emg_half, 64.0)
    assert X.shape[0] == 3 and abs(X.shape[1] - 300 * au.FS) <= 1


def test_gate_sleep_accepteert_een_geheel_hypnogram():
    ev = [(10.0, 15.0), (40.0, 45.0), (100.0, 104.0)]
    assert au.gate_sleep(ev, ["W", "N2", "N2", "R"]) == [(40.0, 45.0), (100.0, 104.0)]
    assert au.gate_sleep(ev, [0, 2, 2, 4]) == [(40.0, 45.0), (100.0, 104.0)]


def test_unet_samenvatting_benoemt_het_artefactbeleid():
    pytest.importorskip("onnxruntime")
    ok, why = au.available()
    if not ok:
        pytest.skip(why)
    n_s = 600
    u = au.detect_arousals_unet(_eeg(n_s), SF, _eeg(n_s, seed=9), SF, _emg(n_s), SF,
                                [2] * (n_s // 30), artifact_epochs=[3, 4])
    assert u["success"], u.get("error")
    s = u["summary"]
    assert "unet_v1" in s["artifact_epochs_policy"]
    assert s["n_artifact_epochs"] == 2
    assert s["models"]["arousal_unet_v1"]["sha256"] == au.meta()["onnx_sha256"]
    for e in u["events"]:
        assert e["stage"] == 2 and "epoch" in e


# ── multi: sf per afleiding ─────────────────────────────────────────────

def test_multi_honoreert_de_samplefrequentie_per_afleiding(monkeypatch):
    gezien = []
    echte = ar.detect_arousals

    def _spion(eeg, sf, hypno, **kw):
        gezien.append(float(sf))
        return echte(eeg, sf, hypno, **kw)

    monkeypatch.setattr(ar, "detect_arousals", _spion)
    der = [("C4-M1", _eeg(n_s=600)), ("O2-M1", _eeg(n_s=600, sf=64.0), 64.0)]
    detect_arousals_multi(der, SF, _hypno(600), lgbm=False)
    assert gezien == [128.0, 64.0], gezien
