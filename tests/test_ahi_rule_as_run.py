"""`ahi_rule` en `hypopnea_criterion` beschrijven de RUN, niet alleen het profiel.

`ahi_rule_conformity(profile)` zei "v3_recommended" zodra het PROFIEL de
arousal-tak kent. Faalde de arousalstap (geen EEG, module niet geladen,
exception in de detector), dan graderde stap 7b op desaturatie alleen en
droeg de uitslag toch het etiket van de aanbevolen regel. Hetzelfde voor de
terugval van de ademteug-detector: de envelope-detector draaide zonder
arousal-tak, en het label bleef breath-conform.

Minimale eerlijke versie: is de tak per profiel effectief maar leverde de
arousalstap in deze run niets, dan is het oordeel `desat_only` met de reden
erbij, en zegt `meta["hypopnea_criterion"]` hetzelfde. Het normale geval
blijft byte-identiek.
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

SF = 32.0
DUR_S = 1200
N = int(DUR_S * SF)


def _raw():
    t = np.arange(N) / SF
    rng = np.random.default_rng(5)
    flow = np.sin(2 * np.pi * 0.25 * t) + 0.02 * rng.normal(size=N)
    spo2 = np.full(N, 96.0)
    for t0 in range(120, DUR_S - 120, 90):
        a, b = int(t0 * SF), int((t0 + 20) * SF)
        flow[a:b] *= 0.3
        spo2[b:b + int(15 * SF)] -= 5.0
    info = mne.create_info(
        ["Resp nasal", "SaO2", "Thorax", "Abdomen", "EEG C4-M1", "EMG chin"],
        SF, ["misc", "misc", "misc", "misc", "eeg", "emg"])
    data = np.vstack([flow, spo2,
                      0.6 * np.sin(2 * np.pi * 0.25 * t),
                      0.6 * np.sin(2 * np.pi * 0.25 * t),
                      rng.normal(0, 20e-6, N), rng.normal(0, 5e-6, N)])
    return mne.io.RawArray(data, info, verbose="ERROR")


HYPNO = ["N2"] * (DUR_S // 30)
PROFIEL = "aasm_v3_breath"


def _run(**kw):
    import psgscoring
    return psgscoring.run_pneumo_analysis(_raw(), hypno=HYPNO,
                                          scoring_profile=PROFIEL, **kw)


def test_het_normale_geval_is_byte_identiek():
    """Arousalstap geslaagd -> precies het profieloordeel, geen extra sleutels."""
    from psgscoring import pipeline as pl
    from psgscoring.constants import SCORING_PROFILES
    if not pl._AROUSAL_AVAILABLE:
        pytest.skip("arousalmodule niet geladen")
    out = _run()
    assert out["arousal"].get("success") is True, out["arousal"].get("error")
    prof = SCORING_PROFILES[PROFIEL]
    assert out["respiratory"]["summary"]["ahi_rule"] == pl.ahi_rule_conformity(prof)
    assert out["meta"]["hypopnea_criterion"] == pl._hypopnea_criterion_str(prof)


def test_een_gefaalde_arousalstap_degradeert_het_oordeel(monkeypatch):
    from psgscoring import pipeline as pl

    def _boom(**kw):
        raise RuntimeError("eeg kapot")

    monkeypatch.setattr(pl, "run_arousal_respiratory_analysis", _boom)
    out = _run()
    assert out["arousal"]["success"] is False
    rule = out["respiratory"]["summary"]["ahi_rule"]
    assert rule["verdict"] == "desat_only", rule
    assert rule["conform"] is False
    assert "arousal" in rule["reason"].lower()
    assert "eeg kapot" in rule["reason"]


def test_het_criterium_in_het_herkomstblok_zegt_hetzelfde(monkeypatch):
    from psgscoring import pipeline as pl

    def _boom(**kw):
        raise RuntimeError("eeg kapot")

    monkeypatch.setattr(pl, "run_arousal_respiratory_analysis", _boom)
    out = _run()
    crit = out["meta"]["hypopnea_criterion"]
    assert "OR arousal" not in crit, crit
    assert "desaturation" in crit


def test_zonder_eeg_kanaal_geldt_hetzelfde():
    """Geen EEG in de montage is de gewone manier waarop de tak uitvalt."""
    import psgscoring
    raw = _raw().drop_channels(["EEG C4-M1", "EMG chin"])
    out = psgscoring.run_pneumo_analysis(raw, hypno=HYPNO, scoring_profile=PROFIEL)
    assert out["arousal"]["success"] is False
    assert out["respiratory"]["summary"]["ahi_rule"]["verdict"] == "desat_only"


def test_de_detectorterugval_staat_in_de_reden(monkeypatch):
    """Faalt de ademteug-detector, dan draaide de envelope-detector -- en op
    dit profiel heeft die geen arousal-tak. Het oordeel hoort dat te zeggen."""
    import psgscoring.breath_scoring as bs

    def _boom(**kw):
        raise RuntimeError("ademteugdetector kapot")

    monkeypatch.setattr(bs, "score_hypopneas_breathwise", _boom)
    out = _run()
    assert out["meta"].get("detector_fallback") == "envelope"
    rule = out["respiratory"]["summary"]["ahi_rule"]
    assert rule["verdict"] == "desat_only", rule
    assert "envelope" in rule["reason"].lower()
