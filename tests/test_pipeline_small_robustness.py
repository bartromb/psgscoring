"""Kleine robuustheidspunten uit de codereview van pipeline.py.

* `_rule1a_gate_param` liet `float(env)` ongevangen: een typefout in een
  meetvlag liet de hele run omvallen, waar elke andere env-helper
  waarschuwt en op het profiel terugvalt.
* `_thermistor_gate` vergeleek de env-waarde letterlijk; " Respiratory_Band "
  werd stil als onbekend genegeerd, waar `_arousal_detector` strip/lower doet.
* een gefaalde ankerbasislijn stond op DEBUG, dus onzichtbaar in productie.
* `_compute_flow_norm` slikte zijn exception zonder spoor.
* `_severity` dupliceerde `_ahi_severity`; `_MNE_AVAILABLE` en
  `artifact_epochs_for_summary` werden nergens gelezen.
"""
import logging
import pathlib

import numpy as np
import pytest

from psgscoring.pipeline import _rule1a_gate_param, _thermistor_gate

SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "psgscoring" / "pipeline.py")


# ── _rule1a_gate_param ──────────────────────────────────────────────────

def test_een_onleesbare_env_waarde_valt_terug_op_het_profiel(caplog):
    with caplog.at_level(logging.WARNING, logger="psgscoring.pipeline"):
        assert _rule1a_gate_param(0.3, "abc") == 0.3
    assert "abc" in caplog.text
    assert "geen getal" in caplog.text


def test_de_bestaande_semantiek_blijft():
    assert _rule1a_gate_param(0.3, None) == 0.3
    assert _rule1a_gate_param(0.3, "") == 0.3
    assert _rule1a_gate_param(0.3, "0.5") == 0.5
    assert _rule1a_gate_param(0.3, " off ") is None
    assert _rule1a_gate_param(None, None) is None


# ── _thermistor_gate ────────────────────────────────────────────────────

def test_de_env_poort_wordt_gestript_en_verkleind(monkeypatch):
    monkeypatch.setenv("PSGSCORING_THERMISTOR_GATE", " Respiratory_Band ")
    assert _thermistor_gate({"THERMISTOR_GATE": "envelope_agreement"}) == \
        "respiratory_band"


def test_een_onbekende_poort_blijft_een_waarschuwing(monkeypatch, caplog):
    monkeypatch.setenv("PSGSCORING_THERMISTOR_GATE", "bestaat_niet")
    with caplog.at_level(logging.WARNING, logger="psgscoring.pipeline"):
        assert _thermistor_gate({"THERMISTOR_GATE": "envelope_agreement"}) == \
            "envelope_agreement"
    assert "bestaat_niet" in caplog.text


# ── ankerbasislijn en flow-normalisatie loggen hun falen ────────────────

def _raw():
    mne = pytest.importorskip("mne")
    sf, dur = 32.0, 300
    n = int(sf * dur)
    t = np.arange(n) / sf
    info = mne.create_info(["Resp nasal", "SaO2"], sf, ["misc", "misc"])
    raw = mne.io.RawArray(np.vstack([np.sin(2 * np.pi * 0.25 * t),
                                     np.full(n, 96.0)]), info, verbose="ERROR")
    return raw, ["N2"] * (dur // 30)


def test_een_gefaalde_ankerbasislijn_staat_op_warning(monkeypatch, caplog):
    import psgscoring
    import psgscoring.signal as sig

    def _boom(*a, **kw):
        raise RuntimeError("anker kapot")

    monkeypatch.setattr(sig, "compute_anchor_baseline", _boom)
    raw, hypno = _raw()
    with caplog.at_level(logging.WARNING, logger="psgscoring.pipeline"):
        out = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                             scoring_profile="aasm_v3_rec")
    assert "anker kapot" in caplog.text
    assert out["anchor_baseline"]["anchor_reliable"] is False
    assert "anker kapot" in out["anchor_baseline"]["error"]


def test_de_flow_normalisatie_logt_haar_exception(monkeypatch, caplog):
    from psgscoring import pipeline as pl

    def _boom(*a, **kw):
        raise ValueError("envelope kapot")

    monkeypatch.setattr(pl, "preprocess_flow", _boom)
    with caplog.at_level(logging.WARNING, logger="psgscoring.pipeline"):
        assert pl._compute_flow_norm(np.ones(100), 10.0) is None
    assert "envelope kapot" in caplog.text


# ── dode code ───────────────────────────────────────────────────────────

def test_de_dode_symbolen_zijn_weg():
    src = SRC.read_text()
    assert "_MNE_AVAILABLE" not in src
    assert "artifact_epochs_for_summary" not in src
    assert "def _severity(" not in src
    assert "_ahi_severity(_ahi_strict)" in src, (
        "de intervalsweep hoort dezelfde ernstbanden te gebruiken als de rest")
