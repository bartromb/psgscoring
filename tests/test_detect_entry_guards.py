"""Twee vangnetten rond `detect_respiratory_events`.

1. EFFORTBANDEN DELEN STILZWIJGEND sf_flow EN DE LENGTE VAN HET FLOWKANAAL.
   Niets controleerde dat. Een thoraxband van de halve lengte liep tot een
   IndexError in `_compute_effort_baseline` (booleaans masker langer dan de
   envelope), de buitenste except ving die, en de hele detectie kwam terug
   als `success=False` met nul events. De apneus en hypopneeën op het
   flowkanaal waren gewoon te scoren; alleen de subtypering had geen effort.
   Nu valt de effortband weg mét reden, en de detectie draait door.

2. DE BUITENSTE except MAAKTE VAN EEN PROGRAMMEERFOUT EEN LEGE NACHT.
   `success=False` plus een lege eventlijst is in de AHI-intervalsweep van de
   pipeline niet te onderscheiden van een nacht zonder events: die arm leest
   als AHI 0. Twee reparaties: het resultaat draagt `error_type`, en met
   `PSGSCORING_STRICT=1` wordt een TypeError/IndexError/ValueError opnieuw
   geworpen zodat een test hem ZIET in plaats van een nul te meten.
"""
import numpy as np
import pytest

import psgscoring.respiratory as R

SF = 32.0
DUR_S = 600.0


def _flow():
    n = int(SF * DUR_S)
    t = np.arange(n) / SF
    flow = np.sin(2 * np.pi * 0.25 * t)
    flow[int(100 * SF):int(115 * SF)] *= 0.02
    return flow


def _hypno():
    return ["N2"] * int(DUR_S // 30)


# ── 1. effortlengte ──────────────────────────────────────────────────────

def test_een_te_korte_effortband_valt_weg_met_reden_en_de_detectie_slaagt():
    flow = _flow()
    thorax = np.sin(2 * np.pi * 0.25 * np.arange(len(flow) // 2) / SF)
    out = R.detect_respiratory_events(
        flow_data=flow, thorax_data=thorax, abdomen_data=thorax.copy(),
        spo2_data=None, sf_flow=SF, sf_spo2=1.0, hypno=_hypno())
    assert out["success"] is True, out.get("error")
    assert out["effort_dropped"]["reason"].startswith("length_mismatch"), out["effort_dropped"]
    assert out["effort_dropped"]["channels"] == ["thorax", "abdomen"]
    assert len(out["events"]) >= 1


def test_een_passende_effortband_wordt_niet_aangeraakt():
    flow = _flow()
    thorax = np.sin(2 * np.pi * 0.25 * np.arange(len(flow)) / SF)
    out = R.detect_respiratory_events(
        flow_data=flow, thorax_data=thorax, abdomen_data=thorax.copy(),
        spo2_data=None, sf_flow=SF, sf_spo2=1.0, hypno=_hypno())
    assert out["success"] is True
    assert "effort_dropped" not in out


def test_alleen_de_afwijkende_band_valt_weg():
    flow = _flow()
    goed = np.sin(2 * np.pi * 0.25 * np.arange(len(flow)) / SF)
    fout = goed[: len(goed) - 500]
    out = R.detect_respiratory_events(
        flow_data=flow, thorax_data=goed, abdomen_data=fout,
        spo2_data=None, sf_flow=SF, sf_spo2=1.0, hypno=_hypno())
    assert out["success"] is True, out.get("error")
    assert out["effort_dropped"]["channels"] == ["abdomen"]


# ── 2. strikte modus en error_type ───────────────────────────────────────

def _breek(monkeypatch, exc):
    def _kapot(*a, **k):
        raise exc("opzettelijk")
    monkeypatch.setattr(R, "preprocess_flow", _kapot)


def test_een_fout_draagt_zijn_type(monkeypatch):
    monkeypatch.delenv("PSGSCORING_STRICT", raising=False)
    _breek(monkeypatch, TypeError)
    out = R.detect_respiratory_events(
        flow_data=_flow(), thorax_data=None, abdomen_data=None,
        spo2_data=None, sf_flow=SF, sf_spo2=1.0, hypno=_hypno())
    assert out["success"] is False
    assert out["error_type"] == "TypeError"
    assert "opzettelijk" in out["error"]


@pytest.mark.parametrize("exc", [TypeError, IndexError, ValueError])
def test_strikt_werpt_programmeerfouten_opnieuw(monkeypatch, exc):
    monkeypatch.setenv("PSGSCORING_STRICT", "1")
    _breek(monkeypatch, exc)
    with pytest.raises(exc):
        R.detect_respiratory_events(
            flow_data=_flow(), thorax_data=None, abdomen_data=None,
            spo2_data=None, sf_flow=SF, sf_spo2=1.0, hypno=_hypno())


def test_strikt_laat_andere_fouten_gedegradeerd_terugkomen(monkeypatch):
    """Een RuntimeError uit een kapotte afhankelijkheid is geen programmeerfout
    van de detector; die blijft het bestaande vangnet houden."""
    monkeypatch.setenv("PSGSCORING_STRICT", "1")
    _breek(monkeypatch, RuntimeError)
    out = R.detect_respiratory_events(
        flow_data=_flow(), thorax_data=None, abdomen_data=None,
        spo2_data=None, sf_flow=SF, sf_spo2=1.0, hypno=_hypno())
    assert out["success"] is False and out["error_type"] == "RuntimeError"
