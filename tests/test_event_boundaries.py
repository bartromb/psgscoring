"""Blok 1B stap 2: eventgrenzen naar ademteuggrenzen snappen.

De meting van 14-08-2026 op PSG-IPA: de cascadedetector start hypopneus 2–5 s
eerder dan menselijke scoorders, dezelfde richting op alle vijf opnames, vier
buiten de menselijke interkwartielafstand. Apneus zitten er wél in. Zie
`PostProcessingRules.event_boundaries`.

De kern van deze suite is dat het snappen aantoonbaar iets DOET en dat het
default uit staat. Een test die groen blijft bij een no-op snapper meet niets.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring.profiles import PROFILES, PostProcessingRules
from psgscoring.respiratory import snap_events_to_breaths


def _breaths(n=200, duur=4.0, start=0.0):
    """Regelmatige ademteugen van `duur` seconden."""
    return [{"onset_s": start + i * duur, "duration_s": duur,
             "amplitude": 1.0} for i in range(n)]


def _ev(onset, duur, type_="hypopnea"):
    return {"type": type_, "onset_s": onset, "duration_s": duur,
            "stage": "N2", "epoch": int(onset // 30)}


# ── het veld ─────────────────────────────────────────────────────────

def test_default_is_envelope():
    assert PostProcessingRules().event_boundaries == "envelope"


def test_geen_enkel_geleverd_profiel_snapt():
    aan = [n for n, p in PROFILES.items()
           if p.post_processing.event_boundaries != "envelope"]
    assert aan == [], f"profielen zouden byte-identiek blijven: {aan}"


def test_veld_bereikt_de_legacy_dict():
    import psgscoring.constants as C
    for naam, d in C.SCORING_PROFILES.items():
        assert d["EVENT_BOUNDARIES"] == "envelope", naam


# ── het snappen ──────────────────────────────────────────────────────

def test_grenzen_verschuiven_werkelijk():
    """Zonder deze test meet de rest niets."""
    # Event begint 1,4 s vóór een ademteugrand en eindigt 0,6 s erna.
    ev = [_ev(38.6, 21.4)]          # 38,6 -> 60,0
    uit, st = snap_events_to_breaths(ev, _breaths())
    assert st["n_snapped"] == 1
    assert uit[0]["onset_s"] == 40.0, uit[0]["onset_s"]
    assert uit[0]["onset_s"] + uit[0]["duration_s"] == 60.0


def test_originelen_worden_niet_gemuteerd():
    ev = [_ev(38.6, 21.4)]
    origineel = dict(ev[0])
    snap_events_to_breaths(ev, _breaths())
    assert ev[0] == origineel


def test_envelope_grenzen_blijven_traceerbaar():
    uit, _ = snap_events_to_breaths([_ev(38.6, 21.4)], _breaths())
    det = uit[0]["classify_detail"]
    assert det["boundaries"] == "breath"
    assert det["envelope_onset_s"] == 38.6
    assert det["envelope_offset_s"] == 60.0


def test_al_uitgelijnde_grenzen_verschuiven_niet():
    uit, st = snap_events_to_breaths([_ev(40.0, 20.0)], _breaths())
    assert uit[0]["onset_s"] == 40.0
    assert uit[0]["duration_s"] == 20.0
    assert st["median_onset_shift_s"] == 0.0


def test_zonder_ademteugen_gebeurt_er_niets():
    ev = [_ev(38.6, 21.4)]
    uit, st = snap_events_to_breaths(ev, [])
    assert uit == ev and st["n_snapped"] == 0 and st["n_unchanged"] == 1


def test_event_korter_dan_een_ademteug_blijft_ongemoeid():
    """Dan is de ademteugdetectie ter plaatse onbetrouwbaar."""
    uit, st = snap_events_to_breaths([_ev(41.0, 1.5)], _breaths())
    assert st["n_snapped"] == 0 and st["n_unchanged"] == 1
    assert uit[0]["onset_s"] == 41.0


def test_nul_gefitte_parameters():
    """Elke nieuwe grens is letterlijk een bestaande ademteugrand.

    Als hier ooit een naschuifconstante bij komt, valt deze test om — en dat
    hoort, want zo'n constante zou op vijf opnames gefit zijn.
    """
    br = _breaths()
    randen = {b["onset_s"] for b in br} | {b["onset_s"] + b["duration_s"] for b in br}
    ev = [_ev(38.6 + 3.1 * i, 17.3) for i in range(12)]
    uit, st = snap_events_to_breaths(ev, br)
    assert st["n_snapped"] >= 10
    for e in uit:
        if (e.get("classify_detail") or {}).get("boundaries") != "breath":
            continue
        assert e["onset_s"] in randen
        assert round(e["onset_s"] + e["duration_s"], 2) in randen


def test_richting_klopt_met_de_meting():
    """De cascade start te VROEG, dus snappen schuift de onset naar LATER."""
    br = _breaths()
    # Events die consequent 1,5 s te vroeg beginnen — de gemeten faalmodus.
    ev = [_ev(40.0 * (i + 1) - 1.5, 18.0) for i in range(8)]
    _uit, st = snap_events_to_breaths(ev, br)
    assert st["median_onset_shift_s"] > 0, (
        "snappen hoort de te vroege onset naar later te schuiven")


def test_apneus_worden_net_zo_behandeld():
    """Het snappen kent geen eventtype; de meting zegt alleen dat apneus het
    niet NODIG hebben, niet dat ze uitgesloten moeten worden."""
    uit, st = snap_events_to_breaths([_ev(38.6, 21.4, "obstructive")], _breaths())
    assert st["n_snapped"] == 1 and uit[0]["type"] == "obstructive"


@pytest.mark.parametrize("ademduur", [2.5, 4.0, 6.0])
def test_werkt_bij_verschillende_ademfrequenties(ademduur):
    br = _breaths(duur=ademduur)
    uit, st = snap_events_to_breaths([_ev(ademduur * 10 + 0.9, ademduur * 5)], br)
    assert st["n_snapped"] == 1
    randen = {b["onset_s"] for b in br} | {b["onset_s"] + b["duration_s"] for b in br}
    assert uit[0]["onset_s"] in randen
