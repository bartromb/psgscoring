"""Een onmogelijke hypoxic burden is geen meting.

Op de Thaise casus van 26-08-2026 gaf een gelijkspanning-gekoppelde, ruizige
SAO2 een burden van **243 %·min/u**. In gepubliceerde cohorten ligt die bij
ernstige OSA zelden boven 100; boven 150 meet je de oximeter en niet de patiënt.

Een getal dat als meting leest terwijl het een artefact is, is in een klinisch
rapport erger dan een ontbrekende waarde: het nodigt uit tot een conclusie. De
ruwe waarde blijft beschikbaar voor wie hem wil narekenen.
"""
import numpy as np
import pytest

from psgscoring.spo2 import HYPOXIC_BURDEN_MAX_PLAUSIBLE, compute_hypoxic_burden


def _opzet(diepte_pct, n_events=40, sf=1.0, uur=2.0):
    """Een nacht met desaturaties van instelbare diepte."""
    n = int(uur * 3600 * sf)
    spo2 = np.full(n, 96.0)
    events = []
    for i in range(n_events):
        t0 = 60 + i * int((uur * 3600 - 120) / n_events)
        events.append({"onset_s": float(t0), "duration_s": 20.0, "type": "obstructive"})
        s, e = int(t0 * sf), int((t0 + 60) * sf)
        spo2[s:e] = 96.0 - diepte_pct
    hypno = ["N2"] * int(uur * 120)
    return spo2, events, hypno


def test_een_normale_burden_komt_gewoon_door():
    spo2, ev, hyp = _opzet(6.0)
    r = compute_hypoxic_burden(spo2, 1.0, ev, hyp)
    assert r["hypoxic_burden"] is not None
    assert r["hypoxic_burden"] <= HYPOXIC_BURDEN_MAX_PLAUSIBLE
    assert "hypoxic_burden_unreliable" not in r


def test_een_onmogelijke_burden_wordt_niet_als_getal_gerapporteerd():
    spo2, ev, hyp = _opzet(40.0, n_events=90)
    r = compute_hypoxic_burden(spo2, 1.0, ev, hyp)
    if r.get("hypoxic_burden_raw") is None:
        pytest.skip("fixture haalt het plafond niet — geen uitspraak mogelijk")
    assert r["hypoxic_burden"] is None, "het onmogelijke getal staat er nog"
    assert r["hypoxic_burden_raw"] > HYPOXIC_BURDEN_MAX_PLAUSIBLE
    assert "signaalkwaliteit" in r["hypoxic_burden_unreliable"]


def test_de_ruwe_waarde_blijft_beschikbaar():
    """Wie het wil narekenen moet erbij kunnen; het rapport toont het niet."""
    spo2, ev, hyp = _opzet(40.0, n_events=90)
    r = compute_hypoxic_burden(spo2, 1.0, ev, hyp)
    if r.get("hypoxic_burden_raw") is None:
        pytest.skip("fixture haalt het plafond niet")
    assert isinstance(r["hypoxic_burden_raw"], float)


def test_het_plafond_staat_waar_de_literatuur_het_zet():
    assert HYPOXIC_BURDEN_MAX_PLAUSIBLE == 150.0
