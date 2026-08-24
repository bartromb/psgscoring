"""Een positie-AHI over een halve minuut is geen index.

Eén klinisch rapport toonde **"AHI Supine 120,0/u"**. De patiënt lag 0,5 min op
de rug en had daar één event: 1 / (0,5/60) = 120. Dat getal staat in dezelfde
tabel als de echte indices en leest als een meting.

`analyze_position` deelde zonder ondergrens, en gaf bij nul minuten **0**
terug — wat leest als "geen events op de rug" terwijl de patiënt er niet
gelegen heeft. Twee verschillende onwaarheden uit dezelfde regel.

De POSA-fenotypering hanteert al ≥ 30 min per groep voordat ze iets beweert
(`_compute_phenotypes`); de rapporttabel hanteerde niets.
"""
import numpy as np
import pytest

from psgscoring.ancillary import analyze_position
from psgscoring.constants import POSITION_MIN_MINUTES

SF = 4.0
EPOCH_S = 30.0


def _run(minuten_per_code: dict, events_per_code: dict):
    """Bouw een nacht met een opgegeven aantal minuten per positiecode."""
    hypno, pos_epochs = [], []
    for code, minuten in minuten_per_code.items():
        n_ep = int(round(minuten * 60 / EPOCH_S))
        hypno += ["N2"] * n_ep
        pos_epochs += [code] * n_ep
    pos_data = np.repeat(np.array(pos_epochs, dtype=float),
                         int(SF * EPOCH_S))
    events = []
    for code, n in events_per_code.items():
        eps = [i for i, c in enumerate(pos_epochs) if c == code][:n]
        events += [{"epoch": e, "onset_s": e * EPOCH_S, "duration_s": 12.0}
                   for e in eps]
    return analyze_position(pos_data, SF, hypno, events)


def test_a_half_minute_supine_yields_no_index():
    out = _run({2: 0.5, 1: 400.0}, {2: 1, 1: 40})
    s = out["summary"]
    assert s["ahi_per_pos"]["Supine"] is None, (
        f"120/u uit één event in 30 s: {s['ahi_per_pos']['Supine']}")
    assert s["sleep_time_min"]["Supine"] == pytest.approx(0.5)


def test_a_position_never_slept_in_is_not_an_ahi_of_zero():
    out = _run({1: 400.0}, {1: 40})
    assert out["summary"]["ahi_per_pos"]["Prone"] is None, (
        "0/u leest als 'geen events buikligging' terwijl de patiënt er niet "
        "gelegen heeft")


def test_enough_time_still_gives_a_number():
    out = _run({2: 120.0, 1: 300.0}, {2: 20, 1: 10})
    ahi = out["summary"]["ahi_per_pos"]["Supine"]
    assert ahi == pytest.approx(20 / 2.0, abs=0.1)


def test_the_boundary_is_the_documented_constant():
    net_onder = _run({2: POSITION_MIN_MINUTES - 0.5, 1: 400.0}, {2: 3, 1: 40})
    net_boven = _run({2: POSITION_MIN_MINUTES + 0.5, 1: 400.0}, {2: 3, 1: 40})
    assert net_onder["summary"]["ahi_per_pos"]["Supine"] is None
    assert net_boven["summary"]["ahi_per_pos"]["Supine"] is not None


def test_the_event_counts_are_published_so_nobody_reconstructs_them():
    """`_compute_phenotypes` leidde het aantal events af uit index x uren.
    Met een index die None kan zijn werkt dat niet meer, en het was sowieso
    een omweg langs een afronding."""
    out = _run({2: 120.0, 1: 300.0}, {2: 20, 1: 10})
    n = out["summary"]["n_events_per_pos"]
    assert n["Supine"] == 20 and n["Left"] == 10
    assert n["Prone"] == 0


def test_the_phenotype_still_works_when_a_group_is_too_short():
    """De POSA-tak reconstrueerde events uit `ahi_per_pos`; een None daar mag
    hem niet laten struikelen."""
    import psgscoring.pipeline as P
    out = _run({2: 120.0, 1: 300.0, 0: 0.5}, {2: 60, 1: 10, 0: 1})
    output = {"position": out,
              "respiratory": {"summary": {"ahi_total": 12.0}}}
    P._compute_phenotypes(output, ["N2"] * 100)
    posa = (output["respiratory"]["summary"].get("phenotypes") or {}).get(
        "positional_osa")
    assert posa is not None and posa.get("flag") is True, posa


def test_a_short_position_still_contributes_its_events_to_the_non_supine_ahi():
    """De valkuil van de reparatie zelf.

    `_compute_phenotypes` telde de niet-rugligevents op als `index x uren`.
    Zodra een korte houding géén index meer heeft, valt haar bijdrage weg
    terwijl haar MINUTEN in de noemer blijven staan. De niet-ruglig-AHI wordt
    dan te laag, de verhouding te hoog, en het POSA-fenotype kan omslaan van
    afwezig naar aanwezig — een reparatie die een klinische uitspraak
    verzint.

    Hier: ruglig 8,0/u, niet-ruglig werkelijk 4,6/u (geen POSA, 8 < 2x4,6),
    maar met de weggevallen buikligevents 1,9/u (wel POSA, 8 >= 2x1,9).
    """
    import psgscoring.pipeline as P

    out = _run({2: 60.0, 1: 300.0, 0: 14.0}, {2: 8, 1: 10, 0: 14})
    s = out["summary"]
    assert s["ahi_per_pos"]["Prone"] is None, "fixture moet onder de grens zitten"
    assert s["n_events_per_pos"]["Prone"] == 14

    output = {"position": out,
              "respiratory": {"summary": {"ahi_total": 12.0}}}
    P._compute_phenotypes(output, ["N2"] * 100)
    posa = (output["respiratory"]["summary"].get("phenotypes") or {}).get(
        "positional_osa")
    assert posa is not None, "POSA-tak niet gedraaid"
    assert posa["ahi_non_supine"] == pytest.approx(24 / (314 / 60), abs=0.15), (
        f"niet-ruglig-AHI {posa['ahi_non_supine']} — de 14 buikligevents zijn "
        f"kwijt")
    assert posa["flag"] is False, (
        "POSA gesteld op een verhouding die uit weggevallen events komt")
