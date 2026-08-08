"""Een index die niet te berekenen is, mag geen getal zijn.

Aanleiding is een echt rapport. Een polygrafie zonder EEG kreeg de neusdruk
als staging-kanaal toegewezen (het formulier eist een EEG-kanaal), YASA
produceerde daarop een hypnogram, en de artefactdetector keurde vervolgens
ALLE 1078 epochs af — terecht, want hij keek naar datzelfde niet-EEG-kanaal.

Daarmee bleef er nul slaaptijd over. De noemer had een ondergrens van 0,001
uur "tegen deling door nul", dus:

    81 hypopnees / 0,001 uur = 81000,0 /u

Dat getal ging ongehinderd door de ernstclassificatie en kwam als
"REI 81000,0/u -> Ernstig SAS -> therapie CPAP" in het rapport.

Twee dingen die deze tests vastleggen:

  * de noemer krijgt geen ondergrens meer, dus zo'n getal kan niet ontstaan;
  * de uitkomst is None, niet 0. Nul leest als "geen events" en is daarmee
    geruststellend fout, wat klinisch erger is dan zichtbaar fout.
"""

import pytest

from psgscoring.respiratory import _compute_summary

E = 30.0  # epochlengte in seconden


def _events(n, stage="N2"):
    """n hypopnees, netjes uit elkaar zodat niets samenvalt."""
    return [{"type": "hypopnea", "onset_s": 100.0 + i * 60,
             "duration_s": 20.0, "stage": stage, "confidence": 0.70}
            for i in range(n)]


# ─────────────────────────────────────────────────────────────
#  Het geval uit het rapport
# ─────────────────────────────────────────────────────────────

def test_all_epochs_rejected_as_artefact_gives_no_index_at_all():
    """Precies de situatie van het echte rapport: 1078 epochs, alles artefact."""
    hypno = ["N2"] * 1078
    s = _compute_summary(_events(81), hypno, artifact_epochs=list(range(1078)))
    assert s["ahi_total"] is None, f"kreeg {s['ahi_total']!r}"
    assert s["indices_computable"] is False
    assert s["index_denominator_h"] == 0


def test_the_event_count_survives_even_when_the_index_cannot_be_computed():
    """De telling is wel degelijk geldig — alleen de deling niet. Die mag je
    de gebruiker niet ook afnemen."""
    s = _compute_summary(_events(81), ["N2"] * 1078,
                         artifact_epochs=list(range(1078)))
    assert s["n_hypopnea"] == 81
    assert s["n_ah_total"] == 81


def test_the_reason_says_which_of_the_three_causes_it_was():
    s = _compute_summary(_events(81), ["N2"] * 1078,
                         artifact_epochs=list(range(1078)))
    assert "artefact" in s["index_unavailable_reason"]
    assert "1078" in s["index_unavailable_reason"]


@pytest.mark.parametrize("hypno,artefacts,fragment", [
    ([], None, "geen hypnogram"),
    (["W"] * 100, None, "uitsluitend wake"),
    (["N2"] * 100, list(range(100)), "artefact"),
])
def test_each_cause_is_named_separately(hypno, artefacts, fragment):
    """Drie verschillende oorzaken, drie verschillende meldingen. "Geen
    hypnogram" vraagt om een andere reactie dan "alles artefact"."""
    s = _compute_summary(_events(5), hypno, artifact_epochs=artefacts)
    assert s["ahi_total"] is None
    assert fragment in s["index_unavailable_reason"]


# ─────────────────────────────────────────────────────────────
#  Nul is geen antwoord
# ─────────────────────────────────────────────────────────────

def test_an_uncomputable_index_is_not_reported_as_zero():
    """Dit is de kern. `AHI 0,0` bij 81 events is geruststellend fout."""
    s = _compute_summary(_events(81), ["W"] * 500)
    assert s["ahi_total"] is not None or s["ahi_total"] is None  # leesbaarheid
    assert s["ahi_total"] != 0, "0 leest als 'geen events' terwijl er 81 zijn"
    assert s["ahi_total"] is None


def test_no_rem_sleep_is_not_a_rem_ahi_of_zero():
    """Geen REM betekent dat er geen REM-AHI bestaat, niet dat hij nul is."""
    s = _compute_summary(_events(10), ["N2"] * 200)
    assert s["ahi_rem"] is None, "geen REM -> geen REM-AHI"
    assert s["ahi_nrem"] is not None


def test_the_old_thousandfold_number_can_no_longer_arise():
    """De regressie zelf: geen enkele index mag het aantal maal duizend zijn."""
    n = 81
    s = _compute_summary(_events(n), ["N2"] * 1078,
                         artifact_epochs=list(range(1078)))
    for k, v in s.items():
        if isinstance(v, (int, float)) and v == n * 1000:
            pytest.fail(f"{k} = {v} — de oude ondergrens is terug")


# ─────────────────────────────────────────────────────────────
#  En een normale opname blijft normaal
# ─────────────────────────────────────────────────────────────

def test_a_normal_recording_is_unaffected():
    """400 epochs slaap = 3,333 uur; 10 events -> 3,0/u."""
    s = _compute_summary(_events(10), ["N2"] * 400)
    assert s["indices_computable"] is True
    assert s["index_denominator_h"] == pytest.approx(400 * E / 3600, abs=0.001)
    assert s["ahi_total"] == pytest.approx(10 / (400 * E / 3600), abs=0.05)
    assert s["index_unavailable_reason"] is None


def test_artefacts_shrink_the_denominator_without_breaking_it():
    """Gedeeltelijke artefactuitsluiting is normaal en hoort gewoon te werken."""
    s = _compute_summary(_events(10), ["N2"] * 400,
                         artifact_epochs=list(range(100)))
    assert s["indices_computable"] is True
    assert s["index_denominator_h"] == pytest.approx(300 * E / 3600, abs=0.001)


# ─────────────────────────────────────────────────────────────
#  Dezelfde ondergrens stond op TWAALF plaatsen
# ─────────────────────────────────────────────────────────────
#
# De eerste reparatie raakte alleen respiratory.py. Daardoor klopte de AHI
# terwijl de arousal-index, PLM-index, ODI, RERA-index en RDI nog steeds het
# aantal maal duizend waren op dezelfde opname. Deze test bewaakt dat er geen
# nieuwe bijkomt.

def test_no_module_floors_a_denominator_at_a_thousandth_of_an_hour():
    """`max(x / 3600, 0.001)` is 3,6 seconden — geen bescherming maar een
    verzinsel. Zie psgscoring/indices.py."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent / "psgscoring"
    overtreders = []
    for f in root.glob("*.py"):
        if f.name == "indices.py":
            continue
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"max\([^)]*3600[^)]*,\s*0\.001\s*\)", line):
                overtreders.append(f"{f.name}:{n}")
    assert overtreders == [], f"ondergrens op de noemer terug: {overtreders}"


def test_per_hour_refuses_a_zero_denominator():
    from psgscoring.indices import per_hour
    assert per_hour(81, 0) is None
    assert per_hour(81, None) is None
    assert per_hour(None, 8.0) is None
    assert per_hour(81, 8.983) == pytest.approx(9.0, abs=0.05)


def test_per_hour_never_produces_the_thousandfold_number():
    from psgscoring.indices import per_hour
    for n in (1, 5, 81, 226):
        assert per_hour(n, 0) != n * 1000
        assert per_hour(n, 0) is None
