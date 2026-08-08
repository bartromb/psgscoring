"""PLM-scoring volgens AASM/WASM — de dunst gedekte module van het pakket.

Dekking was **14 %** (132 van 154 regels ongetest) terwijl de PLM-index in élk
rapport staat, en de noemer van die index is vandaag nog aangepast zonder dat
één test dat afdekte. Dat is de verkeerde volgorde.

De regels die hier getoetst worden, met hun bron in de code:

    LM_MIN_DUR_S       0,5 s    een beenbeweging duurt minstens een halve seconde
    LM_MAX_DUR_S      10,0 s    langer is geen beenbeweging maar iets anders
    PLM_MIN_INTERVAL_S 5,0 s    twee bewegingen binnen 5 s zijn één beweging
    PLM_MAX_INTERVAL_S 90,0 s   verder uit elkaar is geen periodiek patroon
    PLM_MIN_SERIES     4        een reeks telt pas vanaf vier op rij

Plus twee dingen die de AASM voorschrijft en die makkelijk misgaan: bewegingen
tijdens waak tellen niet mee, en bewegingen die aan het einde van een
respiratoir event vastzitten zijn arousal-gerelateerd en horen niet in de
PLM-index.
"""

import numpy as np
import pytest

from psgscoring.plm import (
    LM_MAX_DUR_S,
    LM_MIN_DUR_S,
    PLM_MAX_INTERVAL_S,
    PLM_MIN_INTERVAL_S,
    PLM_MIN_SERIES,
    _classify_plmi,
    _detect_series,
    _exclude_resp_associated,
    analyze_plm,
)

SF = 64.0
DUR_S = 3600
N = int(DUR_S * SF)
HYPNO = ["N2"] * (DUR_S // 30)


# ─────────────────────────────────────────────────────────────
#  Signaalopbouw
# ─────────────────────────────────────────────────────────────

def _emg(bursts, amp_uv=40.0, rust_uv=2.0, seed=0):
    """Rustig tibialis-EMG met bursts op de opgegeven (onset, duur)-paren.

    Amplitudes in microvolt; `analyze_plm` schaalt zelf op eenheid.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, rust_uv, N)
    for onset, dur in bursts:
        a, b = int(onset * SF), int((onset + dur) * SF)
        if a >= N:
            continue
        b = min(b, N)
        x[a:b] += rng.normal(0, amp_uv, b - a)
    return x


def _reeks(n, start=100.0, interval=20.0, dur=1.5):
    """n bewegingen met een vast interval — het klassieke PLM-patroon."""
    return [(start + i * interval, dur) for i in range(n)]


def _run(bursts, **kw):
    return analyze_plm(_emg(bursts), None, SF, HYPNO, leg_unit="uV", **kw)


# ─────────────────────────────────────────────────────────────
#  De reeksregel: vier op rij, 5–90 s uit elkaar
# ─────────────────────────────────────────────────────────────

def _lms(onsets, dur=1.5):
    return [{"onset_s": float(o), "duration_s": dur} for o in onsets]


def test_four_movements_in_rhythm_make_a_series():
    series, count = _detect_series(_lms([100, 120, 140, 160]))
    assert len(series) == 1
    assert count == 4
    assert series[0]["n_lms"] == 4


def test_three_is_not_a_series():
    """PLM_MIN_SERIES = 4. Drie bewegingen zijn geen periodiek patroon."""
    series, count = _detect_series(_lms([100, 120, 140]))
    assert series == [] and count == 0


def test_an_interval_below_five_seconds_breaks_the_chain():
    """Twee bewegingen binnen 5 s gelden als één; de reeks knapt daar."""
    _, count = _detect_series(_lms([100, 120, 123, 143, 163]))
    assert count == 0, "een te kort interval hoort de reeks te breken"


def test_an_interval_above_ninety_seconds_breaks_the_chain():
    _, count = _detect_series(_lms([100, 120, 140, 260, 280, 300]))
    assert count == 0, "een gat van 120 s is geen periodiek patroon"


def test_two_separate_series_are_both_counted():
    onsets = [100, 120, 140, 160] + [500, 520, 540, 560]
    series, count = _detect_series(_lms(onsets))
    assert len(series) == 2
    assert count == 8


def test_a_long_run_is_one_series_not_several():
    series, _ = _detect_series(_lms([100 + i * 20 for i in range(12)]))
    assert len(series) == 1
    assert series[0]["n_lms"] == 12


@pytest.mark.parametrize("interval,telt_mee", [
    (PLM_MIN_INTERVAL_S - 0.5, False),
    (PLM_MIN_INTERVAL_S + 0.5, True),
    (PLM_MAX_INTERVAL_S - 1.0, True),
    (PLM_MAX_INTERVAL_S + 1.0, False),
])
def test_the_interval_window_is_inclusive_where_it_says_it_is(interval, telt_mee):
    onsets = [100 + i * interval for i in range(PLM_MIN_SERIES)]
    _, count = _detect_series(_lms(onsets))
    assert (count > 0) is telt_mee


def test_the_series_spans_from_first_onset_to_last_end():
    series, _ = _detect_series(_lms([100, 120, 140, 160], dur=2.0))
    assert series[0]["start_s"] == 100
    assert series[0]["end_s"] == 162.0


# ─────────────────────────────────────────────────────────────
#  Respiratoir gekoppelde bewegingen tellen niet mee
# ─────────────────────────────────────────────────────────────

def test_a_movement_at_the_end_of_a_respiratory_event_is_excluded():
    """AASM: een beenbeweging die bij het einde van een apneu hoort is
    arousal-gerelateerd, geen PLM."""
    lms = _lms([100, 120, 140, 160])
    eligible, n_resp = _exclude_resp_associated(lms, resp_ends=[120.0])
    assert n_resp == 1
    assert [lm["onset_s"] for lm in eligible] == [100, 140, 160]


def test_the_exclusion_window_is_narrow():
    """Een halve seconde aan weerszijden — niet het hele event."""
    lms = _lms([100, 110])
    eligible, n_resp = _exclude_resp_associated(lms, resp_ends=[105.0])
    assert n_resp == 0, "5 s ervandaan hoort niet uitgesloten te worden"
    assert len(eligible) == 2


def test_exclusion_can_dissolve_a_series():
    """Vier bewegingen waarvan één respiratoir: drie blijven over, en drie is
    geen reeks. Dit is precies waarom de uitsluiting vóór de reeksdetectie
    hoort te gebeuren."""
    lms = _lms([100, 120, 140, 160])
    eligible, _ = _exclude_resp_associated(lms, resp_ends=[140.0])
    _, count = _detect_series(eligible)
    assert count == 0


def test_every_movement_is_labelled_either_way():
    """Ook de niet-uitgesloten bewegingen dragen het label, zodat een lezer
    kan zien dat de toets is uitgevoerd."""
    lms = _lms([100, 120])
    _exclude_resp_associated(lms, resp_ends=[120.0])
    assert lms[0]["resp_associated"] is False
    assert lms[1]["resp_associated"] is True


# ─────────────────────────────────────────────────────────────
#  De ernstclassificatie
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("plmi,verwacht", [
    (None, "unknown"),
    (0, "normal"),
    (4.9, "normal"),
    (5.0, "mild"),
    (14.9, "mild"),
])
def test_the_severity_bands(plmi, verwacht):
    assert _classify_plmi(plmi) == verwacht


def test_an_absent_index_is_unknown_not_normal():
    """None komt voor: geen slaaptijd, geen beenkanaal. Dat is een ONTBREKENDE
    uitslag, geen schone. "normal" leest als het tweede en gaf dus een clean
    bill of health op grond van niets — dezelfde verwarring die elders
    "AHI 0,0" naast 81 events opleverde."""
    assert _classify_plmi(None) == "unknown"
    assert _classify_plmi(0) == "normal", "nul gemeten bewegingen is wél normaal"


# ─────────────────────────────────────────────────────────────
#  De volledige analyse op een synthetisch signaal
# ─────────────────────────────────────────────────────────────

def test_a_clean_plm_pattern_is_found_end_to_end():
    r = _run(_reeks(10, start=200, interval=20))
    assert r.get("success"), r.get("error")
    s = r["summary"]
    assert s["n_lm_sleep"] >= 8, f"maar {s['n_lm_sleep']} bewegingen gevonden"
    assert s["n_plm"] >= 8, f"maar {s['n_plm']} in reeks"
    assert s["n_plm_series"] == 1
    assert s["plm_index"] is not None and s["plm_index"] > 0


def test_a_quiet_recording_yields_no_plms():
    r = analyze_plm(_emg([], seed=7), None, SF, HYPNO, leg_unit="uV")
    assert r.get("success")
    assert (r["summary"].get("n_plm") or 0) == 0


def test_movements_during_wake_do_not_count():
    """De index gaat per uur SLAAP; bewegingen in waak horen er niet in."""
    wakker = ["W"] * (DUR_S // 30)
    r = analyze_plm(_emg(_reeks(10, start=200)), None, SF, wakker, leg_unit="uV")
    assert r.get("success")
    assert (r["summary"].get("n_plm") or 0) == 0


def test_a_missing_channel_is_reported_not_crashed():
    r = analyze_plm(None, None, SF, HYPNO)
    assert r.get("success") is False
    assert r.get("error")


# ─────────────────────────────────────────────────────────────
#  De noemer — vandaag aangepast, hier vastgelegd
# ─────────────────────────────────────────────────────────────

def test_without_sleep_the_index_is_none_not_a_thousandfold_number():
    """`max(uren, 0.001)` maakte de index het aantal maal duizend. plm.py was
    een van de twaalf plaatsen waar die ondergrens stond."""
    wakker = ["W"] * (DUR_S // 30)
    r = analyze_plm(_emg(_reeks(10, start=200)), None, SF, wakker, leg_unit="uV")
    idx = r.get("summary", {}).get("plm_index")
    assert idx in (None, 0), f"kreeg {idx!r}"
    if idx is not None:
        assert idx < 1000


def test_the_index_divides_by_sleep_hours():
    """Eén uur slaap, tien bewegingen in reeks -> ongeveer tien per uur."""
    r = _run(_reeks(10, start=200, interval=20))
    s = r["summary"]
    if s.get("n_plm"):
        verwacht = s["n_plm"] / (DUR_S / 3600)
        assert s["plm_index"] == pytest.approx(verwacht, rel=0.02)


# ─────────────────────────────────────────────────────────────
#  Duurgrenzen
# ─────────────────────────────────────────────────────────────

def test_a_movement_shorter_than_half_a_second_is_not_a_leg_movement():
    """LM_MIN_DUR_S = 0,5 s. Kortere pieken zijn ruis of artefact."""
    r = _run([(200 + i * 20, LM_MIN_DUR_S / 2) for i in range(10)])
    assert r.get("success")
    assert (r["summary"].get("n_plm") or 0) == 0


@pytest.mark.xfail(reason=(
    "OPEN VRAAG, geen vastgestelde fout. De AASM begrenst een beenbeweging op "
    "10 s juist om aanhoudende activatie uit te sluiten. Gemeten: acht bursts "
    "van 15 s leveren twaalf beenbewegingen op — de lange activatie valt "
    "uiteen in stukken die elk wél binnen [0,5, 10] vallen, en de duurgrens "
    "verwerpt dus niets. Of dat over-detectie is hangt af van de vraag of een "
    "tonische contractie in de praktijk werkelijk ononderbroken boven de "
    "drempel blijft; dat is op echte opnames niet gemeten. Deze xfail houdt "
    "de vraag zichtbaar in plaats van hem weg te asserten."), strict=True)
def test_a_sustained_activation_should_not_become_a_run_of_plms():
    r = _run([(200 + i * 30, LM_MAX_DUR_S * 1.5) for i in range(8)])
    assert (r["summary"].get("n_plm") or 0) == 0


def test_a_sustained_activation_currently_fragments_into_several_movements():
    """Het gemeten gedrag, zodat een wijziging eraan zichtbaar wordt."""
    r = _run([(200 + i * 30, LM_MAX_DUR_S * 1.5) for i in range(8)])
    assert (r["summary"].get("n_lm_total") or 0) > 8, (
        "acht lange activaties leverden eerder twaalf beenbewegingen op; "
        "verandert dit, herzie de xfail hierboven")
