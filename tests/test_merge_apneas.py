"""Apneus van twee flowsensoren samenvoegen zonder dubbel te tellen.

Kiezen tussen thermistor en nasale druk is een valse keuze: de druk mist
mondademhaling, de thermistor is te traag voor korte events. Een apneu die op
een van beide zichtbaar is, is een apneu — maar hij mag niet twee keer in de
AHI belanden.

Ontdubbelen gebeurt op OVERLAP, niet op gelijke tijden: temperatuur volgt
trager dan druk, dus dezelfde gebeurtenis krijgt op de twee sensoren
verschoven grenzen.
"""
import pytest

from psgscoring.postprocess import merge_apnea_events


def ev(onset, dur=20.0, conf=0.8, etype="obstructive"):
    return {"onset_s": float(onset), "duration_s": float(dur),
            "confidence": conf, "type": etype}


# ─────────────────────────────────────────────────────────────────
#  Niet dubbel tellen
# ─────────────────────────────────────────────────────────────────

def test_the_same_event_on_both_sensors_counts_once():
    a = [ev(100, 20)]
    b = [ev(103, 22)]                      # thermistor loopt achter
    out, d = merge_apnea_events(a, b)
    assert len(out) == 1
    assert d["n_merged"] == 1
    assert out[0]["seen_on_both_sensors"] is True


def test_shifted_boundaries_still_merge():
    """Temperatuur volgt traag; dezelfde apneu ligt verschoven."""
    a = [ev(200, 25)]
    for shift in (0, 3, 6, 9):
        out, d = merge_apnea_events(a, [ev(200 + shift, 25)])
        assert len(out) == 1, f"verschuiving {shift}s brak de match"
        assert d["n_merged"] == 1


def test_events_far_apart_are_both_kept():
    out, d = merge_apnea_events([ev(100)], [ev(500)])
    assert len(out) == 2
    assert d["n_merged"] == 0
    assert d["n_secondary_only_added"] == 1


def test_output_is_sorted_by_onset():
    out, _ = merge_apnea_events([ev(500), ev(100)], [ev(300)])
    assert [e["onset_s"] for e in out] == [100.0, 300.0, 500.0]


# ─────────────────────────────────────────────────────────────────
#  Welk event overleeft de overlap
# ─────────────────────────────────────────────────────────────────

def test_keep_longest_takes_the_longer_event():
    out, _ = merge_apnea_events([ev(100, 15)], [ev(101, 30)], keep="longest")
    assert out[0]["duration_s"] == 30.0


def test_keep_primary_always_takes_the_aasm_sensor():
    out, _ = merge_apnea_events([ev(100, 15)], [ev(101, 30)], keep="primary")
    assert out[0]["duration_s"] == 15.0


def test_keep_confident_takes_the_higher_confidence():
    out, _ = merge_apnea_events([ev(100, 15, conf=0.4)],
                                [ev(101, 30, conf=0.95)], keep="confident")
    assert out[0]["confidence"] == 0.95


# ─────────────────────────────────────────────────────────────────
#  De klinische knop
# ─────────────────────────────────────────────────────────────────

def test_secondary_only_events_can_be_included_or_excluded():
    """Alleen-op-de-tweede-sensor is mondademhaling of een losse neusbril.

    Welke van de twee het is, kan de software niet weten — daarom een knop
    en geen aanname.
    """
    a = [ev(100)]
    b = [ev(100), ev(400)]
    incl, d_in = merge_apnea_events(a, b, secondary_only=True)
    excl, d_ex = merge_apnea_events(a, b, secondary_only=False)
    assert len(incl) == 2 and d_in["n_secondary_only_added"] == 1
    assert len(excl) == 1 and d_ex["n_secondary_only_added"] == 0
    assert incl[-1]["secondary_sensor_only"] is True


def test_excluding_secondary_only_still_merges_shared_events():
    out, d = merge_apnea_events([ev(100)], [ev(102)], secondary_only=False)
    assert len(out) == 1
    assert d["n_merged"] == 1


# ─────────────────────────────────────────────────────────────────
#  Degeneratiegevallen
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,n", [
    ([], [], 0),
    ([ev(100)], [], 1),
    ([], [ev(100)], 1),
    (None, None, 0),
])
def test_empty_inputs(a, b, n):
    out, d = merge_apnea_events(a, b)
    assert len(out) == n == d["n_total"]


def test_inputs_are_not_mutated():
    a = [ev(100, 15)]
    b = [ev(101, 30)]
    before = dict(a[0])
    merge_apnea_events(a, b)
    assert a[0] == before, "de aanroeper zijn events mogen niet veranderen"


def test_merge_diagnostics_account_for_every_event():
    a = [ev(100), ev(300)]
    b = [ev(102), ev(700)]
    _out, d = merge_apnea_events(a, b)
    assert d["n_total"] == d["n_primary"] + d["n_secondary"] - d["n_merged"]


# ─────────────────────────────────────────────────────────────────
#  Corroboratie: de tweede sensor falsifieert, hij telt niet op
# ─────────────────────────────────────────────────────────────────

from psgscoring.postprocess import corroborate_apnea_events  # noqa: E402


def test_nothing_is_rejected_without_a_licence():
    """Bij twijfel over de tweede sensor blijft een apneu een apneu.

    Falsifieren met een sensor die je niet vertrouwt is precies hoe een echte
    opname 83 centrale apneus verloor. De veilige kant is daarom de default.
    """
    out, d = corroborate_apnea_events([ev(100), ev(300)], [ev(102), ev(700)])
    assert d["corroboration_licensed"] is False
    assert len(out) == 3, "zonder licentie mag niets wegvallen"
    assert d["n_both"] == 1


def test_only_corroborated_apneas_are_kept_with_a_licence():
    """Een OF-regel zou juist de valse apneus toevoegen; dit is een EN."""
    therm = [ev(100), ev(300)]
    press = [ev(102), ev(700)]
    out, d = corroborate_apnea_events(therm, press, corroboration_licensed=True)
    assert d["n_both"] == 1
    assert d["n_thermistor_only"] == 1
    assert d["n_pressure_only"] == 1
    assert len(out) == 1
    assert out[0]["corroboration"] == "both"


def test_pressure_only_is_treated_as_mouth_breathing():
    """Neusdruk zakt weg terwijl de thermistor flow ziet = geen apneu.

    Dit is precies de overdetectie waar de AASM de thermistor voor
    voorschrijft, en die een naieve unie zou binnenhalen.
    """
    out, d = corroborate_apnea_events([], [ev(100), ev(200)],
                                      corroboration_licensed=True)
    assert d["n_pressure_only"] == 2
    assert out == []


def test_thermistor_only_is_treated_as_suspect_artefact():
    """De thermistor voelt oraal en nasaal; nasale flow hoort hij te zien."""
    out, d = corroborate_apnea_events([ev(100)], [],
                                      corroboration_licensed=True)
    assert d["n_thermistor_only"] == 1
    assert out == []


def test_the_thermistor_defines_the_boundaries_of_a_shared_event():
    """Bij overeenstemming wint de AASM-sensor voor apneus."""
    out, _ = corroborate_apnea_events([ev(100, 25)], [ev(103, 40)])
    assert out[0]["onset_s"] == 100.0
    assert out[0]["duration_s"] == 25.0


def test_shifted_boundaries_still_corroborate():
    for shift in (0, 3, 6, 9):
        _out, d = corroborate_apnea_events([ev(200, 25)], [ev(200 + shift, 25)])
        assert d["n_both"] == 1, f"verschuiving {shift}s brak de kruiscontrole"


def test_each_pressure_event_corroborates_at_most_one_thermistor_event():
    """Anders zou een lange druk-apneu twee thermistor-events legitimeren."""
    out, d = corroborate_apnea_events([ev(100, 20), ev(104, 20)], [ev(102, 20)],
                                      corroboration_licensed=True)
    assert d["n_both"] == 1
    assert d["n_thermistor_only"] == 1
    assert len(out) == 1


@pytest.mark.parametrize("kt,kp,expected", [
    (False, False, 1),      # corroboratie
    (True,  False, 2),      # + verdachte thermistor-events
    (False, True,  2),      # + mondademhaling
    (True,  True,  3),      # naieve unie
])
def test_the_knobs_control_what_comes_in(kt, kp, expected):
    out, _ = corroborate_apnea_events(
        [ev(100), ev(300)], [ev(102), ev(700)], corroboration_licensed=True,
        keep_thermistor_only=kt, keep_pressure_only=kp)
    assert len(out) == expected


def test_corroboration_diagnostics_account_for_every_event():
    therm = [ev(100), ev(300), ev(500)]
    press = [ev(101), ev(900)]
    _out, d = corroborate_apnea_events(therm, press)
    assert d["n_both"] + d["n_thermistor_only"] == d["n_thermistor"]
    assert d["n_both"] + d["n_pressure_only"] == d["n_pressure"]


def test_inputs_are_not_mutated_by_corroboration():
    t = [ev(100)]
    before = dict(t[0])
    corroborate_apnea_events(t, [ev(101)])
    assert t[0] == before


def test_an_unreliable_second_sensor_cannot_remove_apneas():
    """De klinische regel: overdetectie tegenhouden mag alleen bij een
    betrouwbaar signaal. Bij twijfel wint goedkeuren."""
    press = [ev(100), ev(300), ev(500)]        # wat de neusdruk ziet
    therm = []                                  # thermistor levert niets
    out, d = corroborate_apnea_events(therm, press)   # geen licentie
    assert len(out) == 3, "een stille tweede sensor mag niets wegnemen"
    assert d["n_pressure_only"] == 3
