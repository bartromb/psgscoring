"""Arousal-detectie: de bouwstenen, niet de hele detector.

Dekking was 57 % — 331 van 766 regels ongetest, de grootste post van het
pakket. `detect_arousals` zelf op synthetisch EEG toetsen is broos: je meet dan
vooral of je nep-EEG toevallig door de banddrempels komt. De onderdelen
eronder zijn wél scherp toetsbaar, en het zijn juist die onderdelen waar de
logica zit die stil fout kan gaan.

Wat hier getoetst wordt:

* **`_union_arousals`** — meerdere afleidingen samenvoegen. Dit is de kern van
  de multi-derivatie-aanpak die sinds v0.9.0 de klinische standaard is. Twee
  kanalen die dezelfde arousal zien mogen niet twee arousals opleveren; een
  kanaal dat er één ziet die de rest mist, moet die wél bijdragen — dat is de
  sensitiviteitswinst waarvoor de aanpak bestaat.
* **`_eog_reject_occipital`** — een occipitaal-only "arousal" die samenvalt met
  een grote oogbeweging is EOG-doorslag, geen arousal. Een menselijke scoorder
  kruist de EOG en herkent dat; deze functie doet hetzelfde. Cross-kanaal
  bevestigde events mag hij niet aanraken.
* **`_recompute_arousal_summary`** — de indices, inclusief de noemer die vandaag
  is aangepast: arousal.py was een van de twaalf plaatsen met de
  `max(uren, 0.001)`-ondergrens die de index het aantal maal duizend maakte.
* **`_classify_arousal_index`** en de koppelingsinterpretatie.
"""

import numpy as np
import pytest

from psgscoring.arousal import (
    _classify_arousal_index,
    _eog_reject_occipital,
    _interpret_arousal_coupling,
    _is_nrem,
    _is_occipital,
    _is_rem,
    _is_sleep,
    _recompute_arousal_summary,
    _union_arousals,
)

SF = 100.0


def _ar(onset, dur, derivation=None, stage="N2", band="alpha"):
    e = {"onset_s": float(onset), "duration_s": float(dur),
         "end_s": float(onset + dur), "stage": stage, "dominant_band": band}
    if derivation:
        e["derivation"] = derivation
    return e


# ─────────────────────────────────────────────────────────────
#  Afleidingen samenvoegen
# ─────────────────────────────────────────────────────────────

def test_two_channels_seeing_the_same_arousal_give_one_event():
    """Anders telt elke extra afleiding de arousal-index omhoog."""
    uit = _union_arousals([[_ar(100, 5, "C4-M1")], [_ar(101, 5, "O2-M1")]])
    assert len(uit) == 1
    assert sorted(uit[0]["derivations"]) == ["C4-M1", "O2-M1"]


def test_the_merged_event_spans_both_contributions():
    """Vroegste begin, laatste einde — anders verlies je de randen."""
    uit = _union_arousals([[_ar(100, 4, "C4-M1")], [_ar(102, 6, "O2-M1")]])
    assert uit[0]["onset_s"] == 100
    assert uit[0]["end_s"] == 108
    assert uit[0]["duration_s"] == 8


def test_an_arousal_only_one_channel_saw_is_kept():
    """Dit IS de sensitiviteitswinst van multi-derivatie. Zou hij wegvallen,
    dan is de hele aanpak zinloos."""
    uit = _union_arousals([[_ar(100, 5, "C4-M1")], [_ar(500, 5, "O2-M1")]])
    assert len(uit) == 2
    assert [e["onset_s"] for e in uit] == [100, 500]


def test_events_that_only_touch_are_not_merged():
    """Aansluitend is niet overlappend: 100-105 en 105-110 zijn twee arousals."""
    uit = _union_arousals([[_ar(100, 5, "C4-M1")], [_ar(105, 5, "O2-M1")]])
    assert len(uit) == 2


def test_the_longest_contributor_supplies_band_and_stage():
    """Bij een fusie moet één van de twee de beschrijving leveren; de langste
    heeft het meeste signaal gezien."""
    kort = _ar(100, 3, "C4-M1", stage="N2", band="alpha")
    lang = _ar(100, 9, "O2-M1", stage="R", band="theta")
    uit = _union_arousals([[kort], [lang]])
    assert uit[0]["dominant_band"] == "theta"
    assert uit[0]["stage"] == "R"


def test_the_result_is_sorted_by_onset():
    uit = _union_arousals([[_ar(500, 5, "A")], [_ar(100, 5, "B")], [_ar(300, 5, "C")]])
    assert [e["onset_s"] for e in uit] == [100, 300, 500]


def test_three_channels_seeing_one_arousal_still_give_one():
    uit = _union_arousals([[_ar(100, 5, "C4-M1")],
                           [_ar(101, 5, "O2-M1")],
                           [_ar(102, 5, "F4-M1")]])
    assert len(uit) == 1
    assert len(uit[0]["derivations"]) == 3


def test_no_input_gives_no_output():
    assert _union_arousals([]) == []
    assert _union_arousals([[], []]) == []


# ─────────────────────────────────────────────────────────────
#  EOG-doorslag op de occipitale elektroden
# ─────────────────────────────────────────────────────────────

def _eog(n_s=600, sf=SF, piek_op=None, rust=1.0, piek=20.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, rust, int(n_s * sf))
    if piek_op is not None:
        a, b = int(piek_op * sf), int((piek_op + 5) * sf)
        x[a:b] += rng.normal(0, piek, b - a)
    return x


def test_an_occipital_only_arousal_on_a_big_eye_movement_is_rejected():
    """Precies wat een scoorder doet: de EOG kruisen en het als oogbeweging
    herkennen."""
    ev = [_ar(100, 5)]
    ev[0]["derivations"] = ["O2-M1"]
    kept, dropped = _eog_reject_occipital(ev, _eog(piek_op=100), SF)
    assert dropped == 1 and kept == []


def test_a_cross_channel_arousal_on_the_same_eye_movement_is_kept():
    """Zag een tweede, niet-occipitaal kanaal hem ook, dan is het geen
    doorslag. Deze functie mag daar niet aankomen."""
    ev = [_ar(100, 5)]
    ev[0]["derivations"] = ["O2-M1", "C4-M1"]
    kept, dropped = _eog_reject_occipital(ev, _eog(piek_op=100), SF)
    assert dropped == 0 and len(kept) == 1


def test_an_occipital_arousal_without_an_eye_movement_is_kept():
    ev = [_ar(100, 5)]
    ev[0]["derivations"] = ["O2-M1"]
    kept, dropped = _eog_reject_occipital(ev, _eog(piek_op=None), SF)
    assert dropped == 0 and len(kept) == 1


def test_without_an_eog_channel_nothing_is_rejected():
    """Geen EOG betekent dat de toets niet uitgevoerd kan worden — dan is
    niets afwijzen het veilige antwoord."""
    ev = [_ar(100, 5)]
    ev[0]["derivations"] = ["O2-M1"]
    for leeg in (None, np.array([])):
        kept, dropped = _eog_reject_occipital(list(ev), leeg, SF)
        assert dropped == 0 and len(kept) == 1


@pytest.mark.parametrize("naam,occipitaal", [
    ("O2-M1", True), ("O1-A2", True), ("C4-M1", False),
    ("F4-M1", False), ("Cz", False),
])
def test_which_derivations_count_as_occipital(naam, occipitaal):
    assert _is_occipital(naam) is occipitaal


# ─────────────────────────────────────────────────────────────
#  Stadia
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("stage,slaap,rem,nrem", [
    ("W",  False, False, False),
    ("N1", True,  False, True),
    ("N2", True,  False, True),
    ("N3", True,  False, True),
    ("R",  True,  True,  False),
])
def test_the_stage_predicates_agree_with_each_other(stage, slaap, rem, nrem):
    """REM en NREM zijn samen precies slaap; wake hoort in geen van drieën."""
    assert _is_sleep(stage) is slaap
    assert _is_rem(stage) is rem
    assert _is_nrem(stage) is nrem
    assert (rem or nrem) is slaap


# ─────────────────────────────────────────────────────────────
#  De indices en hun noemer
# ─────────────────────────────────────────────────────────────

def test_the_arousal_index_divides_by_sleep_hours():
    """120 epochs N2 = 1 uur; 20 arousals -> 20 per uur."""
    hypno = ["N2"] * 120
    s = _recompute_arousal_summary([_ar(i * 100, 5) for i in range(20)], hypno, set())
    assert s["n_arousals"] == 20
    assert s["arousal_index"] == pytest.approx(20.0, rel=0.02)


def test_without_sleep_there_is_no_index_rather_than_a_huge_one():
    """arousal.py was een van de twaalf plaatsen met `max(uren, 0.001)`, die
    de index het aantal maal duizend maakte."""
    s = _recompute_arousal_summary([_ar(i * 100, 5) for i in range(20)],
                                   ["W"] * 120, set())
    assert s["arousal_index"] is None, f"kreeg {s['arousal_index']!r}"
    assert s["n_arousals"] == 20, "de telling blijft wel geldig"


def test_no_rem_means_no_rem_index_not_zero():
    """Geen REM is iets anders dan geen arousals in REM."""
    s = _recompute_arousal_summary([_ar(i * 100, 5, stage="N2") for i in range(10)],
                                   ["N2"] * 120, set())
    assert s["rem_arousal_index"] is None
    assert s["nrem_arousal_index"] is not None


def test_artefact_epochs_shrink_the_denominator():
    """Uitgesloten epochs tellen niet mee in de noemer, anders verdunt ruis
    de index."""
    hypno = ["N2"] * 120
    vol = _recompute_arousal_summary([_ar(i * 100, 5) for i in range(20)], hypno, set())
    half = _recompute_arousal_summary([_ar(i * 100, 5) for i in range(20)],
                                      hypno, set(range(60)))
    assert half["arousal_index"] == pytest.approx(2 * vol["arousal_index"], rel=0.02)


@pytest.mark.parametrize("ai,verwacht", [
    (None, "unknown"),   # niet te berekenen is geen schone uitslag
    (0.0, "normal"),     # nul gemeten arousals IS een schone uitslag
    (5.0, "normal"),
])
def test_an_absent_index_is_unknown_not_normal(ai, verwacht):
    """Het onderscheid dat elders "AHI 0,0 naast 81 events" opleverde."""
    assert _classify_arousal_index(ai) == verwacht


def test_both_classifiers_treat_a_missing_index_the_same_way():
    """`_classify_plmi(None)` gaf "normal" terwijl arousal "unknown" gaf — een
    PLM-index die niet berekend kon worden kreeg dus een schone verklaring op
    grond van ontbrekende gegevens."""
    from psgscoring.plm import _classify_plmi
    assert _classify_plmi(None) == _classify_arousal_index(None) == "unknown"
    assert _classify_plmi(0) == _classify_arousal_index(0) == "normal"


def test_a_high_index_is_not_called_normal():
    assert _classify_arousal_index(45.0) != "normal"


# ─────────────────────────────────────────────────────────────
#  De klinische interpretatie van de koppeling
# ─────────────────────────────────────────────────────────────

def test_no_arousals_gives_one_plain_message():
    msgs = _interpret_arousal_coupling(0, 0, 0, 10, None)
    assert len(msgs) == 1
    assert msgs[0]["level"] == "info"


def test_every_message_carries_a_level_and_a_text():
    """Het rapport rendert deze lijst; een ontbrekend veld valt daar stil weg."""
    for args in [(0, 0, 0, 10, None), (50, 5, 40, 50, 3.0), (5, 50, 3, 50, 12.0)]:
        for m in _interpret_arousal_coupling(*args):
            assert m.get("level") in ("info", "warning", "danger", "success"), m
            assert isinstance(m.get("msg"), str) and m["msg"]


def test_a_zero_event_count_does_not_divide_by_zero():
    """n_ev_total = 0 komt voor op een nacht zonder respiratoire events."""
    msgs = _interpret_arousal_coupling(10, 10, 0, 0, None)
    assert isinstance(msgs, list)
