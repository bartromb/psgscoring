"""De CAISR-lezer, tegen synthetische CSV's in het gepubliceerde formaat.

Deze toetsen raken geen scoringspad. Ze pinnen de vertaling van een
labelvector van 1 Hz naar het eventschema, en vooral de gevallen waarin een
stille aanname de vergelijking ongeldig zou maken.
"""

from __future__ import annotations

import pytest

from psgscoring.compare import (
    CAISR_RESP_CODES, ahi_from_events, labels_to_events,
    read_caisr_resp_csv, verify_code_mapping,
)


def _write_csv(tmp_path, labels, fs=200):
    p = tmp_path / "rec.csv"
    lines = ["start_idx,end_idx,resp"]
    for i, v in enumerate(labels):
        lines.append(f"{i * fs},{(i + 1) * fs},{v}")
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# Lezen
# ---------------------------------------------------------------------------

def test_reads_labels_and_derives_one_second_rows(tmp_path):
    p = _write_csv(tmp_path, [0, 0, 1, 1, 1, 0])
    labels, spr = read_caisr_resp_csv(p)
    assert labels == [0, 0, 1, 1, 1, 0]
    assert spr == 1.0


def test_rejects_a_file_whose_rows_do_not_tile_the_timeline(tmp_path):
    """Overlappende of gapende rijen maken elke afgeleide onset verdacht.

    Liever falen dan een eventlijst produceren die er goed uitziet en
    systematisch verschoven is.
    """
    p = tmp_path / "bad.csv"
    p.write_text("start_idx,end_idx,resp\n0,300,1\n200,500,1\n")
    with pytest.raises(ValueError, match="tijdbasis"):
        read_caisr_resp_csv(p)


def test_rejects_a_foreign_csv(tmp_path):
    p = tmp_path / "other.csv"
    p.write_text("onset,duration,label\n0,10,apnea\n1,10,apnea\n")
    with pytest.raises(ValueError, match="CAISR-resp-CSV"):
        read_caisr_resp_csv(p)


# ---------------------------------------------------------------------------
# Run-length encoding
# ---------------------------------------------------------------------------

def test_consecutive_rows_become_one_event():
    ev = labels_to_events([0, 1, 1, 1, 0, 0], 1.0)
    assert len(ev) == 1
    assert ev[0]["type"] == "obstructive"
    assert ev[0]["onset_s"] == 1.0
    assert ev[0]["duration_s"] == 3.0


def test_zero_is_never_an_event():
    assert labels_to_events([0, 0, 0], 1.0) == []


def test_touching_events_of_different_code_stay_separate():
    """Codes 4 en 6 zijn allebei 'hypopnea' maar twee verschillende
    beslissingen (3%- en 4%-tak). Samenvoegen zou een telling wegpoetsen.
    """
    ev = labels_to_events([4, 4, 6, 6], 1.0)
    assert [e["caisr_code"] for e in ev] == [4, 6]
    assert all(e["type"] == "hypopnea" for e in ev)


def test_every_class_maps_to_a_type_family():
    """De typenamen moeten door validate_psgipa.type_family() heen komen,
    anders matcht type-bewuste matching ze met niets en zakt de F1 stil.
    """
    from validate_psgipa import type_family
    for code, name in CAISR_RESP_CODES.items():
        if name is None or name == "rera":
            continue
        assert type_family(name) in ("apnea", "hypopnea"), (code, name)


def test_confidence_is_present_but_empty():
    """CAISR levert geen per-event vertrouwen. Dat moet zichtbaar zijn als
    afwezig, niet als een verzonnen default.
    """
    ev = labels_to_events([1, 1], 1.0)
    assert "confidence" in ev[0]
    assert ev[0]["confidence"] is None


def test_min_duration_filter():
    ev = labels_to_events([1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 1.0,
                          min_duration_s=10.0)
    assert [e["type"] for e in ev] == ["hypopnea"]


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def test_reras_do_not_count_towards_the_ahi():
    ev = labels_to_events([1] * 12 + [0] + [5] * 12, 1.0)
    assert len(ev) == 2
    assert ahi_from_events(ev, sleep_hours=1.0) == 1.0


def test_ahi_is_undefined_without_a_denominator():
    ev = labels_to_events([1] * 12, 1.0)
    assert ahi_from_events(ev, sleep_hours=0) is None


# ---------------------------------------------------------------------------
# Mapping-verificatie
# ---------------------------------------------------------------------------

def test_unknown_codes_are_reported_not_swallowed():
    """Een nieuwe CAISR-versie met een zesde klasse mag niet stil als
    'geen event' doorgaan.
    """
    out = verify_code_mapping([0, 1, 4, 9, 9])
    assert out["unknown_codes"] == [9]
    assert out["row_counts_by_code"][9] == 2
