"""De DOMINO-lezer, tegen synthetische exports in het ASCII-formaat van
SOMNOmedics ("Export analysis data").

Geen scoringspad wordt geraakt. De toetsen pinnen de vertaling van de
tekstbestanden naar het event- en hypnogramschema van psgscoring, en vooral
de plekken waar een stille aanname de vergelijking ongeldig zou maken: de
tijdbasis over middernacht, het 30-s-raster van het slaapprofiel dat niet op
de opnamestart ligt, en labels die niet gekend zijn.
"""

from __future__ import annotations

import zipfile

import pytest

from psgscoring.compare import (
    DOMINO_RESP_LABELS,
    ahi_from_domino_events,
    estimate_clock_offset,
    hypnogram_kappa,
    read_domino_export,
)

CRLF = "\r\n"
DATUM = "31-12-2025"
DATUM_NA = "01-01-2026"


def _txt(signal_id, start, kind, unit="", rows=(), extra=()):
    kop = [f"Signal ID: {signal_id}", f"Start Time: {start}", f"Unit: {unit}"]
    if kind:
        kop.append(f"Signal Type: {kind}")
    kop.extend(extra)
    return CRLF.join(kop) + CRLF + CRLF + CRLF.join(rows) + CRLF


def _export_files():
    """Een opname die om 23:59:50 begint en over middernacht loopt.

    Slaapprofiel op een 30-s-raster dat 20 s vóór de start ligt (zoals
    DOMINO het schrijft), met Wake, N2, REM, artefact. Events na middernacht.
    """
    return {
        "Markers.txt": _txt("User markers", f"{DATUM} 23:59:50", None, rows=[
            "23:59:50,000; Start",
            "00:00:40,000; Lights Off",
            "00:03:30,000; End",
        ]),
        "Sleep profile.txt": _txt(
            "SchlafProfil\\profil", f"{DATUM} 23:59:30", "Discret",
            extra=["Events list: N3,N2,N1,REM,Wake,Artefact", "Rate: 30 s"],
            rows=[
                "23:59:30,000; A",
                "00:00:00,000; Wake",
                "00:00:30,000; N2",
                "00:01:00,000; N2",
                "00:01:30,000; REM",
                "00:02:00,000; N3",
                "00:02:30,000; Wake",
                "00:03:00,000; Wake",
            ]),
        "Flow Events.txt": _txt("FlowD\\flow", f"{DATUM} 23:59:50", "Impuls", "s", rows=[
            "00:00:10,000-00:00:25,500; 15;Hypopnea",               # Wake
            "00:00:40,246-00:00:55,121; 15;Obstructive Hypopnea",   # N2
            "00:01:05,000-00:01:20,000; 15;Obstructive Apnea",      # N2
            "00:01:35,000-00:01:47,000; 12;Central Apnea",          # REM
            "00:01:50,000-00:02:02,000; 12;Mixed Apnea",            # REM -> N3
            "00:02:10,000-00:02:12,000; 2;Body event",              # geen resp. event
            "00:02:20,000-00:02:35,000; 15;Vreemd Label",           # onbekend
        ]),
        "SpO2 Events.txt": _txt("Spo2\\SpO2_Analyse", f"{DATUM} 23:59:50", "Impuls", "%", rows=[
            "00:00:50,500-00:01:05,000; 3;Desaturation",
            "00:01:40,750-00:02:00,500; 7;Desaturation",
        ]),
        "PLM Events.txt": _txt("PLM\\plm", f"{DATUM} 23:59:50", "Impuls", "dB*s", rows=[
            "00:00:05,859-00:00:07,250; 17;P-Marker",
            "00:01:10,102-00:01:12,180; 31;I-Marker",
            "00:01:50,141-00:01:50,672; 7;LM Resp",
        ]),
        "Classification Arousals.txt": _txt(
            "KorrelationMA\\MAK", f"{DATUM} 23:59:50", "Impuls", "s", rows=[
                "00:00:42,242-00:00:52,992; 10;Arousal (Autonomic)",
                "00:01:08,242-00:01:12,617; 4;Arousal",
                "00:01:52,000-00:01:57,000; 5;LM Arousal",
            ]),
        "SpO2.txt": _txt("Spo2\\SpO2_Mittlere", f"{DATUM} 23:59:50", "Analog", "%", rows=[
            "23:59:50,000; A",
            "23:59:51,000; 95",
            "23:59:52,000; 94,5",
            "00:00:00,000; 93",
        ]),
        "Position.txt": _txt(
            "Koerperlage\\lage", f"{DATUM} 23:59:50", "Discret",
            extra=["Events list: Prone,Upright,Left,Right,Upright,Supine"],
            rows=["23:59:50,000; Upright", "00:00:30,000; Supine"]),
        # DOMINO schrijft µV als één byte (cp1252/latin-1), geen UTF-8.
        "REM.txt": _txt("EOG\\REM", f"{DATUM} 23:59:50", "Impuls", "µV", rows=[
            "00:00:15,727-00:00:15,969; 387;Blink",
        ]),
    }


@pytest.fixture
def export_dir(tmp_path):
    d = tmp_path / "export"
    d.mkdir()
    for name, text in _export_files().items():
        (d / name).write_bytes(text.encode("cp1252"))
    return d


@pytest.fixture
def export_zip(tmp_path, export_dir):
    z = tmp_path / "psg.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in sorted(export_dir.iterdir()):
            zf.write(p, p.name)
    return z


# ---------------------------------------------------------------------------
# Lezen
# ---------------------------------------------------------------------------

def test_directory_and_zip_read_identically(export_dir, export_zip):
    a = read_domino_export(export_dir)
    b = read_domino_export(export_zip)
    assert sorted(a.signals) == sorted(b.signals)
    assert a.respiratory_events() == b.respiratory_events()
    assert a.start_s == b.start_s


def test_header_fields_including_cp1252_units(export_dir):
    ex = read_domino_export(export_dir)
    rem = ex.signal("REM")
    assert rem.kind == "Impuls"
    assert rem.unit == "µV"
    prof = ex.signal("Sleep profile")
    assert prof.kind == "Discret"
    assert prof.rate_s == 30.0
    assert "Artefact" in prof.events_list
    assert ex.signal("SpO2").kind == "Analog"


def test_rejects_a_directory_without_domino_files(tmp_path):
    (tmp_path / "x.txt").write_text("onset,duration\n0,10\n")
    with pytest.raises(ValueError, match="DOMINO"):
        read_domino_export(tmp_path)


# ---------------------------------------------------------------------------
# Tijdbasis
# ---------------------------------------------------------------------------

def test_reference_is_the_start_marker_and_times_unwrap_past_midnight(export_dir):
    """De opname begint om 23:59:50; een event om 00:00:10 ligt 20 s later,
    niet 86380 s eerder. En het slaapprofiel, dat 20 s vóór de start begint,
    krijgt een negatieve maar kleine relatieve tijd."""
    ex = read_domino_export(export_dir)
    assert ex.markers_s() == {"Start": 0.0, "Lights Off": 50.0, "End": 220.0}
    ev = ex.respiratory_events()
    assert ev[0]["onset_s"] == 20.0
    assert ev[0]["duration_s"] == 15.5
    prof = ex.sleep_profile()
    assert prof[0] == (-20.0, "A")
    assert prof[1] == (10.0, "W")


def test_an_explicit_t0_shifts_every_relative_time(export_dir):
    """Een EDF die 10 s na de DOMINO-start begint: alles schuift 10 s op."""
    ex = read_domino_export(export_dir)
    t0 = ex.start_s + 10.0
    assert ex.respiratory_events(t0=t0)[0]["onset_s"] == 10.0
    assert ex.lights_off_s(t0=t0) == 40.0


def test_a_clock_time_can_be_given_as_t0(export_dir):
    ex = read_domino_export(export_dir)
    assert ex.respiratory_events(t0="00:00:00")[0]["onset_s"] == 10.0


# ---------------------------------------------------------------------------
# Respiratoire events
# ---------------------------------------------------------------------------

def test_labels_map_to_psgscoring_types_and_carry_the_human_stage(export_dir):
    ex = read_domino_export(export_dir)
    ev = ex.respiratory_events()
    assert [e["type"] for e in ev] == [
        "hypopnea", "hypopnea", "obstructive", "central", "mixed"]
    assert [e["label"] for e in ev][:2] == ["Hypopnea", "Obstructive Hypopnea"]
    # stadium = het DOMINO-epoch waarin het event BEGINT (zoals de Excel-kolom
    # 'Sleep stage'): 00:00:10 valt in het Wake-epoch 00:00:00–00:00:30.
    assert [e["stage"] for e in ev] == ["W", "N2", "N2", "R", "R"]
    assert all(e["source"] == "domino" and e["confidence"] is None for e in ev)


def test_body_event_is_excluded_and_unknown_labels_are_reported(export_dir):
    """'Body event' is geen respiratoir event en valt eruit; een label dat de
    lezer niet kent mag niet stil verdwijnen én niet stil als hypopneu
    doorgaan."""
    ex = read_domino_export(export_dir)
    labels = [e["label"] for e in ex.respiratory_events()]
    assert "Body event" not in labels
    assert "Vreemd Label" not in labels
    v = ex.verify_label_mapping()
    assert v["unmapped"] == {"Vreemd Label": 1}
    assert v["excluded"] == {"Body event": 1}
    assert v["flow_labels"]["Obstructive Hypopnea"] == 1
    # en met include_unmapped=True komt hij terug, zichtbaar zonder type
    alle = ex.respiratory_events(include_unmapped=True)
    assert [e for e in alle if e["label"] == "Vreemd Label"][0]["type"] is None


def test_every_mapped_type_passes_the_type_family_and_category_rules():
    """Een typenaam die validate_psgipa.type_family() niet kent matcht
    type-bewust met niets, en agreement.event_category() mag een hypopneu
    niet als apneu tellen."""
    from psgscoring.agreement import event_category
    from validate_psgipa import type_family
    for label, name in DOMINO_RESP_LABELS.items():
        if name is None or name == "rera":
            continue
        assert type_family(name) in ("apnea", "hypopnea"), (label, name)
        verwacht = "hypopnea" if "Hypopnea" in label else "apnea"
        assert event_category({"type": name}) == verwacht, (label, name)


def test_events_in_sleep_filter_uses_the_human_stage(export_dir):
    ex = read_domino_export(export_dir)
    ev = ex.respiratory_events()
    slaap = ex.events_in_sleep(ev)
    assert [e["label"] for e in slaap] == [
        "Obstructive Hypopnea", "Obstructive Apnea", "Central Apnea", "Mixed Apnea"]


# ---------------------------------------------------------------------------
# Hypnogram
# ---------------------------------------------------------------------------

def test_hypnogram_epochs_follow_the_machine_grid_not_the_domino_grid(export_dir):
    """De EDF begint om 23:59:50; het DOMINO-raster ligt op :00/:30. Een
    machine-epoch [23:59:50, 00:00:20) heeft zijn midden om 00:00:05, in het
    Wake-epoch van DOMINO. Het tweede (midden 00:00:35) valt in N2."""
    ex = read_domino_export(export_dir)
    assert ex.hypnogram_epochs(n_epochs=5) == ["W", "N2", "N2", "R", "N3"]
    # 1-s-reeks: seconde 0 (23:59:50) ligt in het artefact-epoch, seconde 10 in Wake
    s = ex.hypnogram_1s(n_seconds=12)
    assert s[0] == "A" and s[9] == "A" and s[10] == "W"
    # buiten het profiel: onbekend, geen verzonnen stadium
    assert ex.hypnogram_1s(n_seconds=400)[-1] == "?"


def test_tst_counts_sleep_epochs_only(export_dir):
    ex = read_domino_export(export_dir)
    # N2, N2, REM, N3 = 4 epochs van 30 s
    assert ex.tst_minutes() == 2.0
    assert ex.stage_counts() == {"A": 1, "W": 3, "N2": 2, "R": 1, "N3": 1}


def test_hypnogram_kappa_identical_is_one_and_unknowns_are_skipped():
    a = ["W", "N2", "N2", "R", "?"]
    b = ["W", "N2", "N2", "R", "N3"]
    k = hypnogram_kappa(a, b)
    assert k["kappa"] == 1.0 and k["accuracy"] == 1.0 and k["n"] == 4
    k2 = hypnogram_kappa(["W", "W", "N2", "N2"], ["W", "N2", "W", "N2"])
    assert k2["accuracy"] == 0.5 and k2["kappa"] == 0.0
    assert k2["confusion"]["W"]["N2"] == 1


# ---------------------------------------------------------------------------
# Overige eventlijsten en reeksen
# ---------------------------------------------------------------------------

def test_desaturations_limb_events_and_arousals(export_dir):
    ex = read_domino_export(export_dir)
    d = ex.desaturations()
    assert d[0]["drop_pct"] == 3.0 and d[0]["onset_s"] == 60.5
    lm = ex.limb_events()
    assert [e["type"] for e in lm] == ["plm", "lm", "lm_resp"]
    assert lm[0]["is_plm"] is True and lm[1]["is_plm"] is False
    ar = ex.arousals()
    assert [e["autonomic"] for e in ar] == [True, False, False]
    assert [e["kind"] for e in ar] == ["Arousal", "Arousal", "LM Arousal"]
    assert ex.eeg_arousals() == [e for e in ar if not e["autonomic"]]


def test_analog_series_decimal_comma_and_artefact(export_dir):
    ex = read_domino_export(export_dir)
    s = ex.series("SpO2")
    assert s == [(0.0, None), (1.0, 95.0), (2.0, 94.5), (10.0, 93.0)]


def test_series_1hz_fills_forward_and_leaves_gaps_none(export_dir):
    ex = read_domino_export(export_dir)
    s = ex.series_1hz("SpO2", n_seconds=12, max_gap_s=5.0)
    assert s[1] == 95.0 and s[2] == 94.5
    assert s[3] == 94.5           # vooruit gevuld binnen max_gap_s
    assert s[8] is None           # gat van 8 s > 5 s
    assert s[10] == 93.0


def test_discrete_series_keeps_states(export_dir):
    ex = read_domino_export(export_dir)
    assert ex.series("Position") == [(0.0, "Upright"), (40.0, "Supine")]


# ---------------------------------------------------------------------------
# Index en uitlijning
# ---------------------------------------------------------------------------

def test_ahi_counts_apneas_and_hypopneas_not_reras_and_needs_a_denominator():
    ev = [{"type": "obstructive"}, {"type": "hypopnea"}, {"type": "rera"},
          {"type": "hypopnea_central"}, {"type": "mixed"}]
    assert ahi_from_domino_events(ev, sleep_hours=2.0) == 2.0
    assert ahi_from_domino_events(ev, sleep_hours=0) is None


def test_estimate_clock_offset_recovers_a_known_shift():
    """De DOMINO-reeks is met een t0 berekend dat 7 s te VROEG ligt: de
    waarde op relatieve seconde i is dan wat de machine op seconde i - 7
    ziet. De schatting moet +7 s zijn: het getal dat bij t0 OPGETELD wordt."""
    import numpy as np
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=4000))
    machine = list(x)
    domino = [None] * 7 + list(x[:-7])      # dom[i] = mach[i - 7]
    best, r, n = estimate_clock_offset(domino, machine, max_lag_s=60)
    assert best == 7
    assert r > 0.99 and n > 3000
    best0, _, _ = estimate_clock_offset(machine, machine, max_lag_s=60)
    assert best0 == 0


def test_estimate_clock_offset_refuses_too_little_overlap():
    with pytest.raises(ValueError, match="overlap"):
        estimate_clock_offset([1.0] * 50, [1.0] * 50, max_lag_s=10)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def test_xlsx_export_gives_the_same_events_as_ascii(tmp_path, export_dir):
    """DOMINO's Excel-export: zelfde kopblok in rijen 1–6, kolomkop in rij 8,
    datetime-cellen, extra kolom 'Sleep stage'. Dezelfde events, dezelfde
    tijden."""
    openpyxl = pytest.importorskip("openpyxl")
    from datetime import datetime
    ascii_ex = read_domino_export(export_dir)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    def dt(s, d=DATUM):
        return datetime.strptime(f"{d} {s}", "%d-%m-%Y %H:%M:%S.%f")

    ws = wb.create_sheet("Markers")
    ws.append(["Time", "Marker"]); ws.append([None, None])
    ws.append([dt("23:59:50.000"), "Start"])
    ws.append([dt("00:00:40.000", DATUM_NA), "Lights Off"])
    ws.append([dt("00:03:30.000", DATUM_NA), "End"])

    ws = wb.create_sheet("Sleep profile")
    ws.append(["Signal ID", "SchlafProfil\\profil"]); ws.append(["Start Time", dt("23:59:30.000")])
    ws.append(["Unit", None]); ws.append(["Signal Type", "Discret"])
    ws.append(["Events list", "N3,N2,N1,REM,Wake,Artefact"]); ws.append(["Rate [s]", 30])
    ws.append([None, None]); ws.append(["Time", "Value"])
    for s, v in [("23:59:30.000", "A")]:
        ws.append([dt(s), v])
    for s, v in [("00:00:00.000", "Wake"), ("00:00:30.000", "N2"), ("00:01:00.000", "N2"),
                 ("00:01:30.000", "REM"), ("00:02:00.000", "N3"), ("00:02:30.000", "Wake"),
                 ("00:03:00.000", "Wake")]:
        ws.append([dt(s, DATUM_NA), v])

    ws = wb.create_sheet("Flow Events")
    ws.append(["Signal ID", "FlowD\\flow"]); ws.append(["Start Time", dt("23:59:30.000")])
    ws.append(["Unit", "s"]); ws.append(["Signal Type", "Impuls"])
    ws.append([None] * 5); ws.append([None] * 5); ws.append([None] * 5)
    ws.append(["Time start", "Time end", "Value", "Event name", "Sleep stage"])
    for a, b, v, lab, st in [
        ("00:00:10.000", "00:00:25.500", 15, "Hypopnea", "Wake"),
        ("00:00:40.246", "00:00:55.121", 15, "Obstructive Hypopnea", "N2"),
        ("00:01:05.000", "00:01:20.000", 15, "Obstructive Apnea", "N2"),
        ("00:01:35.000", "00:01:47.000", 12, "Central Apnea", "REM"),
        ("00:01:50.000", "00:02:02.000", 12, "Mixed Apnea", "REM"),
        ("00:02:10.000", "00:02:12.000", 2, "Body event", "N3"),
        ("00:02:20.000", "00:02:35.000", 15, "Vreemd Label", "N3"),
    ]:
        ws.append([dt(a, DATUM_NA), dt(b, DATUM_NA), v, lab, st])

    ws = wb.create_sheet("SpO2")
    ws.append(["Signal ID", "Spo2\\SpO2_Mittlere"]); ws.append(["Start Time", dt("23:59:30.000")])
    ws.append(["Unit", "%"]); ws.append(["Signal Type", "Analog"])
    ws.append([None] * 3); ws.append(["Rate [s]", 30]); ws.append([None] * 3)
    ws.append(["Time", "Value", "Sleep stage"])
    ws.append([dt("23:59:30.000"), "A", "Out!"]); ws.append([dt("00:00:00.000", DATUM_NA), 94, "Out!"])
    p = tmp_path / "Map.xlsx"
    wb.save(p)

    ex = read_domino_export(p)
    assert ex.source == "xlsx"
    assert ex.start_s == ascii_ex.start_s
    assert ex.respiratory_events() == ascii_ex.respiratory_events()
    assert ex.hypnogram_epochs(n_epochs=5) == ascii_ex.hypnogram_epochs(n_epochs=5)
    assert ex.markers_s() == ascii_ex.markers_s()
    assert ex.verify_label_mapping()["unmapped"] == {"Vreemd Label": 1}
    # de 30-s-reeks van Excel: artefact -> None, en de tijd t.o.v. de start
    assert ex.series("SpO2") == [(-20.0, None), (10.0, 94.0)]


# ---------------------------------------------------------------------------
# Defecten uit de verificatie van 04-11-2026 (faalden eerst)
# ---------------------------------------------------------------------------

def test_a_bad_row_raises_instead_of_silently_dropping_the_file(tmp_path):
    """D1: één onbegrepen regel in Flow Events.txt liet het hele bestand uit
    de export verdwijnen — het script meldde dan 'DOMINO 0 events' zonder
    waarschuwing. Een echt vreemd bestand (geen 'Signal ID') mag wél stil
    overgeslagen worden."""
    d = tmp_path / "e"
    d.mkdir()
    for name, text in _export_files().items():
        (d / name).write_bytes(text.encode("cp1252"))
    (d / "notities.txt").write_text("gewoon een tekstbestand\n")
    assert read_domino_export(d).has("Flow Events")          # vreemd bestand: genegeerd
    fe = (d / "Flow Events.txt").read_bytes().decode("cp1252")
    (d / "Flow Events.txt").write_bytes((fe + "dit is geen rij\r\n").encode("cp1252"))
    with pytest.raises(ValueError, match="onbegrepen rij"):
        read_domino_export(d)


def test_a_gap_of_more_than_twelve_hours_does_not_go_back_a_day():
    """D2: rijen werden t.o.v. de vorige gelegd met een drempel van 12 u, dus
    een ijl eventbestand (rij 20:30, volgende rij 09:00) viel 24 u te vroeg.
    Alleen een sprong TERUG in de kloktijd is middernacht."""
    from psgscoring.compare import parse_domino_txt
    sig = parse_domino_txt(_txt("X\\x", f"{DATUM} 20:00:00", "Impuls", "s", rows=[
        "20:30:00,000-20:30:10,000; 10;Obstructive Apnea",
        "09:00:00,000-09:00:10,000; 10;Obstructive Apnea",
    ]), "Flow Events")
    assert [r.t_start - 72000 for r in sig.rows] == [1800.0, 46800.0]
    # eerste rij 13 u na de Start Time: ook vooruit
    sig = parse_domino_txt(_txt("X\\x", f"{DATUM} 20:00:00", "Impuls", "s", rows=[
        "09:00:00,000-09:00:10,000; 10;Obstructive Apnea"]), "Flow Events")
    assert sig.rows[0].t_start - 72000 == 46800.0
    # en een kleine stap terug (profiel op :30 vóór de start, of 1 s jitter) blijft dezelfde dag
    sig = parse_domino_txt(_txt("X\\x", f"{DATUM} 20:00:10", "Analog", "", rows=[
        "20:00:00,000; 1", "20:00:01,000; 2", "20:00:00,500; 3"]), "Y")
    assert [round(r.t_start - 72010, 1) for r in sig.rows] == [-10.0, -9.5, -9.0]


def test_non_monotonic_source_rows_are_sorted():
    """T2: 'Phase Angle.txt' is in de bron zelf 21× niet-monotoon (1 s
    terug); series_1hz zoekt met bisect en heeft een gesorteerde lijst nodig."""
    from psgscoring.compare import parse_domino_txt
    sig = parse_domino_txt(_txt("X\\x", f"{DATUM} 20:00:00", "Analog", "", rows=[
        "20:00:00,000; 1", "20:00:02,000; 3", "20:00:01,000; 2"]), "Y")
    assert [r.value for r in sig.rows] == [1.0, 2.0, 3.0]


def test_a_header_without_blank_line_still_yields_rows():
    """D3: de koplus liep tot de eerste lege regel; zonder die regel werden
    de datarijen als kopregels gelezen en kwam er stil niets uit."""
    from psgscoring.compare import parse_domino_txt
    text = CRLF.join(["Signal ID: FlowD\\flow", f"Start Time: {DATUM} 20:00:00", "Unit: s",
                      "Signal Type: Impuls",
                      "20:00:10,000-20:00:25,000; 15;Obstructive Apnea"]) + CRLF
    sig = parse_domino_txt(text, "Flow Events")
    assert len(sig.rows) == 1 and sig.rows[0].label == "Obstructive Apnea"


def test_excel_sheet_rate_in_ms_and_time_cells():
    """D4/D5: 'Rate [ms]' werd als seconden gelezen en een blad met
    `datetime.time`-cellen gaf stil nul rijen."""
    from datetime import datetime, time
    from psgscoring.compare import parse_domino_sheet
    rows = [("Signal ID", "X\\x"), ("Start Time", datetime(2025, 12, 31, 20, 0, 0)), ("Unit", "%"),
            ("Signal Type", "Analog"), (None, None), ("Rate [ms]", 500), (None, None),
            ("Time", "Value", "Sleep stage"),
            (time(20, 0, 0), 95, "Out!"), (time(20, 0, 0, 500000), 96, "Out!")]
    sig = parse_domino_sheet("SpO2", rows)
    assert sig.rate_s == 0.5
    assert [(r.t_start - 72000, r.value) for r in sig.rows] == [(0.0, 95), (0.5, 96)]
    rows_i = [("Signal ID", "FlowD\\flow"), ("Start Time", datetime(2025, 12, 31, 20, 0, 0)),
              ("Unit", "s"), ("Signal Type", "Impuls"), (None,) * 5, (None,) * 5, (None,) * 5,
              ("Time start", "Time end", "Value", "Event name", "Sleep stage"),
              (time(20, 0, 10), time(20, 0, 25), 15, "Obstructive Apnea", "N2")]
    sig = parse_domino_sheet("Flow Events", rows_i)
    assert (sig.rows[0].t_start - 72000, sig.rows[0].t_end - 72000, sig.rows[0].label) == (10.0, 25.0, "Obstructive Apnea")


def test_series_1hz_default_holds_a_value_until_the_next_sample(export_dir):
    """T1: DOMINO schrijft een Analog-rij bij elke verandering (SpO2: 4273
    rijen op 38 000 s). Standaard geldt een waarde dus tot het volgende
    monster; alleen een artefactrij ('A') onderbreekt dat."""
    ex = read_domino_export(export_dir)
    s = ex.series_1hz("SpO2", n_seconds=12)
    assert s[0] is None            # artefactrij op t=0
    assert s[1] == 95.0 and s[3] == 94.5 and s[8] == 94.5 and s[10] == 93.0
