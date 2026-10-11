"""Bemonsteringsfrequentie per kanaal, en de kanaalmap zoals hij gebruikt is.

Vier bevindingen uit dezelfde hoek:

(a) `get()` gaf voor elk kanaal `raw.info["sfreq"]` terug. MNE bemonstert
    een EDF met gemengde frequenties op naar de hoogste, dus een SpO2-kanaal
    van 0,25 Hz kwam als 256 Hz binnen en de AASM-toets "SpO2 < 0,33 Hz"
    kon nooit vuren. De NATIEVE frequentie staat in de EDF-kop en MNE bewaart
    hem in `_raw_extras` (`n_samps` / `record_length`).
(b) de CSR-detector kreeg de envelope van het REFERENTIEkanaal met de
    frequentie van het APNEUkanaal.
(c) een hard gecodeerde 256 als terugval voor `sfreq`.
(d) `meta["channels_used"]` werd vastgelegd vóór `_resolve_flow_channels`
    de thermistor afkeurde, en toonde hem dus als gebruikt.
"""
import pathlib

import numpy as np
import pytest

mne = pytest.importorskip("mne")

SF = 32.0
DUR_S = 600
N = int(DUR_S * SF)
HYPNO = ["N2"] * (DUR_S // 30)
SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "psgscoring" / "pipeline.py")


# ── EDF met gemengde frequenties, met de hand geschreven ─────────────────

def _write_edf(path, signals, record_dur_s, n_records):
    """signals: (label, fs, data, phys_min, phys_max, eenheid). Geen edfio of
    pyedflib in de omgeving; de EDF-kop is 256 + 256*ns bytes ASCII en de
    records zijn int16 little-endian, dus dit is genoeg."""
    ns = len(signals)

    def fld(s, n):
        return str(s)[:n].ljust(n).encode("ascii")

    hdr = b"".join([
        fld("0", 8), fld("X X X X", 80), fld("Startdate X X X X", 80),
        fld("01.01.01", 8), fld("22.00.00", 8), fld(256 * (ns + 1), 8),
        fld("", 44), fld(n_records, 8), fld(f"{record_dur_s:g}", 8), fld(ns, 4),
    ])
    hdr += b"".join(fld(lab, 16) for lab, *_ in signals)
    hdr += b"".join(fld("", 80) for _ in signals)
    hdr += b"".join(fld(unit, 8) for *_, unit in signals)
    hdr += b"".join(fld(f"{pmin:g}", 8) for _, _, _, pmin, _, _ in signals)
    hdr += b"".join(fld(f"{pmax:g}", 8) for _, _, _, _, pmax, _ in signals)
    hdr += b"".join(fld(-32768, 8) for _ in signals)
    hdr += b"".join(fld(32767, 8) for _ in signals)
    hdr += b"".join(fld("", 80) for _ in signals)
    hdr += b"".join(fld(int(round(fs * record_dur_s)), 8) for _, fs, *_ in signals)
    hdr += b"".join(fld("", 32) for _ in signals)
    assert len(hdr) == 256 * (ns + 1)
    with open(path, "wb") as f:
        f.write(hdr)
        for r in range(n_records):
            for _lab, fs, data, pmin, pmax, _unit in signals:
                n = int(round(fs * record_dur_s))
                seg = np.asarray(data[r * n:(r + 1) * n], dtype=float)
                if len(seg) < n:
                    seg = np.pad(seg, (0, n - len(seg)))
                dig = np.round((seg - pmin) / (pmax - pmin) * 65535 - 32768)
                f.write(np.clip(dig, -32768, 32767).astype("<i2").tobytes())


@pytest.fixture(scope="module")
def mixed_rate_edf(tmp_path_factory):
    path = tmp_path_factory.mktemp("edf") / "mixed.edf"
    t = np.arange(N) / SF
    flow = np.sin(2 * np.pi * 0.25 * t)
    n_spo2 = int(DUR_S * 0.25)
    _write_edf(path, [
        ("Flow", SF, flow, -2.0, 2.0, ""),
        ("SpO2", 0.25, np.full(n_spo2, 96.0), 0.0, 100.0, "%"),
        ("Thorax", SF, 0.6 * flow, -2.0, 2.0, ""),
        ("Abdomen", SF, 0.6 * flow, -2.0, 2.0, ""),
    ], record_dur_s=4.0, n_records=DUR_S // 4)
    return path


@pytest.fixture(scope="module")
def edf_run(mixed_rate_edf):
    import psgscoring
    raw = mne.io.read_raw_edf(mixed_rate_edf, preload=True, verbose="ERROR")
    assert raw.info["sfreq"] == SF                  # MNE bemonstert op naar 32 Hz
    out = psgscoring.run_pneumo_analysis(
        raw, hypno=HYPNO, scoring_profile="aasm_v3_rec",
        channel_map={"flow_pressure": "Flow", "spo2": "SpO2",
                     "thorax": "Thorax", "abdomen": "Abdomen"})
    return raw, out


def test_de_natieve_frequenties_staan_in_het_herkomstblok(edf_run):
    _raw, out = edf_run
    assert out["meta"]["channel_rates"] == {
        "Flow": 32.0, "SpO2": 0.25, "Thorax": 32.0, "Abdomen": 32.0}
    assert out["meta"]["channel_rates_source"] == "edf_header"


def test_de_spo2_frequentietoets_kan_nu_vuren(edf_run):
    """0,25 Hz = middeling over 4 s; AASM staat maximaal 3 s toe."""
    _raw, out = edf_run
    assert out["spo2"]["spo2_native_samplerate"] == 0.25
    assert out["spo2"]["spo2_low_samplerate"] is True
    # De frequentie van de ARRAY die geanalyseerd is blijft wat hij was.
    assert out["spo2"]["spo2_samplerate"] == SF


def test_de_helper_leest_de_edf_kop(mixed_rate_edf):
    from psgscoring.pipeline import _native_channel_rates
    raw = mne.io.read_raw_edf(mixed_rate_edf, preload=False, verbose="ERROR")
    rates, bron = _native_channel_rates(raw)
    assert rates == {"Flow": 32.0, "SpO2": 0.25, "Thorax": 32.0, "Abdomen": 32.0}
    assert bron == "edf_header"


def test_zonder_edf_kop_valt_de_helper_terug_op_info_sfreq():
    """Een RawArray draagt geen natieve frequenties; dan is info["sfreq"] het
    eerlijke antwoord, en dat staat erbij."""
    import psgscoring
    from psgscoring.pipeline import _native_channel_rates
    t = np.arange(N) / SF
    info = mne.create_info(["Resp nasal", "SaO2"], SF, ["misc", "misc"])
    raw = mne.io.RawArray(np.vstack([np.sin(2 * np.pi * 0.25 * t),
                                     np.full(N, 96.0)]), info, verbose="ERROR")
    rates, bron = _native_channel_rates(raw)
    assert rates == {"Resp nasal": SF, "SaO2": SF}
    assert bron == "info_sfreq"
    out = psgscoring.run_pneumo_analysis(raw, hypno=HYPNO,
                                         scoring_profile="aasm_v3_rec")
    assert out["meta"]["channel_rates_source"] == "info_sfreq"
    assert "spo2_low_samplerate" not in out["spo2"]
    assert out["spo2"]["spo2_native_samplerate"] == SF


# ── (b) de CSR-detector krijgt de frequentie van het referentiekanaal ───

def test_de_csr_detector_krijgt_sf_ref(monkeypatch):
    """`aasm_v3_dual` leest het hypopnee-kanaal als referentie. Halveer dat
    kanaal (consistent: data én frequentie) en de CSR-detector moet de
    gehalveerde frequentie zien, niet die van het apneukanaal."""
    import psgscoring
    from psgscoring import pipeline as pl

    echt_resolve = pl._resolve_flow_channels

    def _halve(*a, **kw):
        ap, hp, sa, sh = echt_resolve(*a, **kw)
        return ap, hp[::2], sa, sh / 2.0

    gezien = {}
    echt_csr = pl.detect_cheyne_stokes

    def _spy(env, sf, hypno, *a, **kw):
        gezien["sf"] = sf
        gezien["n"] = len(env)
        return echt_csr(env, sf, hypno, *a, **kw)

    monkeypatch.setattr(pl, "_resolve_flow_channels", _halve)
    monkeypatch.setattr(pl, "detect_cheyne_stokes", _spy)

    t = np.arange(N) / SF
    info = mne.create_info(["Resp nasal", "SaO2"], SF, ["misc", "misc"])
    raw = mne.io.RawArray(np.vstack([np.sin(2 * np.pi * 0.25 * t),
                                     np.full(N, 96.0)]), info, verbose="ERROR")
    out = psgscoring.run_pneumo_analysis(raw, hypno=HYPNO,
                                         scoring_profile="aasm_v3_dual")
    assert out["meta"]["flow_channels"]["reference_sensor"] == "Resp nasal"
    assert gezien, "de CSR-detector is niet aangeroepen"
    assert gezien["n"] == N // 2
    assert gezien["sf"] == SF / 2.0, gezien


# ── (c) geen hard gecodeerde 256 ────────────────────────────────────────

def test_geen_hard_gecodeerde_256_als_sfreq_terugval():
    src = SRC.read_text()
    assert 'raw.info.get("sfreq", 256)' not in src
    assert "else 256" not in src


# ── (d) channels_used toont de map zoals hij gebruikt is ────────────────

def _raw_with_dead_thermistor():
    rng = np.random.default_rng(9)
    t = np.arange(N) / SF
    pres = np.sin(2 * np.pi * 0.25 * t) + 0.05 * rng.normal(size=N)
    therm = rng.normal(size=N)                       # ruis, geen ademhaling
    spo2 = np.full(N, 96.0)
    info = mne.create_info(["Pressure Flow", "Flow Th.", "SPO2"], SF,
                           ["misc"] * 3)
    raw = mne.io.RawArray(np.vstack([pres, therm, spo2]), info, verbose="ERROR")
    cmap = {"flow_pressure": "Pressure Flow", "flow_thermistor": "Flow Th.",
            "spo2": "SPO2"}
    return raw, cmap


def test_een_afgekeurde_thermistor_staat_niet_als_gebruikt_in_meta():
    import psgscoring
    raw, cmap = _raw_with_dead_thermistor()
    out = psgscoring.run_pneumo_analysis(raw, hypno=HYPNO, channel_map=cmap,
                                         scoring_profile="aasm_v3_rec")
    fc = out["meta"]["flow_channels"]
    assert fc["thermistor_rejected"] == "Flow Th.", (
        "fixture: de poort heeft de thermistor niet afgekeurd")
    used = out["meta"]["channels_used"]
    assert "flow_thermistor" not in used, used
    assert used["flow_thermistor_rejected"] == "Flow Th."
    assert used["flow_pressure"] == "Pressure Flow"
