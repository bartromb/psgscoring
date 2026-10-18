"""Twee productiedefecten van 18-10-2026 (laatste tien uploads op slaapkliniek.be).

1. Het Somnomedics-houdingskanaal `Pos.` is een code 1–6 op 4 Hz. Na de
   volledige EDF-lading brengt MNE het naar 256 Hz met FFT-herbemonstering;
   de ringing op elke overgang maakt van zes niveaus één doorlopende brug,
   `_position_levels` ziet één breed plateau en valt terug op de
   duurafhankelijke percentielmethode -- op alle tien nachten.
2. Onder de additieve duale vereniging bleef een DODE thermistor (bandvermogen
   0,00, vlak onder een CPAP-masker) de primaire apneusensor: 165 van 166
   apneus `thermistor_only`, OAHI 21,5 tegen < 4 manueel.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import resample

from psgscoring import ancillary, pipeline


# ───────────────────────────── houding ─────────────────────────────

def _somnomedics_pos(sf_native=4.0, sf_out=256.0, minutes=120, seed=3):
    """Code 1–6 op 4 Hz, zoals de recorder schrijft, daarna ge-upsampled zoals
    MNE dat bij een volledige lading doet (FFT-resample, dus ringing)."""
    rng = np.random.default_rng(seed)
    n = int(sf_native * 60 * minutes)
    codes = np.empty(n)
    i = 0
    blokken = [6, 3, 4, 2, 6, 1, 3, 6, 4, 2]
    while i < n:
        c = blokken[(i // 1000) % len(blokken)]
        d = int(rng.integers(60 * 4, 40 * 60 * 4))       # 1–40 min per houding
        codes[i:i + d] = c
        i += d
    up = resample(codes, int(n * sf_out / sf_native))
    return codes, up


def test_levels_overleven_de_herbemonstering():
    codes, up = _somnomedics_pos()
    assert len(np.unique(np.round(up, 2))) > 50, "fixture moet ringing dragen"
    niveaus = ancillary._position_levels(up[~np.isnan(up)])
    assert niveaus is not None, "ringing maakte van de niveaus één plateau"
    assert [round(v) for v in niveaus] == sorted(set(int(c) for c in codes))


def test_mapping_methode_is_levels_en_niet_percentile():
    _, up = _somnomedics_pos()
    mapped, methode = ancillary._map_position_signal(up)
    assert methode == "levels"
    assert set(np.unique(mapped)) <= {0, 1, 2, 3, 4}


def test_een_echt_continu_signaal_blijft_percentile():
    rng = np.random.default_rng(1)
    hoek = np.cumsum(rng.standard_normal(60 * 60 * 4)) / 10.0     # accelerometerhoek
    _, methode = ancillary._map_position_signal(hoek)
    assert methode == "percentile"


def test_codetabel_geeft_methode_coded_en_de_tabelnamen():
    codes, up = _somnomedics_pos(minutes=60)
    sf = 256.0
    hypno = ["N2"] * int(len(up) / sf / 30)
    tabel = {1: "Upright", 2: "Supine", 3: "Left", 4: "Right", 6: "Prone"}
    res = ancillary.analyze_position(up, sf, hypno, [], position_code_map=tabel)
    assert res["success"], res.get("error")
    s = res["summary"]
    assert s["position_mapping_method"] == "coded"
    # Code 1 is in de fixture zeldzaam en heet in de tabel Upright: de
    # tabel wint van de rangorde (waar de laagste waarde "Prone" zou heten).
    pct = s["sleep_pct"]
    aandeel_code = {c: float(np.mean(np.round(up) == c)) * 100 for c in tabel}
    for c, naam in tabel.items():
        assert abs(pct[naam] - aandeel_code[c]) < 1.5, (naam, pct[naam], aandeel_code[c])


def test_codetabel_die_niet_past_valt_eerlijk_terug():
    _, up = _somnomedics_pos(minutes=30)
    hypno = ["N2"] * int(len(up) / 256.0 / 30)
    res = ancillary.analyze_position(up, 256.0, hypno, [],
                                     position_code_map={10: "Supine", 11: "Left"})
    assert res["summary"]["position_mapping_method"] == "levels"


def test_pipeline_schaalt_naar_de_eenheid_van_de_edf_voor_de_codetabel():
    """MNE levert mV-kanalen in volt; de codetabel staat in de eenheid van de
    recorder. `position_unit_scale` brengt 0,006 V terug naar code 6."""
    codes, up = _somnomedics_pos(minutes=30)
    hypno = ["N2"] * int(len(up) / 256.0 / 30)
    res = ancillary.analyze_position(up * 1e-3, 256.0, hypno, [],
                                     position_code_map={1: "Upright", 2: "Supine", 3: "Left", 4: "Right", 6: "Prone"},
                                     position_unit_scale=1e3)
    assert res["summary"]["position_mapping_method"] == "coded"


# ───────────────────── dode thermistor onder de additieve vereniging ─────────────────────

def _flow(sf=32.0, minutes=20, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sf * 60 * minutes)) / sf
    return np.sin(2 * np.pi * 0.25 * t) + 0.02 * rng.standard_normal(t.size)


def _resolve(pressure, therm, sf=32.0, additive=True):
    output = {"meta": {}}
    ch = {"flow_pressure": "Pressure Flow", "flow_thermistor": "Flow Th."}
    pipeline._resolve_flow_channels(
        pressure, sf, pressure, sf, therm, sf, ch, output,
        additive_thermistor=additive, thermistor_gate="respiratory_band")
    return output["meta"]["flow_channels"]


def test_een_vlakke_thermistor_wordt_ook_additief_afgewezen():
    pressure = _flow()
    therm = np.full_like(pressure, 1.5e-5)                 # de CPAP-nacht
    fc = _resolve(pressure, therm)
    assert fc["apnea_sensor"] == "Pressure Flow"
    assert fc["thermistor_rejected"] == "Flow Th."
    assert fc["dual_sensor"] is False
    assert "dood" in str(fc["thermistor_check"].get("reason", "")).lower() or \
           "geen ademhaling" in str(fc["thermistor_check"].get("reason", "")).lower()


def test_een_ruis_thermistor_onder_de_dode_vloer_wordt_additief_afgewezen():
    rng = np.random.default_rng(5)
    pressure = _flow()
    therm = 0.3 * rng.standard_normal(pressure.size)       # witte ruis: ademfractie ~0,10
    fc = _resolve(pressure, therm)
    assert fc["thermistor_rejected"] == "Flow Th."
    assert fc["thermistor_check"]["band_power"] < pipeline.THERMISTOR_BAND_POWER_DEAD


def test_een_levende_maar_zwakke_thermistor_blijft_additief_in_gebruik():
    """Het oude gedrag voor alles boven de dode vloer: onder 0,70 maar levend
    telt additief mee (dat is de 07-10-beslissing, niet deze reparatie)."""
    rng = np.random.default_rng(7)
    pressure = _flow()
    therm = 0.25 * pressure + 0.30 * rng.standard_normal(pressure.size)   # ademfractie ~0,3
    fc = _resolve(pressure, therm)
    bp = fc["thermistor_check"]["band_power"]
    assert pipeline.THERMISTOR_BAND_POWER_DEAD <= bp < 0.70, bp
    assert fc["thermistor_rejected"] is None
    assert fc["apnea_sensor"] == "Flow Th."


# ───────────────────────────── CPAP-druk ─────────────────────────────

def test_cpap_drukkanaal_wordt_herkend_en_gemeld():
    import mne
    sf = 32.0
    n = int(sf * 3600 * 2)
    druk = np.zeros(n); druk[n // 2:] = 8.0 + 0.3 * np.sin(np.arange(n - n // 2) / 50.0)
    info = mne.create_info(["Pressur"], sf, ["misc"])
    raw = mne.io.RawArray(druk[None, :], info, verbose="ERROR")
    uit = pipeline._detect_cpap_pressure(raw)
    assert uit["detected"] is True
    assert uit["channel"] == "Pressur"
    assert abs(uit["onset_s"] - 3600.0) < 120
    assert 0.45 < uit["therapy_fraction"] < 0.55


def test_zonder_drukkanaal_geen_cpap():
    import mne
    info = mne.create_info(["Pressure Flow", "Flow Th."], 32.0, ["misc", "misc"])
    raw = mne.io.RawArray(np.zeros((2, 32 * 600)), info, verbose="ERROR")
    assert pipeline._detect_cpap_pressure(raw)["detected"] is False
