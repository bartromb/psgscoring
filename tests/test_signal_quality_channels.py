"""Per-kanaal kwaliteit: de drempels maten het bestand, niet de sensor.

Drie defecten in `signal_quality_channels.py`, allemaal van het soort dat
`tests/test_gate_metrics_are_file_invariant.py` voor de RIP-poort al twee
keer heeft gevangen:

1. `_flat_threshold_for_role` was een ABSOLUTE standaarddeviatie in ruwe
   eenheden. Een RIP-kanaal dat zijn eenheid als µV declareert (MNE rekent
   naar V, 1e-5x zo klein) was daardoor "100 % vlak"; de golden-case
   `mv_scale_effort`, gebouwd om eenhedenfouten te vangen, droeg precies dit
   als gezegend resultaat.
2. `_count_clipped` telde elk herhaald maximum als ADC-rail. Een SpO2 die
   minutenlang 96 % aanwijst, een positiekanaal, een gekwantiseerde flow op
   512 Hz: allemaal "clipping". SpO2 en positie stonden daardoor in ELKE
   golden-case op "poor".
3. De uitkomst was gesleuteld op kanaalNAAM. Twee rollen op één fysiek kanaal
   (`flow` en `flow_thermistor` op dezelfde thermistor) overschreven elkaar.

De uitvoer is informatief (een logregel en `output["channel_quality"]`), dus
geen profielveld. Er bestonden geen directe unit-tests voor deze module.
"""
import numpy as np
import pytest

from psgscoring.signal_quality_channels import (
    _assess_channel,
    _count_clipped,
    assess_signal_quality,
)

SF = 32.0
DUUR_S = 600.0
N = int(DUUR_S * SF)
T = np.arange(N) / SF


def _adem(amp=1.0, sf=SF, seed=0):
    t = np.arange(int(DUUR_S * sf)) / sf
    rng = np.random.default_rng(seed)
    return amp * (np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.01, t.size))


# ── 1. vlakheid is schaalvrij ────────────────────────────────────────

@pytest.mark.parametrize("schaal", [1.0, 1e-5, 1e3])
def test_de_vlakheid_hangt_niet_van_de_eenheid_af(schaal):
    r = _assess_channel(_adem() * schaal, SF, "THORAX", "thorax")
    assert r["flat_pct"] == 0.0, (schaal, r)
    assert r["quality_grade"] == "good", (schaal, r)


def test_een_echte_uitval_telt_nog_steeds():
    x = _adem()
    x[int(100 * SF):int(300 * SF)] = 0.0            # 200 van 600 s dood
    r = _assess_channel(x, SF, "FLOW", "flow")
    assert r["flat_pct"] == pytest.approx(33.3, abs=1.0), r
    assert r["quality_grade"] == "poor"
    assert r["n_disconnects"] == 1


def test_een_volledig_dood_kanaal_is_vlak_en_niet_geclipt():
    r = _assess_channel(np.zeros(N), SF, "FLOW", "flow")
    assert r["flat_pct"] == 100.0
    assert r["clip_pct"] == 0.0, "constant is vlak, niet op de rail"
    assert r["quality_grade"] == "poor"


# ── 2. clipping op discrete en gekwantiseerde kanalen ───────────────

def test_spo2_op_een_constante_96_is_niet_poor():
    r = _assess_channel(np.full(N, 96.0), SF, "SPO2", "spo2")
    assert r["clip_pct"] == 0.0, r
    assert r["quality_grade"] == "good", r


def test_spo2_met_echte_uitval_wordt_wel_gezien():
    x = np.full(N, 96.0)
    x[int(100 * SF):int(300 * SF)] = 0.0            # sensor los: 0 %
    r = _assess_channel(x, SF, "SPO2", "spo2")
    assert r["flat_pct"] == pytest.approx(33.3, abs=1.0), r
    assert r["quality_grade"] == "poor"


def test_een_positiekanaal_is_niet_poor():
    pos = np.repeat([0.0, 1.0, 2.0, 1.0, 3.0, 0.0], N // 6)[:N]
    r = _assess_channel(pos, SF, "Position", "position")
    assert r["clip_pct"] == 0.0 and r["flat_pct"] == 0.0, r
    assert r["quality_grade"] == "good"


def test_een_gekwantiseerd_kanaal_op_hoge_sf_clipt_niet():
    """Een sinus die in stappen van 5 % is opgeslagen (STAPPEN in
    test_gate_metrics_are_file_invariant) blijft bij 512 Hz honderden
    monsters per ademtop op zijn hoogste niveau hangen. Dat is kwantisatie,
    geen rail: het niveau eronder duurt even lang."""
    sf = 512.0
    t = np.arange(int(DUUR_S * sf)) / sf
    x = np.round(np.sin(2 * np.pi * 0.25 * t) / 0.05) * 0.05
    assert _count_clipped(x) == 0
    r = _assess_channel(x, sf, "FLOW", "flow")
    assert r["clip_pct"] == 0.0, r
    assert r["quality_grade"] == "good", r


def test_echte_clipping_wordt_gezien():
    x = np.clip(_adem(), -0.7, 0.7)                  # ~25 % op de rails
    r = _assess_channel(x, SF, "FLOW", "flow")
    assert r["clip_pct"] > 10.0, r
    assert r["quality_grade"] == "poor"


def test_een_eenheidsschaling_verandert_de_clipping_niet():
    x = np.clip(_adem(), -0.7, 0.7)
    a = _assess_channel(x, SF, "FLOW", "flow")["clip_pct"]
    b = _assess_channel(x * 1e-5, SF, "FLOW", "flow")["clip_pct"]
    assert a == pytest.approx(b, abs=0.2), (a, b)


# ── 3. per rol gesleuteld ────────────────────────────────────────────

def _raw(mne, **kanalen):
    namen = list(kanalen)
    info = mne.create_info(namen, SF, ["misc"] * len(namen))
    return mne.io.RawArray(np.vstack([kanalen[n] for n in namen]), info,
                           verbose=False)


def test_twee_rollen_op_een_kanaal_overschrijven_elkaar_niet():
    mne = pytest.importorskip("mne")
    raw = _raw(mne, FLOW=_adem(), THORAX=_adem(seed=1))
    out = assess_signal_quality(
        raw, {"flow": "FLOW", "flow_thermistor": "FLOW", "thorax": "THORAX"})
    assert set(out["channels"]) == {"flow", "flow_thermistor", "thorax"}, \
        out["channels"].keys()
    assert out["channels"]["flow"]["channel"] == "FLOW"
    assert out["channels"]["flow_thermistor"]["channel"] == "FLOW"
    assert out["overall_grade"] == "good"


def test_een_spo2_en_een_positiekanaal_trekken_de_opname_niet_omlaag():
    """Dit is wat elke golden-case tot nu toe droeg: 'acceptable' door een
    SpO2 die gewoon 96 % aanwees."""
    mne = pytest.importorskip("mne")
    raw = _raw(mne, FLOW=_adem(), SPO2=np.full(N, 96.0),
               POS=np.repeat([0.0, 1.0], N // 2)[:N])
    out = assess_signal_quality(
        raw, {"flow": "FLOW", "spo2": "SPO2", "position": "POS"})
    assert out["overall_grade"] == "good", out
