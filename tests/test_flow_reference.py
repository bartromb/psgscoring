"""Welk flowkanaal lezen de analyses die géén apneus detecteren?

Vijf afnemers naast de apneudetectie nemen een flowsignaal: de
AHI-intervalsweep, de anker-basislijn, de arousal/RERA-koppeling, de
Cheyne-Stokes-detectie en de ventilatory burden. Ze lazen allemaal het
apneukanaal.

Onder een vervangend profiel is dat ongevaarlijk — het apneukanaal *is* de flow
van de opname. Onder `aasm_v3_dual` niet: dat profiel houdt de thermistor
bewust als nominaal apneukanaal aan, óók wanneer die de kwaliteitstoets niet
haalt, omdat de tweede detectiepas een slecht kanaal onschadelijk maakt vóór de
apneutelling. Die vijf kennen geen tweede pas en erfden zo precies het
vervangende falen dat de additieve opzet vermijdt.

Op een echte opname las de ventilatory burden 20,4% in plaats van 42,6% — onder
de ≤25%-referentie — bij een patiënt die 94,7% van de nacht onder 90%
saturatie lag.
"""
import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES, get_profile

mne = pytest.importorskip("mne")

SF = 32.0
DUR_S = 600
N = int(DUR_S * SF)
T = np.arange(N) / SF


# ─────────────────────────────────────────────────────────────────
#  Het profielveld
# ─────────────────────────────────────────────────────────────────

def test_the_default_is_the_apnea_channel_so_nothing_moves():
    """Byte-identiek gedrag voor elk bestaand profiel is een harde eis."""
    for name in PROFILES:
        # De duale profielen lezen bewust de neusdruk: onder een additieve
        # thermistor beschermt de tweede detectiepas alleen de apneutelling,
        # niet de vijf afgeleide analyses. aasm_v3_fusion is een duale variant
        # en erft die redenering ongewijzigd.
        if name in ("aasm_v3_dual", "aasm_v3_pressure", "aasm_v3_fusion",
                    "aasm_v3_breath_dual", "aasm_v3_prob_dual"):
            continue
        assert get_profile(name).post_processing.flow_reference == "apnea", name


def test_the_dual_profile_reads_the_hypopnea_channel():
    assert get_profile("aasm_v3_dual").post_processing.flow_reference == "hypopnea"


def test_the_field_reaches_the_legacy_dict_the_pipeline_reads():
    assert SCORING_PROFILES["aasm_v3_dual"]["FLOW_REFERENCE"] == "hypopnea"
    assert SCORING_PROFILES["aasm_v3_rec"]["FLOW_REFERENCE"] == "apnea"


# ─────────────────────────────────────────────────────────────────
#  Het gedrag op een montage met een dode thermistor
# ─────────────────────────────────────────────────────────────────

def _breathing(amp=1.0, rate=0.25, seed=0):
    rng = np.random.default_rng(seed)
    mod = 1.0 + 0.4 * np.sin(2 * np.pi * 0.01 * T)
    return amp * mod * np.sin(2 * np.pi * rate * T) + 0.05 * amp * rng.normal(size=N)


def _shallow(sig, t0, t1, factor=0.3):
    """Kleine ademteugen — wat de ventilatory burden hoort te tellen."""
    out = sig.copy()
    out[int(t0 * SF):int(t1 * SF)] *= factor
    return out


#: (start, eind) van de apneus, met telkens een desaturatie erna — zonder
#: gescoorde events is de sweep leeg en toetst een vergelijking van zijn armen
#: niets.
APNEAS = [(60, 80), (180, 200), (300, 320), (450, 470)]


def _raw_with_dead_thermistor():
    """Neusdruk ademt en heeft een lange periode kleine ademteugen; de
    thermistor is ruis, zoals op de Somnomedics-montage waar de
    envelope-overeenstemming ≈ 0 was."""
    pres = _shallow(_breathing(seed=1), 120, 420)
    for t0, t1 in APNEAS:                      # flow valt weg, effort blijft
        pres = _shallow(pres, t0, t1, factor=0.02)
    therm = np.random.default_rng(9).normal(size=N)
    thorax = _breathing(amp=0.6, seed=2)
    abdomen = _breathing(amp=0.6, seed=3)
    spo2 = np.full(N, 96.0)
    for t0, t1 in APNEAS:                      # nadir ~10 s na het event
        a, b = int((t1 + 2) * SF), int((t1 + 18) * SF)
        spo2[a:min(b, N)] = 91.0

    data = np.vstack([pres, therm, thorax, abdomen, spo2])
    info = mne.create_info(["Pressure Flow", "Flow Th.", "THORAX", "ABDOMEN", "SPO2"],
                           SF, ch_types=["misc"] * 5)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    hypno = ["W"] + ["N2"] * (DUR_S // 30 - 1)
    cmap = {"flow_pressure": "Pressure Flow", "flow_thermistor": "Flow Th.",
            "thorax": "THORAX", "abdomen": "ABDOMEN", "spo2": "SPO2"}
    return raw, hypno, cmap


def _run(profile):
    import psgscoring
    raw, hypno, cmap = _raw_with_dead_thermistor()
    return psgscoring.run_pneumo_analysis(raw, hypno=hypno, channel_map=cmap,
                                          scoring_profile=profile)


@pytest.fixture(scope="module")
def dual_out():
    return _run("aasm_v3_dual")


@pytest.fixture(scope="module")
def rec_out():
    return _run("aasm_v3_rec")


def test_the_dual_profile_still_takes_the_thermistor_as_apnea_sensor(dual_out):
    """De additieve keuze zelf blijft staan — die is niet wat hier verandert."""
    fc = dual_out["meta"]["flow_channels"]
    assert fc["apnea_sensor"] == "Flow Th."
    assert fc["thermistor_check"]["usable"] is False


def test_but_the_reference_channel_is_the_nasal_pressure(dual_out):
    assert dual_out["meta"]["flow_channels"]["reference_sensor"] == "Pressure Flow"


def test_the_default_profile_keeps_reading_the_apnea_channel(rec_out):
    fc = rec_out["meta"]["flow_channels"]
    assert fc["reference_sensor"] == fc["apnea_sensor"]


def test_ventilatory_burden_is_measured_on_a_channel_that_breathes(dual_out):
    """Op een dood kanaal is er niets te meten en zakt het getal onder de norm.

    De opname heeft 5 van de 10 minuten kleine ademteugen, dus de burden hoort
    substantieel te zijn — niet de ~20% waarmee een rapport "normaal" meldt.
    """
    vb = dual_out["respiratory"]["summary"].get("ventilatory_burden")
    assert vb is not None
    assert vb > 25.0, f"VB {vb} — gemeten op de thermistor in plaats van de druk?"


def test_the_two_profiles_agree_on_the_ventilatory_burden(dual_out, rec_out):
    """Het profiel gaat over apneudetectie; de burden hoort er niet aan te hangen."""
    vb_d = dual_out["respiratory"]["summary"].get("ventilatory_burden")
    vb_r = rec_out["respiratory"]["summary"].get("ventilatory_burden")
    assert vb_d is not None and vb_r is not None
    assert vb_d == pytest.approx(vb_r, abs=1.0)


def test_the_sweep_scores_the_same_channel_under_both_profiles(dual_out, rec_out):
    """De sweep is een enkelsensor-herscoring en zijn strict-arm belandt als
    "Rule 1B" in het rapport. Op de thermistor stortte die in.

    De vergelijking is alleen zinvol als er events zijn: een lege sweep geeft
    overal 0 en zou net zo goed slagen zonder de reparatie.
    """
    iv_d = dual_out.get("ahi_interval", {})
    iv_r = rec_out.get("ahi_interval", {})
    assert "error" not in iv_d and "error" not in iv_r
    for arm in ("strict", "standard", "sensitive"):
        a = (iv_d.get(arm) or {}).get("ahi")
        b = (iv_r.get(arm) or {}).get("ahi")
        assert a is not None and b is not None, arm
        assert a > 0, f"{arm}-arm vond niets — dan toetst deze test niets"
        assert a == pytest.approx(b, abs=0.6), arm


def test_a_single_channel_montage_is_unaffected():
    """Zonder tweede kanaal valt de referentie samen met het apneukanaal."""
    import psgscoring
    pres = _shallow(_breathing(seed=4), 120, 420)
    data = np.vstack([pres, _breathing(amp=0.6, seed=5),
                      _breathing(amp=0.6, seed=6), np.full(N, 96.0)])
    info = mne.create_info(["FLOW", "THORAX", "ABDOMEN", "SPO2"], SF,
                           ch_types=["misc"] * 4)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    out = psgscoring.run_pneumo_analysis(
        raw, hypno=["W"] + ["N2"] * (DUR_S // 30 - 1),
        channel_map={"flow": "FLOW", "thorax": "THORAX",
                     "abdomen": "ABDOMEN", "spo2": "SPO2"},
        scoring_profile="aasm_v3_dual")
    fc = out["meta"]["flow_channels"]
    assert fc["reference_sensor"] == fc["apnea_sensor"] == "FLOW"
    assert out["meta"]["dual_sensor_fallback"]["performed"] is False
