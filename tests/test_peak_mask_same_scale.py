"""De ademteugregel van het hypopneemasker mengt twee schalen.

Het hypopneemasker is de vereniging van een envelopemasker en een
ademteugmasker ("piekmasker"): minstens drie opeenvolgende teugen waarvan de
amplitude onder 70 % van de basislijn ligt. De teugamplitude is piek-tot-dal
van het bandgefilterde RUWE kanaal; de basislijn is het P95 van de ENVELOPE.
Gemeten op 10-11-2026 op een zuivere sinus:

* één flowkanaal: verhouding 2,0 bij rustige ademhaling, bij elke eenheid --
  "onder 0,70" vraagt dan geen 30 % maar ongeveer 65 % daling;
* apart drukkanaal: de envelope komt van het wortel-gelineariseerde signaal,
  de teugamplitude niet, en de verhouding schaalt met de wortel van de
  eenheid. Het masker is leeg of dekt vrijwel de hele opname, afhankelijk van
  de eenheid waarin het kanaal is opgeslagen.

`_breath_amp_between` verbiedt die menging in zijn docstring uitdrukkelijk.

Veld `peak_mask_same_scale`: aan = de amplitude van een teug is de mediaan van
de envelope over die teug, uit dezelfde reeks als de basislijn. **Default
`False` op ALLE profielen**: dit verschuift de hypopneedetectie op elke opname
en hoort eerst gemeten te worden op PSG-IPA en MESA voordat een profiel het
aanzet.
"""
import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES, PostProcessingRules
from psgscoring.respiratory import _peak_mask

SF = 32.0
N = int(600 * SF)
BASISLIJN = np.ones(N)                   # envelope-basislijn: amplitude 1,0


def _teugen(factor_per_teug):
    """Teugen van 4 s; amplitude piek-tot-dal = 2 x de sinusamplitude."""
    return [{"onset_s": 20.0 + 4.0 * i, "duration_s": 4.0, "amplitude": 2.0 * f}
            for i, f in enumerate(factor_per_teug)]


def _envelope(teugen):
    """De envelope die bij deze teugen hoort: de helft van piek-tot-dal."""
    env = np.ones(N)
    for b in teugen:
        a, e = int(b["onset_s"] * SF), int((b["onset_s"] + b["duration_s"]) * SF)
        env[a:e] = b["amplitude"] / 2.0
    return env


def _masker(teugen, same_scale):
    return _peak_mask(teugen, BASISLIJN, SF, N, hypop_thresh=0.70,
                      apnea_thresh=0.10, exclude_apnea_band=True,
                      min_consecutive=3, same_scale=same_scale,
                      envelope=_envelope(teugen))


# Twaalf gewone teugen, vijf op 60 % (een daling van 40 %), twaalf gewone.
PATROON = [1.0] * 12 + [0.6] * 5 + [1.0] * 12


def test_het_oude_gedrag_ziet_een_daling_van_veertig_procent_niet():
    assert not _masker(_teugen(PATROON), same_scale=False).any()


def test_op_dezelfde_schaal_wordt_een_daling_van_veertig_procent_gemarkeerd():
    m = _masker(_teugen(PATROON), same_scale=True)
    t = np.arange(N) / SF
    assert m[(t >= 68.5) & (t < 87.5)].all()         # de vijf gereduceerde teugen
    assert not m[t < 67.5].any() and not m[t > 88.5].any()


def test_gewone_ademhaling_blijft_ongemarkeerd_op_dezelfde_schaal():
    assert not _masker(_teugen([1.0] * 40), same_scale=True).any()


def test_twee_teugen_zijn_geen_reeks():
    patroon = [1.0] * 12 + [0.6] * 2 + [1.0] * 12
    assert not _masker(_teugen(patroon), same_scale=True).any()


def test_de_apneuband_blijft_uitgesloten():
    patroon = [1.0] * 12 + [0.05] * 5 + [1.0] * 12
    assert not _masker(_teugen(patroon), same_scale=True).any()


def test_een_diepe_daling_markeert_op_beide_schalen():
    """65 % daling of meer haalt de drempel ook op de gemengde schaal."""
    patroon = [1.0] * 12 + [0.30] * 5 + [1.0] * 12
    assert _masker(_teugen(patroon), same_scale=False).any()
    assert _masker(_teugen(patroon), same_scale=True).any()


def test_dezelfde_schaal_vraagt_de_envelope():
    with pytest.raises(ValueError):
        _peak_mask(_teugen(PATROON), BASISLIJN, SF, N, hypop_thresh=0.70,
                   apnea_thresh=0.10, exclude_apnea_band=True,
                   min_consecutive=3, same_scale=True)


# ── De klinische montage: apart drukkanaal, envelope na de wortel ─────────

def _drukmontage(schaal):
    """Tien minuten neusdruk in een willekeurige eenheid, met één stuk van
    32 s waarin de DRUK naar 30 % zakt (flow naar ~55 %: een hypopneu)."""
    from psgscoring.breath import detect_breaths
    from psgscoring.signal import (
        bandpass_flow,
        compute_dynamic_baseline,
        preprocess_flow,
    )
    t = np.arange(N) / SF
    druk = np.sin(2 * np.pi * 0.25 * t)
    druk[(t >= 300) & (t < 332)] *= 0.30
    druk = druk * schaal
    teugen = detect_breaths(bandpass_flow(druk, SF), SF)
    env = preprocess_flow(druk, SF, is_nasal_pressure=True)
    return teugen, env, compute_dynamic_baseline(env, SF)


def _drukmasker(schaal, same_scale):
    teugen, env, basislijn = _drukmontage(schaal)
    return _peak_mask(teugen, basislijn, SF, N, hypop_thresh=0.70,
                      apnea_thresh=0.10, exclude_apnea_band=True,
                      min_consecutive=3, same_scale=same_scale, envelope=env)


SCHALEN = [1e-3, 0.03, 1.0, 100.0]


def test_het_oude_masker_hangt_af_van_de_eenheid_van_het_drukkanaal():
    """Legt het bestaande gedrag vast waartegen het veld gebouwd is: dezelfde
    ademhaling, vier eenheden, en het masker loopt van leeg (verhouding
    onder de apneuband of boven de drempel) tot vrijwel de hele opname."""
    dekking = [float(_drukmasker(s, same_scale=False).mean()) for s in SCHALEN]
    assert max(dekking) - min(dekking) > 0.5, dekking


def test_op_dezelfde_schaal_is_het_masker_eenheidsvrij():
    maskers = [_drukmasker(s, same_scale=True) for s in SCHALEN]
    for m in maskers[1:]:
        assert np.array_equal(m, maskers[0])


def test_op_dezelfde_schaal_markeert_het_masker_alleen_de_hypopneu():
    m = _drukmasker(1e-3, same_scale=True)
    t = np.arange(N) / SF
    assert m[(t >= 306) & (t < 326)].all()
    assert not m[(t < 290) | (t > 345)].any()


def test_default_uit_op_elk_profiel():
    assert PostProcessingRules().peak_mask_same_scale is False
    aan = {n for n, p in PROFILES.items() if p.post_processing.peak_mask_same_scale}
    assert aan == set()
    for naam, d in SCORING_PROFILES.items():
        assert d["PEAK_MASK_SAME_SCALE"] is False, naam


@pytest.mark.parametrize("vlag", [False, True])
def test_het_profielveld_bereikt_het_masker(monkeypatch, vlag):
    pytest.importorskip("mne")
    import psgscoring
    from psgscoring import respiratory
    from tests.test_golden_output import CASES, make_raw

    gezien = []
    echt = respiratory._peak_mask

    def spion(*a, **kw):
        gezien.append(kw.get("same_scale"))
        return echt(*a, **kw)

    monkeypatch.setattr(respiratory, "_peak_mask", spion)
    monkeypatch.setitem(SCORING_PROFILES["aasm_v3_rec"],
                        "PEAK_MASK_SAME_SCALE", vlag)
    cfg = dict(CASES["hypopnea_clean"])
    cfg.pop("profile")
    raw, hypno, channel_map = make_raw(**cfg)
    psgscoring.run_pneumo_analysis(raw, hypno=hypno, channel_map=channel_map,
                                   scoring_profile="aasm_v3_rec")
    # De eerste aanroep is de primaire pas; de intervalarmen (strict,
    # sensitive) lezen hun eigen profiel en volgen daarna.
    assert gezien and gezien[0] is vlag
