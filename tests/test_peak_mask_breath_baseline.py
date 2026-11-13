"""Ademteugmasker tegen een basislijn van TEUGAMPLITUDES (13-11-2026).

Het ademteugmasker deelt de teugamplitude (piek-tot-dal van het ruwe kanaal)
door het 95e percentiel van de ENVELOPE. Die twee staan niet op dezelfde
schaal; op de klinische drukmontage hangt de verhouding af van de eenheid van
het kanaal. `peak_mask_same_scale` (10-11) las de amplitude uit de envelope
zelf, maar is op 13-11 gemeten en weerlegd: de envelope ligt op echte opnames
mediaan op 0,2-0,6 van zijn eigen 95e-percentielbasislijn, het masker dekte
42-89 % van de nacht en de F1 van `aasm_v3_rec` zakte van 0,451 naar 0,310.

Veld `peak_mask_breath_baseline`: de teug wordt vergeleken met de teugen in de
twee minuten ervoor (mediaan van de teugen boven het onderste kwartiel van dat
venster). Teller en noemer zijn dan dezelfde grootheid: de verhouding is 1,0
bij rustige ademhaling, bij elke eenheid, en "onder 0,70" is een daling van
30 %. **Default `False` op ALLE profielen** tot het gemeten is.
"""
import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES, PostProcessingRules
from psgscoring.respiratory import _peak_mask

SF = 32.0
N = int(600 * SF)


def _teugen(factor_per_teug, schaal=1.0):
    return [{"onset_s": 20.0 + 4.0 * i, "duration_s": 4.0, "amplitude": 2.0 * f * schaal}
            for i, f in enumerate(factor_per_teug)]


def _masker(teugen, breath_baseline, basislijn=None):
    return _peak_mask(teugen, np.ones(N) if basislijn is None else basislijn, SF, N,
                      hypop_thresh=0.70, apnea_thresh=0.10, exclude_apnea_band=True,
                      min_consecutive=3, breath_baseline=breath_baseline)


# Veertig gewone teugen, vijf op 60 % (een daling van 40 %), twintig gewone.
PATROON = [1.0] * 40 + [0.6] * 5 + [1.0] * 20
T = np.arange(N) / SF
BEGIN, EIND = 20.0 + 4.0 * 40, 20.0 + 4.0 * 45


def test_het_oude_gedrag_ziet_een_daling_van_veertig_procent_niet():
    assert not _masker(_teugen(PATROON), breath_baseline=False).any()


def test_tegen_de_teugbasislijn_wordt_een_daling_van_veertig_procent_gemarkeerd():
    m = _masker(_teugen(PATROON), breath_baseline=True)
    assert m[(T >= BEGIN + 0.5) & (T < EIND - 0.5)].all()
    assert not m[T < BEGIN - 0.5].any() and not m[T > EIND + 0.5].any()


def test_gewone_ademhaling_blijft_ongemarkeerd():
    assert not _masker(_teugen([1.0] * 100), breath_baseline=True).any()


def test_een_daling_van_twintig_procent_is_geen_hypopneu():
    patroon = [1.0] * 40 + [0.8] * 5 + [1.0] * 20
    assert not _masker(_teugen(patroon), breath_baseline=True).any()


def test_twee_teugen_zijn_geen_reeks():
    patroon = [1.0] * 40 + [0.6] * 2 + [1.0] * 20
    assert not _masker(_teugen(patroon), breath_baseline=True).any()


def test_de_apneuband_blijft_uitgesloten():
    patroon = [1.0] * 40 + [0.05] * 5 + [1.0] * 20
    assert not _masker(_teugen(patroon), breath_baseline=True).any()


@pytest.mark.parametrize("schaal", [1e-3, 0.03, 1.0, 100.0])
def test_het_masker_is_eenheidsvrij(schaal):
    """Dezelfde ademhaling in vier eenheden: hetzelfde masker, wat de
    envelope-basislijn ook is (die doet op dit pad niet meer mee)."""
    ref = _masker(_teugen(PATROON), breath_baseline=True)
    m = _masker(_teugen(PATROON, schaal), breath_baseline=True,
                basislijn=np.full(N, 7.0))
    assert np.array_equal(m, ref) and ref.any()


def test_het_oude_masker_hangt_wel_af_van_de_eenheid():
    dekking = [float(_masker(_teugen(PATROON, s), breath_baseline=False).mean())
               for s in (0.03, 0.2, 1.0, 100.0)]
    assert max(dekking) - min(dekking) > 0.4, dekking


def test_een_lange_daling_blijft_gemarkeerd_zolang_het_venster_gewone_teugen_ziet():
    """Twintig teugen (80 s) op 50 %: het venster van twee minuten bevat nog
    genoeg gewone teugen, dus de basislijn zakt niet mee."""
    patroon = [1.0] * 40 + [0.5] * 20 + [1.0] * 20
    m = _masker(_teugen(patroon), breath_baseline=True)
    assert m[(T >= BEGIN + 0.5) & (T < BEGIN + 80 - 0.5)].all()


def test_te_weinig_voorgeschiedenis_markeert_niets():
    """De eerste teugen van de opname hebben geen basislijn: niet gereduceerd."""
    patroon = [0.5] * 4 + [1.0] * 40
    m = _masker(_teugen(patroon), breath_baseline=True)
    assert not m[T < 20.0 + 16.0].any()


def test_de_twee_velden_sluiten_elkaar_uit():
    with pytest.raises(ValueError):
        _peak_mask(_teugen(PATROON), np.ones(N), SF, N, hypop_thresh=0.70,
                   apnea_thresh=0.10, exclude_apnea_band=True, min_consecutive=3,
                   same_scale=True, envelope=np.ones(N), breath_baseline=True)


def test_default_uit_op_elk_profiel():
    assert PostProcessingRules().peak_mask_breath_baseline is False
    aan = {n for n, p in PROFILES.items() if p.post_processing.peak_mask_breath_baseline}
    assert aan == set()
    for naam, d in SCORING_PROFILES.items():
        assert d["PEAK_MASK_BREATH_BASELINE"] is False, naam


@pytest.mark.parametrize("vlag", [False, True])
def test_het_profielveld_bereikt_het_masker(monkeypatch, vlag):
    pytest.importorskip("mne")
    import psgscoring
    from psgscoring import respiratory
    from tests.test_golden_output import CASES, make_raw

    gezien = []
    echt = respiratory._peak_mask

    def spion(*a, **kw):
        gezien.append(kw.get("breath_baseline"))
        return echt(*a, **kw)

    monkeypatch.setattr(respiratory, "_peak_mask", spion)
    monkeypatch.setitem(SCORING_PROFILES["aasm_v3_rec"],
                        "PEAK_MASK_BREATH_BASELINE", vlag)
    cfg = dict(CASES["hypopnea_clean"])
    cfg.pop("profile")
    raw, hypno, channel_map = make_raw(**cfg)
    psgscoring.run_pneumo_analysis(raw, hypno=hypno, channel_map=channel_map,
                                   scoring_profile="aasm_v3_rec")
    assert gezien and gezien[0] is vlag
