"""De thermistorpoort stelde een amplitudevraag over een timingprobleem.

`assess_flow_sensor_agreement` correleert de amplitude-enveloppes. Een
thermistor is thermisch en verzadigt; neusdruk schaalt vóór linearisatie als
flow-kwadraat. Hun amplitudedynamiek verschilt dus legitiem terwijl beide
dezelfde ademteugen volgen. `assess_breath_coherence` vraagt dat laatste
rechtstreeks.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring.profiles import PROFILES
from psgscoring.signal_quality import (
    THERMISTOR_COHERENCE_MIN,
    assess_breath_coherence,
    assess_flow_sensor_agreement,
)

SF = 32.0
N = int(1800 * SF)
_T = np.arange(N) / SF


def _adem(amp=1.0, seed=0, ruis=0.05):
    """Ademhaling met ONREGELMATIGE periode.

    Een zuivere sinus is hier een te brave fixture: die is na tijdomkering nog
    steeds dezelfde sinus en dus per definitie coherent met zichzelf (gemeten:
    0,735). Echte ademhaling is niet stationair, en juist die onregelmatigheid
    maakt gedeelde timing meetbaar. Met een wandelende periode zakt de
    omgekeerde variant naar 0,018, zoals op de MESA-opnames.
    """
    rng = np.random.default_rng(seed)
    per = 0.25 + 0.05 * np.cumsum(rng.normal(0, 0.002, N))
    mod = 1.0 + 0.4 * np.sin(2 * np.pi * 0.005 * _T)
    return amp * mod * np.sin(2 * np.pi * np.cumsum(per) / SF) + rng.normal(0, ruis, N)


def _verzadigd(x, knie=0.4):
    """Thermistorgedrag: correcte timing, samengedrukt bereik."""
    return np.tanh(x / knie) * knie


# ── het veld ─────────────────────────────────────────────────────────

GEPIND = ("mesa_shhs", "chicago_1999")


def test_geen_enkel_profiel_gebruikt_de_coherentiepoort():
    """De poort is gerepareerd maar staat NIET default aan.

    Kort aan geweest op 14-08-2026 en teruggedraaid na de meting: op MESA
    (n=40) ging de bias van −5,18 naar −8,13 en verdwenen 700 events, omdat
    de poort bij mono-profielen blokkeert en apneus bij openzetten van de
    neusdruk naar de tragere thermistor verhuizen. Zie de CHANGELOG.

    Dat maakt de envelope-poort niet juist — die meet aantoonbaar de verkeerde
    grootheid. Ze deed het goede om de verkeerde reden.
    """
    aan = [n for n, p in PROFILES.items()
           if p.post_processing.thermistor_gate == "breath_coherence"]
    assert aan == [], (
        f"aanzetten kostte op MESA 2,95/u aan bias; niet zonder nieuwe "
        f"meting default maken: {aan}")


def test_de_band_power_profielen_zijn_niet_meeverhuisd():
    """Die stellen een andere vraag (eenkanaals bandvermogen); ze omzetten
    zou twee wijzigingen door elkaar halen."""
    band = {n for n, p in PROFILES.items()
            if p.post_processing.thermistor_gate == "respiratory_band"}
    assert band == {"aasm_v3_breath_dual", "aasm_v3_prob_dual"}, band


# ── de kern: verzadiging mag niet afkeuren ───────────────────────────

def test_verzadigde_thermistor_zakt_voor_de_oude_poort_niet_voor_de_nieuwe():
    """Dit is het defect. Zonder deze test meet de rest niets."""
    p = _adem(amp=1.0, seed=1)
    t = _verzadigd(p * 3.0)          # zelfde timing, plat geslagen amplitude
    oud = assess_flow_sensor_agreement(p, SF, t, SF)
    nieuw = assess_breath_coherence(p, SF, t, SF)
    assert nieuw["usable"] is True, nieuw
    assert nieuw["coherence"] > 0.5, nieuw
    # De oude poort mag hem best halen; wat telt is dat de nieuwe véél hoger
    # scoort op een paar dat aantoonbaar dezelfde ademhaling volgt.
    assert nieuw["coherence"] > (oud["agreement"] or 0.0)


def test_twee_verschillende_ademhalingen_scoren_laag():
    """Twee onafhankelijke ademhalingen delen geen timing.

    De drempel is laag (0,015), dus de eis hier is dat de coherentie ver ONDER
    die van een echt paar ligt, niet dat ze onder de drempel valt. Dat de
    drempel weinig marge heeft is een gemeten eigenschap en staat in de
    CHANGELOG, niet iets wat deze test moet verhullen.
    """
    a, b = _adem(seed=2), _adem(seed=17)
    echt = assess_breath_coherence(a, SF, _verzadigd(a * 3.0), SF)["coherence"]
    los = assess_breath_coherence(a, SF, b, SF)["coherence"]
    assert los < 0.25 * echt, (los, echt)


def test_omgekeerde_tijd_zakt_ver_terug():
    """Zelfde amplitudestatistiek, timing kapot.

    Op echte MESA-signalen zakt dit naar 0,002 en valt het ruim onder de
    drempel. Op een synthetische fixture blijft er meer spectrale structuur
    over (gemeten 0,018 tegen een drempel van 0,015), dus de eis hier is de
    ORDE van de terugval, niet het passeren van de drempel. Dat de drempel
    weinig marge heeft is een gemeten eigenschap; zie de CHANGELOG.
    """
    p = _adem(seed=5)
    echt = assess_breath_coherence(p, SF, _verzadigd(p * 3.0), SF)["coherence"]
    rev = assess_breath_coherence(p, SF, p[::-1].copy(), SF)["coherence"]
    assert rev < 0.10 * echt, (rev, echt)


def test_coherentie_is_schaalinvariant():
    p = _adem(seed=6)
    t = _verzadigd(p * 2.0)
    a = assess_breath_coherence(p, SF, t, SF)["coherence"]
    b = assess_breath_coherence(p * 1e6, SF, t * 1e-6, SF)["coherence"]
    assert abs(a - b) < 1e-6


# ── randgevallen ─────────────────────────────────────────────────────

def test_ontbrekend_kanaal():
    assert assess_breath_coherence(None, SF, _adem(), SF)["usable"] is False
    assert assess_breath_coherence(_adem(), SF, None, SF)["usable"] is False


def test_verschillende_samplefrequenties():
    r = assess_breath_coherence(_adem(), SF, _adem(), SF * 2)
    assert r["usable"] is False and "samplefrequenties" in r["reason"]


def test_te_kort_signaal():
    kort = _adem()[:int(60 * SF)]
    r = assess_breath_coherence(kort, SF, kort, SF)
    assert r["usable"] is False and "te kort" in r["reason"]


def test_kanaal_zonder_variatie():
    r = assess_breath_coherence(_adem(), SF, np.zeros(N), SF)
    assert r["usable"] is False


def test_witte_ruis_wordt_afgekeurd():
    """Regressiepin op de bias-correctie.

    Zonder correctie is magnitude-squared coherentie omhoog vertekend bij
    weinig middelingsvensters: op een opname van 10 minuten gaf pure ruis
    0,037 en kwam die door een drempel van 0,017 heen. De drempel hing dus aan
    de OPNAMEDUUR. Deze test pint dat een korte opname met een ruiskanaal
    afgekeurd blijft.
    """
    rng = np.random.default_rng(11)
    kort = int(600 * SF)
    p = _adem(seed=12)[:kort]
    r = assess_breath_coherence(p, SF, rng.normal(0, 1, kort), SF)
    assert r["usable"] is False, r
    assert r["coherence"] < THERMISTOR_COHERENCE_MIN, r


def test_drempel_ligt_in_het_gemeten_gat():
    """Boven het strengste geconstrueerde negatief, onder het zwakste
    werkelijk waargenomen paar."""
    assert 0.008 < THERMISTOR_COHERENCE_MIN < 0.026
