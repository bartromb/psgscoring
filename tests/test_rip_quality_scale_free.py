"""De RIP-poort mat de eenhedendeclaratie in plaats van de sensor.

De kern van deze suite is `test_oude_poort_valt_om_op_schaal_nieuwe_niet`:
één ademsignaal, twee eenhedenconventies, en de oude poort geeft twee
verschillende oordelen. Zonder die test is de reparatie een bewering.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring.profiles import PROFILES, PostProcessingRules
from psgscoring.signal_quality import (
    BREATH_FRACTION_FAILED_BELOW,
    BREATH_FRACTION_WEAK_BELOW,
    assess_rip_channel,
    compare_rip_pair,
    rip_shape_metrics,
)

SF = 32.0
N = int(2 * 3600 * SF)          # twee uur is ruim genoeg voor de PSD
_T = np.arange(N) / SF


def _adem(amplitude=1.0, freq=0.25, ruis=0.1, seed=1):
    rng = np.random.default_rng(seed)
    return amplitude * (np.sin(2 * np.pi * freq * _T) + rng.normal(0, ruis, N))


# ── het veld ─────────────────────────────────────────────────────────

GEPIND = ("mesa_shhs", "chicago_1999")


def test_default_is_aan():
    """Sinds 13-08-2026 erft een nieuw profiel de gerepareerde poort."""
    assert PostProcessingRules().rip_quality_scale_free is True


def test_gepinde_profielen_blijven_op_het_oude_gedrag():
    """mesa_shhs draagt de reproductie van paper v31/v37, chicago_1999 de
    historische criteria. Beide moeten byte-identiek blijven."""
    for naam in GEPIND:
        assert naam in PROFILES, f"{naam} bestaat niet meer — pin opnieuw"
        assert PROFILES[naam].post_processing.rip_quality_scale_free is False, (
            f"{naam} mag niet meebewegen met poortreparaties")


def test_precies_die_twee_zijn_gepind():
    """Een derde profiel dat stilletjes uit gaat is net zo fout als een
    gepind profiel dat aan gaat."""
    uit = {n for n, p in PROFILES.items()
           if not p.post_processing.rip_quality_scale_free}
    assert uit == set(GEPIND), f"onverwachte pinning: {uit}"


def test_veld_bereikt_de_legacy_dict():
    import psgscoring.constants as C
    for naam, d in C.SCORING_PROFILES.items():
        verwacht = naam not in GEPIND
        assert d["RIP_QUALITY_SCALE_FREE"] is verwacht, naam


# ── de bug zelf ──────────────────────────────────────────────────────

def test_oude_poort_valt_om_op_schaal_nieuwe_niet():
    """Eén signaal, twee eenheden, twee oordelen — dat is het defect.

    `groot` staat voor een EDF met eenheid `n/a` (PSG-IPA, MAD ~200),
    `klein` voor hetzelfde signaal in mV na omrekening naar V (MESA).
    """
    groot = _adem(amplitude=200.0)
    klein = groot * 1e-5

    oud_groot = assess_rip_channel(groot, SF, scale_free=False)["status"]
    oud_klein = assess_rip_channel(klein, SF, scale_free=False)["status"]
    assert oud_groot == "ok"
    assert oud_klein == "failed", (
        "als dit niet failt, reproduceert de fixture de bug niet en meet de "
        "rest van deze suite niets")

    nieuw_groot = assess_rip_channel(groot, SF, scale_free=True)["status"]
    nieuw_klein = assess_rip_channel(klein, SF, scale_free=True)["status"]
    assert nieuw_groot == nieuw_klein == "ok"


@pytest.mark.parametrize("factor", [1e-8, 1e-4, 1.0, 1e3, 1e7])
def test_oordeel_is_schaalinvariant(factor):
    sig = _adem(amplitude=1.0) * factor
    assert assess_rip_channel(sig, SF, scale_free=True)["status"] == "ok"


def test_paar_degradeert_niet_meer_op_schaal():
    thor, abd = _adem(seed=1) * 1e-5, _adem(seed=2) * 1e-5
    assert compare_rip_pair(thor, abd, SF,
                            scale_free=False)["recommended_mode"] == "unreliable"
    assert compare_rip_pair(thor, abd, SF,
                            scale_free=True)["recommended_mode"] == "bilateral"


# ── dood blijft dood ─────────────────────────────────────────────────

def test_vlakke_lijn_faalt():
    q = assess_rip_channel(np.zeros(N), SF, scale_free=True)
    assert q["status"] == "failed"
    assert "vlakke lijn" in q["reason"]


def test_witte_ruis_faalt():
    rng = np.random.default_rng(7)
    q = assess_rip_channel(rng.normal(0, 1.0, N), SF, scale_free=True)
    assert q["status"] == "failed", "ruis is geen ademhaling"


def test_netfrequentie_faalt():
    rng = np.random.default_rng(8)
    sig = np.sin(2 * np.pi * 50 * _T) + rng.normal(0, 0.1, N)
    assert assess_rip_channel(sig, SF, scale_free=True)["status"] == "failed"


def test_lege_en_2d_signalen_blijven_falen():
    assert assess_rip_channel(np.array([]), SF, scale_free=True)["status"] == "failed"
    assert assess_rip_channel(np.zeros((3, N)), SF,
                              scale_free=True)["status"] == "failed"


# ── de grootheden zelf ───────────────────────────────────────────────

def test_ruis_landt_op_de_bandbreedteverhouding():
    """0,10-0,50 Hz binnen 0,02-4,0 Hz is ~0,10 van de band.

    Dat dit klopt is de reden dat de faaldrempel op 0,27 mag liggen: het
    dode-kanaal-uiteinde is analytisch bekend en niet gefit.
    """
    rng = np.random.default_rng(11)
    bf, _flat = rip_shape_metrics(rng.normal(0, 1.0, N), SF)
    verwacht = (0.50 - 0.10) / (4.0 - 0.02)
    assert abs(bf - verwacht) < 0.03, f"{bf} vs analytisch {verwacht}"


def test_drempels_liggen_in_het_gemeten_gat():
    """Onder de zwakste echte meting (0,371), boven de sterkste dode (0,174)."""
    assert 0.174 < BREATH_FRACTION_FAILED_BELOW < 0.371
    assert BREATH_FRACTION_FAILED_BELOW < BREATH_FRACTION_WEAK_BELOW <= 0.371


def test_vlakke_fractie_telt_identieke_monsters():
    half = np.concatenate([np.zeros(N // 2), _adem()[N // 2:]])
    _bf, flat = rip_shape_metrics(half, SF)
    assert 0.45 < flat < 0.55


def test_ademfractie_is_schaalvrij():
    a = _adem(amplitude=1.0)
    bf1, _ = rip_shape_metrics(a, SF)
    bf2, _ = rip_shape_metrics(a * 1e9, SF)
    assert abs(bf1 - bf2) < 1e-9


def test_nan_gaten_laten_de_meting_overeind():
    a = _adem()
    a[: N // 10] = np.nan
    bf, _flat = rip_shape_metrics(a, SF)
    assert bf > BREATH_FRACTION_WEAK_BELOW


# ── de oude tak blijft ongemoeid ─────────────────────────────────────

def test_absolute_tak_is_onveranderd():
    """Default-pad: dezelfde velden, dezelfde uitkomst als voorheen."""
    q = assess_rip_channel(_adem(amplitude=200.0), SF, scale_free=False)
    assert q["status"] == "ok"
    assert "breath_fraction" not in q, (
        "de oude tak mag geen nieuwe sleutels leveren; consumenten pinnen hem")
