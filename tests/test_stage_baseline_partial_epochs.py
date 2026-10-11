"""De stadiumbasislijn draaide op de meeste echte opnames nooit.

`compute_stage_baseline` bouwde zijn stadiummasker met
`np.repeat(stage_ep, spe)[:n]`. Dat masker is KORTER dan de envelope zodra
`len(hypno) * spe < n` -- en dat is het normale geval: YASA en de NSRR-
hypnogrammen zijn vloerlengte (617 s signaal, 20 epochs), de envelope niet.
NumPy weigert een booleaans masker van de verkeerde lengte met een IndexError,
`respiratory.py` vangt die op DEBUG-niveau af en scoort stil verder op de
dynamische basislijn alleen. `stage_baseline_used` stond dan op False zonder
dat iemand het zag.

De reparatie (masker expliciet op lengte n, staart op de globale basislijn)
laat een gedocumenteerde tak draaien waar hij eerder niet draaide: de
getallen op de klinische profielen bewegen. Daarom achter
`stage_baseline_partial_epochs` (default aan), met `mesa_shhs` en
`chicago_1999` gepind op het oude gedrag.
"""
import logging

import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES, PostProcessingRules
from psgscoring.signal import compute_dynamic_baseline, compute_stage_baseline

SF = 32.0
EPOCHS = 20                          # 600 s hypnogram
DUUR_S = 617.0                       # ... op een envelope van 617 s


def _envelope(duur_s=DUUR_S, seed=2):
    t = np.arange(int(duur_s * SF)) / SF
    rng = np.random.default_rng(seed)
    return 1.0 + 0.1 * np.sin(2 * np.pi * 0.01 * t) + rng.normal(0, 0.02, t.size)


def test_een_hypnogram_op_vloerlengte_laat_de_stadiumbasislijn_niet_omvallen():
    env = _envelope()
    bl = compute_stage_baseline(env, SF, ["N2"] * EPOCHS)
    assert bl.shape == env.shape
    assert np.all(np.isfinite(bl)) and np.all(bl > 0)


def test_de_pijplijnstap_meldt_dat_de_stadiumbasislijn_gebruikt_is():
    """Het symptoom zoals het in de uitvoer staat: `stage_baseline_used`."""
    from psgscoring.respiratory import _apply_stage_baseline
    env = _envelope()
    dyn = compute_dynamic_baseline(env, SF)
    result: dict = {}
    uit = _apply_stage_baseline(env, SF, ["N2"] * EPOCHS, [], dyn, result)
    assert result["stage_baseline_used"] is True, (
        "de stadiumbasislijn is stil overgeslagen op een hypnogram van "
        "vloerlengte")
    assert uit.shape == env.shape


def test_de_staart_zonder_hypnogram_krijgt_de_globale_basislijn():
    """Na het laatste epoch is er geen stadium; daar hoort de dynamische
    basislijn te staan, niet een geëxtrapoleerd stadiumgetal."""
    env = _envelope()
    dyn = compute_dynamic_baseline(env, SF)
    bl = compute_stage_baseline(env, SF, ["N2"] * EPOCHS, dynamic_baseline=dyn)
    # De functie strijkt het geheel met 5 s glad (ongewijzigd gedrag); voorbij
    # die overgang moet de uitkomst de evenzo gladgestreken globale zijn.
    win = int(SF * 5)
    glad = np.maximum(np.convolve(dyn, np.ones(win) / win, mode="same"), 1e-6)
    staart = slice(int((EPOCHS * 30 + 6) * SF), None)
    np.testing.assert_allclose(bl[staart], glad[staart], rtol=1e-9, atol=1e-12)


def test_een_dekkend_hypnogram_geeft_in_beide_modi_hetzelfde():
    """De golden-cases hebben een dekkend hypnogram; daar mag niets bewegen."""
    env = _envelope(duur_s=EPOCHS * 30.0)
    hypno = ["W"] + ["N2"] * (EPOCHS - 1)
    oud = compute_stage_baseline(env, SF, hypno, partial_epochs=False)
    nieuw = compute_stage_baseline(env, SF, hypno, partial_epochs=True)
    np.testing.assert_array_equal(oud, nieuw)


def test_het_oude_gedrag_blijft_opvraagbaar_voor_de_gepinde_profielen():
    with pytest.raises(IndexError):
        compute_stage_baseline(_envelope(), SF, ["N2"] * EPOCHS,
                               partial_epochs=False)


def test_een_leeg_hypnogram_valt_terug_met_een_waarschuwing(caplog):
    """Terugvallen mag; stil terugvallen is wat hier misging."""
    env = _envelope()
    dyn = compute_dynamic_baseline(env, SF)
    with caplog.at_level(logging.WARNING, logger="psgscoring.signal"):
        bl = compute_stage_baseline(env, SF, [], dynamic_baseline=dyn)
    np.testing.assert_array_equal(bl, dyn)
    assert any("stadiumbasislijn" in r.getMessage().lower()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


# ── het veld ─────────────────────────────────────────────────────────

GEPIND = ("mesa_shhs", "chicago_1999")


def test_default_is_aan():
    assert PostProcessingRules().stage_baseline_partial_epochs is True


def test_gepinde_profielen_blijven_op_het_oude_gedrag():
    for naam in GEPIND:
        assert PROFILES[naam].post_processing.stage_baseline_partial_epochs \
            is False, naam


def test_precies_die_twee_zijn_gepind():
    uit = {n for n, p in PROFILES.items()
           if not p.post_processing.stage_baseline_partial_epochs}
    assert uit == set(GEPIND), uit


def test_veld_bereikt_de_legacy_dict():
    for naam, d in SCORING_PROFILES.items():
        assert d["STAGE_BASELINE_PARTIAL_EPOCHS"] is (naam not in GEPIND), naam
