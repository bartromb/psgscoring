"""De bandvermogen-poort decimeerde zonder anti-aliasfilter.

`respiratory_band_power` bemonstert naar ~8 Hz met `x[::q]`. Alles boven 4 Hz
vouwt dan terug in het spectrum: netbrom op 50 Hz landt op 2 Hz, midden in de
noemer (0,02-4 Hz) van de breuk. Een thermistor die zuiver op 0,20 Hz ademt
met een 50 Hz-toon van 0,7x de amplitude erbovenop haalt zo 0,671 in plaats
van 1,000 -- bij ELKE bemonsteringsfrequentie waarop 50 Hz representeerbaar
is, want de toon vouwt steeds in de band. De drempel is 0,70, dus een gezonde
sensor wordt afgewezen op `aasm_v3_breath_dual`, de productiestandaard.

De reparatie zit achter `thermistor_band_power_antialias` (default aan,
`mesa_shhs` en `chicago_1999` gepind op het oude gedrag). De negen-montage-
ijking van THERMISTOR_BAND_POWER_MIN is MET aliasing gemeten en hoort opnieuw
bepaald te worden; `scripts/calibrate_thermistor_band_power.py` doet dat op
de server waar die opnames staan.
"""
from pathlib import Path

import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES, PostProcessingRules
from psgscoring.signal_quality import (
    THERMISTOR_BAND_POWER_MIN,
    assess_thermistor_band_power,
    respiratory_band_power,
)

DUUR_S = 600.0
ADEM_HZ = 0.20
TOON_HZ = 50.0                 # netbrom
TOON_AMP = 0.7                 # 0,7x de ademamplitude -> 0,49x het vermogen
#: Alleen frequenties waarop 50 Hz onder Nyquist ligt; bij 32/64 Hz zou de
#: toon al bij de opname zelf vouwen en meet de test iets anders.
SFS = (128.0, 200.0, 250.0, 256.0, 512.0)
#: Wat de oude decimatie oplevert: 1 / (1 + 0,7^2).
ALIAS_UITKOMST = 1.0 / (1.0 + TOON_AMP ** 2)


def _adem(sf, toon=True, seed=3):
    t = np.arange(int(DUUR_S * sf)) / sf
    x = np.sin(2 * np.pi * ADEM_HZ * t)
    if toon:
        x = x + TOON_AMP * np.sin(2 * np.pi * TOON_HZ * t)
    return x + np.random.default_rng(seed).normal(scale=0.01, size=x.size)


def _witte_ruis(sf, seed=5):
    return np.random.default_rng(seed).normal(size=int(DUUR_S * sf))


# ── de fout ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("sf", SFS)
def test_een_toon_buiten_de_band_lekt_niet_in_de_noemer(sf):
    """Zonder antialias: 0,671 bij elke sf. Met: ~1,000."""
    fr = respiratory_band_power(_adem(sf), sf)["fraction"]
    assert fr > 0.95, (
        f"sf={sf}: ademband-fractie {fr:.3f}; een 50 Hz-toon is in de "
        f"noemer gevouwen (aliasuitkomst {ALIAS_UITKOMST:.3f})")


def test_de_oude_decimatie_blijft_opvraagbaar_en_geeft_de_aliasuitkomst():
    """Het gepinde gedrag, met het getal erbij zodat het niet vaag blijft."""
    fr = respiratory_band_power(_adem(256.0), 256.0, antialias=False)["fraction"]
    assert fr == pytest.approx(ALIAS_UITKOMST, abs=0.02), fr


def test_de_poort_keurt_een_gezonde_thermistor_met_netbrom_goed():
    """Dit is de klinische consequentie: 0,671 < 0,70 = afgewezen."""
    r = assess_thermistor_band_power(_adem(256.0), 256.0)
    assert r["usable"] is True, r["reason"]
    assert r["band_power"] > THERMISTOR_BAND_POWER_MIN


# ── wat NIET mag veranderen ──────────────────────────────────────────

@pytest.mark.parametrize("sf", SFS)
def test_witte_ruis_blijft_witte_ruis(sf):
    """Het filter mag een slecht kanaal niet opeens goedkeuren: wit blijft
    ~de bandbreedteverhouding, ver onder de drempel, in beide modi."""
    oud = respiratory_band_power(_witte_ruis(sf), sf, antialias=False)["fraction"]
    nieuw = respiratory_band_power(_witte_ruis(sf), sf, antialias=True)["fraction"]
    assert oud < THERMISTOR_BAND_POWER_MIN and nieuw < THERMISTOR_BAND_POWER_MIN
    assert abs(oud - nieuw) < 0.03, (oud, nieuw)


@pytest.mark.parametrize("sf", SFS)
def test_zonder_toon_maakt_het_filter_geen_verschil(sf):
    x = _adem(sf, toon=False)
    oud = respiratory_band_power(x, sf, antialias=False)["fraction"]
    nieuw = respiratory_band_power(x, sf, antialias=True)["fraction"]
    assert abs(oud - nieuw) < 0.01, (oud, nieuw)
    assert nieuw > THERMISTOR_BAND_POWER_MIN


def test_onder_8_hz_valt_er_niets_te_decimeren():
    """q == 1: geen filter, geen verschil, geen fout."""
    x = _adem(8.0, toon=False)
    assert respiratory_band_power(x, 8.0, antialias=True)["fraction"] == \
        respiratory_band_power(x, 8.0, antialias=False)["fraction"]


# ── het veld ─────────────────────────────────────────────────────────

GEPIND = ("mesa_shhs", "chicago_1999")


def test_default_is_aan():
    assert PostProcessingRules().thermistor_band_power_antialias is True


def test_gepinde_profielen_blijven_op_het_oude_gedrag():
    for naam in GEPIND:
        assert PROFILES[naam].post_processing.thermistor_band_power_antialias \
            is False, naam


def test_precies_die_twee_zijn_gepind():
    uit = {n for n, p in PROFILES.items()
           if not p.post_processing.thermistor_band_power_antialias}
    assert uit == set(GEPIND), uit


def test_veld_bereikt_de_legacy_dict():
    for naam, d in SCORING_PROFILES.items():
        assert d["THERMISTOR_BAND_POWER_ANTIALIAS"] is (naam not in GEPIND), naam


# ── de herijking ─────────────────────────────────────────────────────

def test_de_drempel_zegt_erbij_dat_hij_met_aliasing_is_geijkt():
    import inspect

    import psgscoring.signal_quality as sq
    bron = inspect.getsource(sq)
    assert "calibrate_thermistor_band_power" in bron, (
        "de docstring van THERMISTOR_BAND_POWER_MIN verwijst niet naar het "
        "herijkscript; de 9-montage-ijking is met aliasing gemeten")


def test_het_herijkscript_bestaat_en_kent_beide_modi():
    pad = Path(__file__).resolve().parent.parent / "scripts" / \
        "calibrate_thermistor_band_power.py"
    assert pad.exists(), pad
    tekst = pad.read_text(encoding="utf-8")
    compile(tekst, str(pad), "exec")
    assert "antialias=False" in tekst and "antialias=True" in tekst
