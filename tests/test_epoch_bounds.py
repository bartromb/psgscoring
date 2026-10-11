"""Epochgrenzen op een niet-gehele samplefrequentie.

Zes maskerbouwers rekenden `spe = int(sf * 30)` en vermenigvuldigden die met
de epochindex. Bij 255,74 Hz is dat 7672 waar een epoch 7672,2 samples duurt:
de fout loopt lineair op, 20 samples per 100 epochs, een halve epoch na een
nacht van 12 uur. Bij 0,25 Hz (een houdingskanaal) is int(7,5) = 7 en zit de
drift na vijftien epochs op een volle epoch. Eén helper, `utils.epoch_bounds`,
rondt per epoch af; waar sf x 30 geheel is, is dat byte-identiek.

`_hr_series_1hz` deed `reshape(n, int(sf))`, wat bij een niet-gehele sf een
ValueError gaf (en bij 0,25 Hz een reshape naar nul kolommen).
"""
import numpy as np
import pytest

from psgscoring.ancillary import analyze_position
from psgscoring.arousal import _build_stage_mask, _hr_series_1hz
from psgscoring.utils import build_sleep_mask, epoch_bounds


@pytest.mark.parametrize("sf", [32.0, 100.0, 200.0, 256.0, 512.0])
def test_byte_identiek_waar_sf_maal_30_geheel_is(sf):
    spe = int(sf * 30)
    n = 1000 * spe
    for ep in (0, 1, 7, 99, 999):
        assert epoch_bounds(ep, sf, n) == (ep * spe, min((ep + 1) * spe, n))


def test_geen_drift_op_255_74_hz():
    sf = 255.74
    s, e = epoch_bounds(100, sf, 10 ** 9)
    assert (s, e) == (round(100 * 30 * sf), round(101 * 30 * sf))
    assert s - 100 * int(sf * 30) == 20              # zo ver liep de oude regel al voor


def test_kwart_hertz_houdt_de_tijd_vast():
    assert epoch_bounds(1, 0.25, 10 ** 6) == (8, 15)
    assert epoch_bounds(4, 0.25, 10 ** 6) == (30, 38)
    assert epoch_bounds(0, 0.25, 5) == (0, 5)       # afgekapt op de lengte


def test_de_maskerbouwers_halen_het_einde_van_de_nacht():
    sf = 255.74
    n_ep = 100
    n = round(n_ep * 30 * sf)
    hypno = ["N2"] * n_ep
    assert build_sleep_mask(hypno, sf, n).all()
    assert _build_stage_mask(hypno, sf, n, {"N2"}).all()


def test_de_houdingsepochs_volgen_de_tijd_op_een_kwart_hertz():
    sf = 0.25
    n_ep = 40
    t = np.arange(int(n_ep * 30 * sf)) / sf
    pos = (np.floor(t / 30.0).astype(int) % 2).astype(float)     # 0,1,0,1 per epoch
    r = analyze_position(pos, sf, ["N2"] * n_ep, [])
    assert r["success"], r.get("error")
    assert r["pos_per_epoch"] == [i % 2 for i in range(n_ep)]


def test_hartslagreeks_per_seconde_op_255_74_hz():
    sf = 255.74
    n_s = 700
    seconde = np.floor(np.arange(int(n_s * sf)) / sf)
    hr = 60.0 + (seconde % 100)        # waarde codeert de seconde, binnen 30-180 bpm
    r, ok = _hr_series_1hz(hr, sf)
    n_hele = int(len(hr) // sf)        # 699: int(700 * 255,74) samples is net geen 700 s
    assert ok and len(r) == n_hele and n_s - 1 <= n_hele <= n_s
    np.testing.assert_array_equal(r, 60.0 + (np.arange(n_hele) % 100))


def test_hartslagreeks_op_een_kwart_hertz_breekt_niet():
    hr = np.linspace(60.0, 80.0, 150)
    r, ok = _hr_series_1hz(hr, 0.25)
    assert len(r) == 600
    assert r[0] == hr[0] and r[4] == hr[1]
    assert np.isnan(r[1])
    assert ok is False          # drie van de vier seconden hebben geen sample


def test_hartslagreeks_op_gehele_sf_is_de_oude_reshape():
    sf = 4.0
    rng = np.random.default_rng(0)
    hr = rng.uniform(50, 90, int(650 * sf))
    r, ok = _hr_series_1hz(hr, sf)
    verwacht = np.nanmedian(hr.reshape(650, 4), axis=1)
    np.testing.assert_array_equal(r, verwacht)
    assert ok
