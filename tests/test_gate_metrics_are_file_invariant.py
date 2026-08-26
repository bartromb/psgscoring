"""Geen kwaliteitspoort mag een eigenschap van het BESTAND meten.

WAAROM DEZE TEST BESTAAT
------------------------
Twee keer is dat misgegaan, allebei met een maat die schaalvrij bedoeld was:

1. De MAD-drempel (`MAD < 0.005`) mat de **eenhedendeclaratie** van het EDF.
   MESA schrijft RIP in mV en kwam ~150x onder de drempel binnen met een
   volstrekt normaal signaal; PSG-IPA schrijft `n/a` en kwam er duizenden malen
   boven uit. Gerepareerd in 0.19 door `breath_fraction` en `flat_fraction`.

2. `flat_fraction` mat vervolgens de **bemonsteringsfrequentie**. Het telde
   identieke buren, wat neerkomt op de verhouding kwantisatiestap/helling per
   monster. Op een BDF van 250 Hz gaf een gezonde thoraxband 0,686 tegen een
   faaldrempel van 0,50, terwijl 88 % van zijn vermogen in de ademband zat.
   Twee goede banden werden afgekeurd, 72 apneus bleven ongetypeerd en een
   ernstige patiënt werd als "mild" gerapporteerd. Gerepareerd in 0.29.0.

Twee keer dezelfde faalwijze is een patroon, geen toeval. Deze test voedt
DEZELFDE fysiologie aan elke poort, gerenderd bij verschillende
bemonsteringsfrequenties, kwantisatiestappen en amplitudes. Een poort die
oordeelt over een SENSOR moet overal hetzelfde getal geven.

Faalt hier iets, kijk dan eerst of de maat een absolute drempel gebruikt of
buren vergelijkt — dat zijn de twee manieren waarop dit misgaat.
"""
import numpy as np
import pytest

from psgscoring.signal_quality import (
    assess_breath_coherence,
    assess_flow_sensor_agreement,
    assess_thermistor_band_power,
    compare_rip_pair,
    respiratory_band_power,
    rip_shape_metrics,
)

DUUR_S = 600.0
ADEM_HZ = 0.20                       # 12 per minuut

#: Bemonsteringsfrequenties die in de praktijk voorkomen: MESA schrijft RIP op
#: 32 Hz, klinische systemen op 128-256, de Thaise BDF op 250, EEG-versterkers
#: tot 512.
SFS = (32.0, 64.0, 128.0, 250.0, 512.0)
#: Kwantisatiestap als FRACTIE van de amplitude. `None` = geen kwantisatie.
STAPPEN = (None, 0.01, 0.05)
#: Amplitudes die duizend maal verschillen — de eenhedenval uit 0.19.
AMPLITUDES = (1.0, 1000.0)

#: Hoeveel een poortmaat hoogstens mag variëren over al die varianten.
MAX_SPREIDING = 0.10


def _signaal(sf, stap_fractie=None, amp=1.0, faze=0.0):
    """Ademhaling met wat drift en ruis, zoals een echte band hem levert."""
    t = np.arange(int(DUUR_S * sf)) / sf
    x = amp * np.sin(2 * np.pi * ADEM_HZ * t + faze)
    x = x + 0.05 * amp * np.sin(2 * np.pi * 0.01 * t)
    x = x + np.random.default_rng(7).normal(scale=0.02 * amp, size=x.size)
    if stap_fractie:
        stap = stap_fractie * amp
        x = np.round(x / stap) * stap
    return x


def _varianten():
    for sf in SFS:
        for stap in STAPPEN:
            for amp in AMPLITUDES:
                yield sf, stap, amp


def _spreiding(fn):
    waarden = []
    for sf, stap, amp in _varianten():
        v = fn(sf, stap, amp)
        assert v is not None and np.isfinite(v), \
            f"poort geeft geen getal bij sf={sf}, stap={stap}, amp={amp}"
        waarden.append(float(v))
    return max(waarden) - min(waarden), waarden


@pytest.mark.parametrize("naam,fn", [
    ("breath_fraction",
     lambda sf, st, amp: rip_shape_metrics(_signaal(sf, st, amp), sf)[0]),
    ("flat_fraction",
     lambda sf, st, amp: rip_shape_metrics(_signaal(sf, st, amp), sf)[1]),
    ("respiratory_band_power",
     lambda sf, st, amp: respiratory_band_power(_signaal(sf, st, amp), sf)["fraction"]),
    ("thermistor_band_power",
     lambda sf, st, amp: assess_thermistor_band_power(
         _signaal(sf, st, amp), sf)["band_power"]),
    ("flow_sensor_agreement",
     lambda sf, st, amp: assess_flow_sensor_agreement(
         _signaal(sf, st, amp), sf,
         _signaal(sf, st, amp * 0.6, faze=0.3), sf)["agreement"]),
    ("breath_coherence",
     lambda sf, st, amp: assess_breath_coherence(
         _signaal(sf, st, amp), sf,
         _signaal(sf, st, amp * 0.6, faze=0.3), sf)["coherence"]),
])
def test_poortmaat_hangt_niet_af_van_het_bestand(naam, fn):
    spreiding, waarden = _spreiding(fn)
    assert spreiding < MAX_SPREIDING, (
        f"{naam} varieert {spreiding:.3f} over bemonstering/kwantisatie/"
        f"amplitude — die maat oordeelt over het bestand, niet over de sensor. "
        f"min {min(waarden):.3f}, max {max(waarden):.3f}")


def test_de_energieratio_van_een_riparenpaar_is_ook_bestandsonafhankelijk():
    """Aparte test: deze poort neemt twee kanalen en heeft een eigen schaal."""
    spreiding, waarden = _spreiding(
        lambda sf, st, amp: compare_rip_pair(
            _signaal(sf, st, amp), _signaal(sf, st, amp * 0.8, faze=0.2),
            sf, scale_free=True)["energy_ratio"])
    assert spreiding < MAX_SPREIDING, (spreiding, min(waarden), max(waarden))


def test_de_poorten_keuren_dit_signaal_gewoon_goed():
    """Een gezonde band moet overal slagen — anders test het bovenstaande
    alleen of een poort consequent verkeerd is."""
    from psgscoring.signal_quality import assess_rip_channel

    for sf, stap, amp in _varianten():
        q = assess_rip_channel(_signaal(sf, stap, amp), sf, "thorax",
                               scale_free=True)
        assert q["status"] == "ok", (sf, stap, amp, q)
