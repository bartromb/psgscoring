"""De vlakheidsmaat mag niet afhangen van de bemonsteringsfrequentie.

WAT ER MIS WAS
--------------
`flat_fraction` telde opeenvolgende IDENTIEKE monsters. Dat meet in feite de
verhouding tussen de kwantisatiestap en de helling per monster — en die hangt af
van de bemonsteringsfrequentie, dus van het BESTAND en niet van de sensor.

Op een BDF van 250 Hz gaf een volstrekt normale thoraxband **0,686** en de
abdomenband **0,565**, allebei boven de faaldrempel van 0,50, terwijl 88 % van
hun vermogen in de ademband zat en beide op 11,6 ademhalingen per minuut
piekten. Beide werden afgekeurd. Gevolg: geen effort-gebaseerde typering, 72
apneus die `uncertain` bleven, en een AHI die een ernstige patiënt als "mild"
rapporteerde.

Dit is dezelfde faalwijze als de MAD-drempel die deze functie moest vervangen:
een maat die schaalvrij bedoeld was maar een eigenschap van het opnamebestand
mat. De reparatie meet over een vaste stap in de TIJD (0,25 s).
"""
import numpy as np
import pytest

from psgscoring.signal_quality import (
    FLAT_FRACTION_FAILED_ABOVE,
    FLAT_STEP_S,
    assess_rip_channel,
    rip_shape_metrics,
)

DUUR_S = 600
ADEM_HZ = 0.19          # 11,4 per minuut


def _band(sf, stap):
    """Een ademsignaal met een grove kwantisatiestap.

    `stap` is de resolutie van de ADC. Hoe hoger de bemonstering, hoe vaker twee
    buren in dezelfde stap vallen — precies het effect dat de oude maat mat.
    """
    t = np.arange(int(DUUR_S * sf)) / sf
    x = np.sin(2 * np.pi * ADEM_HZ * t)
    return np.round(x / stap) * stap


def test_dezelfde_ademhaling_geeft_dezelfde_vlakheid_bij_elke_frequentie():
    """DE invariant. Zonder deze eigenschap keurt de poort bestanden af op hun
    bemonstering in plaats van op hun sensor."""
    waarden = {sf: rip_shape_metrics(_band(sf, 0.02), sf)[1]
               for sf in (32.0, 64.0, 128.0, 250.0)}
    assert max(waarden.values()) - min(waarden.values()) < 0.10, waarden
    assert all(v < FLAT_FRACTION_FAILED_ABOVE for v in waarden.values()), waarden


def test_een_hoog_bemonsterde_band_wordt_niet_meer_afgekeurd():
    """Het geval uit de Thaise casus: 250 Hz, grove stap."""
    x = _band(250.0, 0.05)
    bf, flat = rip_shape_metrics(x, 250.0)
    assert flat < FLAT_FRACTION_FAILED_ABOVE, flat
    assert bf > 0.5, bf
    assert assess_rip_channel(x, 250.0, "thorax", scale_free=True)["status"] == "ok"


def test_een_werkelijk_losgeraakte_band_faalt_nog_steeds():
    """De maat moet blijven doen waarvoor ze bestaat."""
    dood = np.full(int(DUUR_S * 250.0), 0.42)
    _bf, flat = rip_shape_metrics(dood, 250.0)
    assert flat == pytest.approx(1.0)
    q = assess_rip_channel(dood, 250.0, "thorax", scale_free=True)
    assert q["status"] == "failed", q


def test_een_band_die_halverwege_losraakt_wordt_gezien():
    sf = 250.0
    goed = _band(sf, 0.02)
    x = np.concatenate([goed[: len(goed) // 2],
                        np.full(len(goed) - len(goed) // 2, goed[len(goed) // 2])])
    _bf, flat = rip_shape_metrics(x, sf)
    assert flat > FLAT_FRACTION_FAILED_ABOVE, flat


def test_de_stap_staat_waar_de_fysiologie_hem_zet():
    """0,25 s: lang genoeg dat een ademsignaal meer dan één kwantisatiestap
    beweegt, kort genoeg om een dode lijn te zien."""
    assert FLAT_STEP_S == 0.25
    # Een ademhaling van 0,19 Hz legt in 0,25 s ruim 25 % van een kwartperiode af.
    assert ADEM_HZ * FLAT_STEP_S * 4 > 0.15
