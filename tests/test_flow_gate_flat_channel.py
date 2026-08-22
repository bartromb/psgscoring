"""Een dood thermistorkanaal moet als dood gemeld worden, niet als oneens.

`assess_flow_sensor_agreement` bewaakt tegen een vlak kanaal met
`float(np.std(a)) == 0`, maar die test staat NA `_breath_envelope`, en
`filtfilt` op een constante levert numerieke ruis van ~1e-15. De std is dan
niet precies nul, de bewaking laat door, en er wordt een correlatie over ruis
berekend.

Op `artefacten.edf` gaf dat `agreement 0,026` met de reden "de thermistor
volgt de ademhaling niet zoals de neusdruk" -- een uitspraak over de
ademhaling, over een kanaal dat niets meet. Het getal belandt in
`thermistor_check` en daarmee in het rapport. Dat het toevallig onder de
drempel van 0,40 uitkomt maakt het niet minder fout: het is ruis die net zo
goed boven de drempel had kunnen liggen.
"""
import numpy as np

from psgscoring.signal_quality import assess_flow_sensor_agreement

SF = 32.0
DUR_S = 600


def _breathing(sf=SF, dur_s=DUR_S, f=0.25, seed=3):
    t = np.arange(int(sf * dur_s)) / sf
    rng = np.random.default_rng(seed)
    return np.sin(2 * np.pi * f * t) + rng.normal(0, 0.02, t.size)


def test_a_constant_thermistor_is_reported_as_dead_not_as_disagreeing():
    press = _breathing()
    therm = np.full(press.size, 0.5)      # constante op de kwantiseringsstap

    out = assess_flow_sensor_agreement(press, SF, therm, SF)

    assert out["usable"] is False, "een dood kanaal is nooit bruikbaar"
    assert out["agreement"] is None, (
        "er staat een correlatiegetal in over een kanaal zonder signaal: "
        f"{out['agreement']} -- dat getal gaat het rapport in")
    assert "ademhaling niet" not in out["reason"], (
        "de reden doet een uitspraak over de ademhaling op een kanaal dat "
        f"niets meet: {out['reason']!r}")


def test_a_near_constant_thermistor_too():
    """Ook net-niet-constant hoort eruit: kwantiseringsruis is geen signaal."""
    press = _breathing()
    rng = np.random.default_rng(7)
    therm = np.full(press.size, 0.5) + rng.normal(0, 1e-13, press.size)

    out = assess_flow_sensor_agreement(press, SF, therm, SF)
    assert out["usable"] is False
    assert out["agreement"] is None, (
        f"correlatie over kwantiseringsruis gerapporteerd: {out['agreement']}")


def test_a_real_thermistor_still_gets_a_number():
    """De bewaking mag geen echte kanalen opeten."""
    press = _breathing(seed=1)
    therm = _breathing(seed=2)
    out = assess_flow_sensor_agreement(press, SF, therm, SF)
    assert out["agreement"] is not None, (
        "een echt ademend kanaal hoort gewoon een getal te krijgen")
