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


# --------------------------------------------------------------------------
# De poortkeuze moet te overrulen zijn om beide armen op één cohort te meten.
# --------------------------------------------------------------------------

def test_the_gate_can_be_overridden_for_a_measurement(monkeypatch):
    from psgscoring.pipeline import _thermistor_gate

    prof = {"THERMISTOR_GATE": "envelope_agreement"}
    assert _thermistor_gate(prof) == "envelope_agreement"

    monkeypatch.setenv("PSGSCORING_THERMISTOR_GATE", "respiratory_band")
    assert _thermistor_gate(prof) == "respiratory_band"

    monkeypatch.setenv("PSGSCORING_THERMISTOR_GATE", "breath_coherence")
    assert _thermistor_gate(prof) == "breath_coherence"


def test_an_unknown_gate_name_does_not_silently_become_the_default(monkeypatch):
    """Een typefout mag geen meting ongeldig maken.

    Deze poort heeft die fout al eens gemaakt: de drempel stond als
    default-argument en werd bij functiedefinitie geevalueerd, waardoor beide
    armen van een vergelijking op hetzelfde kanaal draaiden zonder dat iemand
    het zag.
    """
    import logging

    from psgscoring.pipeline import _thermistor_gate

    prof = {"THERMISTOR_GATE": "respiratory_band"}
    monkeypatch.setenv("PSGSCORING_THERMISTOR_GATE", "respiratory_bnad")

    seen = []

    class _Grab(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    lg = logging.getLogger("psgscoring.pipeline")
    h = _Grab(level=logging.WARNING)
    lg.addHandler(h)
    # Ook het niveau zetten: een andere test in de suite kan de logger op
    # ERROR hebben gezet, en dan komt de waarschuwing hier nooit aan. Zonder
    # dit slaagde deze test los en viel hij om in de volle suite.
    old_level = lg.level
    lg.setLevel(logging.WARNING)
    try:
        got = _thermistor_gate(prof)
    finally:
        lg.removeHandler(h)
        lg.setLevel(old_level)

    assert got == "respiratory_band", "profielwaarde hoort te blijven staan"
    assert any("respiratory_bnad" in m for m in seen), (
        f"onbekende poortnaam bleef stil; gezien: {seen}")
