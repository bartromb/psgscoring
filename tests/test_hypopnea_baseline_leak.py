"""De hypopnee-basislijn mag niet van het apneu-kanaal komen.

``_setup_hypop_channel()`` hergebruikte de voorberekende basislijn zodra de
SAMPLEFREQUENTIES gelijk waren. Bij twee flowkanalen uit dezelfde EDF is dat
per definitie zo, dus kreeg de hypopnee-envelope — correct uit de neusdruk
berekend — de basislijn van de thermistor eroverheen.

Zichtbaar op MESA: opnames met nul apneus in beide armen hielden een
AHI-verschil van 34,0 tegen 5,7. Dat kon alleen van de hypopnees komen,
terwijl die in beide armen op de neusdruk gescoord werden.

Enkelkanaals opnames raken dit nooit: daar IS het hetzelfde signaal.
"""
import numpy as np
import pytest

from psgscoring.respiratory import _setup_hypop_channel

SF = 32.0
T = np.arange(0, 600, 1 / SF)


def _sig(amp=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return amp * np.sin(2 * np.pi * 0.25 * T) + 0.02 * amp * rng.normal(size=T.size)


def _call(hypop_flow, flow_env, baseline, same=False):
    """Draai alleen de hypopnee-kanaalopbouw."""
    result = {}
    return _setup_hypop_channel(
        hypop_flow, SF, flow_env, baseline, np.ones(T.size), SF,
        ["N2"] * int(len(T) / SF / 30), [], [], None, None, result,
        precomputed_hypop_baseline=baseline,
        hypop_is_same_channel=same,
    )


def test_a_different_channel_gets_its_own_baseline():
    """Kern van het lek: een andere sensor, dus een andere schaal."""
    apnea_env = np.abs(_sig(amp=10.0, seed=1))       # thermistor, grote amplitude
    apnea_bl = np.full(T.size, 10.0)
    hypop_flow = _sig(amp=1.0, seed=2)               # neusdruk, kleine amplitude

    _env, _norm, hypop_bl, _sf = _call(hypop_flow, apnea_env, apnea_bl)

    assert not np.allclose(hypop_bl, apnea_bl), (
        "de basislijn van het apneu-kanaal mag niet overgenomen worden")
    assert float(np.median(hypop_bl)) < 5.0, (
        "de basislijn hoort bij de schaal van de neusdruk te passen")


def test_the_same_signal_still_reuses_the_baseline():
    """Bij een enkel flowkanaal blijft hergebruik juist en goedkoop."""
    flow = _sig(amp=1.0, seed=3)
    env = np.abs(flow)
    bl = np.full(T.size, 1.0)
    _env, _norm, hypop_bl, _sf = _call(flow, env, bl, same=True)
    assert np.allclose(hypop_bl, bl)


def test_the_normalised_ratio_stays_physical_across_sensors():
    """hypop_env / hypop_bl hoort rond 1 te liggen bij normale ademhaling.

    Met de lekkende basislijn schaalde die verhouding met de VERHOUDING van de
    twee sensoramplitudes, waardoor de hypopnee-drempel op een willekeurig
    punt kwam te liggen.
    """
    apnea_env = np.abs(_sig(amp=10.0, seed=4))
    apnea_bl = np.full(T.size, 10.0)
    hypop_flow = _sig(amp=1.0, seed=5)

    _env, hypop_norm, _bl, _sf = _call(hypop_flow, apnea_env, apnea_bl)
    med = float(np.median(hypop_norm[np.isfinite(hypop_norm)]))
    assert 0.2 < med < 2.0, (
        f"genormaliseerde flow mediaan {med:.2f} — de schalen zijn gemengd")


@pytest.mark.parametrize("apnea_amp", [0.1, 1.0, 10.0, 100.0])
def test_hypopnea_result_is_independent_of_the_apnea_channel_scale(apnea_amp):
    """De keuze van apneu-sensor mag de hypopnee-detectie niet sturen."""
    hypop_flow = _sig(amp=1.0, seed=6)
    apnea_env = np.abs(_sig(amp=apnea_amp, seed=7))
    apnea_bl = np.full(T.size, apnea_amp)
    _env, hypop_norm, _bl, _sf = _call(hypop_flow, apnea_env, apnea_bl)
    med = float(np.median(hypop_norm[np.isfinite(hypop_norm)]))
    assert 0.2 < med < 2.0, (
        f"apneu-amplitude {apnea_amp} stuurde de hypopnee-normalisatie "
        f"naar {med:.2f}")
