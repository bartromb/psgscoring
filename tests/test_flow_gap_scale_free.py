"""Een losgeraakte canule hoort geen apneus te worden.

`_detect_signal_gaps` markeert uitval bij `|x| < 1e-5` (ABSOLUUT, op het RUWE
signaal) of `diff == 0`. Beide falen op echte data:

- absoluut: gemeten op mesa-sleep-0001 vuurt 1e-5 op 0,00 % van de
  `Pres`-samples en 6,44 % van de `Therm`-samples -- dezelfde opname, alleen
  een andere amplitudeschaal;
- `diff == 0`: echte ADC-data heeft geen exact gelijke opeenvolgende samples.

Een losgeraakte canule RUIST rond nul en haalt dus geen van beide. Er is dan
per definitie geen ademhaling, dus de detector scoort aaneengesloten apneus.

Default blijft de absolute variant; de schaalvrije staat achter
`flow_gap_scale_free`.
"""
import numpy as np

from psgscoring.respiratory import _detect_signal_gaps

SF = 32.0
DUR_S = 600
UITVAL = (200, 260)


def _signaal(ruis_uitval=0.002, seed=1):
    n = int(SF * DUR_S)
    t = np.arange(n) / SF
    rng = np.random.default_rng(seed)
    x = np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.01, n)
    a, b = int(UITVAL[0] * SF), int(UITVAL[1] * SF)
    x[a:b] = rng.normal(0, ruis_uitval, b - a)     # ruist, niet exact vlak
    return x, a, b


def test_the_absolute_rule_misses_a_noisy_dropout():
    """Legt het defect vast: zonder deze test meet de volgende niets."""
    x, a, b = _signaal()
    mask, _ = _detect_signal_gaps(x, SF)
    dekking = mask[a:b].mean()
    assert dekking < 0.05, (
        f"de absolute regel dekt {dekking:.1%} van de uitval -- als hij dit "
        "wel vangt, toont deze fixture het probleem niet")


def test_the_scale_free_rule_catches_it():
    x, a, b = _signaal()
    mask, _ = _detect_signal_gaps(x, SF, scale_free=True)
    dekking = mask[a:b].mean()
    assert dekking > 0.90, f"schaalvrij dekt maar {dekking:.1%} van de uitval"


def test_the_scale_free_rule_does_not_eat_normal_breathing():
    x, a, b = _signaal()
    mask, _ = _detect_signal_gaps(x, SF, scale_free=True)
    na = b + int(15 * SF)                       # ná de herstelramp
    buiten = np.concatenate([mask[:a], mask[na:]])
    assert buiten.mean() < 0.01, (
        f"schaalvrij markeert {buiten.mean():.1%} van de normale ademhaling "
        "als uitval")


def test_it_survives_a_change_of_unit():
    """De kern: dezelfde beslissing bij een duizendvoudige schaalwissel."""
    x, a, b = _signaal()
    m1, _ = _detect_signal_gaps(x, SF, scale_free=True)
    m2, _ = _detect_signal_gaps(x * 1000.0, SF, scale_free=True)
    assert np.array_equal(m1, m2), (
        "de schaalvrije regel verandert van oordeel als de eenheid verandert")

    # en de absolute regel doet dat wél -- dat is het hele punt
    a1, _ = _detect_signal_gaps(x * 1e-4, SF)
    a2, _ = _detect_signal_gaps(x * 1e4, SF)
    assert not np.array_equal(a1, a2), (
        "de absolute regel blijkt schaal-invariant; dan klopt de aanname van "
        "deze test niet")


def test_the_default_is_still_the_absolute_rule():
    from psgscoring.profiles import get_profile, list_profiles

    for name in list_profiles():
        assert get_profile(name).post_processing.flow_gap_scale_free is False, (
            f"{name} staat schaalvrij aan; dat neemt events weg en verlaagt de AHI")
