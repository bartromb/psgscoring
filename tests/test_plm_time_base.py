"""PLM: de omzetting van vensterindex naar tijd.

`_detect_lm_channel` berekent RMS over vensters van `win = int(sf * 0.1)`
STALEN en zet de index daarna om met `idx * 0.1`. Bij 256 Hz is win = 25
stalen = 0,09766 s, dus loopt de gerapporteerde tijd 2,3 % voor -- lineair
oplopend tot ruim tien minuten aan het eind van een nacht.

Het bijt alleen bij sample rates waarvoor `sf * 0.1` niet geheel is: 256 Hz
(2,3 %) en 128 Hz (6,3 %) wel, 100/200/500 Hz niet. Dat laatste verklaart
waarom het nooit opviel, en de laatste test hieronder legt het vast.

Zie docs/plm_tijdbasis_bevinding.md.
"""
import numpy as np
import pytest

from psgscoring.plm import _detect_lm_channel

BURST_S = 1.5


def _emg(sf, at_s, dur_h=2.0, seed=1):
    """Rustig EMG met één beweging op `at_s`."""
    n = int(dur_h * 3600 * sf)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)                     # ~1 µV rust
    s, e = int(at_s * sf), int((at_s + BURST_S) * sf)
    x[s:e] += rng.normal(0.0, 60.0, e - s)          # ruim boven 8 µV
    return x                                        # al in µV


def _onsets(sf, at_s, **kw):
    lms = _detect_lm_channel(_emg(sf, at_s), sf, unit="uV", **kw)
    return [lm["onset_s"] for lm in lms]


def _dichtst(got, doel):
    return min(got, key=lambda o: abs(o - doel))


def test_current_time_base_drifts_at_256hz():
    """Karakterisering, geen goedkeuring: bij 256 Hz loopt de tijd voor.

    Deze test hoort te BREKEN zodra de tijdbasis default gerepareerd wordt --
    dan is dat precies de winst, en hoort de verwachting mee te veranderen.
    """
    sf, at = 256.0, 6000.0
    got = _onsets(sf, at)
    assert got, "geen beweging gedetecteerd -- fixture is inert"
    dichtst = min(got, key=lambda o: abs(o - at))
    win = int(sf * 0.1)
    verwacht = at * 0.1 / (win / sf)          # 6000 -> ~6144 s
    assert abs(dichtst - verwacht) < 2.0, (
        f"onset {dichtst:.1f}s; verwacht ~{verwacht:.1f}s bij de huidige tijdbasis"
    )
    assert dichtst - at > 100.0, (
        f"drift van {dichtst - at:.1f}s -- verwacht ruim 140 s op t=6000 s"
    )


def test_no_drift_at_a_rate_that_divides_evenly():
    """Bij 200 Hz is `sf * 0.1` geheel en is er geen fout.

    Zonder deze test lijkt de vorige een eigenschap van de detector in plaats
    van van de sample rate.
    """
    sf, at = 200.0, 6000.0
    got = _onsets(sf, at)
    assert got, "geen beweging gedetecteerd -- fixture is inert"
    dichtst = min(got, key=lambda o: abs(o - at))
    assert abs(dichtst - at) < 1.0, f"onset {dichtst:.1f}s, verwacht {at}s"


@pytest.mark.parametrize("sf", [128.0, 256.0])
def test_drift_scales_with_the_window_rounding(sf):
    """De drift is exact het afrondingsverschil, niet iets anders."""
    at = 3000.0
    win = int(sf * 0.1)
    factor = 0.1 / (win / sf)
    got = _onsets(sf, at)
    assert got
    dichtst = min(got, key=lambda o: abs(o - at * factor))
    assert abs(dichtst - at * factor) < 2.0, (
        f"sf={sf}: onset {dichtst:.1f}s, voorspeld {at * factor:.1f}s"
    )


# ── de reparatie ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("sf", [128.0, 200.0, 256.0])
@pytest.mark.parametrize("at", [1200.0, 6000.0])
def test_fixed_time_base_lands_on_the_movement(sf, at):
    """Met `time_base_fix` staat de onset op de echte tijd, bij elke rate.

    Dit is de eigenschap die telt: de fout hoort niet klein te zijn maar
    afwezig, ongeacht sample rate en ongeacht hoe laat in de nacht.
    """
    got = _onsets(sf, at, time_base_fix=True)
    assert got, f"sf={sf}: geen beweging gedetecteerd -- fixture is inert"
    d = _dichtst(got, at)
    assert abs(d - at) < 1.0, f"sf={sf}, t={at}: onset {d:.1f}s"


def test_the_fix_changes_nothing_where_the_window_divides_evenly():
    """Bij 200 Hz is `int(sf*0.1)/sf` exact 0,1, dus mag de vlag niets doen."""
    sf, at = 200.0, 6000.0
    zonder = _detect_lm_channel(_emg(sf, at), sf, unit="uV")
    met = _detect_lm_channel(_emg(sf, at), sf, unit="uV", time_base_fix=True)
    assert zonder == met


def test_the_fix_matters_where_it_does_not():
    """Bij 256 Hz scheelt het op t=6000 s ruim twee minuten."""
    sf, at = 256.0, 6000.0
    zonder = _dichtst(_onsets(sf, at), at * 0.1 / (int(sf * 0.1) / sf))
    met = _dichtst(_onsets(sf, at, time_base_fix=True), at)
    assert zonder - met > 100.0, (
        f"verschil {zonder - met:.1f}s -- verwacht ruim 140 s"
    )


def test_profile_flag_reaches_the_profile_dict():
    from psgscoring.constants import SCORING_PROFILES
    for name, d in SCORING_PROFILES.items():
        assert "PLM_TIME_BASE" in d, name
        assert isinstance(d["PLM_TIME_BASE"], bool), name
    aan = [n for n, d in SCORING_PROFILES.items() if d["PLM_TIME_BASE"]]
    assert not aan, f"vlag hoort nergens default aan te staan: {aan}"
