"""PLM: de omzetting van vensterindex naar tijd.

`_detect_lm_channel` berekent RMS over vensters van `win = int(sf * 0.1)`
STALEN en zet de index daarna om met `idx * 0.1`. Bij 256 Hz is win = 25
stalen = 0,09766 s, dus loopt de gerapporteerde tijd 2,3 % voor -- lineair
oplopend tot ruim tien minuten aan het eind van een nacht.

Het bijt alleen bij sample rates waarvoor `sf * 0.1` niet geheel is: 256 Hz
(2,3 %) en 128 Hz (6,3 %) wel, 100/200/500 Hz niet. Dat laatste verklaart
waarom het nooit opviel.

**Sinds 21-08-2026 is `time_base_fix` default True**, in de functie én op
alle profielen behalve `mesa_shhs` en `chicago_1999`. De drift is daarmee de
tak die je expliciet moet opvragen; de tests hieronder die haar vastleggen
geven `time_base_fix=False` mee. Ze blijven staan omdat die twee profielen
haar nog gebruiken en er dus iets moet omvallen als ze verandert.

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


def test_legacy_time_base_drifts_at_256hz():
    """De oude tak, die `mesa_shhs` en `chicago_1999` nog gebruiken.

    Karakterisering, geen goedkeuring: bij 256 Hz loopt de tijd voor. Zolang
    twee profielen hierop gepind staan, hoort dit gedrag vast te liggen --
    verandert het, dan verandert de reproductie van paper v31/v37 mee.
    """
    sf, at = 256.0, 6000.0
    got = _onsets(sf, at, time_base_fix=False)
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
    got = _onsets(sf, at, time_base_fix=False)
    assert got, "geen beweging gedetecteerd -- fixture is inert"
    dichtst = min(got, key=lambda o: abs(o - at))
    assert abs(dichtst - at) < 1.0, f"onset {dichtst:.1f}s, verwacht {at}s"


@pytest.mark.parametrize("sf", [128.0, 256.0])
def test_drift_scales_with_the_window_rounding(sf):
    """De drift is exact het afrondingsverschil, niet iets anders."""
    at = 3000.0
    win = int(sf * 0.1)
    factor = 0.1 / (win / sf)
    got = _onsets(sf, at, time_base_fix=False)
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
    zonder = _detect_lm_channel(_emg(sf, at), sf, unit="uV", time_base_fix=False)
    met = _detect_lm_channel(_emg(sf, at), sf, unit="uV", time_base_fix=True)
    assert zonder == met


def test_the_fix_matters_where_it_does_not():
    """Bij 256 Hz scheelt het op t=6000 s ruim twee minuten."""
    sf, at = 256.0, 6000.0
    zonder = _dichtst(_onsets(sf, at, time_base_fix=False),
                      at * 0.1 / (int(sf * 0.1) / sf))
    met = _dichtst(_onsets(sf, at, time_base_fix=True), at)
    assert zonder - met > 100.0, (
        f"verschil {zonder - met:.1f}s -- verwacht ruim 140 s"
    )


def test_the_function_default_is_the_repaired_one():
    """Wie de bibliotheek rechtstreeks aanroept, krijgt de juiste tijd.

    De profielvlag regelt de pipeline; deze test dekt de API zelf.
    """
    sf, at = 256.0, 6000.0
    assert abs(_dichtst(_onsets(sf, at), at) - at) < 1.0


def test_profile_flag_is_on_everywhere_except_the_pinned_two():
    """Reparatie, geen criteriumwijziging: dus ook op de historische profielen.

    `mesa_shhs` en `chicago_1999` reproduceren paper v31/v37 en blijven op de
    oude tijdbasis. Gaat een van beide toch aan, dan verschuiven gepubliceerde
    cijfers en hoort dit om te vallen.
    """
    from psgscoring.constants import SCORING_PROFILES
    for name, d in SCORING_PROFILES.items():
        assert "PLM_TIME_BASE" in d, name
        assert isinstance(d["PLM_TIME_BASE"], bool), name
    uit = {n for n, d in SCORING_PROFILES.items() if not d["PLM_TIME_BASE"]}
    assert uit == {"mesa_shhs", "chicago_1999"}, (
        f"verwacht alleen de twee gepinde profielen uit, kreeg {sorted(uit)}"
    )


def test_the_pipeline_actually_forwards_the_profile_flag(monkeypatch):
    """Afleverkant: een vlag in het profiel is niets waard tot de pipeline
    hem doorgeeft.

    De unit-tests hierboven dekken `analyze_plm` en de profieldict. Deze dekt
    het stuk ertussen -- `pipeline.py` leest `PLM_TIME_BASE` en zet het om in
    het `time_base_fix`-argument. Zonder deze test kan die regel wegvallen
    zonder dat iets omvalt, en dan draait productie stil op de oude tijdbasis.
    """
    import psgscoring.pipeline as pipeline

    gezien = {}

    def _vang(*a, **kw):
        gezien.update(kw)
        return {"success": False, "events": [], "summary": {}, "error": "test"}

    monkeypatch.setattr(pipeline, "analyze_plm", _vang)

    # roep de aanroep na zoals pipeline.py hem samenstelt
    import inspect
    bron = inspect.getsource(pipeline)
    blok = bron.split('output["plm"] = _run_step("plm", lambda: analyze_plm(')[1]
    blok = blok.split("))")[0]
    assert "time_base_fix=_plm_tb" in blok.replace(" ", "").replace("\n", ""), (
        "pipeline.py geeft time_base_fix niet door aan analyze_plm"
    )
    assert 'profile.get("PLM_TIME_BASE"' in bron, (
        "pipeline.py leest PLM_TIME_BASE niet uit het profiel"
    )
    assert 'PSGSCORING_PLM_TIME_BASE' in bron, (
        "de env-override ontbreekt"
    )
