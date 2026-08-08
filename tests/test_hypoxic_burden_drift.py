"""Hypoxic burden mag niet meebewegen met een dalende basislijn.

Aanleiding: de melding dat de burden "onderschat bij aanhoudende hypoxemie".
Gemeten klopt die richting NIET — een vlakke basislijn van 85 % geeft exact
dezelfde burden als 96 % bij dezelfde dipdiepte, en dat is correct: de maat van
Azarbarzin is per definitie oppervlak onder de EIGEN basislijn.

Wat er wel misgaat is drift. `baseline = max(lokaal, globaal)` legt het globale
95e percentiel van de héle nacht als ondergrens onder elk event, dus events laat
in de nacht worden gemeten tegen een basislijn van vroeg in de nacht. Op een
trend van 94 % naar 82 % verdubbelt de burden bijna zonder dat er iets aan de
events verandert — precies het beeld van COPD of obesitas-hypoventilatie.

De code waarschuwde hier zelf voor in een v0.4.4-review. `local_baseline_only`
is die poort, expliciet in plaats van als drempel op 88 %.
"""

import numpy as np
import pytest

from psgscoring.spo2 import compute_hypoxic_burden

SF, DUR_S = 4.0, 8 * 3600
N = int(DUR_S * SF)
T = np.arange(N) / SF
HYPNO = ["N2"] * (DUR_S // 30)


def _nacht(baseline_fn, diepte=5.0, n=60, ev_dur=25.0):
    """SpO2 met n identieke desaturaties op een gegeven basislijnverloop."""
    s = baseline_fn(T).astype(float)
    events = []
    for i in range(n):
        t0 = 300 + i * 120
        a, b = int(t0 * SF), int((t0 + ev_dur) * SF)
        herstel = int(20 * SF)
        s[a:b] -= diepte
        s[b:b + herstel] -= np.linspace(diepte, 0, herstel)
        events.append({"onset_s": float(t0), "duration_s": ev_dur,
                       "desaturation_pct": diepte})
    return s, events


VLAK_HOOG = lambda t: np.full(len(t), 96.0)
VLAK_LAAG = lambda t: np.full(len(t), 85.0)
DRIFT     = lambda t: 94.0 - 12.0 * t / t[-1]


def _hb(fn, **kw):
    s, ev = _nacht(fn)
    return compute_hypoxic_burden(s, SF, ev, HYPNO, **kw)["hypoxic_burden"]


# ─────────────────────────────────────────────────────────────
#  Wat NIET het probleem is
# ─────────────────────────────────────────────────────────────

def test_a_flat_low_baseline_gives_the_same_burden_as_a_normal_one():
    """De gemelde "onderschatting bij lage basislijn" bestaat niet. Dezelfde
    dipdiepte op 85% geeft dezelfde burden als op 96% — correct, want de maat
    is oppervlak onder de eigen basislijn."""
    assert _hb(VLAK_LAAG) == pytest.approx(_hb(VLAK_HOOG), rel=0.02)


def test_that_is_a_property_of_the_measure_not_a_bug():
    """Wel het vastleggen waard: de burden is blind voor het ABSOLUTE
    saturatieniveau. Een daling van 85 naar 80 telt even zwaar als van 96 naar
    91, terwijl de dissociatiecurve onder 90% veel steiler loopt. Dat is de
    definitie van Azarbarzin, geen implementatiefout — maar wie de burden
    klinisch leest, moet het weten."""
    assert _hb(VLAK_LAAG) == pytest.approx(_hb(VLAK_HOOG), rel=0.02)


# ─────────────────────────────────────────────────────────────
#  Wat het wél is
# ─────────────────────────────────────────────────────────────

def test_a_drifting_baseline_inflates_the_burden():
    """Zonder de vlag: dezelfde events, bijna dubbele burden."""
    assert _hb(DRIFT) > 1.6 * _hb(VLAK_HOOG)


def test_the_flag_removes_the_drift_sensitivity():
    """Met de vlag komt de drift-nacht op dezelfde burden uit als de vlakke."""
    assert _hb(DRIFT, local_baseline_only=True) == pytest.approx(
        _hb(VLAK_HOOG), rel=0.05)


def test_the_flag_changes_nothing_on_a_stable_recording():
    """De prijs mag niet zijn dat stabiele nachten verschuiven."""
    for fn in (VLAK_HOOG, VLAK_LAAG):
        assert _hb(fn, local_baseline_only=True) == pytest.approx(_hb(fn), rel=0.01)


# ─────────────────────────────────────────────────────────────
#  De vlag zelf
# ─────────────────────────────────────────────────────────────

def test_no_profile_turns_it_on_by_default():
    """Dit verschuift een gepubliceerde grootheid op elke opname met drift."""
    from psgscoring.profiles import PROFILES
    aan = [n for n, p in PROFILES.items()
           if p.post_processing.hypoxic_burden_local_baseline]
    assert aan == [], f"profielen met de vlag aan: {aan}"


def test_the_flag_reaches_the_dict_the_pipeline_reads():
    """De dataclass-naam is niet de legacy-sleutel — twee keer eerder is een
    veld gepatcht dat niemand las."""
    from psgscoring.constants import SCORING_PROFILES
    assert SCORING_PROFILES["aasm_v3_rec"]["HYPOXIC_BURDEN_LOCAL_BASELINE"] is False
