"""Fix 1 (post-apneu-hyperpneu) herrekent de basislijn als CONTINUE lijn.

WAT ER MIS WAS
--------------
`_recompute_baseline_with_recovery_excluded` schreef per anker een VLAK
blok van 10 s in een basislijn die verder lineair geïnterpoleerd is. Op elke
blokrand springt de basislijn, en `hypop_norm = env / basislijn` springt
mee -- precies op de plekken waar de hyperpneu de basislijn opblaast, dus
rond de events zelf. Bovendien rekende het blok met een vast venster van
300 s en een vaste P95 (`0,3 * p95`), terwijl de basislijn van het profiel
BASELINE_WINDOW_S/BASELINE_PERCENTILE volgt (mesa_shhs: 120 s / P85), en
schreef het een kale P95 terug op een plek waar de hypopneebasislijn
(dynamisch + stadium) / 2 was.

De reparatie (`recovery_baseline_mode="interpolated"`): herreken de ijle
ankers met de profielparameters, zonder de recovery-samples, één np.interp,
en pas daarna dezelfde stadium-/positiemenging toe die de oorspronkelijke
basislijn kreeg. "block" is het oude gedrag, gepind op `mesa_shhs` en
`chicago_1999`.
"""
import numpy as np
import pytest

from psgscoring.respiratory import (
    _build_postapnea_recovery_mask,
    _recompute_baseline_with_recovery_excluded,
)
from psgscoring.signal import compute_dynamic_baseline

SF = 32.0
DUR_S = 2400.0          # 40 min


def _envelope_met_apneus():
    """Amplitude ~1 met trage modulatie; elke 90 s een apneu van 15 s
    (0,02) gevolgd door 30 s hyperpneu (1,8)."""
    n = int(SF * DUR_S)
    t = np.arange(n) / SF
    env = 1.0 + 0.1 * np.sin(2 * np.pi * t / 600.0)
    events = []
    for s in range(120, int(DUR_S) - 120, 90):
        a, b = int(s * SF), int((s + 15) * SF)
        env[a:b] = 0.02
        env[b:int((s + 45) * SF)] = 1.8
        events.append({"onset_s": float(s), "duration_s": 15.0})
    return env, events


@pytest.fixture(scope="module")
def opzet():
    env, events = _envelope_met_apneus()
    origineel = compute_dynamic_baseline(env, SF, window_s=300, percentile=95.0)
    masker = _build_postapnea_recovery_mask(events, len(env), SF, recovery_s=30.0)
    return env, origineel, masker


def _max_sprong(bl):
    return float(np.max(np.abs(np.diff(bl))))


def test_interpolated_is_continu_en_block_niet(opzet):
    env, origineel, masker = opzet
    blok = _recompute_baseline_with_recovery_excluded(
        env, SF, masker, origineel, mode="block")
    glad = _recompute_baseline_with_recovery_excluded(
        env, SF, masker, origineel, mode="interpolated",
        window_s=300, percentile=95.0)
    grens = 3.0 * _max_sprong(origineel)
    assert _max_sprong(glad) <= grens, (_max_sprong(glad), grens)
    assert _max_sprong(blok) > grens, (_max_sprong(blok), grens)


def test_de_herrekening_haalt_de_hyperpneu_uit_de_basislijn(opzet):
    """Anders is de continuïteit triviaal: een kopie van het origineel is
    ook continu."""
    env, origineel, masker = opzet
    glad = _recompute_baseline_with_recovery_excluded(
        env, SF, masker, origineel, mode="interpolated",
        window_s=300, percentile=95.0)
    assert float(np.mean(glad[masker])) < 0.8 * float(np.mean(origineel[masker]))


def test_zonder_recovery_samples_is_het_de_profielbasislijn_zelf(opzet):
    env, origineel, _ = opzet
    leeg = np.zeros(len(env), dtype=bool)
    glad = _recompute_baseline_with_recovery_excluded(
        env, SF, leeg, origineel, mode="interpolated",
        window_s=300, percentile=95.0)
    assert np.allclose(glad, origineel)


def test_de_profielparameters_bereiken_de_herrekening(opzet):
    env, origineel, masker = opzet
    a = _recompute_baseline_with_recovery_excluded(
        env, SF, masker, origineel, mode="interpolated",
        window_s=300, percentile=95.0)
    b = _recompute_baseline_with_recovery_excluded(
        env, SF, masker, origineel, mode="interpolated",
        window_s=120, percentile=85.0)
    assert not np.allclose(a, b)


def test_de_menging_wordt_opnieuw_toegepast(opzet):
    env, origineel, masker = opzet
    gezien = {}

    def meng(dyn):
        gezien["dyn"] = dyn
        return dyn * 0.5

    uit = _recompute_baseline_with_recovery_excluded(
        env, SF, masker, origineel, mode="interpolated",
        window_s=300, percentile=95.0, blend=meng)
    assert "dyn" in gezien
    assert np.allclose(uit, gezien["dyn"] * 0.5)


def test_onbekende_modus_wordt_geweigerd(opzet):
    env, origineel, masker = opzet
    with pytest.raises(ValueError):
        _recompute_baseline_with_recovery_excluded(
            env, SF, masker, origineel, mode="smooth")


# ── het profielveld ──────────────────────────────────────────────────────

def test_veld_default_en_pins():
    from psgscoring.profiles import PROFILES, PostProcessingRules
    assert PostProcessingRules().recovery_baseline_mode == "interpolated"
    for naam in ("mesa_shhs", "chicago_1999"):
        assert PROFILES[naam].post_processing.recovery_baseline_mode == "block", naam
    for naam in ("aasm_v3_rec", "aasm_v3_breath_dual"):
        assert PROFILES[naam].post_processing.recovery_baseline_mode == "interpolated", naam


def test_veld_bereikt_de_legacy_dict_en_de_detector():
    import inspect

    from psgscoring import respiratory
    from psgscoring.constants import SCORING_PROFILES
    assert SCORING_PROFILES["aasm_v3_rec"]["RECOVERY_BASELINE_MODE"] == "interpolated"
    assert SCORING_PROFILES["mesa_shhs"]["RECOVERY_BASELINE_MODE"] == "block"
    assert "RECOVERY_BASELINE_MODE" in inspect.getsource(
        respiratory.detect_respiratory_events)
