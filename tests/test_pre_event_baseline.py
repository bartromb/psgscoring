"""
tests/test_pre_event_baseline.py — AASM-conforme pre-event baseline.

De AASM meet de >=30% daling t.o.v. de ademhaling **vóór** het event.
`compute_dynamic_baseline()` gebruikt een GECENTREERD venster van 5 minuten
en neemt daarmee de recovery-hyperpnea ná het event mee. Die inflatie
verhoogt de baseline en verkleint de gemeten daling — precies waar Fix 1 en
Fix 6 patches op zijn.

De kerntest hieronder is `test_rolling_inflated_by_recovery_pre_event_not`:
die bouwt een signaal met een bekende pre-event amplitude en een sterke
recovery-piek erna, en toont dat de twee baselines uiteenlopen in de
verwachte richting en orde van grootte.
"""
import numpy as np
import pytest

from psgscoring.profiles import get_profile
from psgscoring.signal import compute_dynamic_baseline, compute_pre_event_baseline

SF = 32.0
BR = 0.25          # 15 ademteugen/min


def _breaths(amps, start_s=0.0, period_s=4.0):
    """Ademteug-dicts zoals breath.detect_breaths ze oplevert."""
    return [
        {"onset_s": start_s + i * period_s, "duration_s": period_s,
         "amplitude": float(a)}
        for i, a in enumerate(amps)
    ]


# ══════════════════════════════════════════════════════════════
#  1. Stabiele vs instabiele ademhaling activeren de juiste tak
# ══════════════════════════════════════════════════════════════

def test_stable_breathing_uses_mean_amplitude():
    b = _breaths([1.00, 1.02, 0.98, 1.01, 0.99, 1.00], start_s=0.0)
    out = compute_pre_event_baseline(onset_s=24.0, breaths=b, sf=SF)
    assert out == pytest.approx(1.0, abs=0.02)


def test_unstable_breathing_uses_n_largest():
    """CV hoog -> gemiddelde van de 3 grootste, niet van alles."""
    amps = [0.2, 0.3, 1.8, 0.25, 1.9, 2.0]
    b = _breaths(amps, start_s=0.0)
    out = compute_pre_event_baseline(onset_s=24.0, breaths=b, sf=SF, n_largest=3)
    assert out == pytest.approx(np.mean([1.8, 1.9, 2.0]), abs=0.01)
    assert out > float(np.mean(amps)), "moet hoger liggen dan het simpele gemiddelde"


def test_stability_threshold_switches_branches():
    amps = [1.0, 1.3, 0.7, 1.2, 0.8, 1.0]          # CV ~ 0.21
    b = _breaths(amps)
    strict = compute_pre_event_baseline(onset_s=24.0, breaths=b, sf=SF,
                                        stability_cv=0.05)   # -> n_largest
    loose = compute_pre_event_baseline(onset_s=24.0, breaths=b, sf=SF,
                                       stability_cv=0.90)    # -> mean
    assert strict > loose


# ══════════════════════════════════════════════════════════════
#  2. Alleen ademteugen VÓÓR het event tellen mee
# ══════════════════════════════════════════════════════════════

def test_only_breaths_before_onset_count():
    before = _breaths([1.0] * 6, start_s=0.0)                 # 0..24 s
    after = _breaths([5.0] * 6, start_s=30.0)                 # recovery, groot
    out = compute_pre_event_baseline(onset_s=28.0, breaths=before + after, sf=SF)
    assert out == pytest.approx(1.0, abs=0.02), "post-event ademteugen lekken door"


def test_window_limits_how_far_back_we_look():
    old = _breaths([9.0] * 6, start_s=0.0)                    # 0..24 s
    recent = _breaths([1.0] * 6, start_s=200.0)               # 200..224 s
    out = compute_pre_event_baseline(onset_s=226.0, breaths=old + recent,
                                     sf=SF, window_s=60.0)
    assert out == pytest.approx(1.0, abs=0.02), "venster van 60 s niet gerespecteerd"


# ══════════════════════════════════════════════════════════════
#  3. Randgevallen — expliciet None, geen verzonnen getal
# ══════════════════════════════════════════════════════════════

def test_too_few_breaths_returns_none():
    assert compute_pre_event_baseline(10.0, _breaths([1.0, 1.0]), SF) is None


def test_no_breaths_at_all_returns_none():
    assert compute_pre_event_baseline(10.0, [], SF) is None
    assert compute_pre_event_baseline(10.0, None, SF) is None


def test_window_entirely_in_wake_returns_none():
    """Wake-ademhaling is geen geldige referentie voor een slaapevent."""
    b = _breaths([1.0] * 8, start_s=0.0)
    hypno = ["W"] * 10
    assert compute_pre_event_baseline(30.0, b, SF, hypno=hypno) is None


def test_sleep_epochs_are_used_when_hypno_given():
    b = _breaths([1.0] * 8, start_s=0.0)
    hypno = ["N2"] * 10
    assert compute_pre_event_baseline(30.0, b, SF, hypno=hypno) == pytest.approx(1.0, abs=0.02)


def test_nonpositive_amplitudes_are_ignored():
    b = _breaths([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    b += [{"onset_s": 25.0, "duration_s": 4.0, "amplitude": 0.0},
          {"onset_s": 26.0, "duration_s": 4.0, "amplitude": -1.0}]
    assert compute_pre_event_baseline(30.0, b, SF) == pytest.approx(1.0, abs=0.02)


# ══════════════════════════════════════════════════════════════
#  4. De kern: rolling wordt opgeblazen door recovery, pre_event niet
# ══════════════════════════════════════════════════════════════

def test_rolling_inflated_by_recovery_pre_event_not():
    """
    Signaal: 200 s rustig ademen (amplitude 1.0), dan een event, dan een
    forse recovery-hyperpnea (amplitude 3.0). De rolling baseline is
    gecentreerd en ziet die piek; de pre-event baseline hoort hem niet te
    zien.
    """
    dur = 400
    n = int(SF * dur)
    t = np.arange(n) / SF
    amp = np.ones(n)
    ev0, ev1 = 200.0, 220.0
    amp[int(ev0 * SF):int(ev1 * SF)] = 0.35          # het event zelf
    amp[int(ev1 * SF):int((ev1 + 60) * SF)] = 3.0    # recovery-hyperpnea

    flow = amp * np.sin(2 * np.pi * BR * t)
    env = np.abs(flow)

    rolling = compute_dynamic_baseline(env, SF, window_s=300, percentile=95.0)
    rolling_at_onset = float(rolling[int(ev0 * SF)])

    # ademteugen tot aan het event: allemaal amplitude ~2.0 peak-to-trough
    pre = _breaths([2.0] * 50, start_s=0.0, period_s=4.0)
    pre_event = compute_pre_event_baseline(ev0, pre, SF, window_s=120.0)

    assert pre_event == pytest.approx(2.0, rel=0.05), \
        "pre-event baseline moet de bekende pre-event amplitude teruggeven"
    assert rolling_at_onset > pre_event, (
        f"rolling ({rolling_at_onset:.3f}) hoort hoger te liggen dan pre_event "
        f"({pre_event:.3f}) doordat de recovery-piek in het gecentreerde venster valt")


# ══════════════════════════════════════════════════════════════
#  5. De profielvlag
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ["aasm_v3_rec", "aasm_v3_strict",
                                  "aasm_v3_sensitive", "mesa_shhs"])
def test_default_stays_rolling(name):
    """Geen enkel profiel mag stilzwijgend omschakelen."""
    assert get_profile(name).post_processing.baseline_mode == "rolling"


def test_baseline_mode_is_exposed_to_the_legacy_dict():
    from psgscoring.constants import SCORING_PROFILES
    assert SCORING_PROFILES["aasm_v3_rec"]["BASELINE_MODE"] == "rolling"
