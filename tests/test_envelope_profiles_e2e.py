"""
tests/test_envelope_profiles_e2e.py — the envelope axis through the scorer.

`test_envelope_methods.py` checks the envelopes themselves. This file checks
that the profile field actually reaches the scoring, and that the one piece of
machinery it invalidates was updated with it.

The preprocessing cache
-----------------------
`detect_respiratory_events` accepts a `_precomputed` dict so the AHI-interval
reruns do not recompute envelopes, baselines and breath detection for every
profile. That is sound exactly as long as the preprocessing is
profile-independent, which is the argument written into the comment above the
cache — and which stopped being true when `envelope_method` was added.

The failure it would cause is quiet: a group mixing envelope arms would score
every arm on whichever arm ran first, the arms would agree closely, and the
robustness interval would come out reassuringly narrow. Nothing raises. The
test that catches it is `test_a_shared_cache_does_not_change_the_answer`.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES

SF = 32.0
DUR_S = 7200                      # 2 h: four 30-minute chunks, so chunking engages
BR = 0.25

ARMS = ["aasm_v3_env_chunked", "aasm_v3_env_rectify",
        "aasm_v3_env_breath", "aasm_v3_env_decimated"]


def _recording(seed: int = 42):
    """Flow / thorax / abdomen / SpO2 with a hypopnea every four minutes."""
    n = int(SF * DUR_S)
    t = np.arange(n) / SF
    rng = np.random.default_rng(seed)

    # Breath-to-breath variability 0.90, as in the golden hypopnea cases. With
    # metronomic breathing the stability filter rejects every candidate and the
    # recording scores zero events — which every comparison below would then
    # pass by comparing nothing to nothing.
    n_breaths = int(DUR_S * BR) + 2
    amps = np.clip(1.0 + rng.normal(0, 0.90, n_breaths), 0.35, 1.9)
    amp_var = amps[np.clip((t * BR).astype(int), 0, n_breaths - 1)]

    events = [(s, s + 16) for s in range(120, DUR_S - 120, 240)]
    flow_amp, eff_amp = np.ones(n), np.ones(n)
    for i, (t0, t1) in enumerate(events):
        # Every third event is an obstructive apnea: flow gone, effort continues.
        flow_f, eff_f = (0.02, 1.0) if i % 3 == 2 else (0.38, 0.45)
        flow_amp[int(t0 * SF):int(t1 * SF)] *= flow_f
        eff_amp[int(t0 * SF):int(t1 * SF)] *= eff_f

    flow = amp_var * flow_amp * np.sin(2 * np.pi * BR * t) + rng.normal(0, 0.005, n)
    thorax = amp_var * eff_amp * np.sin(2 * np.pi * BR * t + 0.05) + rng.normal(0, 0.005, n)
    abdomen = 0.9 * amp_var * eff_amp * np.sin(2 * np.pi * BR * t + 0.20) + rng.normal(0, 0.005, n)

    spo2 = np.full(n, 96.0)
    for t0, t1 in events:
        d0, nadir, d1 = int((t1 - 2) * SF), int((t1 + 4) * SF), int((t1 + 10) * SF)
        spo2[d0:nadir] = np.linspace(96.0, 91.0, nadir - d0)
        spo2[nadir:d1] = np.linspace(91.0, 96.0, d1 - nadir)

    hypno = ["W"] + ["N2"] * (DUR_S // 30 - 1)
    return flow, thorax, abdomen, spo2, hypno


@pytest.fixture(scope="module")
def rec():
    return _recording()


def _score(rec, profile_name, precomputed=None):
    from psgscoring.respiratory import detect_respiratory_events

    flow, thorax, abdomen, spo2, hypno = rec
    return detect_respiratory_events(
        flow_data=flow, thorax_data=thorax, abdomen_data=abdomen,
        spo2_data=spo2, sf_flow=SF, sf_spo2=SF, hypno=hypno,
        scoring_profile=SCORING_PROFILES[profile_name],
        _precomputed=precomputed,
    )


def _fingerprint(result):
    """Enough of the scoring output to notice a changed envelope."""
    events = result.get("events", []) or []
    return (
        len(events),
        round(float(result.get("summary", {}).get("ahi_total", 0.0)), 6),
        tuple(round(float(e.get("onset_s", 0.0)), 3) for e in events),
        tuple(round(float(e.get("duration_s", 0.0)), 3) for e in events),
    )


# ─────────────────────────────────────────────────────────────────────────
# The cache
# ─────────────────────────────────────────────────────────────────────────

class TestSharedPreprocessingCache:

    @pytest.mark.parametrize("arm", ARMS)
    def test_a_shared_cache_does_not_change_the_answer(self, rec, arm):
        """
        Score the reference first into a shared cache, then the arm into that
        same cache, and compare against the arm scored on its own. Equal means
        the cache is keyed on the envelope; unequal means the arm silently
        inherited the reference's envelope.

        This is the test that fails if `_ENV_KEY` is dropped from the cache
        keys in respiratory.py — verified by doing exactly that.
        """
        shared: dict = {}
        _score(rec, "aasm_v3_rec", precomputed=shared)
        via_cache = _fingerprint(_score(rec, arm, precomputed=shared))
        alone = _fingerprint(_score(rec, arm))

        assert via_cache == alone, (
            f"{arm} scored differently after aasm_v3_rec warmed the shared "
            f"cache: it is reading the reference's envelope")

    def test_the_cache_holds_one_entry_per_envelope(self, rec):
        """
        Direct view of the same defect: after two arms share a dict there must
        be two distinct flow envelopes in it, not one.
        """
        shared: dict = {}
        _score(rec, "aasm_v3_rec", precomputed=shared)
        _score(rec, "aasm_v3_env_rectify", precomputed=shared)

        envs = {k: v for k, v in shared.items() if k.startswith("flow_env")}
        assert len(envs) == 2, (
            f"expected one cached envelope per method, found {sorted(envs)}")
        a, b = list(envs.values())
        assert not np.allclose(a, b), "two cache entries holding the same envelope"

    def test_the_cache_still_saves_work_for_the_interval_profiles(self, rec):
        """
        The keys must not become so specific that the interval reruns stop
        sharing anything — that would quietly triple the runtime of every
        clinical report. strict/rec/sensitive all use the default envelope, so
        they must land on the same entry.
        """
        shared: dict = {}
        for name in ("aasm_v3_strict", "aasm_v3_rec", "aasm_v3_sensitive"):
            _score(rec, name, precomputed=shared)
        envs = [k for k in shared if k.startswith("flow_env")]
        assert len(envs) == 1, f"the interval arms stopped sharing: {envs}"


# ─────────────────────────────────────────────────────────────────────────
# The arms reach the scoring
# ─────────────────────────────────────────────────────────────────────────

class TestArmsAffectScoring:

    @pytest.mark.parametrize("arm", ARMS)
    def test_each_arm_scores_without_crashing(self, rec, arm):
        result = _score(rec, arm)
        assert result.get("success") is not False, result.get("error")
        assert "summary" in result

    @pytest.mark.parametrize("arm", ARMS)
    def test_each_arm_reaches_the_envelope_the_scorer_uses(self, rec, arm):
        """
        Checked at the envelope the scorer actually consumed, not at the event
        list. An aggregate can absorb a real difference — `aasm_v3_env_decimated`
        scores identically to the reference on this fixture — and a test that
        only looked at events would then read as "not wired through".
        """
        cache: dict = {}
        _score(rec, arm, precomputed=cache)
        env = next(v for k, v in cache.items() if k.startswith("flow_env"))

        ref_cache: dict = {}
        _score(rec, "aasm_v3_rec", precomputed=ref_cache)
        ref_env = next(v for k, v in ref_cache.items() if k.startswith("flow_env"))

        assert not np.allclose(env, ref_env, rtol=1e-9, atol=1e-12), (
            f"{arm} produced the reference envelope: the profile field never "
            f"reached preprocess_flow")

    def test_the_two_hilbert_variants_barely_move_the_events(self, rec):
        """
        Both stay within a sample or two of the reference on synthetic data.
        Pinned because it is the result, not a gap in the test: an interior
        envelope deviation around 1e-4 shifts one boundary by 0.1 s out of 19
        events, and at 32 Hz the decimation factor is 3 rather than the 20-50
        it would be at a clinical 200-512 Hz.
        """
        ref = _fingerprint(_score(rec, "aasm_v3_rec"))
        chunked = _fingerprint(_score(rec, "aasm_v3_env_chunked"))
        decimated = _fingerprint(_score(rec, "aasm_v3_env_decimated"))

        assert chunked[0] == ref[0], "chunking changed the event count"
        assert chunked[:2] == ref[:2] and chunked != ref, (
            "chunking no longer moves any boundary — either it stopped "
            "engaging, or it became exact and no longer needs its own profile")
        assert decimated == ref, (
            "decimation started moving events on this fixture; that may be "
            "correct, but it is a change worth reading before re-pinning")

    @pytest.mark.parametrize("arm", ["aasm_v3_env_rectify", "aasm_v3_env_breath"])
    def test_the_two_method_changes_move_the_events(self, rec, arm):
        """These are different envelopes, not cheaper ones, and it shows."""
        assert _fingerprint(_score(rec, arm)) != _fingerprint(_score(rec, "aasm_v3_rec"))

    def test_the_reference_is_reproducible(self, rec):
        """Twice through the reference must give the same answer, or nothing
        measured against it means anything."""
        assert _fingerprint(_score(rec, "aasm_v3_rec")) == \
               _fingerprint(_score(rec, "aasm_v3_rec"))

    def test_a_short_recording_makes_the_chunked_arm_the_reference(self):
        """
        Below one chunk there is nothing to chunk, so the arm collapses onto
        the reference. Worth pinning: it is why the golden harness (600 s
        cases) cannot referee this axis, and why anyone validating the arm on a
        short clip would conclude, correctly but uselessly, that it changes
        nothing.
        """
        short = _recording()
        short = tuple(x[:int(SF * 900)] if isinstance(x, np.ndarray) else x[:30]
                      for x in short)
        assert _fingerprint(_score(short, "aasm_v3_env_chunked")) == \
               _fingerprint(_score(short, "aasm_v3_rec"))


# ─────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────

def test_the_arms_are_not_offered_as_clinical_profiles():
    """
    None of these has been measured against human scoring. They must not turn
    up in a clinician's dropdown, which reads the family.
    """
    for arm in ARMS:
        assert PROFILES[arm].family == "exploratory", (
            f"{arm} is offered as a clinical profile without validation")
