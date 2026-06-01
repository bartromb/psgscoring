"""
Regression guard for the rejected-hypopnea-candidate invariant.

Rejected hypopnea candidates are not just thrown away — they are re-read by
downstream consumers:
  * Rule 1B reinstatement reads cand["stage"] / cand["epoch"];
  * the mesa_shhs ML re-classifier promotes them straight into the accepted
    list, which then flows into _compute_summary (reads e["type"]).

Three separate production crashes (KeyError 'stage', 'epoch', 'type') have all
come from a `rejected.append(...)` site in respiratory.py omitting one of these
keys. This test pins the invariant so a future append site can't reintroduce
the family of bugs. It asserts structure only (keys present), not numbers, so
it is environment-independent and needs no LightGBM.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_golden_output import make_raw, EVENTS_HYP  # reuse the synthetic generator

logging.getLogger("psgscoring").setLevel(logging.ERROR)

REQUIRED_KEYS = ("type", "stage", "epoch")


@pytest.mark.parametrize("variability", [0.05, 0.90])
def test_rejected_candidates_carry_required_keys(variability):
    """Every rejected hypopnea candidate must carry type/stage/epoch.

    Two fixtures exercise different rejection paths: low variability triggers
    the stable-breathing filter; high variability triggers the local-reduction
    and Rule-1A rejections.
    """
    import psgscoring

    raw, hypno, cmap = make_raw(events=EVENTS_HYP, variability=variability)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = psgscoring.run_pneumo_analysis(
            raw, hypno, channel_map=cmap, scoring_profile="aasm_v3_rec"
        )

    rejected = out["respiratory"].get("rejected_hypopneas", []) or []
    if not rejected:
        pytest.skip("fixture produced no rejected candidates on this build")

    for r in rejected:
        missing = [k for k in REQUIRED_KEYS if k not in r]
        assert not missing, (
            f"rejected hypopnea candidate is missing {missing} "
            f"(reason={r.get('reject_reason')!r}): {sorted(r)}"
        )
