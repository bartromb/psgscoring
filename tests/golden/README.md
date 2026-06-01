# Golden-output regression harness

This guards the **numbers** the pipeline produces. It does not check that the
scoring is *correct* — it checks that scoring output does not change *by
accident*. That matters because psgscoring's results feed a validation paper:
an unintended shift in AHI must never slip through unnoticed.

There are two layers.

## 1. Synthetic harness (committed, runs in CI)

`tests/test_golden_output.py` builds six tiny, deterministic, PHI-free
recordings with `mne.io.RawArray`, runs the full pipeline, and compares a
reduced digest (AHI family, event list, SpO2 indices, channel-quality grades)
against the blessed baseline `synthetic_baseline.json`.

| case | exercises |
|------|-----------|
| `apnea_clean`    | obstructive + central apnea detection, OA/CA classification |
| `hypopnea_clean` | hypopnea detection, Rule 1A desaturation, stability filter |
| `flat_dropout`   | a dead flow segment during sleep (review finding #1) |
| `spo2_dropout`   | an SpO2 sensor-dropout gap (review finding #5) |
| `cms_arousal`    | `cms_medicare` profile + arousal events / Rule 1B (finding #3) |
| `poor_quality`   | degraded channels → `channel_quality` grading (finding #2) |

This is a same-environment **characterization** test, so it is **not part of
the default CI matrix** — pinning exact scoring output across an unpinned
four-version Python matrix is inherently fragile (the precise numbers, and even
whether the channel-quality module runs cleanly, shift with dependency
versions). It is **skipped unless `PSGSCORING_GOLDEN` is set**:

```bash
PSGSCORING_GOLDEN=1 pytest tests/test_golden_output.py -v
```

Run it locally before/after any output-changing fix, or wire it into a
dedicated pinned-environment CI job. When an **intended** output-changing fix
lands, the relevant case(s) fail with a readable diff. Confirm the change is
expected, then re-bless:

```bash
python tests/test_golden_output.py bless    # regenerate synthetic_baseline.json
python tests/test_golden_output.py show     # print current digests (no write)
```

Always read the diff before re-blessing — re-blessing is how you *consciously
accept* a numbers change. Commit the updated `synthetic_baseline.json` together
with the code change so the new behaviour is recorded in git history.

**Caveat:** an `mne.RawArray` has one shared sample rate for all channels, so
these cases do **not** exercise the mixed sample-rate hypopnea-baseline branch
(review finding #4). Use the real-EDF tool for that.

## 2. Real-EDF tool (not committed, for measuring paper impact)

`scripts/golden_snapshot.py` does the same thing on real EDFs — the definitive
way to answer *"how much does this fix move the actual numbers?"* before and
after an output-changing change.

```bash
# freeze the current numbers for a fixed set of recordings
python scripts/golden_snapshot.py bless --manifest mani.json --out base.json

# ... apply a fix ...

# measure the per-recording AHI delta
python scripts/golden_snapshot.py check --manifest mani.json --baseline base.json
```

The manifest (a JSON list of `{name, edf, hypno, profile, channel_map?,
arousals?}`) and the baseline reference local PHI paths, so **neither is
committed** — keep them outside the repo or in a git-ignored location. Point it
at a small fixed slice of the MESA/SHHS set to validate any Tier-1 fix against
the real distribution before resubmission.
