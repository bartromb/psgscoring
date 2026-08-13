"""
Golden-output regression harness (synthetic).
=============================================

Purpose
-------
Freeze the *current* scoring output of the full pipeline on a small set of
deterministic synthetic recordings, so that any code change which moves the
numbers shows up as a failing test with a readable diff — instead of silently
shifting the paper's results.

This is NOT a correctness test (the synthetic signals are not clinically
realistic enough for that). It is a *change detector*. Each case is engineered
to be deterministic and to exercise a specific code path:

    apnea_clean    obstructive + central apnea detection / OA-CA classification
    hypopnea_clean hypopnea detection, Rule 1A desaturation, stability filter
    flat_dropout   a dead flow segment during sleep  (review finding #1)
    spo2_dropout   an SpO2 sensor-dropout gap        (review finding #5)
    cms_arousal    cms_medicare profile + arousal_events / Rule 1B path (#3)
    poor_quality   degraded channels → channel_quality grading (#2 fix lock)

When an *intended* output-changing fix lands, the relevant case(s) will fail.
Inspect the diff, confirm the change is expected, then re-bless:

    python tests/test_golden_output.py bless      # regenerate the baseline
    python tests/test_golden_output.py show       # print current summaries

Running under pytest
--------------------
This is a same-environment *characterization* test: the baseline is blessed on
one interpreter, so it is NOT part of the default CI matrix (pinning exact
numbers across an unpinned 4-version matrix is inherently fragile). It is
SKIPPED unless PSGSCORING_GOLDEN is set:

    PSGSCORING_GOLDEN=1 pytest tests/test_golden_output.py -v

The bless/show CLI above always runs regardless of that variable.

Caveat: an mne.RawArray has a single shared sample rate for all channels, so
these cases do NOT exercise the *mixed* sample-rate hypopnea-baseline branch
(review finding #4). Use scripts/golden_snapshot.py on real EDFs for that.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# Quiet the pipeline + mne so pytest output stays readable.
logging.getLogger("psgscoring").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

GOLDEN_PATH = Path(__file__).parent / "golden" / "synthetic_baseline.json"

SF = 32.0
DUR_S = 600
BR = 0.25  # breathing frequency (Hz) ~ 15 breaths/min
FLOAT_ATOL = 0.25  # tolerance for float leaves when comparing to golden


# ─────────────────────────────────────────────────────────────────────────
# Synthetic PSG generator
# ─────────────────────────────────────────────────────────────────────────

EVENTS_APNEA = [
    (60, 75, "obstructive"),
    (120, 135, "obstructive"),
    (240, 255, "central"),
    (300, 315, "obstructive"),
    (360, 375, "central"),
]
EVENTS_HYP = [
    (60, 76, "hypopnea"),
    (120, 135, "hypopnea"),
    (180, 196, "hypopnea"),
    (300, 315, "hypopnea"),
    (360, 376, "hypopnea"),
]


def make_raw(events, *, variability, flat_segments=(), spo2_dropouts=(),
             degrade_all=False, noise=0.005, seed=42,
    eeg=False, desats=True, effort_scale=1.0,
):
    """Build a deterministic synthetic recording.

    Returns (raw, hypno, channel_map). All randomness is seeded.
    """
    import mne
    mne.set_log_level("ERROR")

    n = int(SF * DUR_S)
    t = np.arange(n) / SF
    rng = np.random.default_rng(seed)

    # per-breath amplitude variability → realistic breath-to-breath CV
    n_breaths = int(DUR_S * BR) + 2
    breath_amps = np.clip(1.0 + rng.normal(0, variability, n_breaths), 0.35, 1.9)
    amp_var = breath_amps[np.clip((t * BR).astype(int), 0, n_breaths - 1)]

    flow_amp = np.ones(n)
    eff_amp = np.ones(n)

    def stamp(arr, t0, t1, factor):
        arr[int(t0 * SF):int(t1 * SF)] *= factor

    for t0, t1, kind in events:
        if kind == "obstructive":
            stamp(flow_amp, t0, t1, 0.02)            # flow gone, effort continues
        elif kind == "central":
            stamp(flow_amp, t0, t1, 0.02)
            stamp(eff_amp, t0, t1, 0.05)             # effort gone too
        elif kind == "hypopnea":
            stamp(flow_amp, t0, t1, 0.38)            # ~62% reduction
            stamp(eff_amp, t0, t1, 0.45)

    flow = amp_var * flow_amp * np.sin(2 * np.pi * BR * t) + rng.normal(0, noise, n)
    thorax = amp_var * eff_amp * np.sin(2 * np.pi * BR * t + 0.05) + rng.normal(0, noise, n)
    abdomen = 0.9 * amp_var * eff_amp * np.sin(2 * np.pi * BR * t + 0.20) + rng.normal(0, noise, n)

    # `effort_scale` bootst na dat een EDF zijn RIP-kanalen in mV declareert:
    # MNE rekent dan naar V en de waarden komen ~1e-5x zo klein binnen. De
    # signaalVORM verandert niet. Zonder zo'n case is de golden blind voor de
    # hele klasse fouten waarin een absolute amplitudedrempel in feite de
    # eenhedendeclaratie meet — de bestaande cases hebben MAD ~0,6 en zitten
    # dus toevallig in het bereik waar zo'n drempel werkt.
    if effort_scale != 1.0:
        thorax = thorax * effort_scale
        abdomen = abdomen * effort_scale

    # dead/flat flow segments (signal dropout)
    for t0, t1 in flat_segments:
        flow[int(t0 * SF):int(t1 * SF)] = 0.0

    # degrade ≥2 channels → drives channel_quality toward "poor"
    if degrade_all:
        sl = slice(int(40 * SF), int(240 * SF))      # ~42% of the record flat
        flow[sl] = 0.0
        thorax[sl] = 0.0
        abdomen[sl] = 0.0

    # SpO2 with desaturations locked to events
    spo2 = np.full(n, 96.0)
    for t0, t1, kind in (events if desats else []):
        d0, nadir, d1 = int((t1 - 2) * SF), int((t1 + 4) * SF), int((t1 + 10) * SF)
        spo2[d0:nadir] = np.linspace(96.0, 91.0, nadir - d0)
        spo2[nadir:d1] = np.linspace(91.0, 96.0, d1 - nadir)
    # sensor-dropout gaps (recorded as 0 → analyze_spo2 treats as implausible)
    for t0, t1 in spo2_dropouts:
        spo2[int(t0 * SF):int(t1 * SF)] = 0.0

    chans = [flow, thorax, abdomen, spo2]
    names = ["FLOW", "THORAX", "ABDOMEN", "SPO2"]
    types = ["misc", "misc", "misc", "misc"]
    if eeg:
        # Achtergrondruis met een alfa-burst kort na elk event: dat is wat de
        # arousal-detectie moet oppikken. Zonder EEG-kanaal slaat pipeline.py
        # de auto-detectie over (`elif eeg_data is not None`) en blijft dit
        # codepad ongetest -- precies het gat dat deze case dicht.
        eeg_sig = rng.normal(0, 20e-6, n)
        for _t0, t1, _kind in events:
            a, b = int((t1 + 1) * SF), int((t1 + 4) * SF)
            b = min(b, n)
            if b > a:
                eeg_sig[a:b] += 40e-6 * np.sin(2 * np.pi * 10.0 * t[a:b])
        chans.append(eeg_sig); names.append("EEG C4-M1"); types.append("eeg")
        chans.append(rng.normal(0, 5e-6, n)); names.append("EMG chin"); types.append("emg")

    data = np.vstack(chans)
    info = mne.create_info(names, SF, ch_types=types)
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    hypno = ["W"] + ["N2"] * (DUR_S // 30 - 1)       # epoch 0 wake, rest sleep
    cmap = {"flow": "FLOW", "thorax": "THORAX", "abdomen": "ABDOMEN", "spo2": "SPO2"}
    return raw, hypno, cmap


def _arousals_for(events):
    """One arousal at the end of each event (seconds)."""
    return [{"onset_s": float(t1), "duration_s": 3.0} for t0, t1, _ in events]


# Each case: kwargs for make_raw + the profile + optional arousal_events.
# Apnea cases use low breath variability (clean 98% reductions → apneas);
# hypopnea cases need high breath-to-breath CV (≥0.45) to survive the
# stable-breathing rejection filter, hence variability=0.90.
CASES = {
    "apnea_clean":    dict(events=EVENTS_APNEA, variability=0.05,
                           profile="aasm_v3_rec"),
    "hypopnea_clean": dict(events=EVENTS_HYP, variability=0.90,
                           profile="aasm_v3_rec"),
    "flat_dropout":   dict(events=EVENTS_APNEA, variability=0.05,
                           flat_segments=[(195, 230)], profile="aasm_v3_rec"),
    "spo2_dropout":   dict(events=EVENTS_HYP, variability=0.90,
                           spo2_dropouts=[(200, 215)], profile="aasm_v3_rec"),
    "cms_arousal":    dict(events=EVENTS_HYP, variability=0.90,
                           profile="cms_medicare", arousals=True),
    "poor_quality":   dict(events=EVENTS_APNEA, variability=0.05,
                           degrade_all=True, profile="aasm_v3_rec"),
    # Effortkanalen in mV-schaal: dezelfde signaalvorm, 1e-5x de amplitude.
    # Elke ANDERE case heeft MAD ~0,6 en zit dus in het bereik waar een
    # absolute amplitudedrempel toevallig werkt; daardoor bleef de golden
    # groen toen bleek dat de RIP-poort in feite de eenhedendeclaratie mat en
    # op MESA 52 van 52 opnames afkeurde. Deze case is het vangnet daarvoor:
    # de apneus horen hier getypeerd te worden, niet `uncertain`.
    "mv_scale_effort": dict(events=EVENTS_APNEA, variability=0.05,
                            effort_scale=1e-5, profile="aasm_v3_rec"),
    # Hypopnees ZONDER desaturatie plus een EEG met arousals: de enige weg
    # naar kwalificatie is de AASM Rule 1A arousal-tak, via de INTERNE
    # arousal-detectie. `cms_arousal` dekt dit niet -- die geeft arousals
    # extern mee en raakt het auto-detectiepad dus nooit.
    #
    # Twee cases, want v0.13.0 maakt dit pad een PROFIELKEUZE
    # (PostProcessingRules.arousal_limb_wired):
    #   - aasm_v3_rec reageert er NIET op, zodat de uitkomst gelijk blijft
    #     aan v0.12.3. Deze case bewaakt dat die belofte blijft staan.
    #   - aasm_v3_breath reageert er WEL op. Deze case bewaakt dat de
    #     reparatie daar daadwerkelijk aankomt.
    "arousal_autodetect": dict(events=EVENTS_HYP, variability=0.90,
                               eeg=True, desats=False, profile="aasm_v3_rec"),
    "arousal_autodetect_breath": dict(events=EVENTS_HYP, variability=0.90,
                                      eeg=True, desats=False,
                                      profile="aasm_v3_breath"),
}


# ─────────────────────────────────────────────────────────────────────────
# Run one case and reduce its output to a stable digest
# ─────────────────────────────────────────────────────────────────────────

def _r(x, nd=1):
    """Round, propagating None/NaN to None."""
    if x is None:
        return None
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def summarize(output: dict) -> dict:
    """Reduce a run_pneumo_analysis output to a small, stable, comparable dict.

    Scoped to the paper-relevant numbers: respiratory indices + event digest,
    SpO2 indices, and channel-quality grades. Position/snore/PLM/HR are
    intentionally excluded (auto-detected ancillary channels are noisy and not
    the focus of the Tier-1 fixes).
    """
    resp = output.get("respiratory", {})
    events = resp.get("events", []) or []
    rs = resp.get("summary", {}) or {}
    sp = (output.get("spo2", {}) or {}).get("summary", {}) or {}
    cq = output.get("channel_quality", {}) or {}

    digest = {
        "profile": output.get("meta", {}).get("scoring_profile"),
        "resp": {
            "success": resp.get("success"),
            "ahi_total": _r(rs.get("ahi_total")),
            "oahi_conf60": _r(rs.get("oahi_conf60")),
            "oahi_all": _r(rs.get("oahi_all")),
            "ahi_excl_noise": _r(rs.get("ahi_excl_noise")),
            "rdi": _r(rs.get("rdi")),
            "n_events": len(events),
            "n_by_type": dict(sorted(Counter(e["type"] for e in events).items())),
            "n_rejected_hypopneas": len(resp.get("rejected_hypopneas", []) or []),
            "rule1b_reinstated": resp.get("rule1b_reinstated", 0),
            "rule1a_arousal_reinstated": resp.get("rule1a_arousal_reinstated", 0),
        },
        "arousal": {
            # Beide sleutels: 'events' is wat de Rule 1A-reinstatement leest,
            # de geneste lijst is waar de detectie ze werkelijk neerzet. Lopen
            # ze uiteen, dan is de doorgifte stuk (issue #16).
            "n_flat": len((output.get("arousal", {}) or {}).get("events", []) or []),
            "n_nested": len((((output.get("arousal", {}) or {})
                              .get("arousals") or {}).get("events", []) or [])),
            # v0.15.2: de arousal-indices stonden hier niet in, terwijl ze wél
            # in het rapport belanden. Daardoor kon 0.14.7 hun afronding van 1
            # naar 3 decimalen verzetten zonder dat deze toets iets merkte.
            "arousal_index": _r((((output.get("arousal", {}) or {})
                                  .get("summary") or {}).get("arousal_index"))),
            "rera_index": _r((((output.get("arousal", {}) or {})
                               .get("reras") or {}).get("summary", {}) or {}
                             ).get("rera_index")),
        },
        "events": sorted(
            [
                {
                    "t": _r(e.get("onset_s")),
                    "d": _r(e.get("duration_s")),
                    "type": e.get("type"),
                    "conf": _r(e.get("confidence"), 2),
                }
                for e in events
            ],
            key=lambda d: (d["t"] if d["t"] is not None else 0.0, d["type"] or ""),
        ),
        "spo2": {
            "odi_3pct": _r(sp.get("odi_3pct")),
            "odi_4pct": _r(sp.get("odi_4pct")),
            "min_spo2": _r(sp.get("min_spo2")),
            "n_desaturations": sp.get("n_desaturations"),
            "hypoxic_burden": _r(sp.get("hypoxic_burden")),
        },
        "quality": {
            "overall_grade": cq.get("overall_grade"),
            "channels": {
                name: {
                    "grade": c.get("quality_grade"),
                    "flat_pct": _r(c.get("flat_pct")),
                    "clip_pct": _r(c.get("clip_pct")),
                }
                for name, c in sorted((cq.get("channels", {}) or {}).items())
            },
        },
    }
    return digest


def run_case(name: str) -> dict:
    """Build the case, run the pipeline, return its digest."""
    import psgscoring

    cfg = dict(CASES[name])
    profile = cfg.pop("profile")
    use_arousals = cfg.pop("arousals", False)
    raw, hypno, cmap = make_raw(**cfg)
    kwargs = dict(channel_map=cmap, scoring_profile=profile)
    if use_arousals:
        kwargs["arousal_events"] = _arousals_for(cfg["events"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = psgscoring.run_pneumo_analysis(raw, hypno, **kwargs)
    return summarize(out)


# ─────────────────────────────────────────────────────────────────────────
# Compare a current digest against the blessed golden
# ─────────────────────────────────────────────────────────────────────────

def _diff(path, golden, current, out):
    """Recursively collect human-readable differences into `out`."""
    if isinstance(golden, dict) and isinstance(current, dict):
        for k in sorted(set(golden) | set(current)):
            if k not in golden:
                out.append(f"{path}.{k}: missing in golden, now {current[k]!r}")
            elif k not in current:
                out.append(f"{path}.{k}: was {golden[k]!r}, now missing")
            else:
                _diff(f"{path}.{k}", golden[k], current[k], out)
    elif isinstance(golden, list) and isinstance(current, list):
        if len(golden) != len(current):
            out.append(f"{path}: length {len(golden)} -> {len(current)}")
        else:
            for i, (g, c) in enumerate(zip(golden, current)):
                _diff(f"{path}[{i}]", g, c, out)
    elif isinstance(golden, (int, float)) and isinstance(current, (int, float)) \
            and not isinstance(golden, bool) and not isinstance(current, bool):
        if abs(float(golden) - float(current)) > FLOAT_ATOL:
            out.append(f"{path}: {golden} -> {current}")
    else:
        if golden != current:
            out.append(f"{path}: {golden!r} -> {current!r}")
    return out


def compare(golden: dict, current: dict) -> list[str]:
    """Return a list of diff strings (empty == match within tolerance)."""
    return _diff("", golden, current, [])


def load_golden() -> dict:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"golden baseline missing: {GOLDEN_PATH}\n"
            "Generate it with:  python tests/test_golden_output.py bless"
        )
    return json.loads(GOLDEN_PATH.read_text())


# ─────────────────────────────────────────────────────────────────────────
# The test
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("PSGSCORING_GOLDEN"),
    reason="golden-output characterization test — set PSGSCORING_GOLDEN=1 to run. "
           "Kept out of the default CI matrix because pinning exact scoring "
           "output across an unpinned multi-version matrix is fragile.",
)
@pytest.mark.parametrize("name", list(CASES))
def test_golden_output(name):
    golden = load_golden()
    assert name in golden, (
        f"case '{name}' not in golden baseline — re-bless: "
        "python tests/test_golden_output.py bless"
    )
    current = run_case(name)
    diffs = compare(golden[name], current)
    assert not diffs, (
        f"golden-output regression in case '{name}':\n  "
        + "\n  ".join(diffs)
        + "\n\nIf this change is intended, re-bless with:\n"
        "  python tests/test_golden_output.py bless"
    )


# ─────────────────────────────────────────────────────────────────────────
# CLI: bless / show
# ─────────────────────────────────────────────────────────────────────────

def _bless():
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    blessed = {name: run_case(name) for name in CASES}
    GOLDEN_PATH.write_text(json.dumps(blessed, indent=2, sort_keys=True) + "\n")
    print(f"blessed {len(blessed)} cases -> {GOLDEN_PATH}")
    for name, d in blessed.items():
        r = d["resp"]
        print(f"  {name:16s} ahi={r['ahi_total']} n={r['n_events']} "
              f"{r['n_by_type']} grade={d['quality']['overall_grade']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bless"
    if cmd == "bless":
        _bless()
    elif cmd == "show":
        for name in CASES:
            print(f"\n=== {name} ===")
            print(json.dumps(run_case(name), indent=2, sort_keys=True))
    else:
        print(__doc__)
        sys.exit(2)
