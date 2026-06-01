#!/usr/bin/env python3
"""
Golden-output snapshot tool for REAL recordings.
================================================

The synthetic harness (tests/test_golden_output.py) locks behaviour on tiny
PHI-free signals and runs in CI. This tool does the same job on real EDFs —
the definitive way to answer "does this code change move the actual numbers?"
before/after an output-changing fix.

It freezes the full per-recording digest for a fixed list of recordings, then
diffs a later run against that baseline and prints a per-recording AHI delta
table. Baseline + manifest are NOT committed (they reference local PHI paths).

Usage
-----
    # 1. Write a manifest describing the recordings (see MANIFEST below)
    # 2. Freeze the current numbers:
    python scripts/golden_snapshot.py bless  --manifest mani.json --out base.json
    # 3. Apply a fix, then measure the impact:
    python scripts/golden_snapshot.py check  --manifest mani.json --baseline base.json

Manifest format (JSON list)
---------------------------
    [
      {
        "name":    "mesa-0001",                       # label
        "edf":     "/data/.../mesa-sleep-0001.edf",   # path to EDF
        "hypno":   "/data/.../0001-hypno.json",       # list[str] json, or .csv/.txt
        "profile": "mesa_shhs",                        # scoring profile
        "channel_map": {"flow_pressure": "Pres", ...}, # optional manual map
        "arousals": "/data/.../0001-arousals.json"     # optional list[{onset_s,duration_s}]
      },
      ...
    ]

Run inside the project venv (needs mne + the pipeline + pytest on the path).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import warnings
from pathlib import Path

# Reuse the exact digest + diff logic from the synthetic harness so the two
# stay in lockstep (single source of truth for the snapshot format).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_golden_output import summarize, compare  # noqa: E402

logging.getLogger("psgscoring").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def _load_hypno(path: str) -> list[str]:
    p = Path(path)
    if p.suffix == ".json":
        return list(json.loads(p.read_text()))
    # one stage per line (.csv / .txt), first column
    out = []
    with p.open() as f:
        for row in csv.reader(f):
            if row:
                out.append(row[0].strip())
    return out


def run_recording(rec: dict) -> dict:
    """Run the pipeline on one real EDF and return its digest."""
    import mne
    import psgscoring

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(rec["edf"], preload=True, verbose="ERROR")
    hypno = _load_hypno(rec["hypno"])

    kwargs = dict(scoring_profile=rec.get("profile", "aasm_v3_rec"))
    if rec.get("channel_map"):
        kwargs["channel_map"] = rec["channel_map"]
    if rec.get("arousals"):
        kwargs["arousal_events"] = json.loads(Path(rec["arousals"]).read_text())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = psgscoring.run_pneumo_analysis(raw, hypno, **kwargs)
    return summarize(out)


def build(manifest: list[dict]) -> dict:
    snap = {}
    for rec in manifest:
        name = rec["name"]
        try:
            snap[name] = run_recording(rec)
            ahi = snap[name]["resp"]["ahi_total"]
            print(f"  {name:24s} ahi={ahi}  n={snap[name]['resp']['n_events']}")
        except Exception as e:  # keep going; record the failure
            snap[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {name:24s} ERROR: {e}")
    return snap


def cmd_bless(args):
    manifest = json.loads(Path(args.manifest).read_text())
    print(f"Blessing {len(manifest)} recordings -> {args.out}")
    snap = build(manifest)
    Path(args.out).write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")


def cmd_check(args):
    manifest = json.loads(Path(args.manifest).read_text())
    baseline = json.loads(Path(args.baseline).read_text())
    print(f"Checking {len(manifest)} recordings against {args.baseline}\n")
    current = build(manifest)

    print(f"\n{'recording':24s} {'base AHI':>9s} {'new AHI':>9s} {'Δ':>7s}  status")
    print("-" * 64)
    any_diff = False
    for rec in manifest:
        name = rec["name"]
        g, c = baseline.get(name), current.get(name)
        if g is None:
            print(f"{name:24s} {'(new)':>9s}"); any_diff = True; continue
        if "error" in (c or {}):
            print(f"{name:24s}  ERROR: {c['error']}"); any_diff = True; continue
        ga = g.get("resp", {}).get("ahi_total")
        ca = c.get("resp", {}).get("ahi_total")
        delta = (ca - ga) if (isinstance(ga, (int, float))
                              and isinstance(ca, (int, float))) else None
        diffs = compare(g, c)
        status = "ok" if not diffs else f"CHANGED ({len(diffs)})"
        if diffs:
            any_diff = True
        ds = f"{delta:+7.2f}" if delta is not None else f"{'--':>7s}"
        print(f"{name:24s} {str(ga):>9s} {str(ca):>9s} {ds}  {status}")
        for d in diffs:
            print(f"    {d}")

    print()
    if any_diff:
        print("RESULT: differences found (see above).")
        sys.exit(1)
    print("RESULT: all recordings match the baseline.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bless", help="freeze current numbers to a baseline file")
    b.add_argument("--manifest", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_bless)
    c = sub.add_parser("check", help="diff a new run against a baseline file")
    c.add_argument("--manifest", required=True)
    c.add_argument("--baseline", required=True)
    c.set_defaults(func=cmd_check)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
