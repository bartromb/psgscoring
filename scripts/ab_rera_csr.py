#!/usr/bin/env python3
"""
A/B validation harness for the RERA/RDI-vs-CSR ordering fix (v0.7.5).

Purpose
-------
Prove on a real cohort (MESA / SHHS) that the fix is *strictly additive*:
  * `ahi_total`, `ahi_incl_uncertain`, and the event count are byte-identical
    between the pre-fix (v0.7.3/0.7.4) and post-fix (v0.7.5) code, and
  * `rdi` / `rera_index` / `rem_ahi` / `nrem_ahi` change from `None` to a real
    value **only** on the recordings where Cheyne-Stokes was detected.

Because this workstation has no MESA/SHHS EDFs, this script is data-source
agnostic: you supply a scoring harness module that exposes

    score_one(recording_id: str, data_dir: str) -> dict
        # returns the psgscoring run_pneumo_analysis output dict

(the same `run_one(sn_id, data_dir)` convention the psgscoring-ab-test
validators already use — just alias it to `score_one`).

Usage
-----
1. Score the cohort on BOTH branches, writing one digest JSON each:

     git checkout main            # or the v0.7.4 branch (pre-fix)
     python scripts/ab_rera_csr.py score \
         --harness mesa_harness --data-dir /path/to/MESA \
         --ids ids.txt --workers 54 --out pre.json

     git checkout fix/rera-rdi-csr-ordering   # post-fix
     python scripts/ab_rera_csr.py score \
         --harness mesa_harness --data-dir /path/to/MESA \
         --ids ids.txt --workers 54 --out post.json

2. Diff the two digests (exits non-zero if any AHI moved):

     python scripts/ab_rera_csr.py diff pre.json post.json

`--ids` is a text file of recording ids (one per line); omit to let the harness
expose `list_ids(data_dir) -> list[str]`.

Z6G4: 56 cores / 125 GB — `--workers 54` leaves headroom; each MESA worker
peaks ~9 GB, so cap workers at ~min(cores-2, RAM_GB // 10) on POOR recordings.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Fields that MUST be identical across the fix (core scoring is untouched).
INVARIANT = ("ahi_total", "ahi_incl_uncertain", "n_events")
# Fields the fix restores on CSR-positive nights.
RERA = ("rdi", "rera_index", "n_rera", "rem_ahi", "nrem_ahi")


def _digest(out: dict) -> dict:
    resp = out.get("respiratory", {}) or {}
    s = resp.get("summary", {}) or {}
    return {
        "csr_detected": bool((out.get("cheyne_stokes", {}) or {}).get("csr_detected")),
        "ahi_total": s.get("ahi_total"),
        "ahi_incl_uncertain": s.get("ahi_incl_uncertain"),
        "n_events": len(resp.get("events", []) or []),
        **{k: s.get(k) for k in RERA},
    }


def _score_worker(args):
    rec_id, harness_name, data_dir = args
    harness = importlib.import_module(harness_name)
    scorer = getattr(harness, "score_one", None) or getattr(harness, "run_one")
    try:
        out = scorer(rec_id, data_dir)
        return rec_id, _digest(out), None
    except Exception as e:  # keep the sweep going; record the failure
        return rec_id, None, f"{type(e).__name__}: {e}"


def cmd_score(a):
    harness = importlib.import_module(a.harness)
    if a.ids:
        ids = [ln.strip() for ln in open(a.ids) if ln.strip()]
    else:
        ids = list(harness.list_ids(a.data_dir))
    print(f"[ab] scoring {len(ids)} recordings on {a.workers} workers…", file=sys.stderr)
    results, errors = {}, {}
    tasks = [(i, a.harness, a.data_dir) for i in ids]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_score_worker, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            rid, dig, err = fut.result()
            if err:
                errors[rid] = err
            else:
                results[rid] = dig
            if n % 10 == 0 or n == len(ids):
                print(f"[ab] {n}/{len(ids)}", file=sys.stderr)
    json.dump({"results": results, "errors": errors},
              open(a.out, "w"), indent=0, sort_keys=True)
    print(f"[ab] wrote {a.out}: {len(results)} ok, {len(errors)} errors", file=sys.stderr)
    if errors:
        print(f"[ab] WARNING: {len(errors)} recordings errored (see 'errors' in {a.out})",
              file=sys.stderr)


def cmd_diff(a):
    pre = json.load(open(a.pre))["results"]
    post = json.load(open(a.post))["results"]
    common = sorted(set(pre) & set(post))
    if not common:
        print("[ab] no common recordings to compare", file=sys.stderr)
        return 2

    ahi_moved, restored, csr_n = [], [], 0
    for rid in common:
        p, q = pre[rid], post[rid]
        if q["csr_detected"]:
            csr_n += 1
        for k in INVARIANT:
            if p.get(k) != q.get(k):
                ahi_moved.append((rid, k, p.get(k), q.get(k)))
        for k in RERA:
            if p.get(k) is None and q.get(k) is not None:
                restored.append((rid, k, q.get(k)))

    print(f"compared {len(common)} recordings ({csr_n} CSR-positive)")
    print(f"  invariant fields moved : {len(ahi_moved)}  (MUST be 0)")
    print(f"  RERA-family restored   : {len(restored)} keys across "
          f"{len({r for r, _, _ in restored})} recordings")
    for rid, k, pv, qv in ahi_moved[:20]:
        print(f"    !! {rid}.{k}: {pv} -> {qv}")
    if ahi_moved:
        print("FAIL: core scoring changed — the fix is not additive.")
        return 1
    print("PASS: no AHI/event/SpO2 number changed; RERA/RDI restored on CSR nights only.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--harness", required=True, help="import name exposing score_one/run_one")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--ids", help="text file of recording ids (else harness.list_ids)")
    s.add_argument("--workers", type=int, default=54)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_score)
    d = sub.add_parser("diff")
    d.add_argument("pre")
    d.add_argument("post")
    d.set_defaults(func=lambda a: sys.exit(cmd_diff(a)))
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
