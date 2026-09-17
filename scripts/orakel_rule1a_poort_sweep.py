#!/usr/bin/env python3
"""Post-hoc poortsweep op de afleidingsrun (orakel-arm C, 40 verse nachten).

Regel en rooster: docs/orakel_rule1a_poort_preregistratie_20260917.md.
De poort is een pure filter vóór de koppeling op velden die de export
draagt; post-hoc filteren van de herstellingen is dus equivalent aan
in-pipeline gating — en dat wordt hier per nacht GECONTROLEERD: zonder poort
moet (A-events ∪ herstellingen) exact de C-eventlijst zijn.

  python scripts/orakel_rule1a_poort_sweep.py docs/orakel_rule1a_afl40.json
"""
import json
import sys
from itertools import product
from pathlib import Path
from statistics import median

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from validate_psgipa import LEGACY_MATCHER, match_events  # noqa: E402
from validate_mesa import parse_nsrr  # noqa: E402
from orakel_rule1a import DATA_MESA  # noqa: E402

MIN_RED = (None, 35, 40, 45, 50, 55, 60)
MAX_DUR = (None, 60, 45, 30)
MIN_LOC = (None, 20, 30, 40)
REF = "aasm15"


def poort_ok(velden, mr, md, ml):
    fr, lr, dur = velden
    if mr is not None and (fr is None or float(fr) < mr):
        return False
    if md is not None and dur > md:
        return False
    if ml is not None and (lr is None or float(lr) < ml):
        return False
    return True


def main(pad):
    d = json.load(open(pad))
    rows = [r for r in d["results"] if "error" not in r and "A" in r["arms"] and "C" in r["arms"]]
    nachten = []
    for r in rows:
        xml = DATA_MESA / "polysomnography" / "annotations-events-nsrr" / f"{r['recording']}-nsrr.xml"
        einde = max([e[1] for e in r["arms"]["A"]["events"]] + [e[1] for e in r["arms"]["C"]["events"]] + [0]) + 3600
        _h, refs, _t = parse_nsrr(xml, einde)
        ref = refs[REF]
        if not ref:
            continue
        A = [tuple(e) for e in r["arms"]["A"]["events"]]
        C = [tuple(e) for e in r["arms"]["C"]["events"]]
        H = [tuple(e) for e in r["arms"]["C"]["reinstated"]]
        V = [tuple(v) for v in r["arms"]["C"]["reinstated_fields"]]
        assert len(H) == len(V), r["recording"]
        # getrouwheid: post-hoc == in-pipeline zonder poort
        if sorted(A + H) != sorted(C):
            raise SystemExit(f"GETROUWHEID FAALT op {r['recording']}: A∪H ≠ C "
                             f"({len(A)}+{len(H)} vs {len(C)}) — post-hoc sweep is dan ongeldig")
        fA = match_events(A, ref, **LEGACY_MATCHER)["f1"]
        nachten.append((r["recording"], ref, A, H, V, fA))
    print(f"{len(nachten)} nachten met referentie; getrouwheid A∪H==C: OK op alle")

    cellen = []
    for mr, md, ml in product(MIN_RED, MAX_DUR, MIN_LOC):
        dF1, tp, n = [], 0, 0
        for rec, ref, A, H, V, fA in nachten:
            kept = [h for h, v in zip(H, V) if poort_ok(v, mr, md, ml)]
            n += len(kept)
            if kept:
                tp += match_events(kept, ref, **LEGACY_MATCHER)["tp"]
            dF1.append(match_events(A + kept, ref, **LEGACY_MATCHER)["f1"] - fA)
        cellen.append({"min_red": mr, "max_dur": md, "min_local": ml,
                       "n_kept": n, "R1": (tp / n) if n else None,
                       "dF1_mean": float(np.mean(dF1)), "dF1_median": float(median(dF1)),
                       "n_beter": int(sum(x > 0 for x in dF1)), "n_nachten": len(dF1),
                       "knoppen": sum(x is not None for x in (mr, md, ml))})
    # keuzeregel
    ok = [c for c in cellen if c["R1"] is not None and c["R1"] >= 0.60]
    if ok:
        keuze = sorted(ok, key=lambda c: (-round(c["dF1_mean"], 4), c["knoppen"],
                                          c["min_red"] or 0, -(c["max_dur"] or 1e9), c["min_local"] or 0))[0]
        status = "plafondcriterium R1>=0,60 gehaald"
    else:
        pos = [c for c in cellen if c["dF1_mean"] > 0 and c["R1"] is not None]
        keuze = sorted(pos, key=lambda c: -c["R1"])[0] if pos else None
        status = "GEEN cel haalt R1>=0,60 — hoogste R1 onder dF1>0 (plafond niet gehaald)"
    uit = {"pad": str(pad), "n_nachten": len(nachten), "cellen": cellen, "keuze": keuze, "status": status}
    Path(pad).with_name("orakel_rule1a_poort_sweep.json").write_text(json.dumps(uit, indent=1))
    print("\n  min_red max_dur min_loc |  kept   R1    dF1mean dF1med beter")
    for c in sorted(cellen, key=lambda c: -(c["R1"] or 0))[:18]:
        print(f"  {str(c['min_red']):>7} {str(c['max_dur']):>7} {str(c['min_local']):>7} | "
              f"{c['n_kept']:5d} {c['R1'] if c['R1'] is None else round(c['R1'],3)!s:>6} "
              f"{c['dF1_mean']:+.4f} {c['dF1_median']:+.4f} {c['n_beter']:3d}/{c['n_nachten']}")
    z = [c for c in cellen if c["knoppen"] == 0][0]
    print(f"\n  zonder poort: kept {z['n_kept']} R1 {z['R1']:.3f} dF1 {z['dF1_mean']:+.4f}")
    print(f"\nSTATUS: {status}\nKEUZE: {keuze}")


if __name__ == "__main__":
    main(sys.argv[1])
