#!/usr/bin/env python3
"""
compare_baseline_arousal.py — fase 4: de viervoudige vergelijking.

{rolling, pre_event} x {arousal-tak uit, aan} op PSG-IPA.

PRIMAIRE UITKOMSTMAAT is niet de AHI maar de **event-level F1 tegenover de
menselijke scoorders**, en vooral het **percentiel** daarvan binnen de
paarsgewijze mens-mens-verdeling. Een betere AHI-bias kan ontstaan door
elkaar opheffende fouten; het percentiel kan dat niet.

De mens-baseline is scoorder-tegen-scoorder en hangt dus niet van de
configuratie af — hij wordt één keer berekend en hergebruikt, met exact
dezelfde matcher-instellingen als de algoritme-F1 (anders is de vergelijking
betekenisloos).

Let op bij het lezen van de tabel: de kolom "arousal uit" is NIET identiek
aan het gedrag van vóór dit traject. De doorgifte-reparatie laat RERA/RDI nu
wél rekenen, ook met de tak uit. AHI is wel vergelijkbaar.

Gebruik:
    python scripts/compare_baseline_arousal.py --data-dir ~/PSG-IPA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mne.set_log_level("ERROR")

from validate_psgipa import (  # noqa: E402
    LEGACY_MATCHER, event_set, find_scorer_files, human_baseline,
    match_events, parse_scorer_file, percentile_of, severity,
)

RECORDINGS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
CONFIGS = [
    ("rolling",   False),
    ("rolling",   True),
    ("pre_event", False),
    ("pre_event", True),
]


def _algo_events(res, dur):
    return [(float(e["onset_s"]), float(e["onset_s"]) + float(e["duration_s"]),
             e.get("type"))
            for e in res.get("respiratory", {}).get("events", [])
            if e.get("onset_s") is not None and e.get("duration_s") is not None
            and float(e["onset_s"]) < dur]


def one(args):
    sn, data_dir, mode, limb = args
    os.environ["PSGSCORING_BASELINE_MODE"] = mode
    os.environ["PSGSCORING_RULE1A_AROUSAL"] = "1" if limb else "0"
    import psgscoring

    data_dir = Path(data_dir)
    raw = mne.io.read_raw_edf(
        str(data_dir / "Resp_events" / "PSG" / f"{sn}_Respiration.edf"),
        preload=True, verbose=False)
    dur = float(raw.times[-1])
    files = find_scorer_files(data_dir, sn)

    ref_ahis, hypno = [], None
    for i, f in enumerate(files):
        ahi, _t, h = parse_scorer_file(f, dur)
        if ahi is not None:
            ref_ahis.append(ahi)
            if i == 0:
                hypno = h
    res = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile="aasm_v3_rec")
    resp = res.get("respiratory", {})
    summ = resp.get("summary", {})
    algo = _algo_events(res, dur)

    f1s = []
    for f in files:
        ref = event_set(f, dur)
        if ref:
            f1s.append(match_events(algo, ref, **LEGACY_MATCHER)["f1"])

    rej = resp.get("rejected_hypopneas", []) or []
    reasons = Counter()
    for r in rej:
        k = str(r.get("reject_reason", "?"))
        reasons["local_reduction" if k.startswith("local_reduction")
                else "pre_event_reduction" if k.startswith("pre_event")
                else k.split("_")[0]] += 1

    ref_med = float(median(ref_ahis)) if ref_ahis else float("nan")
    ahi = float(summ.get("ahi_total") or 0.0)
    st = resp.get("rule1a_arousal_stats", {}) or {}
    return {
        "recording": sn, "mode": mode, "limb": limb,
        "algo_f1_median": float(median(f1s)) if f1s else None,
        "f1_list": f1s,
        "ahi": ahi, "ref_median": ref_med, "delta": ahi - ref_med,
        "sev_match": severity(ahi) == severity(ref_med),
        "rdi": summ.get("rdi"),
        "n_hyp": sum(1 for e in algo if "hypopnea" in str(e[2])),
        "n_rejected": len(rej),
        "reject_reasons": dict(reasons),
        "arousal_qualified": st.get("n_qualified", 0),
        "arousal_coupled": st.get("n_arousal_coupled", 0),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="~/PSG-IPA")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()
    data_dir = str(Path(args.data_dir).expanduser())

    # ── mens-baseline: configuratie-onafhankelijk, één keer ────────
    print("Mens-mens baseline (zelfde matcher als het algoritme)...", flush=True)
    human = {}
    for sn in RECORDINGS:
        raw = mne.io.read_raw_edf(
            str(Path(data_dir) / "Resp_events" / "PSG" / f"{sn}_Respiration.edf"),
            preload=False, verbose=False)
        dur = float(raw.times[-1])
        hb = human_baseline(find_scorer_files(data_dir, sn), dur, **LEGACY_MATCHER)
        human[sn] = hb
        s = hb["summary"]
        print(f"  {sn}: mediaan {s['median']:.3f}  IQR [{s['p25']:.3f}-{s['p75']:.3f}]  "
              f"n={s['n_pairs']} paren", flush=True)

    all_rows = []
    for mode, limb in CONFIGS:
        jobs = [(sn, data_dir, mode, limb) for sn in RECORDINGS]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(one, jobs))
        for r in rows:
            pairs = [p["f1"] for p in human[r["recording"]]["pairs"]]
            r["percentile"] = percentile_of(r["algo_f1_median"], pairs)
            r["human_median"] = human[r["recording"]]["summary"]["median"]
        all_rows += rows
        print(f"  klaar: baseline={mode} arousal={'aan' if limb else 'uit'}", flush=True)

    # ── rapport ────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("FASE 4 — viervoudige vergelijking (PSG-IPA, aasm_v3_rec, legacy matcher)")
    print("=" * 96)
    print(f"\n{'baseline':10s} {'arousal':8s} {'F1 med':>7s} {'pct':>5s} "
          f"{'bias':>7s} {'MAE':>6s} {'sev':>4s} {'hyp':>5s} {'qual':>5s} {'Fix6':>6s}")
    print("-" * 96)
    summary = {}
    for mode, limb in CONFIGS:
        rows = [r for r in all_rows if r["mode"] == mode and r["limb"] == limb]
        f1 = float(median([r["algo_f1_median"] for r in rows if r["algo_f1_median"] is not None]))
        pct = float(median([r["percentile"] for r in rows if r["percentile"] is not None]))
        deltas = [r["delta"] for r in rows]
        bias = float(np.mean(deltas)); mae = float(np.mean(np.abs(deltas)))
        sev = sum(1 for r in rows if r["sev_match"])
        hyp = sum(r["n_hyp"] for r in rows)
        qual = sum(r["arousal_qualified"] for r in rows)
        fix6 = sum(r["reject_reasons"].get("local_reduction", 0) for r in rows)
        key = f"{mode}/{'on' if limb else 'off'}"
        summary[key] = {"f1_median": f1, "percentile": pct, "bias": bias,
                        "mae": mae, "severity_match": sev, "n_hypopneas": hyp,
                        "arousal_qualified": qual, "fix6_rejections": fix6}
        print(f"{mode:10s} {'aan' if limb else 'uit':8s} {f1:7.3f} p{pct:<4.0f} "
              f"{bias:+7.2f} {mae:6.2f} {sev:>2d}/5 {hyp:5d} {qual:5d} {fix6:6d}")

    print(f"\n{'':10s} mens-mediaan F1 per opname: " +
          "  ".join(f"{sn} {human[sn]['summary']['median']:.3f}" for sn in RECORDINGS))
    print("\nPer opname, F1 (percentiel):")
    print(f"{'rec':5s} " + "".join(f"{m[:4]}/{'on' if l else 'off':<3s}".rjust(14)
                                   for m, l in CONFIGS))
    for sn in RECORDINGS:
        cells = []
        for mode, limb in CONFIGS:
            r = next(x for x in all_rows if x["recording"] == sn
                     and x["mode"] == mode and x["limb"] == limb)
            cells.append(f"{r['algo_f1_median']:.3f} (p{r['percentile']:.0f})".rjust(14))
        print(f"{sn:5s} " + "".join(cells))

    if args.output_json:
        args.output_json.write_text(json.dumps(
            {"rows": all_rows, "summary": summary,
             "human": {k: v["summary"] for k, v in human.items()}}, indent=2))
        print(f"\nJSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
