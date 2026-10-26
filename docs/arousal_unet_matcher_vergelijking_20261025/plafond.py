#!/usr/bin/env python3
"""Het menselijke arousalplafond op PSG-IPA onder twee koppelregels en als 1 s-κ.

Herneemt docs/arousal_menselijk_plafond.md (25-08-2026: mediane event-F1 0,679 over
330 scoorderparen, IoU 0,20) met dezelfde bron en dezelfde bibliotheekmatcher, en
voegt toe: F1 met elke overlap, 1 s-κ over de hele opname en over de slaapseconden
van het meerderheidshypnogram van de twaalf scoorders.

Bron (leest alleen): /srv/DATA/PSG-IPA/EEG_arousals/Annotations/manual/SNx_EEGarousals_manual_scorer{1..12}.txt
  arousal = regel met "arousal" in de annotatiekolom; duur 0 → 3 s (zoals
  scripts/sweep_event_locked_window._psgipa_spans); opnameduur uit de EDF-kop.
Uitvoer: plafond.json naast dit script. Publieke dataset, geen PHI.
"""
from __future__ import annotations
import json, math, sys
from itertools import combinations
from pathlib import Path
from statistics import median

import mne
mne.set_log_level("ERROR")
sys.path.insert(0, "/srv/CODE/psgscoring")
from psgscoring.agreement import compare_event_sets, _mask_1s  # noqa: E402

ROOT = Path("/srv/DATA/PSG-IPA/EEG_arousals")
HIER = Path(__file__).resolve().parent
SNS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
SLEEP = {"N1", "N2", "N3", "R"}
STAGES = [("stage n3", "N3"), ("stage 3", "N3"), ("stage 4", "N3"), ("stage n2", "N2"), ("stage 2", "N2"),
          ("stage n1", "N1"), ("stage 1", "N1"), ("stage rem", "R"), ("stage r", "R"), ("stage w", "W"), ("wake", "W")]


def stage(txt: str):
    t = txt.lower()
    for k, v in STAGES:
        if k in t:
            return v
    return None


def lees_scoorder(p: Path, dur: float):
    """(arousals als eventdicts, hypnogram per 30 s)"""
    n_ep = int(math.ceil(dur / 30.0)); h = ["W"] * n_ep; ar = []
    with open(p, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            q = [x.strip() for x in line.split(",")]
            if len(q) < 5:
                continue
            try:
                o, d = float(q[2]), float(q[3])
            except ValueError:
                continue
            if not (0 <= o < dur):
                continue
            if "arousal" in q[4].lower():
                ar.append({"onset_s": o, "duration_s": d or 3.0, "type": "arousal"})
                continue
            st = stage(q[4])
            if st is not None:
                e0 = int(o // 30.0)
                for i in range(max(1, round(d / 30.0))):
                    if 0 <= e0 + i < n_ep:
                        h[e0 + i] = st
    return ar, h


def kappa_masked(a, b, n, keep):
    ma, mb = _mask_1s(a, n), _mask_1s(b, n)
    n11 = n10 = n01 = n00 = 0
    for x, y, k in zip(ma, mb, keep):
        if not k:
            continue
        if x and y: n11 += 1
        elif x: n10 += 1
        elif y: n01 += 1
        else: n00 += 1
    m = n11 + n10 + n01 + n00
    if m == 0:
        return None
    po = (n11 + n00) / m
    pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (m * m)
    return (po - pe) / (1 - pe) if 1 - pe > 0 else None


def f1_of(r):
    s = 2 * r["n_shared"]
    return s / (r["n_a"] + r["n_b"]) if (r["n_a"] + r["n_b"]) else None


def main():
    uit = {"per_sn": {}, "paren": []}
    for sn in SNS:
        hdr = mne.io.read_raw_edf(ROOT / "PSG" / f"{sn}_EEGarousals.edf", preload=False, verbose=False)
        dur = hdr.n_times / hdr.info["sfreq"]; n = int(math.ceil(dur))
        files = sorted((ROOT / "Annotations" / "manual").glob(f"{sn}_EEGarousals_manual_scorer*.txt"),
                       key=lambda p: int(p.stem.rsplit("scorer", 1)[1]))
        sc = {int(p.stem.rsplit("scorer", 1)[1]): lees_scoorder(p, dur) for p in files}
        n_ep = min(len(h) for _, h in sc.values())
        maj = [max(set(col), key=col.count) for col in ([sc[k][1][i] for k in sc] for i in range(n_ep))]
        slaap = [(t // 30) < n_ep and maj[int(t // 30)] in SLEEP for t in range(n)]
        rows = []
        for i, j in combinations(sorted(sc), 2):
            a, b = sc[i][0], sc[j][0]
            ci = compare_event_sets(a, b, matcher="iou", duration_s=dur)
            ca = compare_event_sets(a, b, matcher="any", duration_s=dur)
            rows.append({"sn": sn, "i": i, "j": j, "n_i": len(a), "n_j": len(b),
                         "f1_iou": f1_of(ci), "f1_any": f1_of(ca), "kappa_1s": ci["kappa_1s"],
                         "kappa_1s_sleep": kappa_masked(a, b, n, slaap),
                         "onset_shift_any": ca["onset_shift_median_s"], "offset_shift_any": ca["offset_shift_median_s"]})
        uit["paren"] += rows
        uit["per_sn"][sn] = {"dur_s": dur, "n_scoorders": len(sc), "n_paren": len(rows),
                             "n_arousals_median": median(len(v[0]) for v in sc.values()),
                             "dur_arousal_median": median(e["duration_s"] for v in sc.values() for e in v[0]),
                             "f1_iou_median": median(r["f1_iou"] for r in rows),
                             "f1_any_median": median(r["f1_any"] for r in rows),
                             "kappa_1s_median": median(r["kappa_1s"] for r in rows),
                             "kappa_1s_sleep_median": median(r["kappa_1s_sleep"] for r in rows)}
    P = uit["paren"]
    uit["alle_paren"] = {"n": len(P),
                         "f1_iou_median": median(r["f1_iou"] for r in P),
                         "f1_any_median": median(r["f1_any"] for r in P),
                         "kappa_1s_median": median(r["kappa_1s"] for r in P),
                         "kappa_1s_sleep_median": median(r["kappa_1s_sleep"] for r in P),
                         "f1_iou_range": [min(r["f1_iou"] for r in P), max(r["f1_iou"] for r in P)],
                         "f1_any_range": [min(r["f1_any"] for r in P), max(r["f1_any"] for r in P)]}
    (HIER / "plafond.json").write_text(json.dumps(uit, indent=1))
    for sn, v in uit["per_sn"].items():
        print(f"{sn}: dur {v['dur_s']/3600:.2f} h, {v['n_scoorders']} scoorders, {v['n_paren']} paren, "
              f"arousals med {v['n_arousals_median']:.0f}, duur med {v['dur_arousal_median']:.1f} s | "
              f"F1 iou {v['f1_iou_median']:.3f} any {v['f1_any_median']:.3f} | κ {v['kappa_1s_median']:.3f} "
              f"κslaap {v['kappa_1s_sleep_median']:.3f}")
    A = uit["alle_paren"]
    print(f"ALLE {A['n']} paren: F1 iou {A['f1_iou_median']:.3f} ({A['f1_iou_range'][0]:.3f}–{A['f1_iou_range'][1]:.3f}) "
          f"any {A['f1_any_median']:.3f} ({A['f1_any_range'][0]:.3f}–{A['f1_any_range'][1]:.3f}) | "
          f"κ {A['kappa_1s_median']:.3f} κslaap {A['kappa_1s_sleep_median']:.3f}")


if __name__ == "__main__":
    main()
