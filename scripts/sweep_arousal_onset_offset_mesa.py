#!/usr/bin/env python3
"""MESA-replicatie van de +2 s onsetverschuiving.

Criteria vooraf: docs/arousal_onsetverschuiving_mesa_preregistratie.md
  1. bij +2 s: gemiddelde paarsgewijze dF1 > 0 EN tekentoets p < 0,05 (>=21/30)
  2. maximum van de reeks op +1/+2/+3 s
  bewaker: telling gelijk over alle verschuivingen

De detectie draait EEN keer per opname; de verschuivingen worden daarna op
dezelfde eventlijst toegepast, zodat elk verschil de verschuiving is.
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean, median
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, mne
mne.set_log_level("ERROR")

# Uitvoermap: PSGSCORING_MEETUITVOER, anders de werkmap. Nooit een
# tijdelijke map -- een harnas in de repo dat naar /tmp schrijft is bij
# de volgende sessie stuk.
_UIT = Path(os.environ.get("PSGSCORING_MEETUITVOER", "."))
sys.path.insert(0, "/home/bart/CODE/psgscoring")
sys.path.insert(0, "/home/bart/CODE/psgscoring/scripts")

MESA = Path("/home/bart/MESA/mesa")
SHIFTS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
N_DEFAULT = 30


def ref_spans(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for ev in root.iter("ScoredEvent"):
        c = (ev.findtext("EventConcept") or "").lower()
        if "arousal" in c:
            try:
                out.append((float(ev.findtext("Start")),
                            max(float(ev.findtext("Duration") or 0.0), 0.1)))
            except (TypeError, ValueError):
                continue
    out.sort()
    return out


def een(stem):
    from validate_mesa import parse_nsrr
    from psgscoring.arousal import detect_arousals_multi
    from psgscoring.pipeline import _pick_eeg_multi
    from psgscoring.agreement import _match

    edf = MESA / "polysomnography" / "edfs" / f"{stem}.edf"
    xml = MESA / "polysomnography" / "annotations-events-nsrr" / f"{stem}-nsrr.xml"
    if not (edf.exists() and xml.exists()):
        return None
    ref = ref_spans(xml)
    if len(ref) < 10:
        return None

    h = mne.io.read_raw_edf(edf, preload=False, verbose=False)
    if "EMG" not in h.ch_names:
        return None
    dur = h.n_times / h.info["sfreq"]
    hypno, _r, _t = parse_nsrr(xml, dur)
    houden = {c for c in h.ch_names
              if c.upper().startswith("EEG") and not c.upper().endswith("_OFF")}
    houden.add("EMG")
    raw = mne.io.read_raw_edf(
        edf, exclude=[c for c in h.ch_names if c not in houden],
        preload=True, verbose=False)
    sf = raw.info["sfreq"]
    emg = raw.get_data(picks=["EMG"])[0]
    derivs = _pick_eeg_multi(raw, {})
    if not derivs:
        return None

    r = detect_arousals_multi(derivs, sf, hypno, emg_data=emg,
                              lgbm=True, lgbm_threshold=0.80)
    ev = r.get("events") or []
    base = [(float(e["onset_s"]), max(float(e.get("duration_s") or 3.0), 0.1))
            for e in ev]
    b = [{"onset_s": o, "duration_s": d, "type": "a"} for o, d in ref]

    uit = {}
    for s in SHIFTS:
        a = [{"onset_s": o + s, "duration_s": d, "type": "a"} for o, d in base]
        if not a:
            uit[s] = 0.0
            continue
        pairs, _oa, _ob = _match(a, b, 0.20)
        tp = len(pairs); fp = len(a) - tp; fn = len(b) - tp
        uit[s] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"rec": stem, "f1": uit, "n_ev": len(ev), "n_ref": len(ref),
            "derivs": [n for n, _d, _s in derivs]}


if __name__ == "__main__":
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    stems = [p.stem for p in sorted(
        (MESA / "polysomnography" / "edfs").glob("mesa-sleep-*.edf"))]
    res = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = {}
        it = iter(stems)
        for _ in range(min(len(stems), n_want + 12)):   # marge voor uitvallers
            try: futs[pool.submit(een, next(it))] = None
            except StopIteration: break
        for f in as_completed(list(futs)):
            try:
                r = f.result()
            except Exception as e:
                print(f"FOUT {type(e).__name__}: {e}", flush=True); continue
            if r is None:
                continue
            res.append(r)
            print(f"{r['rec'].replace('mesa-sleep-','')} ok  "
                  f"ev {r['n_ev']:>4} ref {r['n_ref']:>4}  "
                  f"F1@0 {r['f1'][0.0]:.3f} F1@+2 {r['f1'][2.0]:.3f} "
                  f"[{len(r['derivs'])} afl]", flush=True)
    res.sort(key=lambda r: r["rec"])   # niet op wie het eerst klaar was:
    res = res[:n_want]                  # dat zou snel-ladende opnames bevoordelen
    n = len(res)
    print(f"\nn = {n}\n")
    print(f"{'verschuiving':<14}{'mediane F1':>12}{'gem. Δ':>10}{'beter':>9}")
    print("-" * 45)
    for s in SHIFTS:
        ff = [r["f1"][s] for r in res]
        dd = [r["f1"][s] - r["f1"][0.0] for r in res]
        beter = sum(1 for x in dd if x > 0)
        merk = "  <- nu" if s == 0.0 else ("  <- besluit" if s == 2.0 else "")
        print(f"{s:>+8.0f} s     {median(ff):>12.4f}{mean(dd):>+10.4f}"
              f"{beter:>6}/{n}{merk}")

    dd2 = [r["f1"][2.0] - r["f1"][0.0] for r in res]
    k = sum(1 for x in dd2 if x > 0)
    # exacte tweezijdige tekentoets, alleen niet-nul verschillen
    nz = sum(1 for x in dd2 if x != 0)
    from math import comb
    staart = max(k, nz - k)
    p = (min(1.0, 2 * sum(comb(nz, i) for i in range(staart, nz + 1)) / 2 ** nz)
         if nz else 1.0)
    top = max(SHIFTS, key=lambda s: mean([r["f1"][s] - r["f1"][0.0] for r in res]))
    print(f"\nVooraf: +2 s moet gem. Δ > 0 én tekentoets p < 0,05, "
          f"én maximum op +1/+2/+3.")
    print(f"  gemeten: Δ {mean(dd2):+.4f}, beter op {k}/{n} "
          f"(niet-nul {nz}), tekentoets p = {p:.2}")
    print(f"  maximum van de reeks op {top:+.0f} s")
    json.dump(res, open(str(_UIT / "arousal_onsetverschuiving_mesa.json"),
                        "w"), indent=1, default=float)
    print("\nwegschreven: docs/arousal_onsetverschuiving_mesa.json")
