#!/usr/bin/env python3
"""Event-niveau-vergelijking van gedetecteerde events met een referentie.

    python bench/evaluate.py pred.csv ref.csv                       # projectmatcher (IoU 0,20)
    python bench/evaluate.py pred.json ref.csv --matcher iou --threshold 0.5
    python bench/evaluate.py pred.csv ref.csv --matcher fraction --threshold 0.5 --type-aware
    python bench/evaluate.py pred.csv ref.csv --matcher onset --threshold 5 --json uit.json
    python bench/evaluate.py --selftest

Invoer (pred en ref): CSV met kolommen `onset_s,offset_s[,type]` (of
`onset_s,duration_s`), of JSON: lijst van dicts met dezelfde sleutels of
lijst van [onset, offset(, type)].

Overlapcriterium (`--matcher`, drempel via `--threshold`):
  project   de matcher van de validatieharnassen (validate_psgipa.match_events,
            LEGACY_MATCHER: IoU 0,20, typeonbewust) — voor vergelijkbaarheid
            met alle gepubliceerde cijfers van dit project; --threshold en
            --type-aware worden dan aan die functie doorgegeven.
  iou       intersectie/unie >= drempel (default 0,20)
  fraction  overlap / referentieduur >= drempel (sensitiviteitsstijl)
  onset     |onset_pred - onset_ref| <= drempel seconden
  any       elke overlap > 0
Matching is één-op-één, gretig op aflopende score (bij `onset`: oplopende
afstand). Uitvoer: TP/FP/FN, sensitiviteit (recall), PPV (precisie), F1,
en per type als beide bestanden een type dragen.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
REPO = HIER.parent


def lees_events(pad: Path) -> list[tuple[float, float, str | None]]:
    """(onset, offset, type|None) uit CSV of JSON; offset uit duration als nodig."""
    def norm(d: dict) -> tuple[float, float, str | None]:
        on = float(d.get("onset_s", d.get("onset")))
        if d.get("offset_s") is not None or d.get("offset") is not None:
            off = float(d.get("offset_s", d.get("offset")))
        else:
            off = on + float(d.get("duration_s", d.get("duration")))
        t = d.get("type")
        return on, off, (str(t) if t not in (None, "") else None)

    tekst = pad.read_text(encoding="utf-8")
    if pad.suffix.lower() == ".json":
        data = json.loads(tekst)
        out = []
        for e in data:
            if isinstance(e, dict):
                out.append(norm(e))
            else:
                on, off = float(e[0]), float(e[1])
                out.append((on, off, str(e[2]) if len(e) > 2 and e[2] not in (None, "") else None))
    else:
        rows = list(csv.DictReader(tekst.splitlines()))
        out = [norm(r) for r in rows if any((v or "").strip() for v in r.values())]
    slecht = [e for e in out if e[1] < e[0]]
    if slecht:
        raise SystemExit(f"{pad}: {len(slecht)} events met offset < onset, bv. {slecht[0]}")
    return sorted(out)


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _score(p, r, matcher):
    inter = max(0.0, min(p[1], r[1]) - max(p[0], r[0]))
    if matcher == "iou":
        return _iou(p, r)
    if matcher == "fraction":
        d = r[1] - r[0]
        return inter / d if d > 0 else 0.0
    if matcher == "any":
        return 1.0 if inter > 0 else 0.0
    if matcher == "onset":
        return -abs(p[0] - r[0])  # hoger = beter
    raise ValueError(matcher)


def match(pred, ref, matcher="iou", threshold=0.20, type_aware=False):
    """Gretige één-op-één-matching; geeft (tp, fp, fn, paren)."""
    kandidaten = []
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            if type_aware and p[2] is not None and r[2] is not None and p[2] != r[2]:
                continue
            s = _score(p, r, matcher)
            ok = (-s <= threshold) if matcher == "onset" else (s >= threshold and s > 0)
            if ok:
                kandidaten.append((s, i, j))
    kandidaten.sort(key=lambda x: (-x[0], x[1], x[2]))
    gebruikt_p, gebruikt_r, paren = set(), set(), []
    for s, i, j in kandidaten:
        if i in gebruikt_p or j in gebruikt_r:
            continue
        gebruikt_p.add(i); gebruikt_r.add(j); paren.append((i, j))
    tp = len(paren)
    return tp, len(pred) - tp, len(ref) - tp, paren


def _project_match(pred, ref, threshold, type_aware):
    sys.path.insert(0, str(REPO))
    from validate_psgipa import LEGACY_MATCHER, match_events  # noqa: E402
    kw = dict(LEGACY_MATCHER)
    if threshold is not None:
        kw["iou_thresh"] = float(threshold)
    kw["type_aware"] = bool(type_aware)
    m = match_events([(a, b, t or "event") for a, b, t in pred],
                     [(a, b, t or "event") for a, b, t in ref], **kw)
    return int(m["tp"]), int(m["fp"]), int(m["fn"]), None


def metriek(tp, fp, fn):
    sens = tp / (tp + fn) if tp + fn else None
    ppv = tp / (tp + fp) if tp + fp else None
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    return {"tp": tp, "fp": fp, "fn": fn, "sensitiviteit": sens, "ppv": ppv, "f1": f1}


def evalueer(pred, ref, matcher, threshold, type_aware):
    if matcher == "project":
        tp, fp, fn, _ = _project_match(pred, ref, threshold, type_aware)
    else:
        tp, fp, fn, _ = match(pred, ref, matcher, threshold, type_aware)
    uit = {"totaal": metriek(tp, fp, fn), "per_type": {}}
    types = sorted({t for *_, t in ref if t} | {t for *_, t in pred if t})
    if types:
        for t in types:
            p_t = [e for e in pred if e[2] == t]; r_t = [e for e in ref if e[2] == t]
            if matcher == "project":
                a, b, c, _ = _project_match(p_t, r_t, threshold, False)
            else:
                a, b, c, _ = match(p_t, r_t, matcher, threshold, False)
            uit["per_type"][t] = metriek(a, b, c)
    return uit


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def toon(uit, label=""):
    t = uit["totaal"]
    print(f"{label}TP {t['tp']}  FP {t['fp']}  FN {t['fn']}  | sens {_fmt(t['sensitiviteit'])}  "
          f"PPV {_fmt(t['ppv'])}  F1 {_fmt(t['f1'])}")
    for k, m in uit["per_type"].items():
        print(f"  {k:>14s}: TP {m['tp']:4d} FP {m['fp']:4d} FN {m['fn']:4d} | sens {_fmt(m['sensitiviteit'])} "
              f"PPV {_fmt(m['ppv'])} F1 {_fmt(m['f1'])}")


def selftest():
    ref = [(10, 30, "hypopnea"), (100, 120, "obstructive"), (200, 215, "hypopnea")]
    pred = [(12, 31, "hypopnea"), (101, 119, "central"), (300, 320, "hypopnea")]
    tp, fp, fn, _ = match(pred, ref, "iou", 0.20)
    assert (tp, fp, fn) == (2, 1, 1), (tp, fp, fn)
    tp, fp, fn, _ = match(pred, ref, "iou", 0.20, type_aware=True)
    assert (tp, fp, fn) == (1, 2, 2), (tp, fp, fn)          # central ≠ obstructive
    tp, fp, fn, _ = match(pred, ref, "onset", 1.5)
    assert (tp, fp, fn) == (1, 2, 2), (tp, fp, fn)          # alleen |101-100| ≤ 1,5
    tp, fp, fn, _ = match(pred, ref, "fraction", 0.5)
    assert (tp, fp, fn) == (2, 1, 1)
    tp, fp, fn, _ = match([(10, 30, None), (11, 29, None)], [(10, 30, None)], "iou", 0.2)
    assert (tp, fp, fn) == (1, 1, 0), "één-op-één: tweede predictie mag niet opnieuw matchen"
    m = metriek(2, 1, 1); assert abs(m["f1"] - 2 / 3) < 1e-9
    try:
        a, b, c, _ = _project_match(pred, ref, None, False)
        assert (a, b, c) == (2, 1, 1), (a, b, c)
        print("selftest OK (incl. projectmatcher)")
    except ImportError:
        print("selftest OK (projectmatcher niet beschikbaar buiten de repo)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred", nargs="?", type=Path)
    ap.add_argument("ref", nargs="?", type=Path)
    ap.add_argument("--matcher", choices=("project", "iou", "fraction", "onset", "any"), default="project")
    ap.add_argument("--threshold", type=float, default=None,
                    help="IoU/fractie-drempel of onset-tolerantie in s (default: project/iou 0,20, fraction 0,5, onset 5)")
    ap.add_argument("--type-aware", action="store_true", help="alleen gelijke types mogen matchen")
    ap.add_argument("--json", type=Path, default=None, help="schrijf de metrieken ook als JSON")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not (a.pred and a.ref):
        ap.error("geef pred en ref, of --selftest")
    thr = a.threshold
    if thr is None:
        thr = {"project": 0.20, "iou": 0.20, "fraction": 0.5, "onset": 5.0, "any": 0.0}[a.matcher]
    pred, ref = lees_events(a.pred), lees_events(a.ref)
    uit = evalueer(pred, ref, a.matcher, thr, a.type_aware)
    uit["opzet"] = {"pred": str(a.pred), "ref": str(a.ref), "n_pred": len(pred), "n_ref": len(ref),
                    "matcher": a.matcher, "threshold": thr, "type_aware": a.type_aware}
    print(f"{a.pred.name} vs {a.ref.name}  (n_pred {len(pred)}, n_ref {len(ref)}, "
          f"matcher {a.matcher} @ {thr}{', type-aware' if a.type_aware else ''})")
    toon(uit)
    if a.json:
        a.json.write_text(json.dumps(uit, indent=1)); print(f"-> {a.json}")


if __name__ == "__main__":
    main()
