#!/usr/bin/env python3
"""Herscoring van de bewaarde U-Net-replicatie (27-09-2026) met twee koppelregels.

Leest alleen de bewaarde eventlijsten van de replicatie — geen EDF, geen model:
  /srv/CODE/docs/arousal_unet_20260927/out/{psgipa,shhs,mesa}/{rec}_{ref,base,unet}.csv
  /srv/CODE/docs/arousal_unet_20260927/out/{cohort}/{rec}_hypno.json   (hypno, dur_s, tst_h)
en koppelt met `psgscoring.agreement.compare_event_sets` (bibliotheek, niet nagebouwd):
  matcher "iou"  — IoU ≥ 0,20, projectstandaard
  matcher "any"  — elke overlap > 0 s, de regel van Pitkänen 2026 / Ehrlich 2024
Per nacht en arm: event-F1 per matcher, 1 s-κ over de hele opname én over de slaapseconden
(hypnogram van de replicatie), mediane begin-/eindverschuiving (voorspelling − referentie,
paren onder "any"), mediane eventduur. Uitvoer: herscoor.json naast dit script.

Geen PHI: NSRR-id's en PSG-IPA-nummers zijn publieke datasetsleutels.
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
from statistics import median

sys.path.insert(0, "/srv/CODE/psgscoring")
from psgscoring.agreement import compare_event_sets, _mask_1s  # noqa: E402

UIT_DIR = Path("/srv/CODE/docs/arousal_unet_20260927/out")
HIER = Path(__file__).resolve().parent
SLEEP = {"N1", "N2", "N3", "R"}
COHORTEN = ("psgipa", "shhs", "mesa")
ARMEN = ("base", "unet")


def lees(p: Path) -> list[dict]:
    with p.open() as fh:
        rows = list(csv.DictReader(fh))
    uit = []
    for r in rows:
        o, e = float(r["onset_s"]), float(r["offset_s"])
        uit.append({"onset_s": o, "duration_s": max(0.0, e - o), "type": "arousal"})
    return uit


def kappa_masked(a: list[dict], b: list[dict], n: int, keep: list[bool]) -> float | None:
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


def f1_of(r: dict) -> float | None:
    s = 2 * r["n_shared"]
    return s / (r["n_a"] + r["n_b"]) if (r["n_a"] + r["n_b"]) else None


def nacht(cohort: str, rec: str) -> dict | None:
    d = UIT_DIR / cohort
    meta = json.loads((d / f"{rec}_hypno.json").read_text())
    ref = lees(d / f"{rec}_ref.csv")
    if not ref:
        return None
    dur = float(meta["dur_s"]); hyp = meta["hypno"]
    n = int(math.ceil(dur))
    slaap = [0 <= (t // 30) < len(hyp) and hyp[int(t // 30)] in SLEEP for t in range(n)]
    rij = {"n_ref": len(ref), "tst_h": meta["tst_h"], "dur_s": dur,
           "ref_median_dur": median(e["duration_s"] for e in ref), "armen": {}}
    for arm in ARMEN:
        pred = lees(d / f"{rec}_{arm}.csv")
        r = {"n": len(pred), "count_ratio": len(pred) / len(ref)}
        for m in ("iou", "any"):
            c = compare_event_sets(ref, pred, matcher=m, duration_s=dur)   # a = ref, b = pred
            r[m] = {"tp": c["n_shared"], "fn": c["n_only_a"], "fp": c["n_only_b"],
                    "f1": f1_of(c),
                    "onset_shift_s": c["onset_shift_median_s"],     # pred − ref
                    "offset_shift_s": c["offset_shift_median_s"]}
        r["kappa_1s"] = compare_event_sets(ref, pred, duration_s=dur)["kappa_1s"]
        r["kappa_1s_sleep"] = kappa_masked(ref, pred, n, slaap)
        r["pred_median_dur"] = median(e["duration_s"] for e in pred) if pred else None
        rij["armen"][arm] = r
    return rij


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(median(xs)) if xs else None


def wilcoxon(d):
    try:
        from scipy.stats import wilcoxon as w
    except Exception:
        return None
    d = [x for x in d if x is not None and x != 0]
    if len(d) < 5:
        return None
    return float(w(d).pvalue)


def main():
    uit = {}
    for coh in COHORTEN:
        d = UIT_DIR / coh
        recs = sorted(p.name[:-8] for p in d.glob("*_ref.csv"))
        nachten, zonder = {}, []
        for rec in recs:
            if not all((d / f"{rec}_{s}").exists() for s in ("hypno.json", "base.csv", "unet.csv")):
                continue
            r = nacht(coh, rec)
            (nachten.__setitem__(rec, r) if r else zonder.append(rec))
        R = sorted(nachten)
        S = {"n": len(R), "zonder_referentie": zonder, "ref_median_dur": med(nachten[r]["ref_median_dur"] for r in R),
             "armen": {}, "delta_unet_min_base": {}}
        for arm in ARMEN:
            A = {"n_events": sum(nachten[r]["armen"][arm]["n"] for r in R),
                 "count_ratio_median": med(nachten[r]["armen"][arm]["count_ratio"] for r in R),
                 "kappa_1s_median": med(nachten[r]["armen"][arm]["kappa_1s"] for r in R),
                 "kappa_1s_sleep_median": med(nachten[r]["armen"][arm]["kappa_1s_sleep"] for r in R),
                 "pred_median_dur": med(nachten[r]["armen"][arm]["pred_median_dur"] for r in R)}
            for m in ("iou", "any"):
                tp = sum(nachten[r]["armen"][arm][m]["tp"] for r in R)
                fp = sum(nachten[r]["armen"][arm][m]["fp"] for r in R)
                fn = sum(nachten[r]["armen"][arm][m]["fn"] for r in R)
                A[m] = {"f1_median": med(nachten[r]["armen"][arm][m]["f1"] for r in R),
                        "f1_pooled": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
                        "precision_pooled": tp / (tp + fp) if tp + fp else None,
                        "recall_pooled": tp / (tp + fn) if tp + fn else None,
                        "onset_shift_median_s": med(nachten[r]["armen"][arm][m]["onset_shift_s"] for r in R),
                        "offset_shift_median_s": med(nachten[r]["armen"][arm][m]["offset_shift_s"] for r in R)}
            S["armen"][arm] = A
        for m in ("iou", "any"):
            dd = [(nachten[r]["armen"]["unet"][m]["f1"] or 0) - (nachten[r]["armen"]["base"][m]["f1"] or 0) for r in R]
            S["delta_unet_min_base"][m] = {"median": med(dd), "beter": sum(x > 0 for x in dd),
                                           "slechter": sum(x < 0 for x in dd), "gelijk": sum(x == 0 for x in dd),
                                           "wilcoxon_p": wilcoxon(dd)}
        dk = [(nachten[r]["armen"]["unet"]["kappa_1s"] or 0) - (nachten[r]["armen"]["base"]["kappa_1s"] or 0) for r in R]
        S["delta_unet_min_base"]["kappa_1s"] = {"median": med(dk), "beter": sum(x > 0 for x in dk),
                                                "slechter": sum(x < 0 for x in dk), "wilcoxon_p": wilcoxon(dk)}
        # vergelijking met de replicatie-eval (project-matcher = validate_psgipa legacy, greedy in a-volgorde)
        ev = json.loads((d / "eval.json").read_text())
        S["replicatie_eval_project"] = {arm: {"f1_median": ev["armen"][arm]["project"]["mediaan_f1"],
                                             "f1_pooled": ev["armen"][arm]["project"]["gepoold"]["f1"]} for arm in ARMEN}
        uit[coh] = {"samenvatting": S, "nachten": nachten}
    (HIER / "herscoor.json").write_text(json.dumps(uit, indent=1))
    for coh in COHORTEN:
        S = uit[coh]["samenvatting"]
        print(f"== {coh} n={S['n']} zonder_ref={len(S['zonder_referentie'])} ref_dur={S['ref_median_dur']:.1f}")
        for arm in ARMEN:
            A = S["armen"][arm]
            print(f"  {arm:5s} iou F1 med {A['iou']['f1_median']:.3f} pool {A['iou']['f1_pooled']:.3f} | "
                  f"any F1 med {A['any']['f1_median']:.3f} pool {A['any']['f1_pooled']:.3f} "
                  f"P {A['any']['precision_pooled']:.3f} R {A['any']['recall_pooled']:.3f} | "
                  f"κ {A['kappa_1s_median']:.3f} κslaap {A['kappa_1s_sleep_median']:.3f} | "
                  f"shift any on {A['any']['onset_shift_median_s']:+.2f} off {A['any']['offset_shift_median_s']:+.2f} | "
                  f"dur {A['pred_median_dur']:.1f} | count {A['count_ratio_median']:.2f}")
        for m in ("iou", "any", "kappa_1s"):
            D = S["delta_unet_min_base"][m]
            print(f"  Δ{m}: med {D['median']:+.3f} beter {D['beter']} slechter {D['slechter']} p={D['wilcoxon_p']}")
        print("  replicatie(project):", S["replicatie_eval_project"])


if __name__ == "__main__":
    main()
