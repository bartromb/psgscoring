#!/usr/bin/env python3
"""Alle kandidaten tegen de PSG-IPA-referentie met bench/evaluate.py.

    .venv/bin/python bench/eeg/common/evaluate_all.py

Twee matchers, exact zoals de opdracht: `project` (validate_psgipa.match_events,
IoU 0,20, typeonbewust) en `onset` (|Δonset| ≤ 5 s). Per opname én gepoold
(gepoold = som van TP/FP/FN over de vijf opnames). Schrijft
bench/eeg/results/resultaten.json en tabel.md.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bench"))
from evaluate import lees_events, evalueer  # noqa: E402

E = REPO / "bench" / "eeg"; REF = E / "ref"; RES = E / "results"
SNS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
MENS = 0.679  # menselijk plafond arousal-F1 PSG-IPA, 330 scoorderparen (docs/arousal_menselijk_plafond.md)
CANDS = {
    "baseline: psgscoring 0.34.2 aasm_v3_rec": "baseline/{sn}_pred.csv",
    "DeepSleep2 model_2 (voorgetraind, τ 0,50)": "deepsleep2/out/{sn}_pred.csv",
    "MSED splitstream (voorgetraind, C3:=C4-M1)": "msed/out_dup/{sn}_pred.csv",
    "MSED splitstream (voorgetraind, C3:=F4-M1)": "msed/out_f4/{sn}_pred.csv",
    "U-Net-50Hz (eigen, MESA-getraind, τ uit MESA-validatie)": "unet50/out/{sn}_pred.csv",
    "DeepSleep2 (τ uit MESA-val, zonder verschuiving)": "deepsleep2/out_mesacal_raw/{sn}_pred.csv",
    "DeepSleep2 (τ + onset/offset-verschuiving uit MESA-val)": "deepsleep2/out_mesacal_corr/{sn}_pred.csv",
}
# MSED met de op MESA-val gekozen drempel (results/mesa_val.json), als die er is
_mv = RES / "mesa_val.json"
if _mv.exists():
    _t = json.loads(_mv.read_text()).get("msed_best_thr_mesa")
    if _t is not None:
        CANDS[f"MSED splitstream (C3:=C4-M1, τ {_t:.2f} uit MESA-val)"] = "msed/out_sweep_psgipa/{sn}_t%.2f.csv" % _t
SWEEPS = {
    "DeepSleep2 drempelveeg": ("deepsleep2/out/{sn}_pred_t{thr:.2f}.csv",
                               [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]),
    "U-Net-50Hz drempelveeg": ("unet50/out/{sn}_pred_t{thr:.2f}.csv", [round(0.15 + 0.05 * i, 2) for i in range(15)]),
    "MSED drempelveeg (C3:=C4-M1)": ("msed/out_sweep_psgipa/{sn}_t{thr:.2f}.csv", [0.30, 0.40, 0.50, 0.55, 0.60, 0.64, 0.70, 0.75, 0.80, 0.85, 0.90]),
}
MATCHERS = [("project", 0.20), ("onset", 5.0)]


def _m(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "sens": tp / (tp + fn) if tp + fn else None,
            "ppv": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def score(pattern: str, thr=None):
    per, pooled = {}, {m: [0, 0, 0] for m, _ in MATCHERS}
    n_pred_tot = n_ref_tot = 0
    for sn in SNS:
        f = E / pattern.format(sn=sn, thr=thr)
        if not f.exists():
            return None
        pred, ref = lees_events(f), lees_events(REF / f"{sn}_ref_arousal.csv")
        n_pred_tot += len(pred); n_ref_tot += len(ref)
        per[sn] = {"n_pred": len(pred), "n_ref": len(ref)}
        for m, t in MATCHERS:
            r = evalueer(pred, ref, m, t, False)["totaal"]
            per[sn][m] = r
            for i, k in enumerate(("tp", "fp", "fn")):
                pooled[m][i] += r[k]
    return {"per_sn": per, "pooled": {m: _m(*v) for m, v in pooled.items()},
            "n_pred": n_pred_tot, "n_ref": n_ref_tot, "count_ratio": n_pred_tot / n_ref_tot}


def f(v, d=3):
    return "—" if v is None else f"{v:.{d}f}".replace(".", ",")


def main():
    RES.mkdir(exist_ok=True)
    res = {name: score(p) for name, p in CANDS.items()}
    sweeps = {}
    for name, (p, thrs) in SWEEPS.items():
        rows = {f"{t:.2f}": score(p, t) for t in thrs}
        sweeps[name] = {k: v for k, v in rows.items() if v}
    (RES / "resultaten.json").write_text(json.dumps({"kandidaten": res, "vegen": sweeps, "mens_plafond": MENS}, indent=1))

    L = [f"# Resultaten (menselijk plafond arousal-F1 PSG-IPA: {f(MENS)}, 330 scoorderparen)\n"]
    L.append("## Gepoold over SN1–SN5 (som van TP/FP/FN)\n")
    L.append("| kandidaat | n_pred | n_ref | ratio | IoU-0,20 sens | PPV | **F1 gepoold** | mediaan F1/opname | onset-5 s sens | PPV | F1 | mens |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in res.items():
        if not r:
            L.append(f"| {name} | *(niet beschikbaar)* ||||||||||| {f(MENS)} |"); continue
        a, b = r["pooled"]["project"], r["pooled"]["onset"]
        med = sorted((r["per_sn"][sn]["project"]["f1"] or 0.0) for sn in SNS)[2]
        L.append(f"| {name} | {r['n_pred']} | {r['n_ref']} | {f(r['count_ratio'],2)} | {f(a['sens'])} | {f(a['ppv'])} | **{f(a['f1'])}** | {f(med)} | "
                 f"{f(b['sens'])} | {f(b['ppv'])} | {f(b['f1'])} | {f(MENS)} |")
    for m, lab in (("project", "IoU 0,20 (projectmatcher)"), ("onset", "onset ±5 s")):
        L.append(f"\n## Per opname — {lab}: sens / PPV / F1\n")
        L.append("| kandidaat | " + " | ".join(SNS) + " |"); L.append("|---|" + "---|" * len(SNS))
        for name, r in res.items():
            if not r:
                continue
            cells = []
            for sn in SNS:
                x = r["per_sn"][sn][m]
                cells.append(f"{f(x['sensitiviteit'],2)} / {f(x['ppv'],2)} / **{f(x['f1'])}** (n {r['per_sn'][sn]['n_pred']})")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
    base = res.get("baseline: psgscoring 0.34.2 aasm_v3_rec")
    if base:
        L.append("\n## Gepaard per opname tegen de baseline (F1, IoU 0,20)\n")
        L.append("| kandidaat | " + " | ".join(f"Δ{sn}" for sn in SNS) + " | beter op | gepoold ΔF1 |")
        L.append("|---|" + "---|" * (len(SNS) + 2))
        for name, r in res.items():
            if not r or r is base:
                continue
            ds = [(r["per_sn"][sn]["project"]["f1"] or 0) - (base["per_sn"][sn]["project"]["f1"] or 0) for sn in SNS]
            L.append(f"| {name} | " + " | ".join(f"{d:+.3f}".replace(".", ",") for d in ds) +
                     f" | {sum(d > 0 for d in ds)}/5 | {(r['pooled']['project']['f1'] - base['pooled']['project']['f1']):+.3f} |".replace(".", ","))
    for name, rows in sweeps.items():
        if not rows:
            continue
        L.append(f"\n## {name} — ORAKEL op PSG-IPA (drempel gekozen mét kennis van de referentie; géén validatiecijfer)\n")
        L.append("| drempel | n_pred | ratio | sens | PPV | F1 (IoU 0,20) | F1 (onset 5 s) |"); L.append("|---:|---:|---:|---:|---:|---:|---:|")
        for t, r in rows.items():
            a, b = r["pooled"]["project"], r["pooled"]["onset"]
            L.append(f"| {t.replace('.', ',')} | {r['n_pred']} | {f(r['count_ratio'],2)} | {f(a['sens'])} | {f(a['ppv'])} | {f(a['f1'])} | {f(b['f1'])} |")
    (RES / "tabel.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
