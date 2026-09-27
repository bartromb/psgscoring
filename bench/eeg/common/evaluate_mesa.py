#!/usr/bin/env python3
"""MESA-validatienachten: psgscoring vs MSED vs U-Net tegen de NSRR-arousals, gepaard.

    .venv/bin/python bench/eeg/common/evaluate_mesa.py

Per nacht event-F1 met de projectmatcher (IoU 0,20) via bench/evaluate.py; gepoold,
mediaan per nacht, Wilcoxon (gepaard, per nacht) en het aantal nachten beter/slechter
dan psgscoring. MESA is één scoorder per nacht: de absolute F1 ligt lager dan op
PSG-IPA; het gaat om het PAAR. Schrijft bench/eeg/results/mesa_val.{json,md}.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bench"))
from evaluate import lees_events, evalueer  # noqa: E402

E = REPO / "bench" / "eeg"; RES = E / "results"
ARMS = {
    "psgscoring 0.34.2": (E / "baseline_mesa", "{rec}_pred.csv"),
    "MSED τ 0,64 (upstream)": (E / "msed" / "out_sweep_mesa", "{rec}_t0.64.csv"),
    "U-Net-50Hz (τ uit MESA-val)": (E / "unet50" / "out_mesa", "{rec}_pred.csv"),
}
MSED_THRS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.64, 0.70, 0.75, 0.80, 0.85, 0.90]


def per_night(pred_dir, pattern, ref_dir, recs):
    out = {}
    for rec in recs:
        f = pred_dir / pattern.format(rec=rec); r = ref_dir / f"{rec}_ref.csv"
        if not (f.exists() and r.exists()):
            continue
        pred, ref = lees_events(f), lees_events(r)
        m = evalueer(pred, ref, "project", 0.20, False)["totaal"]
        out[rec] = {"tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "f1": m["f1"] or 0.0, "n_pred": len(pred), "n_ref": len(ref)}
    return out


def pooled(d):
    tp = sum(v["tp"] for v in d.values()); fp = sum(v["fp"] for v in d.values()); fn = sum(v["fn"] for v in d.values())
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "sens": tp / (tp + fn) if tp + fn else None, "ppv": tp / (tp + fp) if tp + fp else None,
            "count_ratio": sum(v["n_pred"] for v in d.values()) / max(1, sum(v["n_ref"] for v in d.values()))}


def fmt(v, d=3):
    return "—" if v is None else f"{v:.{d}f}".replace(".", ",")


def main():
    RES.mkdir(exist_ok=True)
    ref_dir = E / "baseline_mesa"
    recs = sorted(p.name[:-9] for p in ref_dir.glob("*_pred.csv"))
    base = per_night(*ARMS["psgscoring 0.34.2"], ref_dir, recs)
    res = {"n_nights": len(recs), "arms": {}, "msed_sweep": {}}
    L = [f"# MESA-validatienachten (n = {len(recs)}, NSRR-referentie, één scoorder per nacht)\n",
         "| arm | n_pred/n_ref | gepoold sens | PPV | **F1** | mediaan F1/nacht | beter/slechter dan psgscoring | Wilcoxon p |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, (d, pat) in ARMS.items():
        pn = per_night(d, pat, ref_dir, recs)
        common = sorted(set(pn) & set(base))
        if not common:
            L.append(f"| {name} | *(niet beschikbaar)* |||||||"); continue
        po = pooled({r: pn[r] for r in common}); med = float(np.median([pn[r]["f1"] for r in common]))
        row = {"pooled": po, "median_f1": med, "n": len(common), "per_night": pn}
        if name != "psgscoring 0.34.2":
            # psgscoring op exact dezelfde nachten als deze arm (n kan kleiner zijn dan 80)
            pb = pooled({r: base[r] for r in common}); medb = float(np.median([base[r]["f1"] for r in common]))
            row["baseline_on_common"] = {"pooled": pb, "median_f1": medb, "n": len(common)}
            L.append(f"| psgscoring 0.34.2 op dezelfde {len(common)} nachten | {sum(base[r]['n_pred'] for r in common)}/{sum(base[r]['n_ref'] for r in common)} | "
                     f"{fmt(pb['sens'])} | {fmt(pb['ppv'])} | **{fmt(pb['f1'])}** | {fmt(medb)} | (referentie voor de rij hieronder) | — |")
            d_ = [pn[r]["f1"] - base[r]["f1"] for r in common]
            try:
                p = float(wilcoxon(d_).pvalue) if any(abs(x) > 0 for x in d_) else 1.0
            except ValueError:
                p = None
            row.update({"delta_median": float(np.median(d_)), "wins": int(sum(x > 0 for x in d_)), "losses": int(sum(x < 0 for x in d_)), "p": p})
            extra = f"{row['wins']}/{row['losses']} (Δmed {fmt(row['delta_median'])}) | {fmt(p, 4) if p is not None else '—'}"
        else:
            extra = "— | —"
        res["arms"][name] = row
        L.append(f"| {name} (n {len(common)}) | {sum(pn[r]['n_pred'] for r in common)}/{sum(pn[r]['n_ref'] for r in common)} | "
                 f"{fmt(po['sens'])} | {fmt(po['ppv'])} | **{fmt(po['f1'])}** | {fmt(med)} | {extra} |")
    L.append("\n## MSED-drempelveeg op MESA-val (hier mag gekozen worden; PSG-IPA blijft schoon)\n")
    L.append("| τ | n_pred/n_ref | sens | PPV | gepoold F1 | mediaan F1/nacht |"); L.append("|---:|---:|---:|---:|---:|---:|")
    for t in MSED_THRS:
        pn = per_night(E / "msed" / "out_sweep_mesa", f"{{rec}}_t{t:.2f}.csv", ref_dir, recs)
        if not pn:
            continue
        po = pooled(pn); med = float(np.median([v["f1"] for v in pn.values()]))
        res["msed_sweep"][f"{t:.2f}"] = {"pooled": po, "median_f1": med, "n": len(pn)}
        L.append(f"| {t:.2f} | {sum(v['n_pred'] for v in pn.values())}/{sum(v['n_ref'] for v in pn.values())} | {fmt(po['sens'])} | {fmt(po['ppv'])} | {fmt(po['f1'])} | {fmt(med)} |".replace("| 0.", "| 0,"))
    if res["msed_sweep"]:
        best = max(res["msed_sweep"], key=lambda k: res["msed_sweep"][k]["pooled"]["f1"] or 0)
        res["msed_best_thr_mesa"] = float(best); L.append(f"\nBeste MSED-drempel op MESA-val (gepoolde F1): **{best.replace('.', ',')}**")
    # tertielen van arousallast (n_ref per nacht), snede op de gesorteerde n_ref van de gemeenschappelijke nachten
    un = res["arms"].get("U-Net-50Hz (τ uit MESA-val)")
    if un:
        common = sorted(set(un["per_night"]) & set(base)); n_ref = np.array([base[r]["n_ref"] for r in common])
        q = np.percentile(n_ref, [33.3, 66.7]); L.append("\n## Per tertiel van arousallast (n_ref per nacht, gemeenschappelijke nachten)\n")
        L.append("| tertiel | n | n_ref-bereik | psgscoring F1 | U-Net F1 | ΔF1 |"); L.append("|---|---:|---|---:|---:|---:|")
        res["tertielen"] = {}
        for lab, msk in (("laag", n_ref <= q[0]), ("midden", (n_ref > q[0]) & (n_ref <= q[1])), ("hoog", n_ref > q[1])):
            idx = [r for r, m_ in zip(common, msk) if m_]
            fb = pooled({r: base[r] for r in idx})["f1"]; fu = pooled({r: un["per_night"][r] for r in idx})["f1"]
            res["tertielen"][lab] = {"n": len(idx), "n_ref_min": int(n_ref[msk].min()), "n_ref_max": int(n_ref[msk].max()), "psgscoring_f1": fb, "unet_f1": fu}
            L.append(f"| {lab} | {len(idx)} | {int(n_ref[msk].min())}–{int(n_ref[msk].max())} | {fmt(fb)} | {fmt(fu)} | {fmt(fu - fb)} |")
    (RES / "mesa_val.json").write_text(json.dumps(res, indent=1)); (RES / "mesa_val.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
