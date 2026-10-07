#!/usr/bin/env python3
"""Lezing volgens docs/resp_unet_preregistratie_20261007.md.
Invoer: out/<cohort>/rows.json (U-Net), out/<cohort>/<rec>_base_<profiel>.csv + baseline_log.jsonl (psgscoring),
out/<cohort>/<rec>_ref.csv (NSRR), ablatie-rows (rows_zero*.json), seed_*/train_log.json, out/psgipa/rows*.json."""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
HIER = Path(__file__).resolve().parent; OUT = HIER / "out"
sys.path.insert(0, str(HIER.parent))
from evaluate import match  # noqa: E402
PLAFOND = {"SN1": 0.826, "SN2": 0.549, "SN3": 0.948, "SN4": 0.553, "SN5": 0.556}


def lees(p):
    with open(p) as fh:
        return [(float(r["onset_s"]), float(r["offset_s"]), r.get("type")) for r in csv.DictReader(fh)]


def f1_of(pred, ref):
    tp, fp, fn, _ = match([(a, b, None) for a, b, _ in pred], [(a, b, None) for a, b, _ in ref], "iou", 0.20)
    return (2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None), tp, fp, fn


def cohort(naam, profielen):
    d = OUT / naam
    rows = {r["rec"]: r for r in json.load(open(d / "rows.json")) if "f1" in r}
    base = {}
    for l in (d / "baseline_log.jsonl").read_text().splitlines() if (d / "baseline_log.jsonl").exists() else []:
        j = json.loads(l)
        if "rec" in j and "error" not in j:
            base[j["rec"]] = j
    print(f"\n######## {naam}: U-Net n={len(rows)}, baselines n={len(base)}")
    uit = {"n_unet": len(rows), "n_base": len(base), "unet": {}, "paren": {}}
    f1u = [r["f1"] for r in rows.values()]; cr = [r["count_ratio"] for r in rows.values() if r.get("count_ratio") is not None]
    bias_u = [r["ahi_pred"] - r["ahi_ref"] for r in rows.values()]
    tp, fp, fn = (sum(r[k] for r in rows.values()) for k in ("tp", "fp", "fn"))
    uit["unet"] = {"f1_med": float(np.median(f1u)), "f1_pooled": 2 * tp / (2 * tp + fp + fn), "count_ratio_med": float(np.median(cr)),
                   "bias_mean": float(np.mean(bias_u)), "mae": float(np.mean(np.abs(bias_u))),
                   "f1_typebewust_med": float(np.median([r["f1_typebewust"] for r in rows.values() if r.get("f1_typebewust") is not None]))}
    print("U-Net:", {k: round(v, 3) for k, v in uit["unet"].items()})
    for prof in profielen:
        paren = []
        for rec, r in rows.items():
            fb = d / f"{rec}_base_{prof}.csv"
            if not fb.exists() or rec not in base:
                continue
            ref = lees(d / f"{rec}_ref.csv"); pb = lees(fb); pu = lees(d / f"{rec}.csv")
            f1b, *_ = f1_of(pb, ref); f1u_, *_ = f1_of(pu, ref)
            if f1b is None or f1u_ is None:
                continue
            ahi_b = base[rec]["profiles"][prof]["ahi"]
            paren.append({"rec": rec, "f1_u": f1u_, "f1_b": f1b, "d": f1u_ - f1b, "ref_ahi": r["ahi_ref"], "bias_u": r["ahi_pred"] - r["ahi_ref"],
                          "bias_b": (ahi_b - r["ahi_ref"]) if ahi_b is not None else None, "n_ref": r["n_ref"]})
        if not paren:
            print(f"  {prof}: geen baselines (nog niet gedraaid)"); continue
        dd = np.array([p["d"] for p in paren]); nz = dd[np.abs(dd) > 1e-12]
        pw = wilcoxon(nz).pvalue if nz.size >= 5 else float("nan")
        beter = int((dd > 1e-12).sum()); slechter = int((dd < -1e-12).sum())
        bb = [p["bias_b"] for p in paren if p["bias_b"] is not None]; bu = [p["bias_u"] for p in paren]
        ref = np.array([p["ref_ahi"] for p in paren]); q1, q2 = np.percentile(ref, [100 / 3, 200 / 3])
        tert = {}
        for lab, m in (("laag", ref <= q1), ("midden", (ref > q1) & (ref <= q2)), ("hoog", ref > q2)):
            tert[lab] = {"n": int(m.sum()), "dF1_mean": float(dd[m].mean()) if m.any() else None}
        res = {"n": len(paren), "f1_u_med": float(np.median([p["f1_u"] for p in paren])), "f1_b_med": float(np.median([p["f1_b"] for p in paren])),
               "dF1_med": float(np.median(dd)), "dF1_mean": float(dd.mean()), "beter": beter, "slechter": slechter, "p": pw,
               "bias_u_mean": float(np.mean(bu)), "bias_b_mean": float(np.mean(bb)) if bb else None, "tertiel": tert}
        regel = (beter >= 90 and pw < 0.05 and 0.80 <= uit["unet"]["count_ratio_med"] <= 1.25
                 and (res["bias_b_mean"] is None or abs(res["bias_u_mean"]) <= abs(res["bias_b_mean"]))
                 and all(t["dF1_mean"] is None or t["dF1_mean"] >= -0.02 for t in tert.values()))
        res["regel_primair"] = bool(regel) if naam == "shhs1" else None
        uit["paren"][prof] = res
        print(f"  vs {prof}: n={len(paren)} F1 {res['f1_u_med']:.3f} vs {res['f1_b_med']:.3f}; ΔF1 med {res['dF1_med']:+.3f} mean {res['dF1_mean']:+.3f}; "
              f"beter/slechter {beter}/{slechter}; p={pw:.1e}; bias {res['bias_u_mean']:+.2f} vs {res['bias_b_mean']:+.2f}; "
              f"tertielen {[(k, round(v['dF1_mean'], 3) if v['dF1_mean'] is not None else None) for k, v in tert.items()]}"
              + (f"; REGEL {'JA' if regel else 'NEE'}" if naam == 'shhs1' else ""))
    return uit


def ablaties():
    d = OUT / "mesa_val"; uit = {}
    basis = {r["rec"]: r["f1"] for r in json.load(open(d / "rows.json")) if "f1" in r}
    for z, lab in (("0", "zonder druk"), ("1", "zonder thermistor"), ("4", "zonder SpO2"), ("23", "zonder effort")):
        f = d / f"rows_zero{z}.json"
        if not f.exists():
            continue
        rows = {r["rec"]: r for r in json.load(open(f)) if "f1" in r}
        dd = [rows[k]["f1"] - basis[k] for k in rows if k in basis]
        cr = [rows[k]["count_ratio"] for k in rows if rows[k].get("count_ratio") is not None]
        uit[lab] = {"f1_med": float(np.median([r["f1"] for r in rows.values()])), "dF1_mean": float(np.mean(dd)), "count_ratio_med": float(np.median(cr)), "n": len(rows)}
        print(f"  ablatie {lab:18}: F1 med {uit[lab]['f1_med']:.3f} (ΔF1 mean {uit[lab]['dF1_mean']:+.3f}), count-ratio {uit[lab]['count_ratio_med']:.2f}")
    return uit


def _psgipa_scorers(sn):
    """De 12 scoordersets zoals eval_cohort ze gebruikt (validate_psgipa.event_set, duur = raw.times[-1])."""
    import mne
    sys.path.insert(0, str(HIER.parents[1]))
    import validate_psgipa as vp
    raw = mne.io.read_raw_edf(f"/srv/DATA/PSG-IPA/Resp_events/PSG/{sn}_Respiration.edf", preload=True, verbose="ERROR")
    dur = float(raw.times[-1]); del raw
    return [vp.event_set(f, dur) for f in sorted(Path("/srv/DATA/PSG-IPA/Resp_events/Annotations/manual").glob(f"{sn}_Respiration_manual_scorer*.edf"))]


def psgipa():
    d = OUT / "psgipa"; uit = {}
    basis = {}
    for prof in ("aasm_v3_rec", "aasm_v3_breath_dual"):
        for sn in PLAFOND:
            fb = d / f"{sn}_base_{prof}.csv"
            if fb.exists():
                refs = _psgipa_scorers(sn); pb = lees(fb)
                f1s = [f1_of(pb, r)[0] for r in refs]; f1s = [x for x in f1s if x is not None]
                basis.setdefault(sn, {})[prof] = {"f1_med": float(np.median(f1s)), "n": len(pb)}
    for tag in ("", "_cpu"):
        f = d / f"rows{tag}.json"
        if not f.exists():
            continue
        rows = json.load(open(f))
        for r in rows:
            sn = r["rec"]
            uit.setdefault(sn, {}).update({("cpu_t_fwd_s" if tag else "t_fwd_s"): r.get("t_fwd_s"), ("cpu_t_total_s" if tag else "t_total_s"): r.get("t_total_s")})
            if not tag:
                uit[sn].update(f1_med=r.get("scorer_f1_median"), f1_min=r.get("scorer_f1_min"), f1_max=r.get("scorer_f1_max"), plafond=PLAFOND[sn],
                               fractie=(r["scorer_f1_median"] / PLAFOND[sn]) if r.get("scorer_f1_median") else None,
                               ahi_pred=r.get("ahi_pred"), ahi_scorer_median=r.get("ahi_scorer_median"), n_pred=r.get("n_pred"), n_scorer_med=r.get("scorer_n_median"))
    for sn, v in uit.items():
        v["baselines"] = basis.get(sn, {})
        bd = basis.get(sn, {}).get("aasm_v3_breath_dual")
        v["niet_lager_dan_breath_dual"] = (v.get("f1_med") is not None and bd is not None and v["f1_med"] >= bd["f1_med"] - 1e-9) if bd else None
    print("\n######## PSG-IPA (mediaan over 12 scoorders, náást het plafond; regel: ≥ 4/5 niet lager dan breath_dual)")
    ok = sum(1 for v in uit.values() if v.get("niet_lager_dan_breath_dual")); print(f"  niet lager dan breath_dual op {ok}/5")
    for sn, v in uit.items():
        print(f"  {sn}: F1 med {v.get('f1_med') and round(v['f1_med'], 3)} (bereik {v.get('f1_min') and round(v['f1_min'], 3)}–{v.get('f1_max') and round(v['f1_max'], 3)}), baselines {{p: round(b['f1_med'], 3) for p, b in v['baselines'].items()}}, plafond {v['plafond']}, fractie {v.get('fractie') and round(v['fractie'], 2)}, "
              f"AHI {v.get('ahi_pred') and round(v['ahi_pred'], 1)} vs scoordermediaan {v.get('ahi_scorer_median') and round(v['ahi_scorer_median'], 1)}, CPU voorwaarts {v.get('cpu_t_fwd_s')} s totaal {v.get('cpu_t_total_s')} s")
    return uit


def seeds():
    uit = {}
    for s in sorted(HIER.glob("seed_*/train_log.json")):
        j = json.load(open(s)); uit[s.parent.name] = j.get("best")
    print("\n######## seeds (MESA-val gepoolde F1):", {"bevroren": json.load(open(HIER / "train_log.json")).get("best"), **uit})
    return uit


if __name__ == "__main__":
    res = {"shhs1": cohort("shhs1", ["aasm_v3_rec", "aasm_v3_breath"]) if (OUT / "shhs1" / "rows.json").exists() else None,
           "mesa_val": cohort("mesa_val", ["aasm_v3_rec", "aasm_v3_breath_dual"]) if (OUT / "mesa_val" / "rows.json").exists() else None}
    print("\n######## ablaties (MESA-val)"); res["ablaties"] = ablaties()
    res["psgipa"] = psgipa(); res["seeds"] = seeds()
    json.dump(res, open(HIER / "samenvatting.json", "w"), indent=1, default=float)
