#!/usr/bin/env python3
"""Lezing volgens docs/breath_dual_mesa_preregistratie_20261007.md.
Invoer: mesa.json (validate_mesa) of mesa.partial.jsonl. Referentie aasm15, matcher zoals het harnas."""
import json, sys, statistics
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

REF = "aasm15"
D = Path(__file__).parent
SENS = set((D / "gevoeligheid_93.txt").read_text().split()) if (D / "gevoeligheid_93.txt").exists() else set()

def load(p):
    p = Path(p)
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    else:
        rows = json.load(open(p)); rows = rows.get("results", rows) if isinstance(rows, dict) else rows
    return [r for r in rows if "error" not in r]

def sev(a): return "normaal" if a < 5 else "licht" if a < 15 else "matig" if a < 30 else "ernstig"

def paar(rows, A, B, naam, bewaker=1.0, min_df1=0.0):
    ok = [r for r in rows if all(l in r["profiles"] and "error" not in r["profiles"][l] for l in (A, B))]
    f1a = np.array([r["profiles"][A]["match"][REF]["f1"] for r in ok]); f1b = np.array([r["profiles"][B]["match"][REF]["f1"] for r in ok])
    ba = np.array([r["profiles"][A]["ahi"] - r["ahi_ref"][REF] for r in ok]); bb = np.array([r["profiles"][B]["ahi"] - r["ahi_ref"][REF] for r in ok])
    d = f1a - f1b; nz = d[np.abs(d) > 1e-12]
    p = wilcoxon(nz).pvalue if nz.size >= 5 else float("nan")
    beter, slechter = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    uit = {"n": len(ok), "f1_A_med": float(np.median(f1a)), "f1_B_med": float(np.median(f1b)), "dF1_med": float(np.median(d)), "dF1_mean": float(d.mean()),
           "beter": beter, "slechter": slechter, "gelijk": len(ok) - beter - slechter, "p": p,
           "bias_A_mean": float(ba.mean()), "bias_B_mean": float(bb.mean()), "mae_A": float(np.abs(ba).mean()), "mae_B": float(np.abs(bb).mean()),
           "sev_A": sum(sev(r["profiles"][A]["ahi"]) == sev(r["ahi_ref"][REF]) for r in ok), "sev_B": sum(sev(r["profiles"][B]["ahi"]) == sev(r["ahi_ref"][REF]) for r in ok)}
    uit["bewaker_ok"] = abs(uit["bias_A_mean"]) <= abs(uit["bias_B_mean"]) + bewaker
    uit["regel"] = bool(uit["dF1_med"] > min_df1 and beter > slechter and p < 0.05 and uit["bewaker_ok"]) if min_df1 == 0.0 else bool(uit["dF1_med"] >= min_df1 and p < 0.05 and uit["bewaker_ok"])
    # per NSRR-AHI-tertiel
    ref = np.array([r["ahi_ref"][REF] for r in ok]); q1, q2 = np.percentile(ref, [100/3, 200/3])
    uit["tertiel"] = {}
    for lab, m in (("laag", ref <= q1), ("midden", (ref > q1) & (ref <= q2)), ("hoog", ref > q2)):
        uit["tertiel"][lab] = {"n": int(m.sum()), "dF1_mean": float(d[m].mean()) if m.any() else None, "dF1_med": float(np.median(d[m])) if m.any() else None,
                               "bias_A": float(ba[m].mean()) if m.any() else None, "bias_B": float(bb[m].mean()) if m.any() else None}
    print(f"\n== {naam}: {A} vs {B} (n={len(ok)}) ==")
    print(f"  F1 med {uit['f1_A_med']:.3f} vs {uit['f1_B_med']:.3f}; ΔF1 med {uit['dF1_med']:+.4f} mean {uit['dF1_mean']:+.4f}; beter/slechter/gelijk {beter}/{slechter}/{uit['gelijk']}; p={p:.2e}")
    print(f"  bias mean {uit['bias_A_mean']:+.2f} vs {uit['bias_B_mean']:+.2f}; MAE {uit['mae_A']:.2f} vs {uit['mae_B']:.2f}; ernst-overeenstemming {uit['sev_A']} vs {uit['sev_B']}; bewaker {'OK' if uit['bewaker_ok'] else 'FAALT'}; regel: {'JA' if uit['regel'] else 'NEE'}")
    for lab, t in uit["tertiel"].items():
        if t["n"]:
            print(f"  tertiel {lab:6} n={t['n']:3} ΔF1 mean {t['dF1_mean']:+.4f} med {t['dF1_med']:+.4f} bias {t['bias_A']:+.2f} vs {t['bias_B']:+.2f}")
    return uit

def per_event(rows, label="aasm_v3_breath_dual@0.50", conf_label="aasm_v3_breath_dual+conf@0.50"):
    """Alleen-druk-apneus van de dual-arm tegen de NSRR-apneus (IoU ≥ 0,20), per klasse en bevestiging."""
    sys.path.insert(0, str(D.parent.parent / "scripts"))
    from validate_mesa import parse_nsrr  # alleen voor de referentie-apneus
    def iou(a0, a1, b0, b1):
        i = max(0.0, min(a1, b1) - max(a0, b0)); u = max(a1, b1) - min(a0, b0); return i / u if u > 0 else 0.0
    klassen = {}
    for r in rows:
        if conf_label not in r["profiles"] or "apneas" not in r["profiles"][conf_label]: continue
        xml = Path("/srv/DATA/MESA/mesa/polysomnography/annotations-events-nsrr") / f"{r['recording']}-nsrr.xml"
        try:
            _h, refs, _t = parse_nsrr(xml, r["duration_h"] * 3600.0)
        except Exception as e:
            print("ref-fout", r["recording"], e); continue
        ref_ap = [(a, b) for a, b, t in refs[REF] if "hypopnea" not in str(t).lower() and "apnea" in str(t).lower()] if refs.get(REF) else []
        if not ref_ap:
            ref_ap = [(a, b) for a, b, t in refs[REF] if "hypopnea" not in str(t).lower()] if refs.get(REF) else []
        ref_all = [(a, b) for a, b, t in refs[REF]] if refs.get(REF) else []
        conf = r["profiles"][conf_label]
        allev = list(conf.get("apneas", [])) + list(((conf.get("dual_sensor_apnea") or {}).get("confirmation") or {}).get("dropped") or [])
        for e in allev:
            if e.get("corroboration") != "pressure_only": continue
            d = e.get("thermistor_drop"); k = "?" if d is None else "A" if d >= 0.72 else "C" if d >= 0.30 else "B"
            c = e.get("dual_confirmation") or ("vervallen" if e in allev[len(conf.get("apneas", [])):] else "n/a")
            a0 = float(e["onset_s"]); a1 = a0 + float(e["duration_s"])
            hit = any(iou(a0, a1, b0, b1) >= 0.20 for b0, b1 in ref_ap)
            hit_any = any(iou(a0, a1, b0, b1) >= 0.20 for b0, b1 in ref_all)
            key = (k, c); klassen.setdefault(key, [0, 0, 0]); klassen[key][0] += 1; klassen[key][1] += int(hit); klassen[key][2] += int(hit_any)
    print("\n== per alleen-druk-apneu (dual+conf-arm): klasse × bevestiging → n, aandeel dat een NSRR-apneu / enig NSRR-event matcht ==")
    for (k, c), (n, h, ha) in sorted(klassen.items()):
        print(f"  {k:2} {c:10} n={n:5}  NSRR-apneu {h/n if n else 0:.2f}  enig NSRR-event {ha/n if n else 0:.2f}")
    return {f"{k}|{c}": {"n": n, "match_apneu": h, "match_enig": ha} for (k, c), (n, h, ha) in klassen.items()}

if __name__ == "__main__":
    rows = load(sys.argv[1] if len(sys.argv) > 1 else D / "mesa.json")
    print(f"{len(rows)} opnames zonder fout")
    uit = {"n": len(rows)}
    uit["primair"] = paar(rows, "aasm_v3_breath_dual+conf@0.50", "aasm_v3_breath_dual@0.50", "PRIMAIR (F1 primair, bias-bewaker 1,0)")
    if SENS:
        uit["primair_gevoeligheid"] = paar([r for r in rows if r["recording"] in SENS], "aasm_v3_breath_dual+conf@0.50", "aasm_v3_breath_dual@0.50", "PRIMAIR op gevoeligheidsset 51–150")
    uit["baseline"] = paar(rows, "aasm_v3_breath_dual@0.50", "aasm_v3_breath@0.50", "BASELINE dual vs breath (geen regel)")
    uit["rec"] = paar(rows, "aasm_v3_breath_dual@0.50", "aasm_v3_rec", "dual vs rec (anker, geen regel)")
    uit["strict_dual"] = paar(rows, "aasm_v3_breath_dual@0.30", "aasm_v3_breath_dual@0.50", "SECUNDAIR strictness 0,30 onder dual", min_df1=0.010)
    uit["strict_conf"] = paar(rows, "aasm_v3_breath_dual+conf@0.30", "aasm_v3_breath_dual+conf@0.50", "SECUNDAIR strictness 0,30 onder dual+conf", min_df1=0.010)
    uit["strict_breath"] = paar(rows, "aasm_v3_breath@0.30", "aasm_v3_breath@0.50", "strictness 0,30 onder breath (replicatie 24-08, geen regel)", min_df1=0.010)
    uit["per_event"] = per_event(rows)
    json.dump(uit, open(D / "samenvatting.json", "w"), indent=1, default=float)
