#!/usr/bin/env python3
"""Doorwerking unet_v1 op aasm_v3_breath_dual, MESA-val 100: gepaard lgbm vs unet uit de twee validate_mesa-JSON's."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
D = Path(__file__).parent; REF = "aasm15"; L = "aasm_v3_breath_dual"
def load(p):
    d = json.load(open(p)); rows = d["results"] if isinstance(d, dict) and "results" in d else d
    return {r["recording"]: r for r in rows if "error" not in r and L in r.get("profiles", {}) and "error" not in r["profiles"][L]}
a = load(D / "mesa_lgbm.json"); b = load(D / "mesa_unet.json"); ids = sorted(set(a) & set(b))
print("gepaard n", len(ids))
def arr(d, f): return np.array([f(d[i]) for i in ids], dtype=float)
ref = arr(a, lambda r: r["ahi_ref"][REF])
uit = {"n": len(ids)}
for naam, f in (("AHI", lambda r: r["profiles"][L]["ahi"]), ("RDI", lambda r: r["profiles"][L]["rdi"] if r["profiles"][L].get("rdi") is not None else np.nan),
                ("arousal_index", lambda r: r["profiles"][L].get("arousal_index") if r["profiles"][L].get("arousal_index") is not None else np.nan),
                ("n_arousals", lambda r: r["profiles"][L].get("n_arousals", np.nan)), ("n_hypopnea", lambda r: r["profiles"][L].get("n_hypopnea", np.nan)),
                ("F1_resp", lambda r: r["profiles"][L]["match"][REF]["f1"])):
    x = arr(a, f); y = arr(b, f); m = ~(np.isnan(x) | np.isnan(y)); d = (y - x)[m]; nz = d[np.abs(d) > 1e-12]
    p = wilcoxon(nz).pvalue if nz.size >= 5 else float("nan")
    q1, q2 = np.percentile(ref[m], [100/3, 200/3]); tert = [round(float(d[(ref[m] <= q1)].mean()), 3), round(float(d[(ref[m] > q1) & (ref[m] <= q2)].mean()), 3), round(float(d[ref[m] > q2].mean()), 3)]
    uit[naam] = {"lgbm_med": float(np.median(x[m])), "unet_med": float(np.median(y[m])), "d_med": float(np.median(d)), "d_mean": float(d.mean()), "d_min": float(d.min()), "d_max": float(d.max()),
                 "hoger": int((d > 1e-12).sum()), "lager": int((d < -1e-12).sum()), "gelijk": int((np.abs(d) <= 1e-12).sum()), "p": p, "tertielen": tert}
    print(f"{naam:14} lgbm med {uit[naam]['lgbm_med']:8.3f} unet med {uit[naam]['unet_med']:8.3f}  Δ med {uit[naam]['d_med']:+.3f} mean {uit[naam]['d_mean']:+.3f} [{uit[naam]['d_min']:+.2f}, {uit[naam]['d_max']:+.2f}]  hoger/lager/gelijk {uit[naam]['hoger']}/{uit[naam]['lager']}/{uit[naam]['gelijk']}  p={p:.2e}  tertielen {tert}")
# bias AHI tegen NSRR
for lab, d_ in (("lgbm", a), ("unet", b)):
    bias = arr(d_, lambda r: r["profiles"][L]["ahi"]) - ref; print(f"AHI-bias {lab}: mean {bias.mean():+.2f}, MAE {np.abs(bias).mean():.2f}")
json.dump(uit, open(D / "samenvatting_mesa.json", "w"), indent=1, default=float)
