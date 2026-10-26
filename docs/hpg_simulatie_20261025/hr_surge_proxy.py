#!/usr/bin/env python
"""hr_surge_proxy.py — kan ΔHR (Huang 2026) de arousal-gedragen hypopneeën markeren?
Preregistratie: docs/hpg_arousalpoot_hr_surge_preregistratie_20261026.md. Leest MESA-`HR` (1 Hz)
en de NSRR-stadia; importeert psgscoring niet (definitie van ancillary.compute_event_hr_surge
nagerekend per event). Schrijft hr_surge_proxy.json + .md."""
from __future__ import annotations
import json, math, statistics as st, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from waakpoort_stap0 import stadia, MESA, EPOCH_S, SLAAP  # noqa: E402

RAW = Path("/srv/CODE/docs/hpg_simulatie_20261025_raw/rijen.json")
BASE_S, POST_S, MIN_BASE = 60, 15, 30
SEED = 20261026


def hr_1hz(rec: str) -> np.ndarray | None:
    import mne
    pad = str(MESA / "edfs" / f"{rec}.edf")
    if "HR" not in mne.io.read_raw_edf(pad, preload=False, verbose="error").ch_names:
        return None
    raw = mne.io.read_raw_edf(pad, include=["HR"], preload=True, verbose="error")
    x = raw.get_data()[0].astype(float); fs = float(raw.info["sfreq"])
    if abs(fs - 1.0) > 1e-6:
        k = int(round(fs)); n = (len(x) // k) * k
        x = x[:n].reshape(-1, k).mean(1) if k >= 1 else x
    # MNE schaalt 'misc' niet, maar wees robuust: waarden buiten 20–250 zijn ongeldig
    x = np.where((x >= 20) & (x <= 250), x, np.nan)
    return x


def delta(hr: np.ndarray, onset: float, dur: float):
    n = len(hr)
    b0, b1 = max(0, int(math.floor(onset - BASE_S))), min(n, int(math.floor(onset)))
    base = hr[b0:b1]; base = base[~np.isnan(base)]
    if len(base) < MIN_BASE: return None
    w0, w1 = max(0, int(math.floor(onset))), min(n, int(math.ceil(onset + dur + POST_S)))
    win = hr[w0:w1]; win = win[~np.isnan(win)]
    if len(win) == 0: return None
    return float(np.max(win) - np.mean(base))


def nacht(row: dict) -> dict:
    rec = row["recording"]; dur_s = row["duration_h"] * 3600.0
    hyp = stadia(MESA / "annotations-events-nsrr" / f"{rec}-nsrr.xml", dur_s)
    slaap = np.array([s in SLAAP for s in hyp]); n_ep = len(hyp)
    hr = hr_1hz(rec)
    uit = {"recording": rec, "hr_aanwezig": hr is not None, "A": [], "B": [], "C": [], "D": [], "B_rule1a": 0, "B_n": 0, "C_mislukt": 0}
    if hr is None: return uit
    ev = row["arms"]["psg"]["events"]; hyps = row["arms"]["psg"]["hypopneas"]
    ev_iv = [(a, b) for a, b, t in ev]
    def in_slaap_sec(t0, t1):
        e0, e1 = int(t0 // EPOCH_S), int(min(t1, dur_s - 1) // EPOCH_S)
        return 0 <= e0 and e1 < n_ep and slaap[e0:e1 + 1].all()
    rng = np.random.default_rng(SEED + int(rec.split("-")[-1]))
    slaap_sec = np.where(np.repeat(slaap, int(EPOCH_S)))[0]
    for h in hyps:
        d = delta(hr, h["onset_s"], h["duration_s"])
        kl = "A" if (h.get("desaturation_pct") or 0) >= 3.0 else "B"
        if kl == "B":
            uit["B_n"] += 1; uit["B_rule1a"] += int(bool(h.get("rule1a_arousal")))
        if d is not None: uit[kl].append(d)
        if kl == "B":
            ok = False
            for _ in range(50):
                s0 = float(rng.choice(slaap_sec)) if len(slaap_sec) else None
                if s0 is None: break
                if not in_slaap_sec(s0 - BASE_S, s0 + h["duration_s"] + POST_S): continue
                if any(a - 30 <= s0 + h["duration_s"] and b + 30 >= s0 for a, b in ev_iv): continue
                dc = delta(hr, s0, h["duration_s"])
                if dc is None: continue
                uit["C"].append(dc); ok = True; break
            if not ok: uit["C_mislukt"] += 1
    for a, b, t in ev:
        if "hypopnea" not in t:
            d = delta(hr, a, b - a)
            if d is not None: uit["D"].append(d)
    return uit


def auc(x, y):
    from scipy.stats import mannwhitneyu
    if len(x) == 0 or len(y) == 0: return None
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(u / (len(x) * len(y)))


def q(xs):
    if not xs: return {"n": 0}
    a = np.asarray(xs); return {"n": int(len(a)), "mediaan": float(np.median(a)), "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
                                 "ge5": float((a >= 5).mean()), "ge7": float((a >= 7).mean()), "ge10": float((a >= 10).mean())}


def main() -> int:
    R = json.load(open(RAW))
    with ProcessPoolExecutor(8) as ex:
        uit = list(ex.map(nacht, R))
    pool = {k: [v for u in uit for v in u[k]] for k in "ABCD"}
    from scipy.stats import wilcoxon
    paren = [(st.median(u["B"]), st.median(u["C"])) for u in uit if len(u["B"]) >= 3 and len(u["C"]) >= 3]
    d = [b - c for b, c in paren]
    S = {"n_nachten": len(uit), "nachten_met_HR": sum(u["hr_aanwezig"] for u in uit),
         "klassen": {k: q(pool[k]) for k in "ABCD"},
         "B_totaal": sum(u["B_n"] for u in uit), "B_rule1a": sum(u["B_rule1a"] for u in uit), "C_mislukt": sum(u["C_mislukt"] for u in uit),
         "AUC_B_vs_C": auc(pool["B"], pool["C"]), "AUC_A_vs_C": auc(pool["A"], pool["C"]), "AUC_D_vs_C": auc(pool["D"], pool["C"]), "AUC_B_vs_A": auc(pool["B"], pool["A"]),
         "per_nacht_paren": len(paren), "per_nacht_dmed_B_min_C": float(np.median(d)) if d else None,
         "per_nacht_B_hoger": int(sum(x > 0 for x in d)), "per_nacht_wilcoxon_p": float(wilcoxon(d).pvalue) if d and any(d) else None}
    (HIER / "hr_surge_proxy.json").write_text(json.dumps({"samenvatting": S, "per_nacht": [{k: (v if not isinstance(v, list) else len(v)) for k, v in u.items()} for u in uit]}, indent=1))
    L = ["# ΔHR per klasse (Huang-definitie, MESA HR 1 Hz, n150 psg-arm)", "",
         "| klasse | n | mediaan | p25 | p75 | ≥ 5 bpm | ≥ 7 | ≥ 10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    naam = {"A": "A hypopneu ≥ 3 % desat", "B": "B hypopneu < 3 % desat (arousal-gedragen)", "C": "C controle (slaap, geen event)", "D": "D apneu"}
    for k in "ABCD":
        v = S["klassen"][k]
        if v["n"]: L.append(f"| {naam[k]} | {v['n']} | {v['mediaan']:.1f} | {v['p25']:.1f} | {v['p75']:.1f} | {v['ge5']:.2f} | {v['ge7']:.2f} | {v['ge10']:.2f} |")
    L += ["", f"AUC B vs C {S['AUC_B_vs_C']:.3f}; A vs C {S['AUC_A_vs_C']:.3f}; D vs C {S['AUC_D_vs_C']:.3f}; B vs A {S['AUC_B_vs_A']:.3f}.",
          f"Per nacht (≥ 3 per klasse, n = {S['per_nacht_paren']}): mediaan(B) − mediaan(C) = {S['per_nacht_dmed_B_min_C']:.1f} bpm, B hoger op {S['per_nacht_B_hoger']}, Wilcoxon p = {S['per_nacht_wilcoxon_p']:.2g}.",
          f"B: {S['B_totaal']} hypopneeën, {S['B_rule1a']} met rule1a_arousal; controle mislukt voor {S['C_mislukt']}; HR-kanaal op {S['nachten_met_HR']}/{S['n_nachten']} nachten."]
    (HIER / "hr_surge_proxy.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
