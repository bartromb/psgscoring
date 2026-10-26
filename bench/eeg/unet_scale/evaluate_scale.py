#!/usr/bin/env python3
"""Evaluatie van de schaalmeting (M / MS / MSF tegen het bevroren model) volgens
`docs/arousal_unet_schaal_preregistratie_20261025.md` §Beslisregel.

Leest de inferentie-uitvoer van `train_scale.py` (`out/<cohort>/<rec>_ref.csv`, `_hypno.json`,
`<rec>_<model>_pred.csv` op de val-τ, `<rec>_<model>_grid.json` op het τ-raster) en de
`tau.json` per model, past de 10 s-samenvoeging van de bibliotheek toe (zoals de replicatie van
27-09) en koppelt met de projectmatcher (IoU 0,20) én met "elke overlap". Gebruikt dus de
bibliotheek: pas draaien als die niet meer in bewerking is.

    .venv/bin/python bench/eeg/unet_scale/evaluate_scale.py [--out /srv/CODE/docs/arousal_unet_schaal_20261025]

Schrijft `evaluatie.json` en `evaluatie.md` in de uitvoermap. Geen PHI (NSRR-/PhysioNet-id's).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
REPO = HIER.parents[2]
for p in (REPO / "scripts", REPO / "bench", REPO):
    sys.path.insert(0, str(p))

from evaluate import evalueer, lees_events  # noqa: E402  (bench/evaluate.py)
import arousal_unet_replicatie as rep  # noqa: E402  (merge_10s, min_interval_s, _metriek, _stage_of, _wilcoxon)

COHORTEN = ("shhs150b", "mesa76", "psgipa")
ARMEN = ("M", "MS", "MSF")
PLAFOND_PSGIPA = 0.679


def _varianten(out: Path, modellen: list[str]) -> list[tuple[str, str, str]]:
    """(label, model, bron) — bron 'pred' = val-τ, anders een τ uit het raster."""
    v = []
    for m in modellen:
        v.append((m, m, "pred"))
        tj = out / ("frozen" if m == "frozen" else m) / "tau.json"
        if tj.exists():
            t = json.loads(tj.read_text())
            if t.get("tau_neutral") is not None:
                v.append((f"{m}@neutraal", m, f"{float(t['tau_neutral']):.2f}"))
    if "frozen" in modellen:
        v.append(("frozen@0.20", "frozen", "0.20"))
        v.append(("frozen@0.35", "frozen", "0.35"))
    return v


def _events(d: Path, rec: str, model: str, bron: str):
    if bron == "pred":
        p = d / f"{rec}_{model}_pred.csv"
        return [(a, b) for a, b, _ in lees_events(p)] if p.exists() else None
    g = d / f"{rec}_{model}_grid.json"
    if not g.exists():
        return None
    grid = json.loads(g.read_text())["events"]
    if bron not in grid:
        return None
    return [(float(a), float(b)) for a, b in grid[bron]]


def _eval_nacht(pred, ref, hypno):
    # bench/evaluate.py koppelt op (onset, offset, type)-tuples; type None = typeloos.
    pred = [(float(a), float(b), None) for a, b in pred]
    ref = [(float(a), float(b), None) for a, b in ref]
    r = {"n": len(pred), "count_ratio": len(pred) / len(ref) if ref else None}
    for m, t in (("project", 0.20), ("any", 0.0)):
        r[m] = evalueer(pred, ref, m, t, False)["totaal"]
    for lbl, st in (("rem", {"R"}), ("nrem", {"N1", "N2", "N3"})):
        p_s = [e for e in pred if rep._stage_of(e, hypno) in st]
        r_s = [e for e in ref if rep._stage_of(e, hypno) in st]
        tot = evalueer(p_s, r_s, "project", 0.20, False)["totaal"]
        r[lbl] = {k: tot[k] for k in ("tp", "fp", "fn")}
    return r


def _f1(x):
    v = x.get("f1")
    return 0.0 if v is None else float(v)


def cohort_eval(out: Path, cohort: str, varianten, interval_s: float) -> dict:
    d = out / "out" / cohort
    nachten = {}
    ontbreekt = {lbl: [] for lbl, _, _ in varianten}
    for f_ref in sorted(d.glob("*_ref.csv")):
        rec = f_ref.name[: -len("_ref.csv")]
        f_h = d / f"{rec}_hypno.json"
        if not f_h.exists():
            continue
        ref = [(a, b) for a, b, _ in lees_events(f_ref)]
        if not ref:
            continue
        meta = json.loads(f_h.read_text()); hypno = meta["hypno"]; tst_h = float(meta["tst_h"])
        rij = {"n_ref": len(ref), "tst_h": tst_h, "ref_index": len(ref) / tst_h if tst_h > 0 else None, "varianten": {}}
        for lbl, model, bron in varianten:
            ev = _events(d, rec, model, bron)
            if ev is None:
                ontbreekt[lbl].append(rec); continue
            rij["varianten"][lbl] = _eval_nacht(rep.merge_10s(ev, interval_s), ref, hypno)
        nachten[rec] = rij
    labels = [lbl for lbl, _, _ in varianten]
    compleet = [r for r in sorted(nachten) if all(l in nachten[r]["varianten"] for l in labels)]
    uit = {"cohort": cohort, "n_nachten": len(nachten), "n_compleet": len(compleet),
           "ontbreekt": {k: v for k, v in ontbreekt.items() if v}, "varianten": {}, "paren": {}, "tertiel": {}}
    recs = compleet
    for lbl in labels:
        V = {}
        for m in ("project", "any"):
            tp = sum(nachten[r]["varianten"][lbl][m]["tp"] for r in recs)
            fp = sum(nachten[r]["varianten"][lbl][m]["fp"] for r in recs)
            fn = sum(nachten[r]["varianten"][lbl][m]["fn"] for r in recs)
            f1s = [_f1(nachten[r]["varianten"][lbl][m]) for r in recs]
            V[m] = {"gepoold": rep._metriek(tp, fp, fn), "mediaan_f1": float(np.median(f1s)) if recs else None,
                    "gemiddelde_f1": float(np.mean(f1s)) if recs else None}
        for st in ("rem", "nrem"):
            tp = sum(nachten[r]["varianten"][lbl][st]["tp"] for r in recs)
            fp = sum(nachten[r]["varianten"][lbl][st]["fp"] for r in recs)
            fn = sum(nachten[r]["varianten"][lbl][st]["fn"] for r in recs)
            V[st] = rep._metriek(tp, fp, fn)
        cr = [nachten[r]["varianten"][lbl]["count_ratio"] for r in recs if nachten[r]["varianten"][lbl]["count_ratio"] is not None]
        V["mediaan_count_ratio"] = float(np.median(cr)) if cr else None
        V["mediaan_abs_count_ratio_min1"] = float(np.median([abs(x - 1) for x in cr])) if cr else None
        V["n_events"] = int(sum(nachten[r]["varianten"][lbl]["n"] for r in recs))
        V["bias_index"] = float(np.mean([nachten[r]["varianten"][lbl]["n"] / nachten[r]["tst_h"] - nachten[r]["ref_index"]
                                         for r in recs if nachten[r]["ref_index"] is not None and nachten[r]["tst_h"] > 0])) if recs else None
        uit["varianten"][lbl] = V
    uit["n_ref_events"] = int(sum(nachten[r]["n_ref"] for r in recs))

    orde = sorted(recs, key=lambda r: (nachten[r]["ref_index"] is None, nachten[r]["ref_index"] or 0.0))
    tertielen = [list(x) for x in np.array_split(np.array(orde, dtype=object), 3)] if recs else [[], [], []]

    def paar(a: str, b: str) -> dict:
        """variant a tegen variant b (a − b), projectmatcher."""
        d_f1 = np.array([_f1(nachten[r]["varianten"][a]["project"]) - _f1(nachten[r]["varianten"][b]["project"]) for r in recs])
        d_any = np.array([_f1(nachten[r]["varianten"][a]["any"]) - _f1(nachten[r]["varianten"][b]["any"]) for r in recs])
        P = {"n": len(recs),
             "mediaan_dF1": float(np.median(d_f1)) if len(d_f1) else None,
             "gemiddelde_dF1": float(d_f1.mean()) if len(d_f1) else None,
             "beter": int((d_f1 > 0).sum()), "slechter": int((d_f1 < 0).sum()), "gelijk": int((d_f1 == 0).sum()),
             "aandeel_beter": float((d_f1 > 0).mean()) if len(d_f1) else None,
             "wilcoxon_p": rep._wilcoxon(d_f1),
             "mediaan_dF1_any": float(np.median(d_any)) if len(d_any) else None,
             "abs_cr_min1_a": uit["varianten"][a]["mediaan_abs_count_ratio_min1"],
             "abs_cr_min1_b": uit["varianten"][b]["mediaan_abs_count_ratio_min1"],
             "tertiel": {}}
        try:
            from scipy.stats import wilcoxon
            dd = d_f1[np.isfinite(d_f1)]
            P["wilcoxon_p_slechter_eenzijdig"] = (float(wilcoxon(dd, alternative="less").pvalue)
                                                  if len(dd) >= 5 and not np.all(dd == 0) else None)
        except Exception:  # noqa: BLE001
            P["wilcoxon_p_slechter_eenzijdig"] = None
        for naam, t in zip(("laag", "midden", "hoog"), tertielen):
            dt = np.array([_f1(nachten[r]["varianten"][a]["project"]) - _f1(nachten[r]["varianten"][b]["project"]) for r in t])
            P["tertiel"][naam] = {"n": len(t), "mediaan_dF1": float(np.median(dt)) if len(dt) else None,
                                  "beter": int((dt > 0).sum()), "slechter": int((dt < 0).sum())}
        return P

    for arm in ARMEN:
        if arm in uit["varianten"] and "frozen" in uit["varianten"]:
            uit["paren"][f"{arm}-frozen"] = paar(arm, "frozen")
    if "MSF" in uit["varianten"] and "MS" in uit["varianten"]:
        uit["paren"]["MSF-MS"] = paar("MSF", "MS")
    uit["nachten"] = {r: {"n_ref": nachten[r]["n_ref"], "tst_h": nachten[r]["tst_h"],
                          "f1_project": {l: _f1(nachten[r]["varianten"][l]["project"]) for l in labels},
                          "count_ratio": {l: nachten[r]["varianten"][l]["count_ratio"] for l in labels}}
                      for r in recs}
    return uit


def beslisregel(E: dict) -> dict:
    """De vooraf vastgelegde regel, per arm."""
    uit = {}
    for arm in ARMEN:
        key = f"{arm}-frozen"
        r = {"primair_shhs150b": None, "secundair_mesa76": None, "secundair_psgipa": None, "geslaagd": None}
        s = E.get("shhs150b", {}).get("paren", {}).get(key)
        if s:
            r["primair_shhs150b"] = {
                "dF1_ge_0.02": s["mediaan_dF1"] is not None and s["mediaan_dF1"] >= 0.02,
                "p_lt_0.01": s["wilcoxon_p"] is not None and s["wilcoxon_p"] < 0.01,
                "beter_ge_60pct": s["aandeel_beter"] is not None and s["aandeel_beter"] >= 0.60,
                "count_ratio_niet_slechter": (s["abs_cr_min1_a"] is not None and s["abs_cr_min1_b"] is not None
                                              and s["abs_cr_min1_a"] <= s["abs_cr_min1_b"] + 0.05),
            }
            r["primair_shhs150b"]["alle"] = all(r["primair_shhs150b"].values())
        m = E.get("mesa76", {}).get("paren", {}).get(key)
        if m:
            r["secundair_mesa76"] = {"dF1_ge_0": m["mediaan_dF1"] is not None and m["mediaan_dF1"] >= 0.0,
                                     "niet_significant_slechter": (m["wilcoxon_p_slechter_eenzijdig"] is None
                                                                   or m["wilcoxon_p_slechter_eenzijdig"] >= 0.05)}
            r["secundair_mesa76"]["alle"] = all(r["secundair_mesa76"].values())
        p = E.get("psgipa", {}).get("paren", {}).get(key)
        if p:
            niet_lager = p["n"] - p["slechter"]
            r["secundair_psgipa"] = {"niet_lager": niet_lager, "n": p["n"], "ge_4_van_5": niet_lager >= 4}
        if r["primair_shhs150b"] and r["secundair_mesa76"] and r["secundair_psgipa"]:
            r["geslaagd"] = bool(r["primair_shhs150b"]["alle"] and r["secundair_mesa76"]["alle"]
                                 and r["secundair_psgipa"]["ge_4_van_5"])
        uit[arm] = r
    f = E.get("shhs150b", {}).get("paren", {}).get("MSF-MS")
    if f:
        uit["flow_MSF_vs_MS"] = {
            "dF1_ge_0.01": f["mediaan_dF1"] is not None and f["mediaan_dF1"] >= 0.01,
            "p_lt_0.01": f["wilcoxon_p"] is not None and f["wilcoxon_p"] < 0.01,
            "beter_ge_60pct": f["aandeel_beter"] is not None and f["aandeel_beter"] >= 0.60,
            "count_ratio_niet_slechter": (f["abs_cr_min1_a"] is not None and f["abs_cr_min1_b"] is not None
                                          and f["abs_cr_min1_a"] <= f["abs_cr_min1_b"] + 0.05)}
        uit["flow_MSF_vs_MS"]["alle"] = all(uit["flow_MSF_vs_MS"].values())
    return uit


def _f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}".replace(".", ",")


def rapport(E: dict, B: dict, taus: dict) -> str:
    L = ["# Schaalmeting arousal-U-Net — evaluatie (projectmatcher IoU 0,20 na 10 s-samenvoeging; 'any' ter vergelijking)\n"]
    L.append("τ per model (val-80): " + ", ".join(f"{k}: val {_f(v.get('tau_val'), 2)} / neutraal {_f(v.get('tau_neutral'), 2)}" for k, v in taus.items()) + "\n")
    for c in COHORTEN:
        if c not in E or not E[c]["n_compleet"]:
            L.append(f"## {c}: geen complete nachten\n"); continue
        e = E[c]
        L.append(f"## {c} — {e['n_compleet']} nachten ({e['n_ref_events']} referentie-arousals)" + (f"; plafond PSG-IPA {PLAFOND_PSGIPA}" if c == "psgipa" else "") + "\n")
        L.append("| variant | F1 med | F1 gepoold | sens | ppv | F1 any med | REM F1 | NREM F1 | count-ratio med | \\|cr−1\\| med | bias index |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for lbl, V in e["varianten"].items():
            g = V["project"]["gepoold"]
            L.append(f"| {lbl} | {_f(V['project']['mediaan_f1'])} | {_f(g['f1'])} | {_f(g['sens'])} | {_f(g['ppv'])} | "
                     f"{_f(V['any']['mediaan_f1'])} | {_f(V['rem']['f1'])} | {_f(V['nrem']['f1'])} | "
                     f"{_f(V['mediaan_count_ratio'], 2)} | {_f(V['mediaan_abs_count_ratio_min1'], 2)} | {_f(V['bias_index'], 2)} |")
        L.append("")
        if e["paren"]:
            L.append("| paar | n | ΔF1 med | gem | beter/slechter/gelijk | Wilcoxon p | p eenzijdig slechter | ΔF1 any med | laag/midden/hoog ΔF1 med |")
            L.append("|---|---:|---:|---:|---|---:|---:|---:|---|")
            for k, P in e["paren"].items():
                t = P["tertiel"]
                L.append(f"| {k} | {P['n']} | {_f(P['mediaan_dF1'])} | {_f(P['gemiddelde_dF1'])} | "
                         f"{P['beter']}/{P['slechter']}/{P['gelijk']} | {_f(P['wilcoxon_p'], 4)} | {_f(P['wilcoxon_p_slechter_eenzijdig'], 4)} | "
                         f"{_f(P['mediaan_dF1_any'])} | {_f(t['laag']['mediaan_dF1'])}/{_f(t['midden']['mediaan_dF1'])}/{_f(t['hoog']['mediaan_dF1'])} |")
            L.append("")
    L.append("## Beslisregel (prereg 25-10)\n")
    L.append("```json\n" + json.dumps(B, indent=1, ensure_ascii=False) + "\n```")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/srv/CODE/docs/arousal_unet_schaal_20261025"))
    ap.add_argument("--models", nargs="+", default=["frozen", *ARMEN])
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTEN))
    a = ap.parse_args()
    interval = rep.min_interval_s()
    taus = {}
    for m in a.models:
        tj = a.out / ("frozen" if m == "frozen" else m) / "tau.json"
        if tj.exists():
            t = json.loads(tj.read_text()); taus[m] = {"tau_val": t.get("tau_val"), "tau_neutral": t.get("tau_neutral")}
    varianten = _varianten(a.out, a.models)
    E = {c: cohort_eval(a.out, c, varianten, interval) for c in a.cohorts if (a.out / "out" / c).exists()}
    B = beslisregel(E)
    md = rapport(E, B, taus)
    (a.out / "evaluatie.json").write_text(json.dumps({"interval_s": interval, "taus": taus, "varianten": varianten,
                                                       "cohorten": E, "beslisregel": B}, indent=1, default=str))
    (a.out / "evaluatie.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
