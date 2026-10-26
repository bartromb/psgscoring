#!/usr/bin/env python
"""analyse.py — inkortingstoets (prereg docs/inkortingstoets_preregistratie_20261026.md). Leest rijen.json; schrijft samenvatting.json + analyse_uitvoer.md."""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import median
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent.parent / "scripts"))
from validate_mesa import wilcoxon_signed_rank, severity  # noqa: E402

def f(v, nd=3): return "—" if v is None else f"{v:.{nd}f}".replace(".", ",")
def sg(v, nd=3): return "—" if v is None else f"{v:+.{nd}f}".replace(".", ",")
def med(x): x = [v for v in x if v is not None]; return float(median(x)) if x else None
def gem(x): x = [v for v in x if v is not None]; return sum(x) / len(x) if x else None
def jaccard(a, b):
    sa, sb = set(a), set(b); u = len(sa | sb); return (len(sa & sb) / u) if u else 1.0
def venster(evs, t): return [tuple(e) for e in evs if e[0] < t]
def tertiel(rows):
    xs = sorted(r["ahi_ref"] for r in rows); n = len(xs); g1, g2 = xs[n // 3], xs[(2 * n) // 3]
    return {r["recording"]: ("laag" if r["ahi_ref"] < g1 else "midden" if r["ahi_ref"] < g2 else "hoog") for r in rows}, (g1, g2)

def vergelijk_impl(rows, new, ref, L, out, label):
    """new tegen ref in het venster van new (implementatietoets)."""
    rel_apn, rel_ev, jac, groot = [], [], [], []
    L.append(f"\n## {label}: {new} tegen {ref} in het {new}-venster\n"); L.append("| nacht | tertiel | T_cut (h) | apneus ref → new | Δ% apneus | events ref → new | Δ% events | Jaccard | sbu ref/new |"); L.append("|---|---|---|---|---|---|---|---|---|")
    T, _ = tertiel(rows)
    for r in rows:
        a, b = r["arms"][ref], r["arms"][new]; t = b["t_cut_s"]
        ea, eb = venster(a["events"], t), venster(b["events"], t)
        apa = sum(1 for e in ea if e[2] in ("obstructive", "central", "mixed", "uncertain")); apb = sum(1 for e in eb if e[2] in ("obstructive", "central", "mixed", "uncertain"))
        ra = (apb - apa) / apa * 100 if apa else (0.0 if apb == 0 else None); re = (len(eb) - len(ea)) / len(ea) * 100 if ea else (0.0 if not eb else None)
        j = jaccard(ea, eb); rel_apn.append(ra); rel_ev.append(re); jac.append(j)
        if ra is not None and abs(ra) > 5: groot.append((r["recording"], apa, apb, round(ra, 1)))
        L.append(f"| {r['recording']} | {T[r['recording']]} | {t/3600:.2f} | {apa} → {apb} | {sg(ra, 1)} | {len(ea)} → {len(eb)} | {sg(re, 1)} | {f(j)} | {a['stage_baseline_used']}/{b['stage_baseline_used']} |")
    S = {"n": len(rows), "rel_apneus_mediaan_abs": med([abs(x) for x in rel_apn if x is not None]), "rel_apneus_mediaan": med(rel_apn), "rel_events_mediaan_abs": med([abs(x) for x in rel_ev if x is not None]),
         "n_gt5pct": len(groot), "gt5pct": groot, "jaccard_mediaan": med(jac), "jaccard_min": min(jac) if jac else None,
         "stabiel": (med([abs(x) for x in rel_apn if x is not None]) or 0) <= 2 and len(groot) <= 3 and (med(jac) or 0) >= 0.95}
    L.append(f"\n- mediaan |Δ% apneus| {f(S['rel_apneus_mediaan_abs'], 1)} (getekend {sg(S['rel_apneus_mediaan'], 1)}), nachten > 5 %: {S['n_gt5pct']} {groot}; mediaan |Δ% events| {f(S['rel_events_mediaan_abs'], 1)}; Jaccard mediaan {f(S['jaccard_mediaan'])}, min {f(S['jaccard_min'])} → **{'STABIEL' if S['stabiel'] else 'NIET STABIEL'}** (regel: ≤ 2 % én ≤ 3/30 > 5 % én Jaccard ≥ 0,95)")
    out[label] = S

def vergelijk_kwal(rows, new, old, L, out, label):
    """detectiekwaliteit: new tegen old tegen aasm15 in hun (gelijke) venster."""
    T, grenzen = tertiel(rows); d, da, rijen = [], [], []; bias_n, bias_o = [], []; ern_n = ern_o = 0; ratio = []
    L.append(f"\n## {label}: {new} tegen {old} tegen `aasm15` (venster {new})\n"); L.append("| nacht | tertiel | AHI ref | AHI old → new | F1 old → new | ΔF1 | apneu-F1 old → new | apneus old → new (ratio) | ernst ref / old / new | sbu old/new |"); L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        n_, o_ = r["arms"][new], r["arms"][old]; dF = n_["match"]["f1"] - o_["match"]["f1"]; d.append(dF)
        if n_["match_apnea"]["f1"] is not None and o_["match_apnea"]["f1"] is not None: da.append(n_["match_apnea"]["f1"] - o_["match_apnea"]["f1"])
        bias_n.append(n_["ahi_w"] - n_["ahi_ref_w"]); bias_o.append(o_["ahi_w"] - o_["ahi_ref_w"]); ern_n += n_["severity_w"] == n_["severity_ref_w"]; ern_o += o_["severity_w"] == o_["severity_ref_w"]
        rt = (o_["n_apnea_w"] / n_["n_apnea_w"]) if n_["n_apnea_w"] else None; ratio.append(rt)
        rijen.append({"recording": r["recording"], "tertiel": T[r["recording"]], "dF1": dF, "ratio_old_new": rt})
        L.append(f"| {r['recording']} | {T[r['recording']]} | {f(n_['ahi_ref_w'], 1)} | {f(o_['ahi_w'], 1)} → {f(n_['ahi_w'], 1)} | {f(o_['match']['f1'])} → {f(n_['match']['f1'])} | {sg(dF)} | {f(o_['match_apnea']['f1'])} → {f(n_['match_apnea']['f1'])} | {o_['n_apnea_w']} → {n_['n_apnea_w']} ({f(rt, 2)}) | {n_['severity_ref_w']} / {o_['severity_w']} / {n_['severity_w']} | {o_['stage_baseline_used']}/{n_['stage_baseline_used']} |")
    beter = sum(1 for x in d if x > 0); slechter = sum(1 for x in d if x < 0); gelijk = len(d) - beter - slechter
    p = (wilcoxon_signed_rank(d)[0] if any(d) else None)  # (p, n_nonzero)
    pt = {t: {"n": sum(1 for x in rijen if x["tertiel"] == t), "dF1_mediaan": med([x["dF1"] for x in rijen if x["tertiel"] == t]), "beter": sum(1 for x in rijen if x["tertiel"] == t and x["dF1"] > 0), "slechter": sum(1 for x in rijen if x["tertiel"] == t and x["dF1"] < 0)} for t in ("laag", "midden", "hoog")}
    S = {"n": len(d), "dF1_mediaan": med(d), "dF1_gem": gem(d), "beter": beter, "slechter": slechter, "gelijk": gelijk, "p_wilcoxon": p, "dF1_apneu_mediaan": med(da), "bias_old": gem(bias_o), "bias_new": gem(bias_n),
         "bias_old_mediaan": med(bias_o), "bias_new_mediaan": med(bias_n), "ernst_old": ern_o, "ernst_new": ern_n, "ratio_old_new_mediaan": med(ratio), "ratio_min": min(x for x in ratio if x is not None), "ratio_max": max(x for x in ratio if x is not None), "tertielgrenzen": grenzen, "per_tertiel": pt,
         "sbu_old_false": sum(1 for r in rows if r["arms"][old]["stage_baseline_used"] is False), "sbu_new_true": sum(1 for r in rows if r["arms"][new]["stage_baseline_used"] is True)}
    uitslag = "stadium-basislijn BETER op MESA" if (S["dF1_mediaan"] or 0) > 0 and p is not None and p < 0.05 and abs(S["bias_new"]) <= abs(S["bias_old"]) else ("stadium-basislijn SLECHTER op MESA" if (S["dF1_mediaan"] or 0) < 0 and p is not None and p < 0.05 and abs(S["bias_new"]) >= abs(S["bias_old"]) else "MESA beslist niet")
    S["uitslag"] = uitslag
    L.append(f"\n- ΔF1 ({new} − {old}): mediaan {sg(S['dF1_mediaan'], 4)}, gemiddeld {sg(S['dF1_gem'], 4)}, beter/slechter/gelijk {beter}/{slechter}/{gelijk}, Wilcoxon p = {f(p, 4) if p is not None else '—'}; apneu-ΔF1 mediaan {sg(S['dF1_apneu_mediaan'], 4)}")
    L.append(f"- bias (AHI − ref) gemiddeld old {sg(S['bias_old'], 2)} → new {sg(S['bias_new'], 2)} (mediaan {sg(S['bias_old_mediaan'], 2)} → {sg(S['bias_new_mediaan'], 2)}); ernst gelijk aan ref old {ern_o}/{len(d)} → new {ern_n}/{len(d)}; apneus old/new ratio mediaan {f(S['ratio_old_new_mediaan'], 2)} (bereik {f(S['ratio_min'], 2)} … {f(S['ratio_max'], 2)}); stage_baseline_used: old False op {S['sbu_old_false']}/{len(d)}, new True op {S['sbu_new_true']}/{len(d)}")
    L.append("- per tertiel van de referentie-AHI (grenzen " + f"{grenzen[0]:.1f} / {grenzen[1]:.1f}): " + "; ".join(f"{t} n={v['n']} ΔF1 mediaan {sg(v['dF1_mediaan'], 4)} beter/slechter {v['beter']}/{v['slechter']}" for t, v in pt.items()))
    L.append(f"- **Lezing (huisregel): {uitslag}**")
    out[label] = S

def main():
    bron = HIER / "rijen.json"
    alle = json.load(open(bron)) if bron.exists() else [json.loads(l) for l in (HIER / "rijen.partial.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in alle if not r.get("error")]; fouten = [r for r in alle if r.get("error")]
    out = {"n": len(rows), "fouten": [(r["recording"], r["error"]) for r in fouten]}; L = [f"# Analyse-uitvoer inkortingstoets (n = {len(rows)}, fouten {len(fouten)})"]
    # 1. controle H0_old ≡ H0_new
    L.append("\n## 1. Controle: H0_old tegen H0_new (dekkend hypnogram)\n"); gelijk = 0; sbu = 0
    for r in rows:
        a, b = r["arms"]["H0_new"], r["arms"]["H0_old"]
        g = a["events"] == b["events"] and a["ahi_full"] == b["ahi_full"]; gelijk += g; sbu += (a["stage_baseline_used"] is True and b["stage_baseline_used"] is True)
        if not g: L.append(f"- {r['recording']}: NIET gelijk (events {len(a['events'])} vs {len(b['events'])}, AHI {a['ahi_full']} vs {b['ahi_full']}, sbu {a['stage_baseline_used']}/{b['stage_baseline_used']})")
    L.append(f"- eventlijst + AHI gelijk op {gelijk}/{len(rows)}; stage_baseline_used True in beide op {sbu}/{len(rows)}"); out["controle"] = {"gelijk": gelijk, "sbu_beide_true": sbu}
    # 2. implementatie
    vergelijk_impl(rows, "H1_new", "H0_new", L, out, "2_implementatie_H1"); vergelijk_impl(rows, "H5_new", "H0_new", L, out, "2b_implementatie_H5")
    # 3. oud pad
    vergelijk_impl(rows, "H1_old", "H1_new", L, out, "3_oud_pad_H1_old_vs_H1_new")
    # 4. kwaliteit
    vergelijk_kwal(rows, "H1_new", "H1_old", L, out, "4_kwaliteit_H1"); vergelijk_kwal(rows, "H5_new", "H5_old", L, out, "4b_kwaliteit_H5")
    # tijden
    L.append(f"\n- rekentijd per nacht (6 armen): mediaan {f(med([r['seconds'] for r in rows]) / 60, 1)} min, max {f(max(r['seconds'] for r in rows) / 60, 1)} min; psgscoring {sorted({r['psgscoring'] for r in rows})}")
    json.dump(out, open(HIER / "samenvatting.json", "w"), indent=1, default=str); (HIER / "analyse_uitvoer.md").write_text("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__": main()
