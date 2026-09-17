#!/usr/bin/env python3
"""Replicatie-analyse van de kandidaatpoort (n150): B_poort en C_poort tegen
arm A uit de orakel-run van 17-09 (zelfde keten, zelfde nachten).

Beslisregel: docs/orakel_rule1a_poort_preregistratie_20260917.md.
  python scripts/orakel_rule1a_poort_analyse.py \\
      /srv/CODE/docs/orakel_rule1a_20260917/orakel_rule1a_mesa_n150.json \\
      docs/orakel_rule1a_poort_n150.json
"""
import json
import sys
from statistics import median

import numpy as np
from scipy.stats import wilcoxon

REF = "aasm15"


def p_w(d):
    d = [x for x in d if x != 0]
    return float(wilcoxon(d).pvalue) if len(d) >= 6 else None


def laad(pad):
    if str(pad).endswith(".jsonl"):
        rows = [json.loads(l) for l in open(pad) if l.strip()]
    else:
        rows = json.load(open(pad))["results"]
    return {r["recording"]: r for r in rows if "error" not in r}


def main(pad_a, pad_poort):
    A = laad(pad_a); P = laad(pad_poort)
    ids = [i for i in A if i in P and A[i]["arms"]["A"]["match"].get(REF)
           and all(a in P[i]["arms"] and "error" not in P[i]["arms"][a] for a in "BC")]
    print(f"gepaarde nachten: {len(ids)}")
    ids.sort(key=lambda i: A[i]["ahi_ref"][REF]); n = len(ids)
    tert = {"T1": ids[:n//3], "T2": ids[n//3:2*n//3], "T3": ids[2*n//3:]}

    def stat(arm_rows, arm, label):
        f1 = [arm_rows[i]["arms"][arm]["match"][REF]["f1"] for i in ids]
        pr = [arm_rows[i]["arms"][arm]["match"][REF]["precision"] for i in ids]
        rc = [arm_rows[i]["arms"][arm]["match"][REF]["recall"] for i in ids]
        bias = [arm_rows[i]["arms"][arm]["ahi"] - A[i]["ahi_ref"][REF] for i in ids]
        n_re = sum(arm_rows[i]["arms"][arm]["n_reinstated"] for i in ids)
        tp = sum(round(arm_rows[i]["arms"][arm]["reinst_precision"][REF] * arm_rows[i]["arms"][arm]["n_reinstated"])
                 for i in ids if arm_rows[i]["arms"][arm].get("reinst_precision"))
        gat = [arm_rows[i]["arms"][arm]["gat_recall"] for i in ids if arm_rows[i]["arms"][arm]["gat_recall"] is not None]
        sev = sum(1 for i in ids if arm_rows[i]["arms"][arm]["severity"] == A[i]["severity_ref"])
        print(f"{label:8s} F1 {median(f1):.3f} P {median(pr):.3f} R {median(rc):.3f} bias {np.mean(bias):+.2f} "
              f"| herst {n_re:5d} R1 {tp/n_re if n_re else float('nan'):.3f} | gat-recall {median(gat):.3f} | ernst=ref {sev}/{n}")
        return f1, bias

    f1A, biasA = stat(A, "A", "A")
    uit = {}
    for arm, naam in (("B", "B_poort"), ("C", "C_poort")):
        f1x, biasx = stat(P, arm, naam)
        d = [x - y for x, y in zip(f1x, f1A)]
        print(f"   dF1({naam}-A): med {median(d):+.4f} mean {np.mean(d):+.4f} beter {sum(x>0 for x in d)}/{n} "
              f"slechter {sum(x<0 for x in d)} p={p_w(d):.2e}")
        for t, tids in tert.items():
            idx = [ids.index(i) for i in tids]
            dt = [d[k] for k in idx]; bA = np.mean([biasA[k] for k in idx]); bX = np.mean([biasx[k] for k in idx])
            rp = [P[i]["arms"][arm]["reinst_precision"][REF] for i in tids if P[i]["arms"][arm].get("reinst_precision")]
            print(f"     {t} (AHI {A[tids[0]]['ahi_ref'][REF]:.1f}-{A[tids[-1]]['ahi_ref'][REF]:.1f}, n={len(tids)}): "
                  f"dF1 {np.mean(dt):+.4f} ({sum(x>0 for x in dt)}/{len(dt)}, p={p_w(dt) or float('nan'):.1e}) "
                  f"bias {bA:+.1f}->{bX:+.1f} R1med {median(rp) if rp else float('nan'):.2f}")
        n_re = sum(P[i]["arms"][arm]["n_reinstated"] for i in ids)
        tp = sum(round(P[i]["arms"][arm]["reinst_precision"][REF] * P[i]["arms"][arm]["n_reinstated"])
                 for i in ids if P[i]["arms"][arm].get("reinst_precision"))
        t1 = [ids.index(i) for i in tert["T1"]]
        uit[naam] = {"R1": tp / n_re if n_re else None, "dF1_mean": float(np.mean(d)), "dF1_median": float(median(d)),
                     "n_beter": int(sum(x > 0 for x in d)), "n": n, "p": p_w(d),
                     "T1_bias_A": float(np.mean([biasA[k] for k in t1])), "T1_bias_X": float(np.mean([biasx[k] for k in t1]))}
    # beslisregel
    c = uit["C_poort"]; b = uit["B_poort"]
    plafond = c["R1"] >= 0.60 and c["n_beter"] > n / 2 and (c["p"] or 1) < 0.05
    praktijk = b["n_beter"] > n / 2 and (b["p"] or 1) < 0.05 and (b["R1"] or 0) >= 0.40 and (b["T1_bias_X"] - b["T1_bias_A"]) <= 1.0
    uit["besluit"] = {"plafond_C_poort_repareert_eligibility": bool(plafond), "praktijk_B_poort_promotiekandidaat": bool(praktijk)}
    print("\nBESLISREGEL:", json.dumps(uit["besluit"]))
    json.dump(uit, open("docs/orakel_rule1a_poort_analyse.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
