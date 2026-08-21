#!/usr/bin/env python3
"""Beoordeel de AHI-impact van de arousalhybride tegen de preregistratie.

Leest de twee armen van `docs/mesa_arousal_ahi_lgbm{0,1}_n150.json` en toetst
de criteria uit `docs/arousal_lgbm_preregistratie.md`:

  Voorwaarde 1  bias mag niet meer dan 1,0/u verslechteren
                event-F1 mag niet meer dan 0,010 dalen

`aasm_v3_rec` is de CONTROLE: `arousal_limb_wired` staat daar uit, dus de twee
armen horen identiek te zijn. Een verschil betekent dat er iets lekt en dat de
toets op `aasm_v3_breath` niet los gelezen mag worden.

Werkt ook op een half afgeronde run: geef `--partial` mee om de
`.partial.jsonl`-checkpoints te lezen in plaats van de eind-JSON.

Gebruik:
    python docs/analyse_arousal_ahi.py [--partial] [--ref aasm15]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

DOCS = Path("/home/bart/CODE/docs")
STEM = "mesa_arousal_ahi_lgbm{arm}_n150"
PROFILES = ("aasm_v3_rec", "aasm_v3_breath")
BIAS_LIMIT = 1.0        # /u verslechtering
F1_LIMIT = 0.010        # daling


def load(arm: int, partial: bool) -> dict:
    base = DOCS / STEM.format(arm=arm)
    if not partial and (p := base.with_suffix(".json")).exists():
        d = json.loads(p.read_text())
        return {"meta": d.get("meta", {}),
                "rows": {r["recording"]: r for r in d["results"]}}
    p = Path(str(base) + ".partial.jsonl")
    rows = {}
    for line in p.read_text().splitlines() if p.exists() else []:
        try:
            r = json.loads(line)
        except Exception:      # noqa: BLE001 — halve regel na een onderbreking
            continue
        rows[r.get("recording")] = r
    return {"meta": {"(partial)": True}, "rows": rows}


def severity(ahi: float) -> str:
    if ahi < 5:
        return "normaal"
    if ahi < 15:
        return "licht"
    if ahi < 30:
        return "matig"
    return "ernstig"


def wilcoxon_p(diffs: list[float]) -> float | None:
    """Tweezijdige Wilcoxon signed-rank; None als scipy ontbreekt."""
    nz = [d for d in diffs if d != 0]
    if len(nz) < 6:
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(nz).pvalue)
    except Exception:      # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--ref", default="aasm15")
    ap.add_argument("--dir", type=Path, default=None,
                    help="map met de twee armen (default docs/)")
    ap.add_argument("--stem", default=None,
                    help="bestandsnaam met {arm} erin, voor een andere run")
    a = ap.parse_args()
    global DOCS, STEM
    if a.dir:
        DOCS = a.dir
    if a.stem:
        STEM = a.stem

    A, B = load(0, a.partial), load(1, a.partial)
    recs = sorted(set(A["rows"]) & set(B["rows"]))
    if not recs:
        print("nog geen gepaarde opnames")
        return
    print(f"gepaard over {len(recs)} opnames, referentie {a.ref}")
    for lbl, m in (("regels", A["meta"]), ("hybride", B["meta"])):
        sha = m.get("git_sha") or "?"
        print(f"  {lbl}: psgscoring {m.get('psgscoring_version','?')} "
              f"sha {sha}{' (vuile werkboom)' if m.get('git_dirty') else ''}")
    print()

    for prof in PROFILES:
        rows = []
        for rc in recs:
            pa, pb = A["rows"][rc]["profiles"].get(prof), B["rows"][rc]["profiles"].get(prof)
            if not pa or not pb:
                continue
            ref = A["rows"][rc]["ahi_ref"][a.ref]
            rows.append((rc, pa["ahi"] - ref, pb["ahi"] - ref,
                         pa["match"][a.ref]["f1"], pb["match"][a.ref]["f1"],
                         pa["ahi"], pb["ahi"], ref))
        if not rows:
            continue
        bias_o = st.mean(r[1] for r in rows)
        bias_n = st.mean(r[2] for r in rows)
        f1_o = st.median(r[3] for r in rows)
        f1_n = st.median(r[4] for r in rows)
        d_bias = abs(bias_n) - abs(bias_o)          # positief = slechter
        d_f1 = f1_n - f1_o
        identiek = all(abs(r[5] - r[6]) < 1e-9 for r in rows)
        beter = sum(1 for r in rows if r[4] > r[3])
        slechter = sum(1 for r in rows if r[4] < r[3])
        p = wilcoxon_p([r[4] - r[3] for r in rows])
        sev = sum(1 for r in rows if severity(r[5]) != severity(r[6]))

        rol = "CONTROLE" if prof == "aasm_v3_rec" else "TOETS"
        print(f"{prof}  [{rol}]  n={len(rows)}")
        if prof == "aasm_v3_rec":
            print(f"   armen identiek: {identiek}"
                  + ("" if identiek else "   <== LET OP: dit profiel gebruikt "
                                         "de arousals niet, dus dit hoort niet te kunnen"))
        print(f"   bias   {bias_o:+7.2f}/u -> {bias_n:+7.2f}/u   "
              f"|bias| {d_bias:+.2f}  ({'binnen' if d_bias <= BIAS_LIMIT else 'BUITEN'} "
              f"de grens van {BIAS_LIMIT:.1f})")
        print(f"   F1     {f1_o:7.3f}   -> {f1_n:7.3f}     "
              f"verschil {d_f1:+.3f}  ({'binnen' if d_f1 >= -F1_LIMIT else 'BUITEN'} "
              f"de grens van {F1_LIMIT:.3f})")
        print(f"   F1 per opname: {beter} beter, {slechter} slechter, "
              f"{len(rows)-beter-slechter} gelijk"
              + (f"   Wilcoxon p = {p:.4g}" if p is not None else ""))
        print(f"   ernstklasse verschuift op {sev}/{len(rows)} opnames")
        if prof != "aasm_v3_rec":
            gehaald = d_bias <= BIAS_LIMIT and d_f1 >= -F1_LIMIT
            print(f"   VOORWAARDE 1: {'GEHAALD' if gehaald else 'NIET GEHAALD'}")
        print()


if __name__ == "__main__":
    main()
