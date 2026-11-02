#!/usr/bin/env python3
"""Analyse laatste tien: L*.json (zeven armen) + optioneel humaan_cijfers.json → analyse_uitvoer.md + samenvatting.json.
Aanroep: python analyse.py <map met L*.json> [humaan_cijfers.json]. Lezing volgens de prereg van 02-11 §5."""
import json, sys
from math import comb
from pathlib import Path
from statistics import mean, median

ARMEN = ["A0", "SB", "ST", "BR", "LG", "ALL", "REC"]; ENKEL = ["SB", "ST", "BR", "LG"]
KLASSEN = ["normaal", "licht", "matig", "ernstig"]


def klasse(ahi):
    if ahi is None: return None
    return "normaal" if ahi < 5 else "licht" if ahi < 15 else "matig" if ahi < 30 else "ernstig"


def tekentoets(diffs):
    d = [x for x in diffs if x is not None and abs(x) > 1e-9]
    n = len(d); k = sum(1 for x in d if x > 0)
    if n == 0: return None, 0, 0
    onder = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n; boven = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(onder, boven)), k, n - k


def f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def lees(map_):
    nachten = {}
    for p in sorted(Path(map_).glob("L*.json")):
        if "_" in p.stem: continue
        d = json.load(open(p)); nachten[d["R"]] = d
    return nachten


def main():
    map_ = Path(sys.argv[1]); nachten = lees(map_)
    mens = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and Path(sys.argv[2]).exists() else {}
    Ls = sorted(nachten); uit = ["# Analyse laatste tien (automatisch)\n"]
    fouten = [L for L in Ls if nachten[L].get("error")]
    uit.append(f"Nachten: {len(Ls)}; fouten: {fouten or 'geen'}; menselijke AHI aanwezig voor {sum(1 for L in Ls if (mens.get(L) or {}).get('ahi_h') is not None)} nachten.\n")
    ahi = {L: {a: (nachten[L]["armen"].get(a) or {}).get("ahi_total") for a in ARMEN} for L in Ls}
    # 1 identiteit
    uit.append("## 1 Identiteit A0 tegen opgeslagen AHI\n\n| L | opgeslagen | A0 | Δ | profiel/versie opgeslagen |\n|---|---|---|---|---|")
    ident = 0
    for L in Ls:
        o = nachten[L].get("ahi_opgeslagen"); a = ahi[L]["A0"]
        d = None if o is None or a is None else a - o; ident += int(d is not None and abs(d) <= 0.1)
        uit.append(f"| {L} | {f(o)} | {f(a)} | {f(d)} | {nachten[L].get('profiel_opgeslagen')} / {nachten[L].get('versie_opgeslagen')} |")
    uit.append(f"\nIdentiek (|Δ| ≤ 0,1): {ident}/{len(Ls)}\n")
    # 2 tabel per nacht
    kop = "| L | mens | " + " | ".join(ARMEN) + " | noemer (min) | TST machine | A0 op volle TST | TST mens |"
    uit.append("## 2 AHI per nacht en arm (noemer = artefactvrije slaap, summary.tst_minutes)\n\n" + kop + "\n|" + "---|" * (len(ARMEN) + 6))
    for L in Ls:
        m = (mens.get(L) or {}); den = ((nachten[L]["armen"].get("A0") or {}).get("summary") or {}).get("tst_minutes"); tm = nachten[L].get("tst_machine_min"); x = ahi[L]["A0"]
        n_ah = ((nachten[L]["armen"].get("A0") or {}).get("summary") or {}).get("n_ah_total")
        volle = None if None in (n_ah, tm) or not tm else n_ah / (tm / 60.0)   # exacte telling op de volle TST
        uit.append(f"| {L} | {f(m.get('ahi_h'))} | " + " | ".join(f(ahi[L][a]) for a in ARMEN) + f" | {f(den)} | {f(tm)} | {f(volle)} | {f(m.get('tst_h_min'), 0)} |")
    uit.append("")
    sam = {"n": len(Ls), "identiek": ident, "armen": {}}
    # 3 tegen de mens
    if any((mens.get(L) or {}).get("ahi_h") is not None for L in Ls):
        uit.append("## 3 Machine − mens per arm (AHI, /u)\n\n| arm | mediaan Δ | gemiddeld | min | max | +/− | tekentoets p | klasse exact | binnen 1 | mediaan Δ % |\n|---|---|---|---|---|---|---|---|---|---|")
        for a in ARMEN:
            diffs, pct, ex, b1, n = [], [], 0, 0, 0
            for L in Ls:
                h = (mens.get(L) or {}).get("ahi_h"); x = ahi[L][a]
                if h is None or x is None: continue
                n += 1; diffs.append(x - h); pct.append(100 * (x - h) / h if h > 0 else None)
                kh, kx = KLASSEN.index(klasse(h)), KLASSEN.index(klasse(x)); ex += int(kh == kx); b1 += int(abs(kh - kx) <= 1)
            if not diffs: continue
            p, kp, kn = tekentoets(diffs); pctv = [v for v in pct if v is not None]
            sam["armen"][a] = {"mediaan": median(diffs), "gemiddeld": mean(diffs), "min": min(diffs), "max": max(diffs), "pos": kp, "neg": kn, "p": p, "klasse_exact": ex, "binnen1": b1, "n": n}
            uit.append(f"| {a} | {f(median(diffs))} | {f(mean(diffs))} | {f(min(diffs))} | {f(max(diffs))} | {kp}/{kn} | {f(p, 3)} | {ex}/{n} | {b1}/{n} | {f(median(pctv), 0) if pctv else '—'} |")
        uit.append("")
        # TST-herrekening
        rijen = []
        for L in Ls:
            m = mens.get(L) or {}; h, th, tm, x = m.get("ahi_h"), m.get("tst_h_min"), nachten[L].get("tst_machine_min"), ahi[L]["A0"]
            den = ((nachten[L]["armen"].get("A0") or {}).get("summary") or {}).get("tst_minutes")   # noemer = artefactvrije slaap
            if None in (h, th, tm, x, den) or th <= 0: continue
            n_ah = ((nachten[L]["armen"].get("A0") or {}).get("summary") or {}).get("n_ah_total") or 0
            rijen.append(f"| {L} | {f(den, 0)} | {f(tm, 0)} | {f(th, 0)} | {f(x)} | {f(n_ah / (tm / 60.0))} | {f(n_ah / (th / 60.0))} | {f(h)} |")
        if rijen:
            uit.append("### 3b A0 herrekend op de volle machine-TST en op de menselijke TST (basis: noemer = artefactvrije slaap)\n\n| L | noemer | TST machine | TST mens | A0 | A0 op TST machine | A0 op TST mens | mens |\n|---|---|---|---|---|---|---|---|"); uit += rijen; uit.append("")
        # tellingen
        rijen = []
        for L in Ls:
            m = mens.get(L) or {}
            if m.get("n_apneu_h") is None and m.get("n_hypopneu_h") is None: continue
            pt = (nachten[L]["armen"].get("A0") or {}).get("per_type") or {}
            ap = sum(v for k, v in pt.items() if "hypopnea" not in k and "rera" not in k.lower()); hy = sum(v for k, v in pt.items() if "hypopnea" in k)
            rijen.append(f"| {L} | {ap} | {f(m.get('n_apneu_h'), 0)} | {hy} | {f(m.get('n_hypopneu_h'), 0)} |")
        if rijen:
            uit.append("### 3c Tellingen A0 tegen mens\n\n| L | apneus A0 | apneus mens | hypopneeën A0 | hypopneeën mens |\n|---|---|---|---|---|"); uit += rijen; uit.append("")
    # 4 drager
    uit.append("## 4 Verschuiving tegen A0 per één-wijziging-terug-arm (A0 − arm, /u)\n\n| arm | mediaan | gemiddeld | min | max | nachten |Δ| ≥ 1 | nachten |Δ| ≥ 5 |\n|---|---|---|---|---|---|---|")
    enkel = {}
    for a in ENKEL + ["ALL", "REC"]:
        d = [ahi[L]["A0"] - ahi[L][a] for L in Ls if ahi[L]["A0"] is not None and ahi[L][a] is not None]
        if not d: continue
        enkel[a] = d; sam.setdefault("tegen_A0", {})[a] = {"mediaan": median(d), "gemiddeld": mean(d), "min": min(d), "max": max(d)}
        uit.append(f"| {a} | {f(median(d))} | {f(mean(d))} | {f(min(d))} | {f(max(d))} | {sum(1 for x in d if abs(x) >= 1)} | {sum(1 for x in d if abs(x) >= 5)} |")
    uit.append("")
    # additiviteit
    rijen = []
    for L in Ls:
        if any(ahi[L][a] is None for a in ["A0", "ALL"] + ENKEL): continue
        som = sum(ahi[L]["A0"] - ahi[L][a] for a in ENKEL); tot = ahi[L]["A0"] - ahi[L]["ALL"]
        rijen.append(f"| {L} | {f(tot)} | {f(som)} | {f(tot - som)} |")
    if rijen:
        uit.append("### 4b Additiviteit: A0 − ALL tegen de som van de vier enkelvoudige verschuivingen\n\n| L | A0 − ALL | som | interactie |\n|---|---|---|---|"); uit += rijen; uit.append("")
    # 5 tellingen per arm (mediaan)
    uit.append("## 5 Tellingen per arm (mediaan over nachten)\n\n| arm | apneus | hypopneeën | hyp ≥ 3 % desat | hyp met arousalvlag | arousals | arousal-index | stage_baseline_used |\n|---|---|---|---|---|---|---|---|")
    for a in ARMEN:
        rows = [nachten[L]["armen"].get(a) or {} for L in Ls]; rows = [r for r in rows if r.get("per_type") is not None]
        if not rows: continue
        ap = [sum(v for k, v in r["per_type"].items() if "hypopnea" not in k and "rera" not in k.lower()) for r in rows]
        hy = [sum(v for k, v in r["per_type"].items() if "hypopnea" in k) for r in rows]
        sbu = sum(1 for r in rows if (r.get("resp_scalars") or {}).get("stage_baseline_used"))
        uit.append(f"| {a} | {f(median(ap), 0)} | {f(median(hy), 0)} | {f(median([r.get('hyp_desat3') or 0 for r in rows]), 0)} | {f(median([r.get('hyp_arousal_vlag') or 0 for r in rows]), 0)} | {f(median([r.get('n_arousals') or 0 for r in rows]), 0)} | {f(median([r.get('arousal_index') or 0 for r in rows]))} | {sbu}/{len(rows)} |")
    uit.append("")
    # 6 per nacht: sensoren, hypopneeroute, arousaldetectoren
    uit.append("## 6 Per nacht: apneusensoren (A0), hypopneeën A0/ST/REC, arousals U-Net tegen lgbm\n\n| L | TST | apneus A0 | druk-only | beide | th-only | thermistor bruikbaar | hyp A0 | hyp ST | hyp REC | hyp ≥3 % A0 | hyp arousal A0 | arousals U-Net (ruw) | arousals lgbm | AI U-Net | AI lgbm |\n|" + "---|" * 16)
    for L in Ls:
        a = nachten[L]["armen"].get("A0") or {}; st_ = nachten[L]["armen"].get("ST") or {}; rec = nachten[L]["armen"].get("REC") or {}; lg = nachten[L]["armen"].get("LG") or {}
        du = a.get("dual") or {}; tc = a.get("thermistor_check") or {}; asum = a.get("arousal_summary") or {}; lsum = lg.get("arousal_summary") or {}
        hyp = lambda r: sum(v for k, v in (r.get("per_type") or {}).items() if "hypopnea" in k)
        uit.append(f"| {L} | {f(nachten[L].get('tst_machine_min'), 0)} | {(a.get('summary') or {}).get('n_apnea_total')} | {du.get('n_pressure_only')} | {du.get('n_both')} | {du.get('n_thermistor_only')} | {tc.get('usable')} ({tc.get('gate')}) | {hyp(a)} | {hyp(st_)} | {hyp(rec)} | {a.get('hyp_desat3')} | {a.get('hyp_arousal_vlag')} | {a.get('n_arousals')} ({asum.get('unet_n_raw')}) | {lg.get('n_arousals')} | {f(a.get('arousal_index'))} | {f(lg.get('arousal_index'))} |")
    uit.append("")
    (map_ / "analyse_uitvoer.md").write_text("\n".join(uit)); json.dump(sam, open(map_ / "samenvatting.json", "w"), indent=1)
    print("\n".join(uit))


if __name__ == "__main__":
    main()
