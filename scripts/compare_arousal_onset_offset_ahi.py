#!/usr/bin/env python3
"""Gepaarde vergelijking: arousal-onsets 0 s tegen +2 s, op dezelfde 30 opnames.

De vraag is NIET of de arousals beter liggen -- dat is al gemeten. De vraag is
wat de verschuiving doet met wat de kliniek leest: AHI, RDI en de ernstklasse.
Arousals bevestigen Rule 1B-hypopneus (venster 15 s) en dragen de RERA-detectie,
dus ze zijn ook INVOER.

Controle: de arousalTELLING moet identiek zijn. Een verschuiving verplaatst
events en maakt er geen. Wijkt hij af, dan meet dit iets anders dan bedoeld.
"""
import json, sys
from pathlib import Path
from statistics import median, mean
from math import comb

AHI = Path(sys.argv[1] if len(sys.argv) > 1 else
           "./mesa_onset_ab")
PROF = "aasm_v3_breath"
CTRL = "aasm_v3_rec"
REF = "aasm15"


def klasse(x, grenzen=(5, 15, 30)):
    if x is None: return None
    for i, g in enumerate(grenzen):
        if x < g: return i
    return len(grenzen)


def tekentoets(d):
    nz = [x for x in d if x != 0]
    if not nz: return 1.0, 0, 0
    k = sum(1 for x in nz if x > 0)
    staart = max(k, len(nz) - k)
    p = min(1.0, 2 * sum(comb(len(nz), i) for i in range(staart, len(nz) + 1))
            / 2 ** len(nz))
    return p, k, len(nz)


def laad(pad):
    d = json.loads(Path(pad).read_text())
    return {r["recording"]: r for r in d["results"]}


a = laad(AHI / "mesa_off0.json")
b = laad(AHI / "mesa_off2.json")
gedeeld = sorted(set(a) & set(b))
print(f"gepaarde opnames: {len(gedeeld)}  (0 s: {len(a)}, +2 s: {len(b)})\n")

for prof in (PROF, CTRL):
    ok = [r for r in gedeeld
          if "error" not in a[r]["profiles"].get(prof, {"error": 1})
          and "error" not in b[r]["profiles"].get(prof, {"error": 1})]
    if not ok:
        print(f"== {prof}: geen bruikbare opnames\n"); continue
    A = [a[r]["profiles"][prof] for r in ok]
    B = [b[r]["profiles"][prof] for r in ok]

    print(f"══ {prof}  (n = {len(ok)})")

    # Controle: telling identiek?
    dn = [y.get("n_arousals", 0) - x.get("n_arousals", 0) for x, y in zip(A, B)]
    anders = sum(1 for x in dn if x != 0)
    vlag = "" if anders == 0 else f"   << WIJKT AF op {anders} opnames"
    print(f"   controle arousaltelling: gelijk op {len(ok)-anders}/{len(ok)}{vlag}")

    for naam, sleutel in (("AHI", "ahi"), ("RDI", "rdi")):
        xa = [x.get(sleutel) for x in A]; xb = [y.get(sleutel) for y in B]
        paren = [(u, v) for u, v in zip(xa, xb) if u is not None and v is not None]
        if not paren:
            print(f"   {naam}: geen waarden"); continue
        d = [v - u for u, v in paren]
        p, k, nz = tekentoets(d)
        kl = sum(1 for u, v in paren if klasse(u) != klasse(v))
        print(f"   {naam:<4} mediaan {median([u for u,_ in paren]):6.2f} → "
              f"{median([v for _,v in paren]):6.2f}  |  gepaarde Δ mediaan "
              f"{median(d):+6.2f}  gem {mean(d):+6.2f}  |  "
              f"anders op {sum(1 for x in d if x!=0)}/{len(d)}, "
              f"tekentoets p={p:.3g}  |  ernstklasse verschuift op {kl}/{len(paren)}")

    # AHI-bias tegen de menselijke referentie
    bias_a, bias_b = [], []
    for r, x, y in zip(ok, A, B):
        ref = (a[r].get("ahi_ref") or {}).get(REF)
        if ref is None or x.get("ahi") is None or y.get("ahi") is None: continue
        bias_a.append(x["ahi"] - ref); bias_b.append(y["ahi"] - ref)
    if bias_a:
        print(f"   AHI-bias tegen {REF}: {median(bias_a):+6.2f} → {median(bias_b):+6.2f} "
              f"(mediaan)")

    # Respiratoire event-F1 tegen de referentie
    fa = [x.get("match", {}).get(REF, {}).get("f1") for x in A]
    fb = [y.get("match", {}).get(REF, {}).get("f1") for y in B]
    par = [(u, v) for u, v in zip(fa, fb) if u is not None and v is not None]
    if par:
        d = [v - u for u, v in par]
        p, k, nz = tekentoets(d)
        print(f"   event-F1 ({REF}): {median([u for u,_ in par]):.4f} → "
              f"{median([v for _,v in par]):.4f}  |  gepaarde Δ {mean(d):+.4f}  "
              f"beter op {k}/{nz} (niet-nul), p={p:.3g}")
    print()

print("Vooraf: dit is een IMPACTmeting, geen criterium. De vraag is of de\n"
      "verschuiving de klinische index onberoerd laat. Verschuift de ernstklasse\n"
      "op een noemenswaardig deel, dan is dat de prijs die naast de +0,014\n"
      "arousal-F1 gelegd moet worden.")
