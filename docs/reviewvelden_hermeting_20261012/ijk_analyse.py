#!/usr/bin/env python3
"""Herijking THERMISTOR_BAND_POWER_MIN: analyse van de server-CSV's
(`ijk_edf.csv`, `ijk_bdf.csv`), volgens het addendum van de preregistratie
12-10-2026. Bestandsnamen worden hier niet getoond; alleen tellingen.

    python ijk_analyse.py ijk_edf.csv [ijk_bdf.csv ...]
"""
from __future__ import annotations

import csv
import statistics as st
import sys

DREMPEL = 0.70
ONDER, BOVEN = 0.40, 0.95
# Post-hoc operationalisatie van "leeg gat dat 0,70 bevat": minstens 0,10 breed
# (het oorspronkelijke gat was 0,53). Vooraf stond alleen "leeg gat".
MIN_GAT = 0.10


def lees(paden):
    rijen = []
    for p in paden:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    a = float(r["fractie_alias"]); b = float(r["fractie_antialias"])
                except (KeyError, ValueError, TypeError):
                    continue
                rijen.append({"sf": float(r.get("sf") or 0), "alias": a, "anti": b,
                              "piek": r.get("piek_hz"), "nv": r.get("n_vensters"),
                              "bron": p})
    return rijen


def dedupe(rijen):
    """Dezelfde opname onder meerdere job-id's geeft identieke getallen."""
    gezien, uit = set(), []
    for r in rijen:
        k = (round(r["alias"], 4), round(r["anti"], 4), r["piek"], r["nv"], r["sf"])
        if k in gezien:
            continue
        gezien.add(k); uit.append(r)
    return uit


def gaten(waarden):
    w = sorted(v for v in waarden if ONDER <= v <= BOVEN)
    out = []
    for lo, hi in zip(w, w[1:]):
        out.append((hi - lo, lo, hi))
    return sorted(out, reverse=True)


def main(paden):
    alles = lees(paden)
    rijen = dedupe(alles)
    print(f"rijen: {len(alles)}, unieke montages (op getallen): {len(rijen)}")
    sfs = sorted({r['sf'] for r in rijen})
    print(f"bemonsterfrequenties: {sfs}")
    for modus in ("alias", "anti"):
        w = sorted(r[modus] for r in rijen)
        onder = sum(1 for v in w if v < DREMPEL)
        print(f"\n{modus}: n={len(w)} mediaan {st.median(w):.3f} min {w[0]:.3f} max {w[-1]:.3f} "
              f"onder {DREMPEL:.2f}: {onder} ({100*onder/len(w):.0f} %)")
        print("  gesorteerd: " + " ".join(f"{v:.2f}" for v in w))
    d = [r["anti"] - r["alias"] for r in rijen]
    print(f"\nverschil anti − alias: mediaan {st.median(d):+.4f}, max |Δ| {max(abs(x) for x in d):.4f}")
    wissel = [r for r in rijen if (r["alias"] < DREMPEL) != (r["anti"] < DREMPEL)]
    print(f"montages die van oordeel wisselen bij {DREMPEL:.2f}: {len(wissel)}")
    for r in wissel:
        print(f"   alias {r['alias']:.3f} -> anti {r['anti']:.3f} (sf {r['sf']:.0f} Hz)")
    # gat-analyse onder anti-alias
    w = sorted(r["anti"] for r in rijen)
    lager = [v for v in w if v < DREMPEL]; hoger = [v for v in w if v >= DREMPEL]
    lo = max(lager) if lager else None; hi = min(hoger) if hoger else None
    print(f"\nonder anti-alias: dichtstbijzijnde waarde onder {DREMPEL:.2f} = "
          f"{'—' if lo is None else f'{lo:.3f}'}, erboven = {'—' if hi is None else f'{hi:.3f}'}"
          + (f", gat {hi - lo:.3f}" if lo is not None and hi is not None else ""))
    g = gaten(w)
    if g:
        print(f"breedste lege opening in [{ONDER:.2f}, {BOVEN:.2f}]: "
              + ", ".join(f"{b:.3f} ({l:.3f}–{h:.3f}, midden {(l+h)/2:.3f})" for b, l, h in g[:3]))
    gat_070 = (lo is not None and hi is not None and (hi - lo) >= MIN_GAT)
    if gat_070:
        print(f"\nBESLISREGEL: 0,70 ligt in een leeg gat van {hi - lo:.3f} (≥ {MIN_GAT}) → drempel BLIJFT 0,70.")
    else:
        b, l, h = g[0] if g else (0, None, None)
        if b >= MIN_GAT:
            print(f"\nBESLISREGEL: 0,70 ligt in een bezette zone; breedste opening {b:.3f} ({l:.3f}–{h:.3f}) "
                  f"→ VOORSTEL aan Bart: {(l+h)/2:.2f} (niet doorgevoerd). "
                  f"Montages die bij {(l+h)/2:.2f} anders beoordeeld worden dan bij 0,70: "
                  f"{sum(1 for v in w if (v < DREMPEL) != (v < (l+h)/2))}.")
        else:
            print(f"\nBESLISREGEL: geen lege opening ≥ {MIN_GAT} in [{ONDER:.2f}, {BOVEN:.2f}] — de verdeling is "
                  f"continu; een drempel-in-het-gat is op deze montages niet af te leiden. Voorleggen aan Bart: "
                  f"de poort heeft op dit cohort geen natuurlijke scheiding; alleen een referentiemeting "
                  f"(welke thermistors scoren apneus die de druk mist) kan hem ijken.")


if __name__ == "__main__":
    main(sys.argv[1:])
