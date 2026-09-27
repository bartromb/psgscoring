#!/usr/bin/env python3
"""Baseline vs MSED op PSG-IPA: wie vindt welke referentie-arousals? (beschrijvend, geen tuning)

Per referentie-arousal: gevonden (IoU ≥ 0,20 met een voorspeld event) door beide, alleen
psgscoring, alleen MSED, of geen van beide. Plus eventduren en de F1 tussen de twee
detectoren onderling. Schrijft bench/eeg/results/complementariteit.{json,md}.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bench"))
from evaluate import lees_events, match  # noqa: E402

E = REPO / "bench" / "eeg"; RES = E / "results"
SNS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
ARMS = {"psgscoring": "baseline/{sn}_pred.csv", "MSED": "msed/out_dup/{sn}_pred.csv"}


def found_mask(pred, ref):
    _, _, _, pairs = match(pred, ref, "iou", 0.20)
    m = np.zeros(len(ref), dtype=bool)
    for _, j in pairs:
        m[j] = True
    return m


def main():
    RES.mkdir(exist_ok=True)
    tot = {"beide": 0, "alleen_psgscoring": 0, "alleen_MSED": 0, "geen": 0, "n_ref": 0}
    rows = []; out = {}
    for sn in SNS:
        ref = lees_events(E / f"ref/{sn}_ref_arousal.csv")
        p = {k: lees_events(E / v.format(sn=sn)) for k, v in ARMS.items()}
        fb, fm = found_mask(p["psgscoring"], ref), found_mask(p["MSED"], ref)
        c = {"beide": int((fb & fm).sum()), "alleen_psgscoring": int((fb & ~fm).sum()), "alleen_MSED": int((~fb & fm).sum()),
             "geen": int((~fb & ~fm).sum()), "n_ref": len(ref)}
        tp, fp, fn, _ = match(p["MSED"], p["psgscoring"], "iou", 0.20)
        f1_onderling = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        dur = {k: float(np.median([b - a for a, b, _ in v])) for k, v in p.items()}; dur["referentie"] = float(np.median([b - a for a, b, _ in ref]))
        # onsetfout van de gematchte events (voorspeld - referentie), mediaan
        onset_err = {}
        for k, v in p.items():
            _, _, _, pairs = match(v, ref, "iou", 0.20)
            d = [v[i][0] - ref[j][0] for i, j in pairs]
            onset_err[k] = float(np.median(d)) if d else None
        out[sn] = {**c, "f1_MSED_vs_psgscoring": f1_onderling, "mediane_duur_s": dur, "mediane_onsetfout_s": onset_err}
        for k in tot:
            tot[k] += c[k]
        rows.append(f"| {sn} | {len(ref)} | {c['beide']} | {c['alleen_psgscoring']} | {c['alleen_MSED']} | {c['geen']} | "
                    f"{f1_onderling:.3f} | {dur['referentie']:.1f} / {dur['psgscoring']:.1f} / {dur['MSED']:.1f} | "
                    f"{onset_err['psgscoring']:+.1f} / {onset_err['MSED']:+.1f} |".replace(".", ","))
    out["totaal"] = tot
    unie = tot["beide"] + tot["alleen_psgscoring"] + tot["alleen_MSED"]
    L = ["# Complementariteit psgscoring × MSED op PSG-IPA (referentie-arousals, IoU 0,20)\n",
         "| opname | n_ref | beide | alleen psgscoring | alleen MSED | geen | F1 MSED↔psgscoring | mediane duur ref / psgscoring / MSED (s) | mediane onsetfout psgscoring / MSED (s) |",
         "|---|---:|---:|---:|---:|---:|---:|---|---|", *rows,
         f"| **totaal** | {tot['n_ref']} | {tot['beide']} | {tot['alleen_psgscoring']} | {tot['alleen_MSED']} | {tot['geen']} | | | |",
         f"\nDekking van de referentie: psgscoring {(tot['beide'] + tot['alleen_psgscoring']) / tot['n_ref']:.3f}, MSED "
         f"{(tot['beide'] + tot['alleen_MSED']) / tot['n_ref']:.3f}, unie {unie / tot['n_ref']:.3f} (bovengrens van een ensemble-recall, "
         f"zonder rekening te houden met de fout-positieven van beide).".replace(".", ",")]
    (RES / "complementariteit.json").write_text(json.dumps(out, indent=1)); (RES / "complementariteit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
