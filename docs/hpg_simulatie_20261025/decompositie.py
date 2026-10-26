#!/usr/bin/env python
"""decompositie.py — post-hoc, beschrijvend: waar komen de extra events van de `hpg`-arm vandaan?

Legt per nacht het NSRR-hypnogram (parse_nsrr, zoals de meting zelf) naast de events van de drie
armen en splitst de `hpg`-events op het slaapstadium van hun onset (W tegen slaap), op ligging in
de waak (vóór slaapbegin / WASO / na het laatste slaapepoch) en op type/duur. Schrijft
decompositie.json en decompositie.md naast rijen.json. Geen hermeting; alleen boekhouding op de
al geschreven eventlijsten.
"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path
import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent.parent / "scripts"))
from validate_mesa import parse_nsrr  # noqa: E402

XML_DIR = Path("/srv/DATA/MESA/mesa/polysomnography/annotations-events-nsrr")
EPOCH_S = 30.0
SLAAP = ("N1", "N2", "N3", "R")


def med(x):
    x = [v for v in x if v is not None]
    return float(st.median(x)) if x else None


def main() -> int:
    RAW = Path("/srv/CODE/docs/hpg_simulatie_20261025_raw")
    R = json.load(open((RAW if (RAW / "rijen.json").exists() else HIER) / "rijen.json"))
    per = []
    for r in R:
        rec = r["recording"]
        dur = r["duration_h"] * 3600.0
        hypno, _refs, tst_h = parse_nsrr(XML_DIR / f"{rec}-nsrr.xml", dur)
        n_ep = len(hypno)
        slaap_idx = [i for i, s in enumerate(hypno) if s in SLAAP]
        eerste, laatste = (slaap_idx[0], slaap_idx[-1]) if slaap_idx else (None, None)
        w_h = sum(1 for s in hypno if s == "W") * EPOCH_S / 3600.0
        w_voor_h = (eerste or 0) * EPOCH_S / 3600.0
        w_na_h = ((n_ep - 1 - laatste) if laatste is not None else 0) * EPOCH_S / 3600.0
        w_waso_h = max(w_h - w_voor_h - w_na_h, 0.0)

        def stadium(t):
            i = min(int(t // EPOCH_S), n_ep - 1)
            return hypno[i], i

        uit = {"recording": rec, "tst_h": r["tst_h"], "tib_h": r["tib_h"], "w_h": w_h,
               "w_voor_h": w_voor_h, "w_waso_h": w_waso_h, "w_na_h": w_na_h,
               "ahi_ref": r["ahi_ref"]["aasm15"], "psg_ahi": r["arms"]["psg"]["ahi"],
               "hpg_hypno_ahi": r["arms"]["hpg_hypno"]["ahi"], "hpg_ahi": r["arms"]["hpg"]["ahi"],
               "psg_n": r["arms"]["psg"]["n_events"], "hpg_hypno_n": r["arms"]["hpg_hypno"]["n_events"]}
        ev = r["arms"]["hpg"]["events"]
        tel = {"W": 0, "slaap": 0, "W_apnea": 0, "W_hyp": 0, "slaap_apnea": 0, "slaap_hyp": 0,
               "W_voor": 0, "W_waso": 0, "W_na": 0}
        duur_w, duur_s = [], []
        for a, b, typ in ev:
            s, i = stadium(a)
            is_ap = "hypopnea" not in typ  # typen: obstructive/central/mixed(/uncertain) tegen hypopnea*
            if s in SLAAP:
                tel["slaap"] += 1; tel["slaap_apnea" if is_ap else "slaap_hyp"] += 1; duur_s.append(b - a)
            else:
                tel["W"] += 1; tel["W_apnea" if is_ap else "W_hyp"] += 1; duur_w.append(b - a)
                if eerste is not None and i < eerste:
                    tel["W_voor"] += 1
                elif laatste is not None and i > laatste:
                    tel["W_na"] += 1
                else:
                    tel["W_waso"] += 1
        na_ap = [b - a for a, b, typ in ev if (lambda st_i: laatste is not None and st_i[1] > laatste and st_i[0] not in SLAAP)(stadium(a)) and "hypopnea" not in typ]
        na_hyp = [b - a for a, b, typ in ev if (lambda st_i: laatste is not None and st_i[1] > laatste and st_i[0] not in SLAAP)(stadium(a)) and "hypopnea" in typ]
        tel["W_na_apnea"] = len(na_ap); tel["W_na_hyp"] = len(na_hyp); uit["duur_W_na_apnea_med"] = med(na_ap)
        uit.update(tel)
        uit["hpg_n"] = len(ev)
        uit["hpg_slaap_per_tst"] = tel["slaap"] / r["tst_h"] if r["tst_h"] else None
        uit["hpg_slaap_per_tib"] = tel["slaap"] / r["tib_h"]
        uit["hpg_w_per_tib"] = tel["W"] / r["tib_h"]
        uit["w_event_rate_per_w_h"] = tel["W"] / w_h if w_h > 0 else None
        uit["duur_w_med"] = med(duur_w); uit["duur_slaap_med"] = med(duur_s)
        uit["psg_apnea"] = r["arms"]["psg"]["n_apnea"]; uit["hpg_hypno_apnea"] = r["arms"]["hpg_hypno"]["n_apnea"]
        per.append(uit)

    tot = lambda k: int(sum(p[k] for p in per))
    S = {
        "n": len(per),
        "hpg_events_totaal": tot("hpg_n"), "hpg_in_W": tot("W"), "hpg_in_slaap": tot("slaap"),
        "hpg_W_apnea": tot("W_apnea"), "hpg_W_hyp": tot("W_hyp"), "hpg_slaap_apnea": tot("slaap_apnea"), "hpg_slaap_hyp": tot("slaap_hyp"),
        "hpg_W_voor_slaapbegin": tot("W_voor"), "hpg_W_waso": tot("W_waso"), "hpg_W_na_laatste_slaap": tot("W_na"),
        "hpg_W_na_apnea": tot("W_na_apnea"), "hpg_W_na_hyp": tot("W_na_hyp"), "duur_W_na_apnea_med_van_medianen": med([p["duur_W_na_apnea_med"] for p in per]),
        "psg_events_totaal": tot("psg_n"), "hpg_hypno_events_totaal": tot("hpg_hypno_n"),
        "psg_apnea_totaal": tot("psg_apnea"), "hpg_hypno_apnea_totaal": tot("hpg_hypno_apnea"),
        "aandeel_hpg_events_in_W": tot("W") / tot("hpg_n"),
        "mediaan_aandeel_W_per_nacht": med([p["W"] / p["hpg_n"] if p["hpg_n"] else None for p in per]),
        "W_uren_totaal": float(sum(p["w_h"] for p in per)), "W_voor_h_med": med([p["w_voor_h"] for p in per]),
        "W_waso_h_med": med([p["w_waso_h"] for p in per]), "W_na_h_med": med([p["w_na_h"] for p in per]),
        "w_event_rate_per_w_h_med": med([p["w_event_rate_per_w_h"] for p in per]),
        "slaap_event_rate_psg_med": med([p["psg_ahi"] for p in per]),
        "duur_W_event_med_van_medianen": med([p["duur_w_med"] for p in per]),
        "duur_slaap_event_med_van_medianen": med([p["duur_slaap_med"] for p in per]),
        # decompositie van ΔAHI(hpg − psg) per nacht: noemer-effect op de slaapevents + waak-events/TIB
        "d_ahi_hpg_psg_med": med([p["hpg_ahi"] - p["psg_ahi"] for p in per]),
        "component_noemer_med": med([p["hpg_slaap_per_tib"] - p["psg_ahi"] for p in per]),
        "component_waak_med": med([p["hpg_w_per_tib"] for p in per]),
        # tegenproef: hpg-slaapevents / TST tegen hpg_hypno-AHI (zelfde kanaalset, wél hypnogram)
        "hpg_slaap_per_tst_minus_hpg_hypno_med": med([p["hpg_slaap_per_tst"] - p["hpg_hypno_ahi"] for p in per]),
        "hpg_slaap_n_minus_hpg_hypno_n_med": med([p["slaap"] - p["hpg_hypno_n"] for p in per]),
        "hpg_slaap_per_tst_minus_psg_med": med([p["hpg_slaap_per_tst"] - p["psg_ahi"] for p in per]),
        "nachten_W_events_ge_50pct": int(sum(1 for p in per if p["hpg_n"] and p["W"] / p["hpg_n"] >= 0.5)),
    }
    d = [p["hpg_slaap_per_tst"] - p["hpg_hypno_ahi"] for p in per]
    from scipy.stats import wilcoxon
    S["tegenproef_wilcoxon_p"] = float(wilcoxon(d).pvalue) if any(v != 0 for v in d) else 1.0
    # tertielen van de W-last (TIB/TST)
    ratio = sorted(p["tib_h"] / p["tst_h"] for p in per)
    g = (ratio[len(ratio) // 3], ratio[2 * len(ratio) // 3])
    S["tib_tst_tertielgrenzen"] = g
    S["d_ahi_hpg_psg_med_per_tib_tst_tertiel"] = {
        "laag": med([p["hpg_ahi"] - p["psg_ahi"] for p in per if p["tib_h"] / p["tst_h"] < g[0]]),
        "midden": med([p["hpg_ahi"] - p["psg_ahi"] for p in per if g[0] <= p["tib_h"] / p["tst_h"] < g[1]]),
        "hoog": med([p["hpg_ahi"] - p["psg_ahi"] for p in per if p["tib_h"] / p["tst_h"] >= g[1]])}
    S["abs_d_ahi_ge5_per_tib_tst_tertiel"] = {
        "laag": int(sum(1 for p in per if p["tib_h"] / p["tst_h"] < g[0] and abs(p["hpg_ahi"] - p["psg_ahi"]) >= 5)),
        "midden": int(sum(1 for p in per if g[0] <= p["tib_h"] / p["tst_h"] < g[1] and abs(p["hpg_ahi"] - p["psg_ahi"]) >= 5)),
        "hoog": int(sum(1 for p in per if p["tib_h"] / p["tst_h"] >= g[1] and abs(p["hpg_ahi"] - p["psg_ahi"]) >= 5))}
    (HIER / "decompositie.json").write_text(json.dumps({"samenvatting": S, "per_nacht": per}, indent=1))
    print(json.dumps(S, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
