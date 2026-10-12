#!/usr/bin/env python3
"""Analyse van de hermeting van de reviewvelden (preregistratie 12-10-2026).

Leest `mesa.json` (of, zolang de run loopt, `mesa.partial.jsonl`) en
`psgipa.json`, en schrijft `analyse_uitvoer.md` met per arm de gepaarde
verschillen tegen `nieuw` volgens de vooraf vastgelegde regels.

    PYTHONPATH=/srv/CODE/psgscoring python docs/reviewvelden_hermeting_20261012/analyse.py
"""
from __future__ import annotations

import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("vm", REPO / "scripts" / "validate_mesa.py")
vm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(vm)
_spec2 = importlib.util.spec_from_file_location("vp", REPO / "validate_psgipa.py")
vp = importlib.util.module_from_spec(_spec2); sys.modules["vp"] = vp; _spec2.loader.exec_module(vp)

REF = "aasm15"
PROF = "aasm_v3_breath_dual"
ARMEN = ["alles_oud", "oud_nadir", "oud_csr", "oud_plm", "oud_odi", "oud_spectraal",
         "oud_herstel", "oud_mixed", "oud_stadium", "oud_fase", "oud_antialias",
         "aan_masker", "aan_lineariseer"]
SUBTYPE_ARMEN = {"oud_csr", "oud_mixed", "oud_spectraal", "alles_oud"}


def sev(a):
    return "normal" if a < 5 else "mild" if a < 15 else "moderate" if a < 30 else "severe"


def wilcoxon(d):
    p, n = vm.wilcoxon_signed_rank(d)
    return p, n


def fmt_p(p):
    return "—" if p is None else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


# ───────────────────────────── MESA ─────────────────────────────

def laad_mesa(naam="mesa"):
    j = HERE / f"{naam}.json"
    if j.exists():
        return json.loads(j.read_text())["results"]
    pj = HERE / f"{naam}.partial.jsonl"
    if not pj.exists():
        return []
    rows = []
    for line in pj.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def paar(rows, lab_a, lab_b):
    """Gepaarde reeksen (nieuw = a, oud/alt = b) over opnames waar beide bestaan."""
    out = []
    for r in rows:
        if "error" in r:
            continue
        pa, pb = r["profiles"].get(lab_a), r["profiles"].get(lab_b)
        if not pa or not pb or "error" in pa or "error" in pb:
            continue
        ref_ahi = r["ahi_ref"][REF]
        out.append({
            "id": r["recording"], "ref": ref_ahi,
            "f1_a": pa["match"][REF]["f1"], "f1_b": pb["match"][REF]["f1"],
            "ahi_a": pa["ahi"], "ahi_b": pb["ahi"],
            "sev_a": sev(pa["ahi"]) == sev(ref_ahi), "sev_b": sev(pb["ahi"]) == sev(ref_ahi),
            "types_a": [a["type"] for a in pa.get("apneas", [])],
            "types_b": [a["type"] for a in pb.get("apneas", [])],
            "n_ev_a": pa["n_events"], "n_ev_b": pb["n_events"],
        })
    return out


def tertielen(paren):
    s = sorted(p["ref"] for p in paren)
    if len(s) < 3:
        return lambda p: "alle"
    g1, g2 = s[len(s) // 3], s[2 * len(s) // 3]
    return lambda p: "laag" if p["ref"] < g1 else ("midden" if p["ref"] < g2 else "hoog")


def regel_mesa(paren):
    """ΔF1 = nieuw − oud (positief = reparatie beter)."""
    d = [p["f1_a"] - p["f1_b"] for p in paren]
    bias_a = st.mean(p["ahi_a"] - p["ref"] for p in paren)
    bias_b = st.mean(p["ahi_b"] - p["ref"] for p in paren)
    sev_a = sum(p["sev_a"] for p in paren); sev_b = sum(p["sev_b"] for p in paren)
    p_val, n_eff = wilcoxon(d)
    identiek = sum(1 for x in d if abs(x) < 1e-12)
    return {
        "n": len(paren), "med": st.median(d) if d else None, "mean": st.mean(d) if d else None,
        "beter": sum(1 for x in d if x > 1e-12), "slechter": sum(1 for x in d if x < -1e-12),
        "gelijk": identiek, "p": p_val, "bias_nieuw": bias_a, "bias_oud": bias_b,
        "sev_nieuw": sev_a, "sev_oud": sev_b,
        "events_nieuw": sum(p["n_ev_a"] for p in paren), "events_oud": sum(p["n_ev_b"] for p in paren),
    }


def oordeel_default_aan(s):
    """Regel 1 van de preregistratie: ter heroverweging als de reparatie schaadt."""
    redenen = []
    if s["med"] is not None and s["med"] < 0 and s["p"] is not None and s["p"] < 0.05:
        redenen.append(f"mediaan ΔF1 {s['med']:+.4f} < 0, p = {fmt_p(s['p'])}")
    if abs(s["bias_nieuw"]) - abs(s["bias_oud"]) >= 1.0:
        redenen.append(f"|bias| {abs(s['bias_oud']):.2f} → {abs(s['bias_nieuw']):.2f} /u")
    if s["sev_oud"] - s["sev_nieuw"] >= 3:
        redenen.append(f"ernstconcordantie {s['sev_oud']} → {s['sev_nieuw']}")
    return ("**HEROVERWEGEN**: " + "; ".join(redenen)) if redenen else "blijft default"


def oordeel_default_uit(s):
    """Regel 2: kandidaat voor default-aan (MESA-deel).

    Let op het teken: `s` is nieuw − arm, en bij een `aan_*`-arm is de ARM de
    ingeschakelde reparatie. Inschakelen wint dus als nieuw − arm < 0 (de arm
    heeft de hogere F1), |bias_arm| ≤ |bias_nieuw| en sev_arm ≥ sev_nieuw.
    (Nalezing 12-10: stond eerst met omgekeerd teken.)"""
    ok = (s["med"] is not None and s["med"] < 0 and s["p"] is not None and s["p"] < 0.05
          and abs(s["bias_oud"]) <= abs(s["bias_nieuw"]) and s["sev_oud"] >= s["sev_nieuw"])
    return "MESA-deel JA — PSG-IPA (binnen bereik) beslist mee" if ok else "blijft uit (MESA-deel NEE)"


def subtype_tel(types):
    t = {"obstructive": 0, "central": 0, "mixed": 0, "uncertain": 0}
    for x in types:
        if x in t:
            t[x] += 1
    return t


def mesa_sectie(rows, PROF=PROF, ARMEN=ARMEN, kop="run 1"):
    out = [f"## MESA {kop}: `{PROF}` (held-out nachten, referentie NSRR aasm15, matcher IoU 0,20 typeonbewust)", ""]
    n_ok = sum(1 for r in rows if "error" not in r)
    out.append(f"Opnames met resultaat: **{n_ok}** (van 30 gevraagd)." + ("" if n_ok == 30 else " *(tussenstand)*"))
    out.append("")
    nieuw = f"{PROF}|nieuw"
    out.append(f"### `{PROF}`: elke arm tegen `nieuw` (ΔF1 = nieuw − arm)")
    out.append("")
    out.append("| arm | n | ΔF1 mediaan | gemiddeld | beter/slechter/gelijk | p | bias nieuw → arm | ernst nieuw → arm | events nieuw → arm | oordeel |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for arm in ARMEN:
        paren = paar(rows, nieuw, f"{PROF}|{arm}")
        if not paren:
            out.append(f"| {arm} | 0 | | | | | | | | nog geen paren |")
            continue
        s = regel_mesa(paren); stats[arm] = (s, paren)
        oordeel = oordeel_default_uit(s) if arm.startswith("aan_") else oordeel_default_aan(s)
        out.append(f"| {arm} | {s['n']} | {s['med']:+.4f} | {s['mean']:+.4f} | {s['beter']}/{s['slechter']}/{s['gelijk']} | {fmt_p(s['p'])} | {s['bias_nieuw']:+.2f} → {s['bias_oud']:+.2f} | {s['sev_nieuw']} → {s['sev_oud']} | {s['events_nieuw']} → {s['events_oud']} | {oordeel} |")
    out.append("")
    out.append("Let op het teken: de tabel vergelijkt `nieuw` met de arm; voor de `oud_*`-armen is een positieve ΔF1 winst van de reparatie, voor de `aan_*`-armen is een **negatieve** ΔF1 winst van het inschakelen.")
    out.append("")
    # rec-anker (alleen zinvol in run 1, waar rec met nieuw/alles_oud meeliep)
    out.append("### `aasm_v3_rec`: alles_oud tegen nieuw (in deze run)")
    paren = paar(rows, "aasm_v3_rec|nieuw", "aasm_v3_rec|alles_oud")
    if paren:
        s = regel_mesa(paren)
        out.append(f"n = {s['n']}: ΔF1 mediaan {s['med']:+.4f} (gemiddeld {s['mean']:+.4f}), beter/slechter/gelijk {s['beter']}/{s['slechter']}/{s['gelijk']}, p = {fmt_p(s['p'])}; bias {s['bias_oud']:+.2f} → {s['bias_nieuw']:+.2f} /u; ernst {s['sev_oud']} → {s['sev_nieuw']}; events {s['events_oud']} → {s['events_nieuw']}. Oordeel: {oordeel_default_aan(s)}")
        anker = paar(rows, "aasm_v3_rec", "aasm_v3_rec|nieuw")
        if anker:
            d = [abs(p["f1_a"] - p["f1_b"]) for p in anker]
            out.append(f"Anker: `aasm_v3_rec` (harnasbasislijn) vs `aasm_v3_rec|nieuw` max |ΔF1| = {max(d):.6f} over {len(anker)} nachten (hoort 0 te zijn).")
    else:
        out.append("nog geen paren")
    out.append("")
    # per tertiel voor alles_oud
    if "alles_oud" in stats:
        s, paren = stats["alles_oud"]
        tert = tertielen(paren)
        out.append(f"### Netto-effect van de review per AHI-tertiel van de referentie (`{PROF}`, nieuw − alles_oud)")
        out.append("")
        out.append("| tertiel | n | ΔF1 mediaan | beter/slechter/gelijk | bias oud → nieuw | ernst oud → nieuw |")
        out.append("|---|---|---|---|---|---|")
        for naam in ("laag", "midden", "hoog", "alle"):
            sub = [p for p in paren if tert(p) == naam]
            if not sub:
                continue
            ss = regel_mesa(sub)
            out.append(f"| {naam} | {ss['n']} | {ss['med']:+.4f} | {ss['beter']}/{ss['slechter']}/{ss['gelijk']} | {ss['bias_oud']:+.2f} → {ss['bias_nieuw']:+.2f} | {ss['sev_oud']} → {ss['sev_nieuw']} |")
        out.append("")
    # subtypes
    out.append("### Apneu-subtypes (tellingen over alle nachten; nieuw → arm)")
    out.append("")
    out.append("| arm | obstructief | centraal | gemengd | uncertain |")
    out.append("|---|---|---|---|---|")
    for arm in ARMEN:
        if arm not in stats:
            continue
        s, paren = stats[arm]
        ta = subtype_tel(sum((p["types_a"] for p in paren), []))
        tb = subtype_tel(sum((p["types_b"] for p in paren), []))
        if ta == tb and arm not in SUBTYPE_ARMEN:
            continue
        out.append(f"| {arm} | {ta['obstructive']} → {tb['obstructive']} | {ta['central']} → {tb['central']} | {ta['mixed']} → {tb['mixed']} | {ta['uncertain']} → {tb['uncertain']} |")
    out.append("")
    return out


# ───────────────────────────── PSG-IPA ─────────────────────────────

def laad_psgipa():
    j = HERE / "psgipa.json"
    return json.loads(j.read_text()) if j.exists() else []


def scorer_sets(sn_id, data_dir=Path("/srv/DATA/PSG-IPA")):
    import mne
    mne.set_log_level("ERROR")
    psg = data_dir / "Resp_events" / "PSG" / f"{sn_id}_Respiration.edf"
    dur = float(mne.io.read_raw_edf(str(psg), preload=False, verbose="ERROR").times[-1])
    return [vp.event_set(f, dur) for f in vp.find_scorer_files(data_dir, sn_id)]


def psgipa_sectie(rows):
    out = ["## PSG-IPA (SN1–SN5, 12 scoorders, hypnogram scoorder 1; plafond respiratoire F1 0,667)", ""]
    ok = [r for r in rows if "error" not in r]
    if not ok:
        out.append("nog geen resultaat"); return out
    labels = sorted({l for r in ok for l in r["profiles"]})
    cache = {}
    def f1_median(r, label):
        ev = r["profiles"][label].get("events")
        if ev is None:
            return None
        if r["recording"] not in cache:
            cache[r["recording"]] = scorer_sets(r["recording"])
        vals = []
        for s in cache[r["recording"]]:
            m = vp.match_events([tuple(e) for e in ev], s, iou_thresh=0.20, type_aware=False)
            vals.append(m["f1"])
        return st.median(vals) if vals else None
    out.append("| arm | bias | MAE | binnen bereik | ernst | event-F1 mediaan (min–max over opnames) |")
    out.append("|---|---|---|---|---|---|")
    for label in labels:
        vals = [(r["profiles"][label].get("ahi"), r) for r in ok]
        if any(v is None for v, _ in vals):
            out.append(f"| {label} | onvolledig | | | | |"); continue
        b = [v - r["ref_median"] for v, r in vals]
        inr = sum(1 for v, r in vals if r["ref_lo"] <= v <= r["ref_hi"])
        sv = sum(1 for v, r in vals if sev(v) == sev(r["ref_median"]))
        f1s = [f1_median(r, label) for r in ok]
        f1s = [x for x in f1s if x is not None]
        f1txt = f"{st.median(f1s):.3f} ({min(f1s):.3f}–{max(f1s):.3f})" if f1s else "—"
        out.append(f"| {label} | {st.mean(b):+.2f} | {st.mean(abs(x) for x in b):.2f} | {inr}/{len(ok)} | {sv}/{len(ok)} | {f1txt} |")
    out.append("")
    out.append("Regel 2 (default-uit-velden): `aan_masker`/`aan_lineariseer` mogen het aantal binnen het bereik niet verlagen t.o.v. `nieuw`.")
    out.append("")
    return out


def main():
    md = ["# Hermeting reviewvelden — analyse-uitvoer", "",
          "Automatisch gegenereerd door `analyse.py`; preregistratie "
          "`docs/reviewvelden_hermeting_preregistratie_20261012.md`.", ""]
    md += mesa_sectie(laad_mesa("mesa"))
    ARMEN_REC = ["alles_oud", "oud_nadir", "oud_csr", "oud_spectraal", "oud_herstel",
                 "oud_mixed", "oud_fase", "aan_masker", "aan_lineariseer"]
    rec_rows = laad_mesa("mesa_rec")
    if rec_rows:
        md += mesa_sectie(rec_rows, PROF="aasm_v3_rec", ARMEN=ARMEN_REC, kop="run 2")
    md += psgipa_sectie(laad_psgipa())
    (HERE / "analyse_uitvoer.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
