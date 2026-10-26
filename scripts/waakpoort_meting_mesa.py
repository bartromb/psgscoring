#!/usr/bin/env python
"""waakpoort_meting_mesa.py — meting van het analysevenster op signaalgeldigheid (prereg
docs/waakpoort_preregistratie_20261026.md) op MESA-nachten zonder EEG/EOG/EMG.

Drie armen per nacht, profiel aasm_v3_breath_dual, kanaalset als de `hpg`-arm van de hPG-simulatie:
  T  huidige polygrafieroute: hypnogram alles "N2", registratietijd als noemer
  V  psgscoring.polygraphy_hypnogram_from_raw (automatisch venster, geen handmatig venster)
  O  orakel: NSRR-hypnogram (= hpg_hypno; het beste wat zonder EEG haalbaar is)
Per arm: AHI, ernst, events, match tegen de NSRR-referenties (validate_mesa); voor V de provenance;
daarnaast per nacht de O-events die V uitsluit, de events per arm met onset in een NSRR-W-epoch en
de NSRR-slaapminuten die V wegknipt. Ruwe rijen buiten git (--raw), samenvatting + rapport in --out.
"""
from __future__ import annotations
import argparse, json, statistics as st, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from hpg_simulatie_mesa import (EDF_DIR, XML_DIR, EPOCH_S, PRIMARY_REFERENCE, _arm_record,  # noqa: E402
                                _is_eeg_like, parse_nsrr, severity)

ARMEN = ("T", "V", "O")
SLAAP = ("N1", "N2", "N3", "R")


def analyse_one(args):
    rec_id, profile = args
    import mne
    mne.set_log_level("ERROR")
    import psgscoring
    t0 = time.time()
    try:
        raw = mne.io.read_raw_edf(str(EDF_DIR / f"{rec_id}.edf"), preload=True, verbose=False)
    except Exception as e:  # noqa: BLE001
        return {"recording": rec_id, "error": f"EDF: {e}"}
    dur = float(raw.times[-1])
    try:
        hypno_o, refs, tst_h = parse_nsrr(XML_DIR / f"{rec_id}-nsrr.xml", dur)
    except Exception as e:  # noqa: BLE001
        return {"recording": rec_id, "error": f"XML: {e}"}
    if tst_h < 1.0:
        return {"recording": rec_id, "error": f"TST {tst_h:.2f} h te kort"}
    n_epochs = int(dur // EPOCH_S)
    drop = [c for c in raw.ch_names if _is_eeg_like(c)]
    raw_hpg = raw.copy().drop_channels(drop) if drop else raw.copy()
    del raw
    hypno_t = ["N2"] * n_epochs
    try:
        hypno_v, prov = psgscoring.polygraphy_hypnogram_from_raw(raw_hpg, n_epochs=n_epochs)
    except Exception as e:  # noqa: BLE001
        return {"recording": rec_id, "error": f"venster: {e}"}
    hyp = {"T": hypno_t, "V": hypno_v, "O": hypno_o}
    out = {"recording": rec_id, "profile": profile, "tst_h": tst_h, "tib_h": n_epochs * EPOCH_S / 3600.0,
           "duration_h": dur / 3600.0, "n_ref": {k: len(v) for k, v in refs.items()},
           "ahi_ref": {k: len(v) / tst_h for k, v in refs.items()},
           "severity_ref": severity(len(refs[PRIMARY_REFERENCE]) / tst_h),
           "channels_dropped": drop, "venster": prov, "arms": {}}
    for arm in ARMEN:
        t1 = time.time()
        try:
            res = psgscoring.run_pneumo_analysis(raw_hpg, hypno=hyp[arm], scoring_profile=profile)
            rec = _arm_record(res, refs, hyp[arm])
        except Exception as e:  # noqa: BLE001
            rec = {"error": str(e)}
        rec["seconds"] = round(time.time() - t1, 1)
        # events met onset in een NSRR-W-epoch
        if "events" in rec:
            rec["n_in_nsrr_W"] = sum(1 for a, b, t in rec["events"]
                                     if not (0 <= int(a // EPOCH_S) < len(hypno_o) and hypno_o[int(a // EPOCH_S)] in SLAAP))
        out["arms"][arm] = rec
    # wat V van het orakel wegknipt
    o_ev = out["arms"]["O"].get("events", [])
    out["o_events_buiten_V"] = sum(1 for a, b, t in o_ev if 0 <= int(a // EPOCH_S) < n_epochs and hypno_v[int(a // EPOCH_S)] == "W")
    out["nsrr_slaap_epochs_buiten_V"] = sum(1 for i in range(min(n_epochs, len(hypno_o))) if hypno_o[i] in SLAAP and hypno_v[i] == "W")
    out["nsrr_W_epochs_buiten_V"] = sum(1 for i in range(min(n_epochs, len(hypno_o))) if hypno_o[i] not in SLAAP and hypno_v[i] == "W")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _f(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}".replace(".", ",")


def samenvatting(rows: list[dict]) -> tuple[dict, str]:
    from scipy.stats import wilcoxon
    ok = [r for r in rows if "error" not in r and all("ahi" in r["arms"][a] and r["arms"][a]["ahi"] is not None for a in ARMEN)]
    fouten = [r for r in rows if r not in ok]
    S = {"n": len(ok), "n_fouten": len(fouten), "fouten": [{"recording": r["recording"], "error": r.get("error") or {a: r["arms"][a].get("error") for a in ARMEN if a in r.get("arms", {})}} for r in fouten]}
    absd = {a: [abs(r["arms"][a]["ahi"] - r["arms"]["O"]["ahi"]) for r in ok] for a in ("T", "V")}
    sev_eq = {a: sum(r["arms"][a]["severity"] == r["arms"]["O"]["severity"] for r in ok) for a in ("T", "V")}
    o_tot = sum(r["arms"]["O"]["n_events"] for r in ok); o_lost = sum(r["o_events_buiten_V"] for r in ok)
    d = [t - v for t, v in zip(absd["T"], absd["V"])]
    p = float(wilcoxon(d).pvalue) if any(d) else 1.0
    S["per_arm"] = {}
    for a in ARMEN:
        f1 = [r["arms"][a]["match"][PRIMARY_REFERENCE]["f1"] for r in ok]
        S["per_arm"][a] = {"ahi_med": st.median(r["arms"][a]["ahi"] for r in ok), "f1_med": st.median(f1),
                           "n_events": sum(r["arms"][a]["n_events"] for r in ok), "n_apnea": sum(r["arms"][a]["n_apnea"] for r in ok),
                           "n_in_nsrr_W": sum(r["arms"][a].get("n_in_nsrr_W", 0) for r in ok),
                           "ernst_gelijk_ref": sum(r["arms"][a]["severity"] == r["severity_ref"] for r in ok),
                           "s_per_nacht_med": st.median(r["arms"][a]["seconds"] for r in ok)}
    S["vs_orakel"] = {a: {"abs_d_med": st.median(absd[a]), "abs_d_ge5": sum(x >= 5 for x in absd[a]), "ernst_gelijk_O": sev_eq[a],
                          "d_med": st.median(r["arms"][a]["ahi"] - r["arms"]["O"]["ahi"] for r in ok)} for a in ("T", "V")}
    S["wilcoxon_absd_T_min_V_p"] = p; S["absd_T_min_V_med"] = st.median(d) if d else None
    S["o_events_totaal"] = o_tot; S["o_events_buiten_V"] = o_lost; S["o_verlies_frac"] = (o_lost / o_tot) if o_tot else None
    S["nsrr_slaap_min_buiten_V_totaal"] = sum(r["nsrr_slaap_epochs_buiten_V"] for r in ok) * EPOCH_S / 60
    S["nachten_overknip_gt_5min"] = sum(1 for r in ok if r["nsrr_slaap_epochs_buiten_V"] * EPOCH_S / 60 > 5)
    S["uitgesloten_min_med"] = {k: st.median(r["venster"]["excluded_min"][k] for r in ok) for k in ("before_window", "after_window", "inside_window", "total")}
    S["venster_method"] = {m: sum(1 for r in ok if r["venster"]["method"] == m) for m in set(r["venster"]["method"] for r in ok)}
    ratio = sorted(r["tib_h"] / r["tst_h"] for r in ok); g = (ratio[len(ratio) // 3], ratio[2 * len(ratio) // 3]) if ratio else (0, 0)
    S["tib_tst_tertielgrenzen"] = g
    def tert(r): x = r["tib_h"] / r["tst_h"]; return "laag" if x < g[0] else "midden" if x < g[1] else "hoog"
    S["per_tib_tst_tertiel"] = {t: {a: {"abs_d_med": st.median([abs(r["arms"][a]["ahi"] - r["arms"]["O"]["ahi"]) for r in ok if tert(r) == t] or [0]),
                                      "abs_d_ge5": sum(abs(r["arms"][a]["ahi"] - r["arms"]["O"]["ahi"]) >= 5 for r in ok if tert(r) == t)} for a in ("T", "V")}
                               for t in ("laag", "midden", "hoog")}
    regel = {"1_ge5_halveert": S["vs_orakel"]["V"]["abs_d_ge5"] <= 0.5 * S["vs_orakel"]["T"]["abs_d_ge5"],
             "2_mediaan_lager_p<0.01": S["vs_orakel"]["V"]["abs_d_med"] < S["vs_orakel"]["T"]["abs_d_med"] and p < 0.01,
             "3_orakelverlies_le_0.5pct": S["o_verlies_frac"] is not None and S["o_verlies_frac"] <= 0.005,
             "4_ernst_gelijk_O_plus10": sev_eq["V"] >= sev_eq["T"] + 10}
    regel["alle"] = all(regel.values()); S["beslisregel"] = regel
    L = [f"# Waakpoort-meting — {len(ok)} nachten (fouten {len(fouten)}), profiel aasm_v3_breath_dual, kanalen zonder EEG/EOG/EMG\n",
         "| arm | AHI med | F1 med (aasm15) | events | apneus | events in NSRR-W | ernst = ref | \\|Δ\\| vs O med | \\|Δ\\| ≥ 5 | ernst = O | s/nacht |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ARMEN:
        x = S["per_arm"][a]; v = S["vs_orakel"].get(a, {})
        L.append(f"| {a} | {_f(x['ahi_med'])} | {_f(x['f1_med'], 3)} | {x['n_events']} | {x['n_apnea']} | {x['n_in_nsrr_W']} | {x['ernst_gelijk_ref']} | {_f(v.get('abs_d_med'))} | {v.get('abs_d_ge5', '—')} | {v.get('ernst_gelijk_O', '—')} | {_f(x['s_per_nacht_med'], 0)} |")
    L += ["", f"Wilcoxon |Δ_T| − |Δ_V|: mediaan {_f(S['absd_T_min_V_med'])}, p = {p:.2g}. O-events buiten V: {o_lost} van {o_tot} ({_f(100 * (S['o_verlies_frac'] or 0))} %). NSRR-slaapminuten weggeknipt: {_f(S['nsrr_slaap_min_buiten_V_totaal'], 0)} totaal, > 5 min op {S['nachten_overknip_gt_5min']} nachten.",
          f"Uitgesloten minuten mediaan: vóór {_f(S['uitgesloten_min_med']['before_window'], 1)}, na {_f(S['uitgesloten_min_med']['after_window'], 1)}, binnen {_f(S['uitgesloten_min_med']['inside_window'], 1)}, totaal {_f(S['uitgesloten_min_med']['total'], 1)}; venstermethode {S['venster_method']}.",
          f"Per TIB/TST-tertiel (grenzen {_f(g[0])} / {_f(g[1])}): " + "; ".join(f"{t}: T |Δ| {_f(S['per_tib_tst_tertiel'][t]['T']['abs_d_med'])} (≥5: {S['per_tib_tst_tertiel'][t]['T']['abs_d_ge5']}) → V {_f(S['per_tib_tst_tertiel'][t]['V']['abs_d_med'])} (≥5: {S['per_tib_tst_tertiel'][t]['V']['abs_d_ge5']})" for t in ("laag", "midden", "hoog")),
          "", "Beslisregel: " + ", ".join(f"{k}: {'JA' if v else 'NEE'}" for k, v in regel.items())]
    return S, "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--profile", default="aasm_v3_breath_dual")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    ids = [l.strip() for l in a.ids_file.read_text().split("\n") if l.strip()]
    if a.n: ids = ids[: a.n]
    a.out.mkdir(parents=True, exist_ok=True); a.raw.mkdir(parents=True, exist_ok=True)
    partial = a.raw / "rijen.partial.jsonl"
    rows = {}
    if partial.exists():
        for l in partial.read_text().split("\n"):
            if l.strip():
                r = json.loads(l); rows[r["recording"]] = r
    todo = [i for i in ids if i not in rows]
    print(f"{len(ids)} nachten, {len(rows)} al gedaan, {len(todo)} te doen, profiel {a.profile}, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, partial.open("a") as fh:
        for r in ex.map(analyse_one, [(i, a.profile) for i in todo]):
            rows[r["recording"]] = r
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"{len(rows)}/{len(ids)} {r['recording']} {r.get('seconds', '?')} s" + (f" FOUT {r['error']}" if "error" in r else ""), flush=True)
    lijst = [rows[i] for i in ids if i in rows]
    (a.raw / "rijen.json").write_text(json.dumps(lijst, indent=1))
    S, md = samenvatting(lijst)
    (a.out / "samenvatting.json").write_text(json.dumps(S, indent=1))
    (a.out / "rapport.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
