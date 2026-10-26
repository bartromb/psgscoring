#!/usr/bin/env python3
"""hPG-simulatie op MESA: wat verliest de scoring zonder EEG, EOG en kin-EMG?

Drie armen per nacht, zelfde profiel (default `aasm_v3_breath_dual`, de
productiestandaard sinds 07-10-2026):

  psg        volledige kanaalset, NSRR-hypnogram               (zoals validate_mesa.py)
  hpg_hypno  zónder EEG/EOG/EMG-kanalen, NSRR-hypnogram        (alleen de arousalpoot valt weg;
                                                                 noemer = TST, dus gelijk aan psg)
  hpg        zónder EEG/EOG/EMG, hypnogram = alle epochs "N2"  (registratietijd als noemer:
                                                                 de polygrafieroute van
                                                                 YASAFlaskified/tasks.py, REI)

Referentie en koppeling: NSRR-events zoals `scripts/validate_mesa.py` ze reconstrueert
(`aasm15` primair: 3 % of arousal, alle apneus; daarnaast `desat3_all`, `oahi3`, `oahi4`),
matcher `LEGACY_MATCHER` uit validate_psgipa. Beschrijvende verschilmeting, geen beslisregel;
preregistratie `docs/hpg_simulatie_preregistratie_20261025.md`.

    OMP_NUM_THREADS=1 PSGSCORING_UNET_THREADS=1 .venv/bin/python scripts/hpg_simulatie_mesa.py \
        --ids-file <lijst> --out docs/hpg_simulatie_20261025 --workers 8

Hervat vanzelf uit `<out>/rijen.partial.jsonl`. Niets van MESA wordt gekopieerd; de uitvoer
bevat alleen getallen en eventgrenzen (geen PHI: MESA is publiek, de id's zijn NSRR-id's).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PSGSCORING_UNET_THREADS", "1")

import numpy as np  # noqa: E402

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
sys.path.insert(0, str(HIER.parent))

from validate_mesa import (  # noqa: E402
    APNEA_TYPES, LEGACY_MATCHER, PRIMARY_REFERENCE, REFERENCES,
    match_events, parse_nsrr, severity, wilcoxon_signed_rank,
)

MESA = Path("/srv/DATA/MESA/mesa")
EDF_DIR = MESA / "polysomnography" / "edfs"
XML_DIR = MESA / "polysomnography" / "annotations-events-nsrr"
EPOCH_S = 30.0
ARMEN = ("psg", "hpg_hypno", "hpg")
# Alles wat een thuispolygrafie niet heeft: EEG, EOG, kin-EMG, ook de *_Off-varianten.
EEG_PREFIX = ("EEG", "EOG", "EMG")


def _is_eeg_like(name: str) -> bool:
    return name.upper().startswith(EEG_PREFIX)


def _arm_record(res: dict, refs: dict, hypno: list) -> dict:
    r = res.get("respiratory", {}) or {}
    summ = r.get("summary", {}) or {}
    ev = [e for e in r.get("events", []) if e.get("onset_s") is not None]
    algo = [(float(e["onset_s"]),
             float(e["onset_s"]) + float(e.get("duration_s") or 0.0),
             str(e.get("type"))) for e in ev]
    ar = res.get("arousal") or {}
    ars = ar.get("summary") or {}
    hyp = [e for e in ev if "hypopnea" in str(e.get("type"))]
    rec = {
        "ahi": float(summ.get("ahi_total") or summ.get("ahi") or 0.0),
        "ahi_incl_uncertain": summ.get("ahi_incl_uncertain"),
        "rdi": (float(summ["rdi"]) if summ.get("rdi") is not None else None),
        "severity": severity(float(summ.get("ahi_total") or summ.get("ahi") or 0.0)),
        "n_events": len(algo),
        "n_apnea": sum(1 for e in ev if str(e.get("type")) in APNEA_TYPES),
        "n_hypopnea": len(hyp),
        "n_hyp_desat3": sum(1 for e in hyp if (e.get("desaturation_pct") or 0.0) >= 3.0),
        "n_hyp_rule1a_arousal": sum(1 for e in hyp if e.get("rule1a_arousal")),
        "n_arousals": len(ar.get("events") or []),
        "arousal_index": ars.get("arousal_index"),
        "arousal_detector": ars.get("detector"),
        "arousal_fallback": ars.get("unet_fallback_reason"),
        "arousal_error": ar.get("error"),
        "denominator_h": sum(1 for s in hypno if s in ("N1", "N2", "N3", "R")) * EPOCH_S / 3600.0,
        "hypoxic_burden": ((res.get("spo2") or {}).get("summary") or {}).get("hypoxic_burden"),
        "flow_channels": {k: (((res.get("meta") or {}).get("flow_channels") or {}).get(k))
                          for k in ("apnea_sensor", "hypopnea_sensor", "reference_sensor",
                                    "thermistor_rejected", "dual_sensor")},
        "match": {},
        "events": [(round(a, 2), round(b, 2), t) for a, b, t in algo],
        "hypopneas": [{"onset_s": round(float(e["onset_s"]), 2),
                       "duration_s": round(float(e.get("duration_s") or 0.0), 2),
                       "desaturation_pct": e.get("desaturation_pct"),
                       "rule1a_arousal": bool(e.get("rule1a_arousal"))} for e in hyp],
    }
    for name, ref in refs.items():
        m = match_events(algo, ref, **LEGACY_MATCHER)
        rec["match"][name] = {k: m[k] for k in ("f1", "precision", "recall", "tp", "fp", "fn")}
    return rec


def analyse_one(args):
    rec_id, profile = args
    import mne
    mne.set_log_level("ERROR")
    import psgscoring
    edf = EDF_DIR / f"{rec_id}.edf"
    xml = XML_DIR / f"{rec_id}-nsrr.xml"
    t0 = time.time()
    try:
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    except Exception as e:  # noqa: BLE001
        return {"recording": rec_id, "error": f"EDF: {e}"}
    dur = float(raw.times[-1])
    try:
        hypno, refs, tst_h = parse_nsrr(xml, dur)
    except Exception as e:  # noqa: BLE001
        return {"recording": rec_id, "error": f"XML: {e}"}
    if tst_h < 1.0:
        return {"recording": rec_id, "error": f"TST {tst_h:.2f} h te kort"}
    n_epochs = int(dur // EPOCH_S)
    out = {"recording": rec_id, "profile": profile, "tst_h": tst_h,
           "tib_h": n_epochs * EPOCH_S / 3600.0, "duration_h": dur / 3600.0,
           "n_ref": {k: len(v) for k, v in refs.items()},
           "ahi_ref": {k: len(v) / tst_h for k, v in refs.items()},
           "severity_ref": severity(len(refs[PRIMARY_REFERENCE]) / tst_h),
           "channels_dropped": [c for c in raw.ch_names if _is_eeg_like(c)],
           "arms": {}}
    drop = out["channels_dropped"]
    raw_hpg = raw.copy().drop_channels(drop) if drop else raw.copy()
    hypno_tib = ["N2"] * n_epochs
    for arm in ARMEN:
        r_in = raw if arm == "psg" else raw_hpg
        h_in = hypno_tib if arm == "hpg" else hypno
        t1 = time.time()
        try:
            res = psgscoring.run_pneumo_analysis(r_in, hypno=h_in, scoring_profile=profile)
            rec = _arm_record(res, refs, h_in)
        except Exception as e:  # noqa: BLE001
            rec = {"error": str(e)}
        rec["seconds"] = round(time.time() - t1, 1)
        out["arms"][arm] = rec
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ───────────────────────────── samenvatting ─────────────────────────────

def _f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}".replace(".", ",")


def _tertielen(ok: list[dict], ref: str) -> dict[str, str]:
    a = sorted(r["ahi_ref"][ref] for r in ok)
    n = len(a)
    g1, g2 = a[n // 3], a[(2 * n) // 3]
    return {r["recording"]: ("laag" if r["ahi_ref"][ref] < g1 else "midden" if r["ahi_ref"][ref] < g2 else "hoog")
            for r in ok}, (g1, g2)


def samenvatting(rows: list[dict], ref: str = PRIMARY_REFERENCE) -> tuple[dict, str]:
    ok = [r for r in rows if "error" not in r
          and all(a in r["arms"] and "error" not in r["arms"][a] for a in ARMEN)]
    fouten = [r for r in rows if r not in ok]
    uit: dict = {"n": len(ok), "n_fouten": len(fouten), "referentie": ref,
                 "fouten": [{"recording": r["recording"],
                             "error": r.get("error") or {a: r["arms"].get(a, {}).get("error") for a in ARMEN}}
                            for r in fouten],
                 "armen": {}, "gepaard": {}, "tertiel": {}, "ernst": {}}
    if not ok:
        return uit, "geen bruikbare nachten"
    tert, grenzen = _tertielen(ok, ref)
    uit["tertielgrenzen"] = grenzen
    refv = np.array([r["ahi_ref"][ref] for r in ok])
    for arm in ARMEN:
        ahi = np.array([r["arms"][arm]["ahi"] for r in ok])
        f1 = [r["arms"][arm]["match"][ref]["f1"] for r in ok]
        d = ahi - refv
        uit["armen"][arm] = {
            "f1_mediaan": float(median(f1)), "f1_gem": float(np.mean(f1)),
            "precisie_mediaan": float(median(r["arms"][arm]["match"][ref]["precision"] for r in ok)),
            "recall_mediaan": float(median(r["arms"][arm]["match"][ref]["recall"] for r in ok)),
            "ahi_bias": float(d.mean()), "ahi_mae": float(np.abs(d).mean()),
            "ahi_r": float(np.corrcoef(ahi, refv)[0, 1]) if len(ok) > 2 else None,
            "ernst_gelijk": int(sum(1 for r in ok if r["arms"][arm]["severity"] == r["severity_ref"])),
            "n_hypopnea_som": int(sum(r["arms"][arm]["n_hypopnea"] for r in ok)),
            "n_apnea_som": int(sum(r["arms"][arm]["n_apnea"] for r in ok)),
            "n_hyp_rule1a_som": int(sum(r["arms"][arm]["n_hyp_rule1a_arousal"] for r in ok)),
            "n_hyp_desat3_som": int(sum(r["arms"][arm]["n_hyp_desat3"] for r in ok)),
            "arousal_detector": sorted({str(r["arms"][arm]["arousal_detector"]) for r in ok}),
            "seconden_mediaan": float(median(r["arms"][arm]["seconds"] for r in ok)),
        }
    for arm in ("hpg_hypno", "hpg"):
        d_ahi = [r["arms"][arm]["ahi"] - r["arms"]["psg"]["ahi"] for r in ok]
        d_f1 = [r["arms"][arm]["match"][ref]["f1"] - r["arms"]["psg"]["match"][ref]["f1"] for r in ok]
        p_ahi, n_ahi = wilcoxon_signed_rank(d_ahi)
        p_f1, n_f1 = wilcoxon_signed_rank(d_f1)
        wissel = {}
        for r in ok:
            a, b = r["arms"]["psg"]["severity"], r["arms"][arm]["severity"]
            if a != b:
                wissel[f"{a}->{b}"] = wissel.get(f"{a}->{b}", 0) + 1
        uit["gepaard"][arm] = {
            "d_ahi_mediaan": float(median(d_ahi)), "d_ahi_gem": float(np.mean(d_ahi)),
            "d_ahi_min": float(min(d_ahi)), "d_ahi_max": float(max(d_ahi)),
            "d_ahi_lager": int(sum(1 for x in d_ahi if x < 0)),
            "d_ahi_hoger": int(sum(1 for x in d_ahi if x > 0)),
            "d_ahi_abs_ge5": int(sum(1 for x in d_ahi if abs(x) >= 5.0)),
            "wilcoxon_ahi_p": p_ahi, "wilcoxon_ahi_n": n_ahi,
            "d_f1_mediaan": float(median(d_f1)), "d_f1_gem": float(np.mean(d_f1)),
            "d_f1_beter": int(sum(1 for x in d_f1 if x > 0)),
            "d_f1_slechter": int(sum(1 for x in d_f1 if x < 0)),
            "wilcoxon_f1_p": p_f1, "wilcoxon_f1_n": n_f1,
            "ernstwissel": wissel, "ernstwissel_n": int(sum(wissel.values())),
            "ernst_gelijk_aan_referentie_psg": uit["armen"]["psg"]["ernst_gelijk"],
            "ernst_gelijk_aan_referentie_arm": uit["armen"][arm]["ernst_gelijk"],
        }
        uit["tertiel"][arm] = {}
        for t in ("laag", "midden", "hoog"):
            sel = [r for r in ok if tert[r["recording"]] == t]
            dd = [r["arms"][arm]["ahi"] - r["arms"]["psg"]["ahi"] for r in sel]
            df = [r["arms"][arm]["match"][ref]["f1"] - r["arms"]["psg"]["match"][ref]["f1"] for r in sel]
            uit["tertiel"][arm][t] = {"n": len(sel), "d_ahi_mediaan": float(median(dd)) if dd else None,
                                      "d_ahi_gem": float(np.mean(dd)) if dd else None,
                                      "d_f1_mediaan": float(median(df)) if df else None,
                                      "ernstwissel": int(sum(1 for r in sel if r["arms"][arm]["severity"] != r["arms"]["psg"]["severity"]))}
    # TIB/TST en de arousal-afhankelijke hypopneeën van de psg-arm
    uit["tib_tst_ratio_mediaan"] = float(median(r["tib_h"] / r["tst_h"] for r in ok))
    uit["aandeel_hyp_zonder_desat3_psg"] = (
        (uit["armen"]["psg"]["n_hypopnea_som"] - uit["armen"]["psg"]["n_hyp_desat3_som"])
        / max(1, uit["armen"]["psg"]["n_hypopnea_som"]))

    L = []
    L.append(f"# hPG-simulatie op MESA — {len(ok)} nachten (fouten {len(fouten)}), referentie `{ref}`\n")
    L.append("| arm | F1 med | F1 gem | prec | rec | AHI-bias | MAE | r | ernst = ref | hyp (som) | hyp rule1a | apneus | detector | s/nacht |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for arm in ARMEN:
        a = uit["armen"][arm]
        L.append(f"| {arm} | {_f(a['f1_mediaan'])} | {_f(a['f1_gem'])} | {_f(a['precisie_mediaan'])} | {_f(a['recall_mediaan'])} | "
                 f"{_f(a['ahi_bias'], 2)} | {_f(a['ahi_mae'], 2)} | {_f(a['ahi_r'])} | {a['ernst_gelijk']}/{len(ok)} | "
                 f"{a['n_hypopnea_som']} | {a['n_hyp_rule1a_som']} | {a['n_apnea_som']} | {', '.join(a['arousal_detector'])} | {_f(a['seconden_mediaan'], 0)} |")
    L.append("")
    L.append("| gepaard t.o.v. psg | ΔAHI med | gem | min … max | lager/hoger | \\|Δ\\| ≥ 5 | Wilcoxon p | ΔF1 med | beter/slechter | p | ernstwissels |")
    L.append("|---|---:|---:|---|---|---:|---:|---:|---|---:|---|")
    for arm in ("hpg_hypno", "hpg"):
        g = uit["gepaard"][arm]
        w = ", ".join(f"{k}: {v}" for k, v in sorted(g["ernstwissel"].items(), key=lambda x: -x[1])) or "geen"
        L.append(f"| {arm} | {_f(g['d_ahi_mediaan'], 2)} | {_f(g['d_ahi_gem'], 2)} | {_f(g['d_ahi_min'], 1)} … {_f(g['d_ahi_max'], 1)} | "
                 f"{g['d_ahi_lager']}/{g['d_ahi_hoger']} | {g['d_ahi_abs_ge5']} | {_f(g['wilcoxon_ahi_p'], 4)} | {_f(g['d_f1_mediaan'])} | "
                 f"{g['d_f1_beter']}/{g['d_f1_slechter']} | {_f(g['wilcoxon_f1_p'], 4)} | {g['ernstwissel_n']} ({w}) |")
    L.append("")
    L.append(f"Tertielgrenzen (NSRR-{ref}-AHI): {_f(grenzen[0], 1)} / {_f(grenzen[1], 1)} /u")
    L.append("")
    L.append("| arm × tertiel | n | ΔAHI med | ΔAHI gem | ΔF1 med | ernstwissels |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for arm in ("hpg_hypno", "hpg"):
        for t in ("laag", "midden", "hoog"):
            x = uit["tertiel"][arm][t]
            L.append(f"| {arm} {t} | {x['n']} | {_f(x['d_ahi_mediaan'], 2)} | {_f(x['d_ahi_gem'], 2)} | {_f(x['d_f1_mediaan'])} | {x['ernstwissel']} |")
    L.append("")
    L.append(f"TIB/TST mediaan {_f(uit['tib_tst_ratio_mediaan'], 2)}; aandeel hypopneeën zonder ≥ 3 % desaturatie in de psg-arm "
             f"{_f(100 * uit['aandeel_hyp_zonder_desat3_psg'], 1)} %.")
    return uit, "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", default="aasm_v3_breath_dual")
    ap.add_argument("--n", type=int, default=None, help="steekproef uit de lijst (seed)")
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-summary", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in a.ids_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if a.n and a.n < len(ids):
        ids = sorted(random.Random(a.seed).sample(ids, a.n))
    partial = a.out / "rijen.partial.jsonl"
    rows = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["recording"]] = r
    todo = [i for i in ids if i not in rows]
    print(f"{len(ids)} nachten, {len(rows)} al gedaan, {len(todo)} te doen, profiel {a.profile}, "
          f"{a.workers} workers", flush=True)
    if todo and not a.only_summary:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex, partial.open("a") as fh:
            for k, r in enumerate(ex.map(analyse_one, [(i, a.profile) for i in todo]), 1):
                rows[r["recording"]] = r
                fh.write(json.dumps(r) + "\n"); fh.flush()
                status = r.get("error") or " ".join(
                    f"{arm}:{r['arms'][arm].get('ahi', 'ERR'):.1f}" if "error" not in r["arms"][arm] else f"{arm}:ERR"
                    for arm in ARMEN)
                print(f"  {k}/{len(todo)} {r['recording']} {status} ({r.get('seconds', 0):.0f} s, "
                      f"{(time.time() - t0) / 60:.1f} min)", flush=True)
    uit, md = samenvatting([rows[i] for i in ids if i in rows])
    (a.out / "rijen.json").write_text(json.dumps([rows[i] for i in ids if i in rows], indent=1))
    (a.out / "samenvatting.json").write_text(json.dumps(uit, indent=1))
    (a.out / "rapport.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
