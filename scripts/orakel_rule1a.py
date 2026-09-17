#!/usr/bin/env python3
"""Orakel-decompositie van de Rule-1A-arousaltak.

Drie armen, alle in-pipeline op dezelfde raw/hypno/profiel (aasm_v3_rec):
  A  tak uit                        (productie)
  B  tak aan, onze arousals         (PSGSCORING_RULE1A_AROUSAL=1 + LIMB_WIRED=1)
  C  tak aan, REFERENTIE-arousals   (run_pneumo_analysis(arousal_events=...);
                                     bron "external" wordt altijd gehonoreerd)
Regel, uitlezingen en beslisregel: docs/orakel_rule1a_preregistratie_20260917.md
— geschreven vóór de eerste run. Dit script meet; het stelt niets af.

  python scripts/orakel_rule1a.py --cohort psgipa --workers 5 \
      --output-json docs/orakel_rule1a_psgipa.json
  python scripts/orakel_rule1a.py --cohort mesa --ids /tmp/n150_ids.txt \
      --workers 20 --output-json docs/orakel_rule1a_mesa_n150.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median

import mne
import numpy as np

mne.set_log_level("ERROR")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from validate_psgipa import (LEGACY_MATCHER, event_set, find_scorer_files,  # noqa: E402
                             match_events, parse_scorer_file, severity)
from validate_mesa import parse_nsrr  # noqa: E402

ARMEN = tuple(os.environ.get("ORAKEL_ARMEN", "ABC"))
PROFIEL = "aasm_v3_rec"
DATA_PSGIPA = Path(os.environ.get("PSGSCORING_DATA_ROOT", "/srv/DATA")) / "PSG-IPA"
DATA_MESA = Path(os.environ.get("PSGSCORING_DATA_ROOT", "/srv/DATA")) / "MESA" / "mesa"


def _zet_env(arm: str) -> None:
    if arm == "A":
        os.environ["PSGSCORING_RULE1A_AROUSAL"] = "0"
        os.environ.pop("PSGSCORING_AROUSAL_LIMB_WIRED", None)
    elif arm == "B":
        os.environ["PSGSCORING_RULE1A_AROUSAL"] = "1"
        os.environ["PSGSCORING_AROUSAL_LIMB_WIRED"] = "1"
    else:  # C: extern pad, de wired-vlag doet er niet toe
        os.environ["PSGSCORING_RULE1A_AROUSAL"] = "1"
        os.environ.pop("PSGSCORING_AROUSAL_LIMB_WIRED", None)


def _tup(e):
    return (float(e["onset_s"]),
            float(e["onset_s"]) + float(e.get("duration_s") or 0.0),
            str(e.get("type")))


def _m(algo, ref):
    if not ref:
        return None
    m = match_events(algo, ref, **LEGACY_MATCHER)
    return {k: m[k] for k in ("f1", "precision", "recall", "tp", "fp", "fn")}


def draai_arm(raw, hypno, arm, ref_arousals):
    import psgscoring
    _zet_env(arm)
    kw = {}
    if arm == "C":
        kw["arousal_events"] = [{"onset_s": float(o), "duration_s": float(d)}
                                for o, d in ref_arousals]
    t0 = time.monotonic()
    res = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile=PROFIEL, **kw)
    r = res.get("respiratory", {}) or {}
    evs = [e for e in (r.get("events") or []) if e.get("onset_s") is not None]
    algo = [_tup(e) for e in evs]
    herst = [_tup(e) for e in evs if e.get("rule1a_arousal")]
    # Debietbewijs per herstelling, voor de post-hoc poortafleiding
    # (docs/orakel_rule1a_poort_preregistratie_20260917.md): dezelfde
    # velden die de kandidaatpoort in de bibliotheek leest.
    herst_velden = [(e.get("flow_reduction"), e.get("local_reduction_pct"),
                     float(e.get("duration_s") or 0.0))
                    for e in evs if e.get("rule1a_arousal")]
    summ = r.get("summary", {}) or {}
    ar = res.get("arousal") or {}
    return {
        "ahi": float(summ.get("ahi_total") or summ.get("ahi") or 0.0),
        "n_events": len(algo),
        "n_reinstated": len(herst),
        "stats": r.get("rule1a_arousal_stats"),
        "arousal_source": ar.get("source"),
        "n_arousals": len(ar.get("events") or []),
        "wall_s": round(time.monotonic() - t0, 1),
        "events": algo,
        "reinstated": herst,
        "reinstated_fields": herst_velden,
    }


# ── PSG-IPA ─────────────────────────────────────────────────────────
def psgipa_een(sn):
    raw = mne.io.read_raw_edf(str(DATA_PSGIPA / "Resp_events" / "PSG"
                                  / f"{sn}_Respiration.edf"),
                              preload=True, verbose=False)
    dur = float(raw.times[-1])
    files = find_scorer_files(DATA_PSGIPA, sn)
    ref_ahis, hypno = [], None
    for i, f in enumerate(files):
        ahi, _t, h = parse_scorer_file(f, dur)
        if ahi is not None:
            ref_ahis.append(ahi)
            if i == 0:
                hypno = h
    ann = mne.read_annotations(str(files[0]))
    ref_ar = [(float(o), float(d)) for o, d, x in
              zip(ann.onset, ann.duration, ann.description)
              if "eeg arousal" in str(x).lower() and 0 <= o < dur]
    refsets = [s for s in (event_set(f, dur) for f in files) if s]
    out = {"recording": sn, "n_scorers": len(ref_ahis),
           "ref_median_ahi": round(float(median(ref_ahis)), 2),
           "ref_range": [round(min(ref_ahis), 2), round(max(ref_ahis), 2)],
           "n_ref_arousals": len(ref_ar), "arms": {}}
    for arm in ARMEN:
        r = draai_arm(raw, hypno, arm, ref_ar)
        per = [match_events(r["events"], s, **LEGACY_MATCHER) for s in refsets]
        r["f1_median"] = float(median(m["f1"] for m in per))
        r["precision_median"] = float(median(m["precision"] for m in per))
        r["recall_median"] = float(median(m["recall"] for m in per))
        if r["reinstated"]:
            rp = [match_events(r["reinstated"], s, **LEGACY_MATCHER)["precision"]
                  for s in refsets]
            r["reinst_precision_median"] = float(median(rp))
        else:
            r["reinst_precision_median"] = None
        r["delta_ahi"] = round(r["ahi"] - out["ref_median_ahi"], 2)
        r["events"] = [(round(a, 2), round(b, 2), t) for a, b, t in r["events"]]
        r["reinstated"] = [(round(a, 2), round(b, 2), t) for a, b, t in r["reinstated"]]
        out["arms"][arm] = r
    return out


# ── MESA ─────────────────────────────────────────────────────────────
def _nsrr_arousals(xml_path, dur):
    root = ET.parse(str(xml_path)).getroot()
    out = []
    for ev in root.iter("ScoredEvent"):
        concept = (ev.findtext("EventConcept") or "").lower()
        ev_type = (ev.findtext("EventType") or "").lower()
        if "arousal" not in concept and "arousal" not in ev_type:
            continue
        try:
            s = float(ev.findtext("Start") or "nan")
            d = float(ev.findtext("Duration") or "nan")
        except ValueError:
            continue
        if np.isfinite(s) and np.isfinite(d) and 0 <= s < dur:
            out.append((s, max(d, 0.5)))
    return sorted(out)


def mesa_een(rec_id):
    edf = DATA_MESA / "polysomnography" / "edfs" / f"{rec_id}.edf"
    xml = DATA_MESA / "polysomnography" / "annotations-events-nsrr" / f"{rec_id}-nsrr.xml"
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
    ref_ar = _nsrr_arousals(xml, dur)
    d3 = {(a, b) for a, b, _ in refs["desat3_all"]}
    gat = [(a, b, t) for a, b, t in refs["aasm15"] if (a, b) not in d3]
    out = {"recording": rec_id, "tst_h": tst_h,
           "n_ref": {k: len(v) for k, v in refs.items()},
           "ahi_ref": {k: len(v) / tst_h for k, v in refs.items()},
           "n_ref_arousals": len(ref_ar), "n_gat": len(gat), "arms": {}}
    for arm in ARMEN:
        try:
            r = draai_arm(raw, hypno, arm, ref_ar)
        except Exception as e:  # noqa: BLE001
            out["arms"][arm] = {"error": str(e)}
            continue
        r["match"] = {k: _m(r["events"], v) for k, v in refs.items()}
        r["reinst_precision"] = ({k: _m(r["reinstated"], v)["precision"]
                                  for k, v in refs.items() if v}
                                 if r["reinstated"] else None)
        r["gat_recall"] = _m(r["events"], gat)["recall"] if gat else None
        r["severity"] = severity(r["ahi"])
        r["events"] = [(round(a, 2), round(b, 2), t) for a, b, t in r["events"]]
        r["reinstated"] = [(round(a, 2), round(b, 2), t) for a, b, t in r["reinstated"]]
        out["arms"][arm] = r
    out["severity_ref"] = severity(out["ahi_ref"]["aasm15"])
    return out


# ── aggregatie ───────────────────────────────────────────────────────
def _p_wilcoxon(deltas):
    d = [x for x in deltas if x != 0]
    if len(d) < 6:
        return None
    from scipy.stats import wilcoxon
    return float(wilcoxon(d).pvalue)


def samenvatting_mesa(rows, ref="aasm15"):
    ok = [r for r in rows if "error" not in r and all("error" not in r["arms"].get(a, {"error": 1}) for a in ARMEN)]
    # Nachten zonder enig referentie-event hebben geen F1 (match is None);
    # die vallen uit de statistiek, en dat aantal hoort zichtbaar te zijn.
    n_zonder_ref = sum(1 for r in ok if r["arms"]["A"]["match"].get(ref) is None)
    ok = [r for r in ok if all(r["arms"][a]["match"].get(ref) is not None for a in ARMEN)]
    S = {"n": len(ok), "n_zonder_referentie_events": n_zonder_ref, "ref": ref, "arms": {}}
    for arm in ARMEN:
        f1 = [r["arms"][arm]["match"][ref]["f1"] for r in ok]
        pr = [r["arms"][arm]["match"][ref]["precision"] for r in ok]
        rc = [r["arms"][arm]["match"][ref]["recall"] for r in ok]
        bias = [r["arms"][arm]["ahi"] - r["ahi_ref"][ref] for r in ok]
        rp = [r["arms"][arm]["reinst_precision"][ref] for r in ok
              if r["arms"][arm]["reinst_precision"]]
        n_re = sum(r["arms"][arm]["n_reinstated"] for r in ok)
        tp_re = sum(round(r["arms"][arm]["reinst_precision"][ref] * r["arms"][arm]["n_reinstated"])
                    for r in ok if r["arms"][arm]["reinst_precision"])
        gr = [r["arms"][arm]["gat_recall"] for r in ok if r["arms"][arm]["gat_recall"] is not None]
        S["arms"][arm] = {
            "f1_median": median(f1), "precision_median": median(pr),
            "recall_median": median(rc), "bias_mean": float(np.mean(bias)),
            "bias_median": median(bias),
            "n_reinstated_total": n_re,
            "R1_reinst_precision_pooled": (tp_re / n_re) if n_re else None,
            "R1_reinst_precision_median": median(rp) if rp else None,
            "R3_gat_recall_median": median(gr) if gr else None,
            "R3_gat_recall_pooled": (sum(r["arms"][arm]["gat_recall"] * r["n_gat"] for r in ok if r["arms"][arm]["gat_recall"] is not None)
                                     / max(1, sum(r["n_gat"] for r in ok if r["arms"][arm]["gat_recall"] is not None))),
            "severity_match": sum(1 for r in ok if r["arms"][arm]["severity"] == r["severity_ref"]),
        }
    for arm in ("B", "C"):
        d = [r["arms"][arm]["match"][ref]["f1"] - r["arms"]["A"]["match"][ref]["f1"] for r in ok]
        S[f"R2_dF1_{arm}_min_A"] = {"median": median(d), "mean": float(np.mean(d)),
                                    "n_beter": sum(1 for x in d if x > 0),
                                    "n_slechter": sum(1 for x in d if x < 0),
                                    "wilcoxon_p": _p_wilcoxon(d)}
    return S


def samenvatting_psgipa(rows):
    S = {"n": len(rows), "arms": {}}
    for arm in ARMEN:
        S["arms"][arm] = {
            "ahi": {r["recording"]: r["arms"][arm]["ahi"] for r in rows},
            "delta_ahi": {r["recording"]: r["arms"][arm]["delta_ahi"] for r in rows},
            "bias_mean": float(np.mean([r["arms"][arm]["delta_ahi"] for r in rows])),
            "f1_median_per_rec": {r["recording"]: round(r["arms"][arm]["f1_median"], 3) for r in rows},
            "f1_median": float(median(r["arms"][arm]["f1_median"] for r in rows)),
            "n_reinstated": {r["recording"]: r["arms"][arm]["n_reinstated"] for r in rows},
            "reinst_precision": {r["recording"]: r["arms"][arm]["reinst_precision_median"] for r in rows},
        }
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=("psgipa", "mesa"), required=True)
    ap.add_argument("--ids", type=Path, default=None, help="MESA: bestand met één rec-id per regel")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--arms", default="ABC", help="welke armen (bv. AC voor de poortafleiding)")
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()

    if a.cohort == "psgipa":
        ids = ["SN1", "SN2", "SN3", "SN4", "SN5"]
        fn = psgipa_een
    else:
        ids = [l.strip() for l in a.ids.read_text().splitlines() if l.strip()]
        fn = mesa_een
    if a.limit:
        ids = ids[:a.limit]
    global ARMEN
    ARMEN = tuple(a.arms)
    os.environ["ORAKEL_ARMEN"] = a.arms   # workers lezen dit bij import

    partial = a.output_json.with_suffix(".partial.jsonl")
    klaar = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                klaar[r["recording"]] = r
    todo = [i for i in ids if i not in klaar]
    print(f"{a.cohort}: {len(ids)} opnames, {len(klaar)} al klaar, {len(todo)} te doen, "
          f"{a.workers} workers", flush=True)
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, partial.open("a") as fh:
        futs = {ex.submit(fn, i): i for i in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                r = {"recording": futs[fut], "error": f"worker: {e}"}
            klaar[r["recording"]] = r
            fh.write(json.dumps(r) + "\n"); fh.flush()
            kort = ("FOUT " + r["error"][:60]) if "error" in r else " ".join(
                f"{arm}:{r['arms'][arm].get('ahi', float('nan')):.1f}/{r['arms'][arm].get('n_reinstated', '?')}"
                for arm in ARMEN if arm in r["arms"])
            print(f"[{n}/{len(todo)} {time.monotonic()-t0:6.0f}s] {r['recording']} {kort}", flush=True)

    rows = [klaar[i] for i in ids if i in klaar]
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    import psgscoring
    S = samenvatting_psgipa(rows) if a.cohort == "psgipa" else samenvatting_mesa(rows)
    uit = {"meta": {"cohort": a.cohort, "profiel": PROFIEL, "psgscoring": psgscoring.__version__,
                    "git_sha": sha, "datum": time.strftime("%F %T"), "matcher": LEGACY_MATCHER,
                    "preregistratie": "docs/orakel_rule1a_preregistratie_20260917.md"},
           "samenvatting": S, "results": rows}
    a.output_json.write_text(json.dumps(uit, indent=1))
    print(json.dumps(S, indent=1, default=str))
    print(f"KLAAR: {a.output_json}")


if __name__ == "__main__":
    main()
