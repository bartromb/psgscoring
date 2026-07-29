#!/usr/bin/env python3
"""
validate_psgipa.py — PSG-IPA validation harness for psgscoring.

Reproduces paper v31 (Rombaut et al. 2026) on the PSG-IPA multi-scorer
reference subset (SN1-SN5, 12 scorers each), per the supplement S3.2
methodology: scorer-1 hypnogram and scorer-1 respiratory events are
read from the same {SN}_Respiration_manual_scorer1.edf file in the
Resp_events/ subtree (single time-axis, no meas_date alignment needed).

Runs all three clinical profiles (aasm_v3_strict / aasm_v3_rec /
aasm_v3_sensitive) on each recording in parallel, computes the
robustness-grade (A/B/C) per recording, aggregate Bland-Altman and
severity-concordance metrics on the standard (aasm_v3_rec) profile,
and SN3 event-level F1 / mean Δt against each scorer's event set.

Outputs:
  - human-readable summary table to stdout
  - JSON to /tmp/validation_results.json (consumed by validation_report.py)

Usage:
    python validate_psgipa.py --data-dir ~/PSG-IPA --workers 5
    python validate_psgipa.py --data-dir ~/PSG-IPA --output-json results.json

Requires: psgscoring, mne, numpy, scikit-learn.

Reference values (paper v31, scorer-median per recording):
    SN1: 5.96 [4.66-6.56]    Mild      12 scorers
    SN2: 4.33 [1.65-6.80]    Normal    12 scorers
    SN3: 53.98 [45.06-55.96] Severe    12 scorers
    SN4: 3.82 [0.17-6.32]    Normal    12 scorers
    SN5: 9.98 [3.56-14.39]   Mild      12 scorers
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median, stdev

import mne
import numpy as np

mne.set_log_level("ERROR")

CLINICAL_PROFILES = {
    "strict":    "aasm_v3_strict",
    "standard":  "aasm_v3_rec",
    "sensitive": "aasm_v3_sensitive",
}

STAGE_KEYS = [
    ("stage n3", "N3"), ("stage 3", "N3"), ("stage 4", "N3"),
    ("stage n2", "N2"), ("stage 2", "N2"),
    ("stage n1", "N1"), ("stage 1", "N1"),
    ("stage rem", "R"), ("stage r", "R"),
    ("stage w", "W"), ("wake", "W"),
]


def classify_stage(desc):
    d = desc.lower().strip()
    if "lights" in d or "stage ?" in d:
        return None
    for k, v in STAGE_KEYS:
        if k in d:
            return v
    return None


def is_resp_event(desc):
    d = desc.lower()
    if "arousal" in d:
        return False
    return any(k in d for k in [
        "obstructive apnea", "central apnea", "mixed apnea",
        "hypopnea", "hypopnoea", "apnea", "apnoea",
    ])


def event_type(desc):
    d = desc.lower()
    if "arousal" in d:
        return None
    if "obstructive" in d and ("apnea" in d or "apnoea" in d):
        return "obstructive"
    if "central" in d and ("apnea" in d or "apnoea" in d):
        return "central"
    if "mixed" in d and ("apnea" in d or "apnoea" in d):
        return "mixed"
    if "hypopnea" in d or "hypopnoea" in d:
        return "hypopnea"
    if "apnea" in d or "apnoea" in d:
        return "apnea_unspec"
    return None


def severity(ahi):
    if ahi is None or (isinstance(ahi, float) and np.isnan(ahi)):
        return "?"
    if ahi < 5:
        return "Normal"
    if ahi < 15:
        return "Mild"
    if ahi < 30:
        return "Moderate"
    return "Severe"


def grade_from_severities(strict_sev, std_sev, sens_sev):
    s = {strict_sev, std_sev, sens_sev}
    if len(s) == 1:
        return "A"
    if len(s) == 2:
        return "B"
    return "C"


def parse_scorer_file(scorer_edf, signal_duration_s):
    try:
        ann = mne.read_annotations(str(scorer_edf))
    except Exception:
        return None, None, None
    n_sleep_epochs = 0
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if onset < 0 or onset >= signal_duration_s:
            continue
        st = classify_stage(str(desc))
        if st in ("N1", "N2", "N3", "R"):
            n_eps = max(1, int(round(float(dur) / 30.0)))
            n_sleep_epochs += n_eps
    tst_h = n_sleep_epochs * 30.0 / 3600.0
    if tst_h < 0.1:
        return None, None, None
    n_events = sum(
        1 for onset, desc in zip(ann.onset, ann.description)
        if 0 <= onset < signal_duration_s and is_resp_event(str(desc))
    )
    ahi = n_events / tst_h
    n_epochs = int(np.ceil(signal_duration_s / 30.0))
    hypno = ["W"] * n_epochs
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if onset < 0 or onset >= signal_duration_s:
            continue
        st = classify_stage(str(desc))
        if st is None:
            continue
        ep_start = int(float(onset) // 30)
        n_eps = max(1, int(round(float(dur) / 30.0)))
        for i in range(n_eps):
            if 0 <= ep_start + i < n_epochs:
                hypno[ep_start + i] = st
    return ahi, tst_h, hypno


def find_scorer_files(data_dir, sn_id):
    d = Path(data_dir) / "Resp_events" / "Annotations" / "manual"
    return sorted(d.glob(f"{sn_id}_Respiration_manual_scorer*.edf"))


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match_events(a_events, b_events, iou_thresh=0.20, type_aware=False, optimal=False):
    """Match twee eventlijsten [(onset, offset, type), ...].

    Returns dict met: tp, fp, fn, precision, recall, f1, mean_dt,
    n_merges, n_splits, matched_pairs.

    De legacy-modus (type_aware=False, optimal=False) reproduceert exact de
    inline-matching van paper v31: greedy in volgorde van a_events, één-op-één,
    IoU-drempel 0.20, eventtype genegeerd. Wijzig die tak niet zonder de
    byte-identieke invariant opnieuw aan te tonen — de gepubliceerde cijfers
    hangen eraan.
    """
    matched_a, matched_b, onset_diffs, pairs = set(), set(), [], []

    for i, (a0, a1, _atype) in enumerate(a_events):
        best_j, best_v = -1, 0.0
        for j, (b0, b1, _btype) in enumerate(b_events):
            if j in matched_b:
                continue
            v = iou(a0, a1, b0, b1)
            if v >= iou_thresh and v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            matched_a.add(i)
            matched_b.add(best_j)
            onset_diffs.append(abs(a_events[i][0] - b_events[best_j][0]))
            pairs.append((i, best_j, best_v))

    tp = len(matched_a)
    fp = len(a_events) - tp
    fn = len(b_events) - len(matched_b)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec_ = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec_,
        "f1": f1,
        "mean_dt": float(np.mean(onset_diffs)) if onset_diffs else None,
        "n_merges": 0,
        "n_splits": 0,
        "matched_pairs": pairs,
    }


def match_events_symmetric(ev_a, ev_b, **match_kw):
    """Match in beide richtingen en middel.

    F1 = 2·TP / (2·TP + FP + FN) is symmetrisch in A en B, want FP + FN =
    |A| + |B| − 2·TP. Het verschil tussen de richtingen zit dus volledig in
    de gevonden TP: greedy loopt over de eerste lijst, dus wie "kandidaat" is
    bepaalt welk paar als eerste een partner inpikt. Voor een mens-tegen-mens
    vergelijking is geen van beide scorers de waarheid, dus is één richting
    kiezen willekeurig. We rapporteren beide plus het gemiddelde.

    Met optimal=True is de toewijzing per constructie richtingsonafhankelijk
    en wordt f1_ab == f1_ba; `asymmetry` is dan 0.
    """
    m_ab = match_events(ev_a, ev_b, **match_kw)
    m_ba = match_events(ev_b, ev_a, **match_kw)
    dts = [m["mean_dt"] for m in (m_ab, m_ba) if m["mean_dt"] is not None]
    return {
        "f1":        (m_ab["f1"] + m_ba["f1"]) / 2.0,
        "f1_ab":     m_ab["f1"],
        "f1_ba":     m_ba["f1"],
        "asymmetry": abs(m_ab["f1"] - m_ba["f1"]),
        "tp_ab":     m_ab["tp"],
        "fp_ab":     m_ab["fp"],
        "fn_ab":     m_ab["fn"],
        "precision_ab": m_ab["precision"],
        "recall_ab":    m_ab["recall"],
        "mean_dt":   float(np.mean(dts)) if dts else None,
        "n_a":       len(ev_a),
        "n_b":       len(ev_b),
    }


def human_baseline(scorer_files, sig_dur_s, **match_kw):
    """Alle paarsgewijze F1's tussen scorers van één opname.

    Zonder deze referentie is een algoritme-F1 oninterpreteerbaar: 0.29 kan
    slecht zijn of gewoon de bovengrens van wat op die opname haalbaar is.
    Bij 12 scorers zijn dit 12·11/2 = 66 paren.
    """
    sets = []
    for f in scorer_files:
        ev = event_set(f, sig_dur_s)
        if ev:
            sets.append((Path(f).name, ev))
    return pairwise_baseline(sets, **match_kw)


def pairwise_baseline(sets, **match_kw):
    """Paarsgewijze F1's over [(naam, eventlijst), ...] — de kern van
    human_baseline(), los van het inlezen zodat hij testbaar is."""
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            name_a, ev_a = sets[i]
            name_b, ev_b = sets[j]
            m = match_events_symmetric(ev_a, ev_b, **match_kw)
            m["scorer_a"] = name_a
            m["scorer_b"] = name_b
            pairs.append(m)

    f1s = sorted(p["f1"] for p in pairs)
    summary = {"n_scorers": len(sets), "n_pairs": len(pairs)}
    if f1s:
        summary.update({
            "median": float(median(f1s)),
            "p25":    float(np.percentile(f1s, 25)),
            "p75":    float(np.percentile(f1s, 75)),
            "min":    float(f1s[0]),
            "max":    float(f1s[-1]),
            "max_asymmetry": float(max(p["asymmetry"] for p in pairs)),
        })
    return {"pairs": pairs, "summary": summary}


def percentile_of(value, population):
    """Percentiel van `value` binnen `population` (ties tellen half mee)."""
    if not population or value is None:
        return None
    below = sum(1 for x in population if x < value)
    equal = sum(1 for x in population if x == value)
    return 100.0 * (below + 0.5 * equal) / len(population)


def event_set(scorer_edf, signal_duration_s):
    try:
        ann = mne.read_annotations(str(scorer_edf))
    except Exception:
        return []
    out = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if onset < 0 or onset >= signal_duration_s:
            continue
        t = event_type(str(desc))
        if t is None:
            continue
        out.append((float(onset), float(onset) + float(dur), t))
    return out


def analyse_one(sn_id, data_dir):
    data_dir = Path(data_dir)
    psg_path = data_dir / "Resp_events" / "PSG" / f"{sn_id}_Respiration.edf"
    if not psg_path.exists():
        return {"recording": sn_id, "error": "PSG not found"}
    try:
        import psgscoring
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
        sig_dur_s = float(raw.times[-1])
        scorer_files = find_scorer_files(data_dir, sn_id)
        if not scorer_files:
            return {"recording": sn_id, "error": "No scorer files"}

        ref_ahis = []
        scorer1_hypno = None
        for i, f in enumerate(scorer_files):
            ahi, _tst, hypno = parse_scorer_file(f, sig_dur_s)
            if ahi is not None:
                ref_ahis.append(ahi)
                if i == 0:
                    scorer1_hypno = hypno
        if not ref_ahis or scorer1_hypno is None:
            return {"recording": sn_id, "error": "No usable scorer data"}

        ref_med = float(median(ref_ahis))
        ref_lo  = float(min(ref_ahis))
        ref_hi  = float(max(ref_ahis))

        algo_ahis = {}
        algo_events_std = None
        for prof_short, prof_canonical in CLINICAL_PROFILES.items():
            results = psgscoring.run_pneumo_analysis(
                raw, hypno=scorer1_hypno, scoring_profile=prof_canonical,
            )
            rsum = results.get("respiratory", {}).get("summary", {})
            ahi = rsum.get("ahi_total", rsum.get("ahi"))
            if ahi is None or (isinstance(ahi, float) and np.isnan(ahi)):
                return {"recording": sn_id, "error": f"No AHI for profile {prof_canonical}"}
            algo_ahis[prof_short] = float(ahi)
            if prof_short == "standard":
                algo_events_std = results.get("respiratory", {}).get("events", [])

        sev_strict = severity(algo_ahis["strict"])
        sev_std    = severity(algo_ahis["standard"])
        sev_sens   = severity(algo_ahis["sensitive"])

        out = {
            "recording": sn_id,
            "psgscoring_version": psgscoring.__version__,
            "signal_hours": round(sig_dur_s / 3600.0, 2),
            "n_scorers": len(ref_ahis),
            "ref_median": round(ref_med, 2),
            "ref_range":  [round(ref_lo, 2), round(ref_hi, 2)],
            "algo_strict":    round(algo_ahis["strict"], 2),
            "algo_standard":  round(algo_ahis["standard"], 2),
            "algo_sensitive": round(algo_ahis["sensitive"], 2),
            "delta_ahi_standard": round(algo_ahis["standard"] - ref_med, 2),
            "severity_ref":      severity(ref_med),
            "severity_strict":   sev_strict,
            "severity_standard": sev_std,
            "severity_sensitive": sev_sens,
            "severity_match_standard": severity(ref_med) == sev_std,
            "robustness_grade": grade_from_severities(sev_strict, sev_std, sev_sens),
        }

        # Event-level metriek voor ELKE opname. Stond hiervoor achter een
        # `sn_id == "SN3"`-gate terwijl het paper SN1-SN5 rapporteert.
        if algo_events_std:
            algo_events = [
                (float(e["onset_s"]), float(e["onset_s"]) + float(e["duration_s"]), e["type"])
                for e in algo_events_std
                if e.get("onset_s") is not None
                and e.get("duration_s") is not None
                and float(e["onset_s"]) < sig_dur_s
            ]
            f1_list, dt_list, tp_list, fp_list, fn_list = [], [], [], [], []
            sym_f1_list = []
            for f in scorer_files:
                ref_events = event_set(f, sig_dur_s)
                if not ref_events:
                    continue
                m = match_events(algo_events, ref_events, iou_thresh=0.20)
                f1_list.append(m["f1"])
                tp_list.append(m["tp"]); fp_list.append(m["fp"]); fn_list.append(m["fn"])
                if m["mean_dt"] is not None:
                    dt_list.append(m["mean_dt"])
                # Zelfde symmetrisering als de mens-paren, zodat het
                # percentiel hieronder appels met appels vergelijkt. Dit
                # vervangt ev_f1_median niet — dat blijft paper v31.
                sym_f1_list.append(
                    match_events_symmetric(algo_events, ref_events, iou_thresh=0.20)["f1"]
                )
            if f1_list:
                out["ev_f1_median"]   = round(float(median(f1_list)), 3)
                out["ev_dt_median_s"] = round(float(median(dt_list)), 2) if dt_list else None
                out["ev_tp_median"]   = int(median(tp_list))
                out["ev_fp_median"]   = int(median(fp_list))
                out["ev_fn_median"]   = int(median(fn_list))
                out["ev_n_scorers"]   = len(f1_list)
                # Oude sleutels blijven bestaan voor SN3: validation_report.py
                # en tests/test_psgipa_reproducibility.py lezen ze, en paper v31
                # rapporteert ze onder deze naam.
                if sn_id == "SN3":
                    out["sn3_f1_median"]   = out["ev_f1_median"]
                    out["sn3_dt_median_s"] = out["ev_dt_median_s"]
                    out["sn3_tp_median"]   = out["ev_tp_median"]
                    out["sn3_fp_median"]   = out["ev_fp_median"]
                    out["sn3_fn_median"]   = out["ev_fn_median"]

                # ── Mens-tegen-mens baseline ────────────────────────
                # Zonder deze referentie is de algoritme-F1 hierboven
                # oninterpreteerbaar. Zelfde matcher-instellingen aan beide
                # kanten — anders is de vergelijking betekenisloos.
                hb = human_baseline(scorer_files, sig_dur_s, iou_thresh=0.20)
                hs = hb["summary"]
                if hs.get("n_pairs"):
                    algo_sym = float(median(sym_f1_list)) if sym_f1_list else None
                    pair_f1s = [p["f1"] for p in hb["pairs"]]
                    out["human_f1_median"]  = round(hs["median"], 3)
                    out["human_f1_p25"]     = round(hs["p25"], 3)
                    out["human_f1_p75"]     = round(hs["p75"], 3)
                    out["human_f1_min"]     = round(hs["min"], 3)
                    out["human_f1_max"]     = round(hs["max"], 3)
                    out["human_f1_n_pairs"] = hs["n_pairs"]
                    out["human_max_asymmetry"] = round(hs["max_asymmetry"], 4)
                    out["algo_f1_symmetric_median"] = (
                        round(algo_sym, 3) if algo_sym is not None else None
                    )
                    out["algo_f1_percentile"] = (
                        round(percentile_of(algo_sym, pair_f1s), 1)
                        if algo_sym is not None else None
                    )
                    out["_human_pairs"] = hb["pairs"]
        return out
    except Exception as e:
        import traceback
        return {
            "recording": sn_id,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def aggregate_metrics(results):
    refs   = [r["ref_median"]    for r in results if "error" not in r]
    algos  = [r["algo_standard"] for r in results if "error" not in r]
    deltas = [r["delta_ahi_standard"] for r in results if "error" not in r]
    if len(deltas) < 2:
        return {}
    bias = float(np.mean(deltas))
    mae  = float(np.mean(np.abs(deltas)))
    sd   = float(stdev(deltas))
    pearson_r = float(np.corrcoef(refs, algos)[0, 1])
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = float(cohen_kappa_score(
            [severity(x) for x in refs],
            [severity(x) for x in algos],
            weights="quadratic",
        ))
    except ImportError:
        kappa = float("nan")
    return {
        "n_recordings":  len(deltas),
        "bias":          round(bias, 3),
        "mae":           round(mae, 3),
        "sd":            round(sd, 3),
        "loa_low":       round(bias - 1.96 * sd, 3),
        "loa_high":      round(bias + 1.96 * sd, 3),
        "pearson_r":     round(pearson_r, 4),
        "weighted_kappa": round(kappa, 3),
    }


def to_report_json(results, agg):
    sn3 = next((r for r in results if r.get("recording") == "SN3"), None)
    payload = {
        "per_recording": {
            r["recording"]: {
                "ref_median":     r["ref_median"],
                "ref_range":      r["ref_range"],
                "algo_strict":    r["algo_strict"],
                "algo_standard":  r["algo_standard"],
                "algo_sensitive": r["algo_sensitive"],
                "robustness_grade": r["robustness_grade"],
            }
            for r in results
            if "error" not in r
        },
        "aggregate": {
            "n_recordings":  agg.get("n_recordings"),
            "bias":          agg.get("bias"),
            "mae":           agg.get("mae"),
            "sd":            agg.get("sd"),
            "loa_low":       agg.get("loa_low"),
            "loa_high":      agg.get("loa_high"),
            "pearson_r":     agg.get("pearson_r"),
            "weighted_kappa": agg.get("weighted_kappa"),
        },
    }
    # Event-level per opname (nieuw). sn3_event_level blijft eronder staan als
    # alias: validation_report.py en de reproduceerbaarheidstest lezen die.
    event_level = {
        r["recording"]: {
            "f1":   r["ev_f1_median"],
            "dt_s": r.get("ev_dt_median_s"),
            "tp":   r.get("ev_tp_median"),
            "fp":   r.get("ev_fp_median"),
            "fn":   r.get("ev_fn_median"),
            "n_scorers": r.get("ev_n_scorers"),
        }
        for r in results
        if "error" not in r and "ev_f1_median" in r
    }
    if event_level:
        payload["event_level"] = event_level

    human = {
        r["recording"]: {
            "f1_median": r["human_f1_median"],
            "f1_p25":    r["human_f1_p25"],
            "f1_p75":    r["human_f1_p75"],
            "f1_min":    r["human_f1_min"],
            "f1_max":    r["human_f1_max"],
            "n_pairs":   r["human_f1_n_pairs"],
            "max_asymmetry":  r.get("human_max_asymmetry"),
            "algo_f1_symmetric": r.get("algo_f1_symmetric_median"),
            "algo_percentile":   r.get("algo_f1_percentile"),
        }
        for r in results
        if "error" not in r and "human_f1_median" in r
    }
    if human:
        payload["human_baseline"] = human

    if sn3 and "sn3_f1_median" in sn3:
        payload["sn3_event_level"] = {
            "f1":   sn3["sn3_f1_median"],
            "dt_s": sn3.get("sn3_dt_median_s"),
            "tp":   sn3.get("sn3_tp_median"),
            "fp":   sn3.get("sn3_fp_median"),
            "fn":   sn3.get("sn3_fn_median"),
        }
    return payload


def print_summary(results, agg):
    print(f"\n{'Rec':4s}  {'sig_h':>5s}  {'n':>2s}  {'Ref':>5s}  {'[range]':>12s}  "
          f"{'Strict':>6s}  {'Std':>5s}  {'Sens':>5s}  {'Grade':>5s}  Sev")
    print("─" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['recording']:4s}  ERROR: {r['error']}")
            continue
        rng = f"[{r['ref_range'][0]:.1f}-{r['ref_range'][1]:.1f}]"
        print(f"{r['recording']:4s}  {r['signal_hours']:>4.2f}h  {r['n_scorers']:>2d}  "
              f"{r['ref_median']:>4.1f}  {rng:>12s}  "
              f"{r['algo_strict']:>5.1f}  {r['algo_standard']:>4.1f}  {r['algo_sensitive']:>4.1f}  "
              f"{r['robustness_grade']:>5s}  {r['severity_ref'][:1]}/{r['severity_standard'][:1]}"
              f" {'✓' if r['severity_match_standard'] else '✗'}")
        if "ev_f1_median" in r:
            print(f"        {r['recording']} events: F1={r['ev_f1_median']:.3f}  "
                  f"Δt={r['ev_dt_median_s']:.2f}s  "
                  f"TP/FP/FN={r['ev_tp_median']}/{r['ev_fp_median']}/{r['ev_fn_median']}")
        if "human_f1_median" in r:
            pct = r.get("algo_f1_percentile")
            print(f"        {r['recording']} human:  F1 median={r['human_f1_median']:.3f}  "
                  f"IQR=[{r['human_f1_p25']:.3f}-{r['human_f1_p75']:.3f}]  "
                  f"range=[{r['human_f1_min']:.3f}-{r['human_f1_max']:.3f}]  "
                  f"n={r['human_f1_n_pairs']} pairs  "
                  f"→ algo {r['algo_f1_symmetric_median']:.3f} at p{pct:.0f}")
    if agg:
        print()
        print("─── Aggregate (standard profile) ─────────────────")
        print(f"  Bias:       {agg['bias']:+.2f} /h")
        print(f"  MAE:        {agg['mae']:.2f} /h")
        print(f"  SD:         {agg['sd']:.2f} /h")
        print(f"  LoA:        [{agg['loa_low']:+.2f}, {agg['loa_high']:+.2f}] /h")
        print(f"  Pearson r:  {agg['pearson_r']:.3f}")
        print(f"  Weighted κ: {agg['weighted_kappa']:.3f}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", required=True,
                   help="PSG-IPA data root (must contain Resp_events/)")
    p.add_argument("--workers", type=int, default=5,
                   help="Parallel workers (default 5)")
    p.add_argument("--recordings", nargs="+",
                   default=["SN1", "SN2", "SN3", "SN4", "SN5"],
                   help="Subset of recordings to validate")
    p.add_argument("--output-json", type=Path,
                   default=Path("/tmp/validation_results.json"),
                   help="Where to write the report JSON")
    args = p.parse_args()

    import psgscoring
    print(f"\npsgscoring v{psgscoring.__version__} — clinical profile sweep "
          f"(strict / standard / sensitive)")
    print(f"Method: paper v31 conventie — stages + events from Resp_events/ "
          f"(single time-axis)\n")

    results = []
    if args.workers > 1 and len(args.recordings) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(analyse_one, sn, args.data_dir): sn
                       for sn in args.recordings}
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for sn in args.recordings:
            results.append(analyse_one(sn, args.data_dir))
    results.sort(key=lambda r: r["recording"])

    agg = aggregate_metrics(results)
    print_summary(results, agg)

    payload = to_report_json(results, agg)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON written: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
