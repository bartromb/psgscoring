#!/usr/bin/env python3
"""
diagnose_hypopnea.py — waarom wijkt de hypopnee-detectie af van de scoorders?

Op PSG-IPA is de event-level F1 voor apneus 0.38-0.93 maar voor hypopneeën
0.17-0.30, terwijl de AHI-overeenstemming uitstekend is (bias +1.8/u,
r 0.997). Dat is de handtekening van compenserende fouten. Dit script
ontleedt ze in drie stappen:

  1. CONSENSUS   Hoeveel van de 12 scoorders bevestigen elk algoritme-event,
                 en welk deel van de menselijke events vindt het algoritme?
                 Scorer-instanties worden geclusterd tot DISTINCTE events —
                 zonder dat telt één event twaalf keer mee en lijken
                 percentages veel steviger dan ze zijn.

  2. AROUSAL     AASM Rule 1A accepteert >=3% desaturatie OF een arousal.
                 Rusten de gemiste events op arousals?

  3. FLOW        Een onafhankelijke flowmaat op de vier groepen, inclusief
                 een CONTROLEGROEP van willekeurige slaapvensters. Zonder
                 die controle is "groep X toont 25% reductie" niet te
                 interpreteren.

⚠️  TIJDAS — LEES DIT VOOR JE ANDERE SUBTREES GEBRUIKT
    De PSG-IPA-subtrees zijn VERSCHILLENDE excerpten van de opname:

        Resp_events/PSG/SN1_Respiration.edf    meas_date 23:46, 23281 s
        EEG_arousals/PSG/SN1_EEGarousals.edf   meas_date 23:04, 25980 s

    Een verschil van 2520 s. Onsets uit EEG_arousals zijn dus NIET
    vergelijkbaar met Resp_events zonder klokcorrectie; wie dat toch doet
    krijgt zelfverzekerde onzin. Dit script gebruikt uitsluitend de
    arousals die IN de Resp_events-scorerbestanden staan (zelfde bestand,
    zelfde tijdas). Die zijn identiek over alle 12 scoorders — het is
    referentiecontext, geen per-scoorder oordeel.

Analyse-only: raakt psgscoring/ niet aan en verandert de scoring niet.

Usage:
    python diagnose_hypopnea.py --data-dir ~/PSG-IPA --recordings SN1
    python diagnose_hypopnea.py --data-dir ~/PSG-IPA --recordings SN1 SN5 \
        --output-json /tmp/hypopnea_diag.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
mne.set_log_level("ERROR")

from validate_psgipa import (  # noqa: E402
    event_set,
    find_scorer_files,
    iou,
    parse_scorer_file,
    type_family,
)

IOU_THRESH = 0.20
DESAT_MIN_PCT = 3.0
AROUSAL_WINDOW = (-5.0, 15.0)      # t.o.v. eventeinde
FLOW_CHANNEL_HINTS = ("resp nasal", "nasal", "flow", "airflow")
ENVELOPE_WIN_S = 3.0
BASELINE_WIN = (-120.0, -10.0)     # pre-event, t.o.v. onset


# ═══════════════════════════════════════════════════════════════
#  Pure helpers (unit-getest)
# ═══════════════════════════════════════════════════════════════

def cluster_events(intervals):
    """Overlappende (onset, offset)-intervallen samenvoegen tot distincte events.

    Twaalf scoorders die hetzelfde event markeren leveren twaalf intervallen;
    die horen als ÉÉN event te tellen. Verwacht een sorteerbare reeks en
    geeft een lijst dicts met onset/offset/members terug.
    """
    out = []
    for onset, offset, *rest in sorted(intervals):
        payload = rest[0] if rest else None
        if out and onset < out[-1]["offset"]:
            out[-1]["offset"] = max(out[-1]["offset"], offset)
            out[-1]["members"].append(payload)
        else:
            out.append({"onset": onset, "offset": offset, "members": [payload]})
    return out


def rolling_p2p(x, win):
    """Peak-to-peak in een rollend venster — ruwe ademhalingsamplitude.

    Bewust géén Hilbert-envelope: die vult de dalen tussen ademhalingen op en
    overschat daardoor de amplitude tijdens een event (dezelfde valkuil die
    de ventilatory burden in v0.12.0 deed overtellen).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.zeros(0)
    win = max(1, min(int(win), n))
    xp = np.pad(x, win // 2, mode="edge")
    out = np.empty(n)
    step = max(1, win // 8)
    for i in range(0, n, step):
        seg = xp[i:i + win]
        out[i:i + step] = seg.max() - seg.min()
    return out[:n]


def median_in(env, sf, t0, t1, duration_s=None):
    """Mediane envelope tussen t0 en t1 (seconden). NaN als het venster leeg is."""
    lo = int(max(0.0, t0) * sf)
    hi = int((min(t1, duration_s) if duration_s else t1) * sf)
    hi = min(hi, env.size)
    if hi <= lo:
        return float("nan")
    return float(np.median(env[lo:hi]))


def flow_reduction_pct(env, sf, onset, offset, duration_s=None,
                       baseline_win=BASELINE_WIN):
    """Reductie (%) van de event-amplitude t.o.v. de pre-event ademhaling.

    LET OP: dit is een ONAFHANKELIJKE maat, bewust niet psgscoring's eigen
    lokale basislijn. Die zou per constructie bevestigen wat het algoritme al
    besloot. Vergelijk de uitkomst dus niet één-op-één met
    `local_baseline_min_reduction_pct`; de vensters verschillen.
    """
    base = median_in(env, sf, onset + baseline_win[0], onset + baseline_win[1],
                     duration_s)
    ev = median_in(env, sf, onset, offset, duration_s)
    if not np.isfinite(base) or base <= 0 or not np.isfinite(ev):
        return float("nan")
    return 100.0 * (1.0 - ev / base)


def overlaps_any(onset, offset, intervals, thresh=IOU_THRESH):
    return any(iou(onset, offset, a, b) >= thresh for a, b in intervals)


def pick_flow_channel(ch_names):
    for hint in FLOW_CHANNEL_HINTS:
        for ch in ch_names:
            if hint in ch.lower():
                return ch
    return None


def summarise(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "pct_ge_30": float(100.0 * np.mean(v >= 30.0)),
    }


# ═══════════════════════════════════════════════════════════════
#  Analyse per opname
# ═══════════════════════════════════════════════════════════════

def analyse(sn_id, data_dir, n_controls=200, seed=20260729, profile="aasm_v3_rec"):
    import psgscoring

    data_dir = Path(data_dir)
    psg = data_dir / "Resp_events" / "PSG" / f"{sn_id}_Respiration.edf"
    if not psg.exists():
        return {"recording": sn_id, "error": "PSG not found"}

    raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
    sf = float(raw.info["sfreq"])
    dur = float(raw.times[-1])
    files = find_scorer_files(data_dir, sn_id)
    if not files:
        return {"recording": sn_id, "error": "no scorer files"}

    _, _, hypno = parse_scorer_file(files[0], dur)
    if hypno is None:
        return {"recording": sn_id, "error": "no usable hypnogram"}

    # ── algoritme ───────────────────────────────────────────────
    res = psgscoring.run_pneumo_analysis(raw, hypno=hypno, scoring_profile=profile)
    algo_all, algo_hyp = [], []
    for e in res.get("respiratory", {}).get("events", []):
        if e.get("onset_s") is None or e.get("duration_s") is None:
            continue
        t0 = float(e["onset_s"])
        if t0 >= dur:
            continue
        t1 = t0 + float(e["duration_s"])
        algo_all.append((t0, t1))
        if type_family(e.get("type")) == "hypopnea":
            d = e.get("desaturation_pct", e.get("desat"))
            algo_hyp.append({"onset": t0, "offset": t1,
                             "desat": None if d is None else float(d)})

    # ── scoorders ───────────────────────────────────────────────
    refs, ref_hyp_per_scorer = [], []
    for f in files:
        ev = event_set(f, dur)
        if not ev:
            continue
        refs.append(ev)
        ref_hyp_per_scorer.append(
            [(a, b) for a, b, t in ev if type_family(t) == "hypopnea"])
    n_scorers = len(refs)

    # arousals: identiek over scoorders? (zelfde tijdas als de events)
    arousal_sets = []
    for f in files:
        ann = mne.read_annotations(str(f))
        arousal_sets.append(sorted(
            float(o) for o, d in zip(ann.onset, ann.description)
            if "arousal" in str(d).lower() and 0 <= o < dur))
    arousals_identical = bool(
        arousal_sets and all(len(s) == len(arousal_sets[0])
                             and np.allclose(s, arousal_sets[0], atol=0.05)
                             for s in arousal_sets))
    arousals = arousal_sets[0] if arousal_sets else []

    def has_arousal(offset):
        return any(offset + AROUSAL_WINDOW[0] <= a <= offset + AROUSAL_WINDOW[1]
                   for a in arousals)

    # ── SpO2 ────────────────────────────────────────────────────
    spo2_ch = next((c for c in raw.ch_names
                    if "sao2" in c.lower() or "spo2" in c.lower()), None)
    spo2 = raw.get_data(picks=[spo2_ch])[0] if spo2_ch else None

    def desat_of(t0, t1):
        if spo2 is None:
            return None
        pre = spo2[max(0, int((t0 - 30) * sf)):max(1, int(t0 * sf))]
        post = spo2[int(t0 * sf):min(spo2.size, int((t1 + 45) * sf))]
        if pre.size == 0 or post.size == 0:
            return None
        return float(np.nanmax(pre) - np.nanmin(post))

    # ── 1. consensus ────────────────────────────────────────────
    for h in algo_hyp:
        h["n_agree"] = sum(1 for s in ref_hyp_per_scorer
                           if overlaps_any(h["onset"], h["offset"], s))

    flat = [(a, b, si) for si, s in enumerate(ref_hyp_per_scorer) for a, b in s]
    human_events = cluster_events(flat)
    for ev in human_events:
        ev["consensus"] = len(set(ev["members"]))
        ev["found_by_algo"] = overlaps_any(
            ev["onset"], ev["offset"], [(h["onset"], h["offset"]) for h in algo_hyp])
        ev["arousal"] = has_arousal(ev["offset"])
        ev["desat"] = desat_of(ev["onset"], ev["offset"])

    bands = [(1, 2), (3, 5), (6, 8), (9, n_scorers)]
    consensus_bands = []
    for lo, hi in bands:
        sel = [e for e in human_events if lo <= e["consensus"] <= hi]
        if sel:
            consensus_bands.append({
                "band": f"{lo}-{hi}", "events": len(sel),
                "found": sum(1 for e in sel if e["found_by_algo"])})

    # ── 2. arousal ──────────────────────────────────────────────
    def evidence(e):
        d = e["desat"] is not None and e["desat"] >= DESAT_MIN_PCT
        return ("both" if d and e["arousal"] else "desat" if d
                else "arousal_only" if e["arousal"] else "neither")

    arousal_table = {}
    for key in ("desat", "both", "arousal_only", "neither"):
        sel = [e for e in human_events if evidence(e) == key]
        if sel:
            arousal_table[key] = {
                "events": len(sel),
                "found": sum(1 for e in sel if e["found_by_algo"])}

    # ── 3. flow ─────────────────────────────────────────────────
    flow_ch = pick_flow_channel(raw.ch_names)
    flow_groups = {}
    if flow_ch:
        env = rolling_p2p(raw.get_data(picks=[flow_ch])[0], int(ENVELOPE_WIN_S * sf))

        A = [(h["onset"], h["offset"]) for h in algo_hyp if h["n_agree"] == 0]
        B = [(h["onset"], h["offset"]) for h in algo_hyp if h["n_agree"] >= 1]
        C = [(e["onset"], e["offset"]) for e in human_events if not e["found_by_algo"]]

        busy = sorted([(a, b) for s in refs for a, b, _ in s] + algo_all)
        med_dur = float(np.median([b - a for a, b in (A + B)])) if (A or B) else 20.0
        rng = np.random.default_rng(seed)
        D, tries = [], 0
        while len(D) < n_controls and tries < 40 * n_controls:
            tries += 1
            t0 = float(rng.uniform(150.0, max(200.0, dur - med_dur - 60.0)))
            ep = int(t0 // 30)
            if ep >= len(hypno) or hypno[ep] not in ("N1", "N2", "N3", "R"):
                continue
            if not any(b0 < t0 + med_dur + 5 and t0 - 5 < b1 for b0, b1 in busy):
                D.append((t0, t0 + med_dur))

        for name, grp in (("A_unconfirmed_algo", A), ("B_confirmed_algo", B),
                          ("C_missed_human", C), ("D_control_sleep", D)):
            flow_groups[name] = summarise(
                [flow_reduction_pct(env, sf, a, b, dur) for a, b in grp])

    return {
        "recording": sn_id,
        "psgscoring_version": psgscoring.__version__,
        "profile": profile,
        "n_scorers": n_scorers,
        "flow_channel": flow_ch,
        "arousals": {"n": len(arousals), "identical_across_scorers": arousals_identical},
        "algo_hypopneas": {
            "n": len(algo_hyp),
            "unconfirmed": sum(1 for h in algo_hyp if h["n_agree"] == 0),
            "broadly_shared": sum(1 for h in algo_hyp if h["n_agree"] >= 6),
        },
        "human_hypopneas": {
            "distinct_events": len(human_events),
            "per_scorer_median": float(np.median(
                [len(s) for s in ref_hyp_per_scorer])) if ref_hyp_per_scorer else 0.0,
            "found_by_algo": sum(1 for e in human_events if e["found_by_algo"]),
            "consensus_histogram": dict(sorted(
                Counter(e["consensus"] for e in human_events).items())),
        },
        "consensus_bands": consensus_bands,
        "arousal_evidence": arousal_table,
        "flow_groups": flow_groups,
    }


# ═══════════════════════════════════════════════════════════════
#  Rapportage
# ═══════════════════════════════════════════════════════════════

def print_report(r):
    if "error" in r:
        print(f"{r['recording']}: ERROR — {r['error']}")
        return
    a, h = r["algo_hypopneas"], r["human_hypopneas"]
    print(f"\n{'═' * 78}\n{r['recording']} — hypopneediagnose "
          f"(psgscoring {r['psgscoring_version']}, {r['profile']})\n{'═' * 78}")
    print(f"  scoorders            : {r['n_scorers']}")
    print(f"  arousals in bestand  : {r['arousals']['n']} "
          f"(identiek over scoorders: {r['arousals']['identical_across_scorers']})")
    print(f"  algoritme-hypopneeën : {a['n']}  "
          f"waarvan door niemand bevestigd: {a['unconfirmed']}")
    print(f"  menselijke events    : {h['distinct_events']} distinct "
          f"(mediaan per scoorder {h['per_scorer_median']:.1f}); "
          f"algoritme raakt er {h['found_by_algo']}")

    print("\n  consensus van de menselijke events:")
    for k, v in h["consensus_histogram"].items():
        print(f"     {k:2d}/{r['n_scorers']} scoorders: {v:3d} events")

    if r["consensus_bands"]:
        print("\n  vindt het algoritme de events waar men het over eens is?")
        for b in r["consensus_bands"]:
            print(f"     {b['band']:>5s} scoorders: {b['found']:3d}/{b['events']:<3d} gevonden")

    if r["arousal_evidence"]:
        print("\n  menselijke events naar bewijsgrond (AASM Rule 1A):")
        for k, v in r["arousal_evidence"].items():
            pct = 100.0 * v["found"] / v["events"]
            print(f"     {k:>12s}: {v['found']:3d}/{v['events']:<3d} gevonden ({pct:3.0f}%)")

    if r["flow_groups"]:
        print(f"\n  flowreductie t.o.v. pre-event ademhaling ({r['flow_channel']}):")
        print(f"     {'groep':>20s} {'n':>4s} {'mediaan':>9s} {'[p25-p75]':>16s} {'>=30%':>7s}")
        for name, s in r["flow_groups"].items():
            if not s:
                continue
            print(f"     {name:>20s} {s['n']:>4d} {s['median']:>8.1f}% "
                  f"{'[' + format(s['p25'], '.1f') + '-' + format(s['p75'], '.1f') + ']':>16s} "
                  f"{s['pct_ge_30']:>6.0f}%")
        print("     (D_control = willekeurige slaapvensters zonder gescoord event;")
        print("      zonder die nullijn zijn de andere getallen niet te duiden)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="PSG-IPA root (bevat Resp_events/)")
    p.add_argument("--recordings", nargs="+", default=["SN1"])
    p.add_argument("--profile", default="aasm_v3_rec")
    p.add_argument("--n-controls", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    results = [analyse(sn, args.data_dir, n_controls=args.n_controls,
                       seed=args.seed, profile=args.profile)
               for sn in args.recordings]
    for r in results:
        print_report(r)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"\nJSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
