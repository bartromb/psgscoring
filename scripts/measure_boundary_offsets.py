#!/usr/bin/env python
"""
measure_boundary_offsets.py — zijn de eventgrenzen systematisch verschoven
ten opzichte van menselijke scoring, of vallen ze binnen de menselijke
spreiding?

Meet, per opname en per profiel, de GETEKENDE onset- en offsetverschillen van
elk gematcht algoritme↔scorer-paar op PSG-IPA, en zet daar de
mens-tegen-mens-verdeling van dezelfde grootheid naast. Zonder die referentie
is "het algoritme wijkt 1,8 s af" niet te interpreteren: menselijke scorers
wijken onderling ook af.

Drie mogelijke uitkomsten, met de conclusie erbij:

  mediaan ≈ 0, spreiding ≈ menselijk   → grenzen zijn goed; niets doen
  mediaan ≠ 0 met vaste richting       → envelope-lag; corrigeer bij de bron
                                          (ademhalings-granulair snappen),
                                          NIET met een gefitte naschuifconstante
  spreiding ≫ menselijk, geen richting → jitter in de flankdetectie

Bijvangst: `n_lost_to_iou` telt referentie-events waarvoor wél een
algoritme-event met overlap bestaat maar de IoU onder de matchdrempel blijft.
Dat getal herclassificeert een deel van de event-F1-kloof van "gemist" naar
"anders afgebakend".

Dit script wijzigt niets aan de scoring. Het is stap 1 van blok 1B; stap 2
(het profielveld `event_boundaries="breath"`) bestaat pas als deze meting een
systematische offset laat zien.

GEBRUIK
-------
    PSGSCORING_AROUSAL_DERIVATION=single PYTHONPATH=$PWD \
        python scripts/measure_boundary_offsets.py \
        --data-dir /path/to/PSG-IPA \
        --profiles aasm_v3_rec aasm_v3_breath \
        --out boundary_offsets

Schrijft {out}_pairs.csv (één rij per gematcht paar) en {out}_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mne  # noqa: E402

from validate_psgipa import (  # noqa: E402
    LEGACY_MATCHER, event_set, find_scorer_files, match_events,
    parse_scorer_file, percentile_of, type_family,
)

SN_IDS = ("SN1", "SN2", "SN3", "SN4", "SN5")


# ---------------------------------------------------------------------------
# Kern: getekende delta's per gematcht paar
# ---------------------------------------------------------------------------

def signed_deltas(a_events, b_events, matcher) -> list[dict]:
    """Per match (a→b): d_onset, d_offset, d_duration, GETEKEND (a − b).

    Getekend is het hele punt: een offset heeft een richting of hij bestaat
    niet. `match_events` retourneert `matched_pairs` als (i, j, iou); de
    delta's worden hier afgeleid in plaats van in de matcher, zodat de
    matcher zelf ongewijzigd blijft en de gepubliceerde F1's blijven staan.
    """
    m = match_events(a_events, b_events, **matcher)
    out = []
    for i, j, iou_v in m["matched_pairs"]:
        a0, a1, at = a_events[i]
        b0, b1, bt = b_events[j]
        out.append({
            "family": type_family(at) or type_family(bt) or "unknown",
            "d_onset": round(a0 - b0, 2),
            "d_offset": round(a1 - b1, 2),
            "d_duration": round((a1 - a0) - (b1 - b0), 2),
            "iou": round(iou_v, 3),
        })
    return out


def lost_to_iou(algo_events, ref_events, iou_thresh: float) -> int:
    """Referentie-events met overlap > 0 naar een algoritme-event maar zonder
    match op de drempel: gevonden, anders afgebakend."""
    n = 0
    for b0, b1, _bt in ref_events:
        best = 0.0
        overlaps = False
        for a0, a1, _at in algo_events:
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            if inter > 0:
                overlaps = True
                union = max(a1, b1) - min(a0, b0)
                best = max(best, inter / union if union > 0 else 0.0)
        if overlaps and best < iou_thresh:
            n += 1
    return n


def _summarise(deltas: list[dict], key: str) -> dict:
    vals = [d[key] for d in deltas]
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": len(vals),
        "median": round(float(np.median(arr)), 2),
        "p25": round(float(np.percentile(arr, 25)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "mean_abs": round(float(np.mean(np.abs(arr))), 2),
    }


# ---------------------------------------------------------------------------
# Per opname
# ---------------------------------------------------------------------------

def analyse_recording(sn_id: str, data_dir: Path, profiles: list[str],
                      matcher: dict, writer: csv.DictWriter) -> dict | None:
    import psgscoring

    psg_path = data_dir / "Resp_events" / "PSG" / f"{sn_id}_Respiration.edf"
    if not psg_path.exists():
        print(f"  {sn_id}: EDF niet gevonden", file=sys.stderr)
        return None
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose="ERROR")
    sig_dur_s = float(raw.times[-1])

    scorer_files = find_scorer_files(data_dir, sn_id)
    if not scorer_files:
        print(f"  {sn_id}: geen scoorderbestanden", file=sys.stderr)
        return None
    _ahi, _tst, hypno1 = parse_scorer_file(scorer_files[0], sig_dur_s)
    scorer_sets = [(f.stem, event_set(f, sig_dur_s)) for f in scorer_files]
    scorer_sets = [(n, ev) for n, ev in scorer_sets if ev]

    # --- menselijke referentieverdeling: zelfde delta's, mens tegen mens.
    # Richtingsconventie is willekeurig tussen mensen; voor de VERDELING
    # (spreiding, IQR) maakt dat niet uit, en de mediaan hoort ~0 te zijn —
    # dat is meteen een sanity-check op de meting zelf.
    human_deltas: list[dict] = []
    for i in range(len(scorer_sets)):
        for j in range(i + 1, len(scorer_sets)):
            human_deltas.extend(
                signed_deltas(scorer_sets[i][1], scorer_sets[j][1], matcher))

    rec_out = {"n_scorers": len(scorer_sets),
               "human": {
                   "onset": _summarise(human_deltas, "d_onset"),
                   "offset": _summarise(human_deltas, "d_offset"),
               },
               "profiles": {}}

    human_abs_onset = [abs(d["d_onset"]) for d in human_deltas]
    human_abs_offset = [abs(d["d_offset"]) for d in human_deltas]

    for prof in profiles:
        results = psgscoring.run_pneumo_analysis(
            raw, hypno=hypno1, scoring_profile=prof)
        events = (results.get("respiratory") or {}).get("events") or []
        algo = []
        for e in events:
            try:
                o, d = float(e["onset_s"]), float(e["duration_s"])
            except (KeyError, TypeError, ValueError):
                continue
            algo.append((o, o + d, e.get("type")))

        all_deltas: list[dict] = []
        n_lost = 0
        for scorer_name, sc_events in scorer_sets:
            ds = signed_deltas(algo, sc_events, matcher)
            for d in ds:
                row = {"recording": sn_id, "profile": prof,
                       "scorer": scorer_name, **d}
                writer.writerow(row)
            all_deltas.extend(ds)
            n_lost += lost_to_iou(algo, sc_events,
                                  matcher.get("iou_thresh", 0.20))

        per_family = {}
        for fam in ("apnea", "hypopnea"):
            fd = [d for d in all_deltas if d["family"] == fam]
            per_family[fam] = {
                "onset": _summarise(fd, "d_onset"),
                "offset": _summarise(fd, "d_offset"),
                "duration": _summarise(fd, "d_duration"),
            }

        med_abs_on = (float(np.median([abs(d["d_onset"]) for d in all_deltas]))
                      if all_deltas else None)
        med_abs_off = (float(np.median([abs(d["d_offset"]) for d in all_deltas]))
                       if all_deltas else None)
        rec_out["profiles"][prof] = {
            "n_algo_events": len(algo),
            "n_matched_pairs": len(all_deltas),
            "n_lost_to_iou_total": n_lost,
            "by_family": per_family,
            # Percentiel van de |delta|-mediaan binnen de menselijke
            # |delta|-verdeling: >50 betekent slechter dan de mediane mens.
            "abs_onset_percentile_vs_humans":
                round(percentile_of(med_abs_on, human_abs_onset), 1)
                if med_abs_on is not None and human_abs_onset else None,
            "abs_offset_percentile_vs_humans":
                round(percentile_of(med_abs_off, human_abs_offset), 1)
                if med_abs_off is not None and human_abs_offset else None,
        }
        pf = rec_out["profiles"][prof]
        print(f"  {sn_id} {prof}: onset med "
              f"{per_family['hypopnea']['onset'].get('median')} s, offset med "
              f"{per_family['hypopnea']['offset'].get('median')} s (hyp), "
              f"lost_to_iou {n_lost}, pct vs mens "
              f"{pf['abs_onset_percentile_vs_humans']}/"
              f"{pf['abs_offset_percentile_vs_humans']}")
    return rec_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--profiles", nargs="+",
                    default=["aasm_v3_rec", "aasm_v3_breath"])
    ap.add_argument("--recordings", nargs="*", default=list(SN_IDS),
                    choices=list(SN_IDS))
    ap.add_argument("--out", default="boundary_offsets",
                    help="basisnaam voor _pairs.csv en _summary.json")
    args = ap.parse_args(argv)

    matcher = dict(LEGACY_MATCHER)
    pairs_path = Path(f"{args.out}_pairs.csv")
    summary_path = Path(f"{args.out}_summary.json")

    summary: dict = {"matcher": matcher, "recordings": {}}
    with pairs_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "recording", "profile", "scorer", "family",
            "d_onset", "d_offset", "d_duration", "iou"])
        writer.writeheader()
        for sn_id in args.recordings:
            rec = analyse_recording(sn_id, args.data_dir, args.profiles,
                                    matcher, writer)
            if rec is not None:
                summary["recordings"][sn_id] = rec

    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nGeschreven: {pairs_path} en {summary_path}")
    print("Lees eerst de 'human'-blokken: de menselijke mediaan hoort ~0 te "
          "zijn (sanity-check), en de IQR is de ruisvloer waartegen elke "
          "algoritme-offset beoordeeld wordt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
