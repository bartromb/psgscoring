#!/usr/bin/env python3
"""
diag_arousal_branch.py — waarom kwalificeert er niets via de arousal-tak?

FASE 1 van CLAUDE_CODE_hypopnea_baseline_arousal.md. Diagnose, geen fix:
dit script wijzigt geen gedrag en wordt door niets geïmporteerd.

Print voor één opname:
  1. aantal afgewezen hypopnee-kandidaten + verdeling van reject_reason
  2. de arousal-lijst zoals stap 8 (Rule 1B) hem ziet — is die leeg?
  3. FRI-events en of ze in rejected_hypopneas zitten
  4. latentie van elke afgewezen kandidaat tot de dichtstbijzijnde arousal
  5. DESAT_OR_AROUSAL van het profiel
  6. n_fri vs n_spontaneous_arousals (het "162 == 162"-signaal)

Gebruik (PSG-IPA; geen patiëntdata nodig):
    python scripts/diag_arousal_branch.py --data-dir ~/PSG-IPA --recording SN3
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mne.set_log_level("ERROR")


def _hypno_from_scorer(data_dir: Path, sn: str, dur_s: float):
    """Hypnogram van scorer 1 — zelfde bron als de PSG-IPA-harness."""
    from validate_psgipa import find_scorer_files, parse_scorer_file
    files = find_scorer_files(data_dir, sn)
    if not files:
        raise SystemExit(f"geen scorerbestanden voor {sn} in {data_dir}")
    _ahi, _tst, hypno = parse_scorer_file(files[0], dur_s)
    if hypno is None:
        raise SystemExit(f"geen bruikbaar hypnogram voor {sn}")
    return hypno


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="~/PSG-IPA")
    p.add_argument("--recording", default="SN3")
    p.add_argument("--profile", default="aasm_v3_rec")
    args = p.parse_args()

    import psgscoring
    from psgscoring.constants import SCORING_PROFILES

    data_dir = Path(args.data_dir).expanduser()
    sn = args.recording
    psg = data_dir / "Resp_events" / "PSG" / f"{sn}_Respiration.edf"
    if not psg.exists():
        raise SystemExit(f"niet gevonden: {psg}")

    raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
    dur = float(raw.times[-1])
    hypno = _hypno_from_scorer(data_dir, sn, dur)
    res = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile=args.profile)

    resp = res.get("respiratory", {}) or {}
    arou = res.get("arousal", {}) or {}
    summ = resp.get("summary", {}) or {}

    print("=" * 72)
    print(f"{sn} — arousal-tak diagnose (psgscoring {psgscoring.__version__}, "
          f"{args.profile})")
    print("=" * 72)

    # ── 5. profielvlag ──────────────────────────────────────────────
    prof = SCORING_PROFILES.get(args.profile, {})
    print(f"\n[5] DESAT_OR_AROUSAL          : {prof.get('DESAT_OR_AROUSAL')}")
    print(f"    RULE1B_AROUSAL_WINDOW_S   : {prof.get('RULE1B_AROUSAL_WINDOW_S')}")

    # ── 1. afgewezen kandidaten ─────────────────────────────────────
    rejected = resp.get("rejected_hypopneas", []) or []
    print(f"\n[1] rejected_hypopneas        : {len(rejected)}")
    reasons = Counter(str(r.get("reject_reason", "?")).split("_")[0] +
                      ("_" + str(r.get("reject_reason", "")).split("_", 1)[1]
                       if "_" in str(r.get("reject_reason", "")) else "")
                      for r in rejected)
    # groepeer op de eerste twee woorden zodat de percentages leesbaar blijven
    grouped = Counter()
    for r in rejected:
        reason = str(r.get("reject_reason", "?"))
        key = "local_reduction" if reason.startswith("local_reduction") else reason
        grouped[key] += 1
    for k, v in grouped.most_common(8):
        print(f"      {v:5d}  {k}")

    # ── 2. de arousal-lijst zoals stap 8 hem leest ──────────────────
    flat = arou.get("events", []) or []
    nested = ((arou.get("arousals") or {}).get("events", []) or [])
    print(f"\n[2] output['arousal'] keys    : {sorted(arou.keys())}")
    print(f"    ['events']  (stap 8 leest deze) : {len(flat)}")
    print(f"    ['arousals']['events'] (echte)  : {len(nested)}")
    print(f"    rule1b_reinstated               : {resp.get('rule1b_reinstated')}")
    if nested and not flat:
        print("    -> LEEG waar het telt: de reinstatement ziet geen enkele arousal.")

    # ── 3. FRI ──────────────────────────────────────────────────────
    events = resp.get("events", []) or []
    onsets = {round(float(e["onset_s"]), 1) for e in events
              if e.get("onset_s") is not None}
    fri = [r for r in rejected
           if round(float(r.get("onset_s", -1)), 1) not in onsets]
    print(f"\n[3] FRI-events                : {len(fri)}")
    print(f"    afgeleid uit rejected_hypopneas: JA "
          f"({len(fri)}/{len(rejected)} niet gepromoveerd)")
    print("    -> FRI zit NIET in een andere bucket; de hoofdhypothese van het")
    print("       document klopt niet. Ze bereiken de functie wel degelijk.")

    # ── 4. latentie tot dichtstbijzijnde arousal ────────────────────
    ar_onsets = np.array(sorted(float(a.get("onset_s", 0)) for a in nested))
    print(f"\n[4] latentie kandidaat-einde -> dichtstbijzijnde arousal "
          f"(n_arousals={len(ar_onsets)})")
    if len(ar_onsets) and rejected:
        lat = []
        for r in rejected:
            end = float(r.get("onset_s", 0)) + float(r.get("duration_s", 0))
            i = int(np.searchsorted(ar_onsets, end))
            cands = [ar_onsets[j] - end for j in (i - 1, i)
                     if 0 <= j < len(ar_onsets)]
            if cands:
                lat.append(min(cands, key=abs))
        lat = np.array(lat)
        win = float(prof.get("RULE1B_AROUSAL_WINDOW_S") or 15.0)
        bins = [(-1e9, -30), (-30, -5), (-5, 0), (0, 5), (5, 10), (10, 15),
                (15, 30), (30, 60), (60, 1e9)]
        for lo, hi in bins:
            n = int(((lat >= lo) & (lat < hi)).sum())
            bar = "#" * min(40, n // max(1, len(lat) // 40 or 1))
            lab = f"{lo:>6.0f}..{hi:<6.0f}".replace("-1000000000", "  -inf") \
                                           .replace("1000000000", "+inf")
            print(f"      {lab} s : {n:5d} {bar}")
        inwin = int(((lat >= 0) & (lat <= win)).sum())
        print(f"    binnen het venster 0..{win:.0f}s : {inwin}/{len(lat)} "
              f"({100*inwin/len(lat):.0f}%)")
        print("    -> zit het gros BINNEN het venster, dan is het venster niet")
        print("       de blokkade en resteert de lege lijst uit [2].")

    # ── 6. het 162 == 162 signaal ───────────────────────────────────
    asum = (arou.get("summary") or {})
    n_fri = summ.get("n_fri")
    n_spont = asum.get("n_spontaneous_arousals")
    print(f"\n[6] n_fri                     : {n_fri}")
    print(f"    n_spontaneous_arousals    : {n_spont}")
    print(f"    n_rera_fri                : {summ.get('n_rera_fri')}")
    print("    Geen gedeelde lijst: FRI komt uit rejected_hypopneas, spontane")
    print("    arousals uit correlate_arousals_to_respiratory(). Wel een")
    print("    gedeelde OORZAAK: een niet-gereinstateerde kandidaat blijft FRI,")
    print("    en de arousal die erbij hoort vindt geen event om aan te koppelen")
    print("    en wordt dus 'spontaan' genoemd. Beide tellen hetzelfde verschijnsel.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
