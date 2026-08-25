#!/usr/bin/env python3
"""
measure_human_arousal_coupling_psgipa.py — hoeveel van de respiratoire events
laat een MENS in een arousal eindigen?

WAAROM
------
`docs/arousal-recall-diagnose.md` neemt als verwachting "60-80 % van de events
eindigt in een arousal" en leidt daaruit af dat de 20 % op de klinische opname
te laag is. Die verwachting draagt de hele diagnose, en ze is nergens gemeten
-- ze staat er als vanzelfsprekendheid.

PSG-IPA kan hem toetsen: dezelfde twaalf scoorders scoorden zowel de
respiratoire events (`Resp_events/`) als de arousals (`EEG_arousals/`). Per
scoorder is dus te berekenen welk deel van ZIJN EIGEN events gevolgd wordt door
ZIJN EIGEN arousal -- met exact de koppeldefinitie die de bibliotheek gebruikt,
niet een nabouw ervan.

Levert dit 60-80 %, dan staat de diagnose. Levert het beduidend minder, dan is
de premisse fout en meet de klinische 20 % niets bijzonders.

Gebruik:
    python scripts/measure_human_arousal_coupling_psgipa.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psgscoring.arousal import arousal_couples_to_event

RECS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
RESP = re.compile(
    r"\b(apnea|apnoea|apneu|hypopnea|hypopnoea|hypopneu|rera)\b", re.IGNORECASE)


def _rows(txt_path: Path):
    """(onset, duur, tekst) uit een PSG-IPA-annotatiebestand."""
    uit = []
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 5:
                continue
            try:
                uit.append((float(p[2]), float(p[3]), p[4]))
            except ValueError:
                continue
    return uit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path.home() / "PSG-IPA"))
    ap.add_argument("--window", type=float, default=15.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.data_dir)

    alles, per_rec = [], {}
    print(f"{'opname':<8}{'scoorders':>10}{'events (med)':>14}"
          f"{'arousals (med)':>16}{'koppeling %':>14}{'spreiding':>16}")
    print("-" * 78)
    for sn in RECS:
        rdir = root / "Resp_events" / "Annotations" / "manual"
        fracties, n_ev, n_ar = [], [], []
        for i in range(1, 13):
            rf = rdir / f"{sn}_Respiration_manual_scorer{i}.txt"
            if not rf.exists():
                continue
            # Events EN arousals uit HETZELFDE bestand. De EEG_arousals-map
            # draagt voor dezelfde scoorder een ander aantal (SN3 scoorder10:
            # 242 hier tegen 188 daar), dus het zijn aparte scoringspassages.
            # Twee subtrees kruisen zou een koppeling meten over twee
            # tijdassen -- precies de valstrik die validate_psgipa.py vermijdt
            # door alleen Resp_events/ te lezen.
            rijen_r = _rows(rf)
            events = [(o, d) for o, d, t in rijen_r if RESP.search(t)]
            ar = [o for o, _d, t in rijen_r if "arousal" in t.lower()]
            if not events:
                continue
            gekoppeld = sum(
                1 for o, d in events
                if any(arousal_couples_to_event(x, o, o + d, a.window)
                       for x in ar))
            fracties.append(gekoppeld / len(events))
            n_ev.append(len(events)); n_ar.append(len(ar))
        if not fracties:
            continue
        per_rec[sn] = {"fracties": fracties, "n_events": n_ev,
                       "n_arousals": n_ar}
        alles += fracties
        print(f"{sn:<8}{len(fracties):>10}{median(n_ev):>14.0f}"
              f"{median(n_ar):>16.0f}{median(fracties)*100:>13.1f}%"
              f"{f'{min(fracties)*100:.0f}-{max(fracties)*100:.0f}%':>16}")

    if not alles:
        return
    print(f"\nMENSELIJKE KOPPELINGSFRACTIE, alle {len(alles)} scoorder-nachten:")
    print(f"   mediaan {median(alles)*100:.1f}%   "
          f"bereik {min(alles)*100:.0f}-{max(alles)*100:.0f}%")
    print("\nDe diagnose neemt 60-80 % als verwachting. Ligt de gemeten\n"
          "mediaan daar ver onder, dan draagt die premisse niet.")
    if a.out:
        Path(a.out).write_text(json.dumps(per_rec, indent=2, default=float))


if __name__ == "__main__":
    main()
