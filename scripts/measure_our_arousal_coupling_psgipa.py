#!/usr/bin/env python3
"""
measure_our_arousal_coupling_psgipa.py — koppelen ONZE arousals aan
respiratoire events zoals die van een mens?

DE VRAAG
--------
Op PSG-IPA geeft werkpunt 0,80 een count-ratio van 1,01 tegen twaalf
scoorders: we vinden het juiste AANTAL arousals. Op de klinische opname eindigt
maar 25,7 % van de respiratoire events in een arousal, terwijl dezelfde twaalf
scoorders 49,2 % halen (mediaan over 60 scoorder-nachten) en 68,2 % op de
opname met een vergelijkbare eventlast.

Het juiste aantal op de verkeerde plek is een localisatieprobleem, en dat is
iets anders dan een drempelprobleem -- een drempel verplaatst geen events.
Maar of die 25,7 % opnamespecifiek is of systematisch, is nooit gemeten.

DE OPZET DIE HET UIT ELKAAR HAALT
---------------------------------
Drie fracties per opname, met exact dezelfde koppeldefinitie
(`arousal_couples_to_event`, event-onset tot 15 s na het einde):

  mens x mens    hun events, hun arousals        -- de maatstaf
  ons  x ons     onze events, onze arousals      -- vergelijkbaar met de mens
  ons  x mens    onze arousals, HUN events       -- isoleert de arousals

Die derde is de sleutel. Koppelen onze arousals slecht aan MENSELIJKE events,
dan ligt het aan de arousals. Koppelen ze goed, dan liggen onze respiratoire
events op andere momenten en is het daar dat de fractie sneuvelt.

Gebruik:
    python scripts/measure_our_arousal_coupling_psgipa.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import median

import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mne.set_log_level("ERROR")

from psgscoring.arousal import arousal_couples_to_event

RECS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
EPOCH_S = 30.0
SLEEP = {"N1", "N2", "N3", "R"}
RESP = re.compile(
    r"\b(apnea|apnoea|apneu|hypopnea|hypopnoea|hypopneu|rera)\b", re.IGNORECASE)


class StageParseError(RuntimeError):
    pass


def _stage(desc: str) -> str | None:
    d = desc.strip().lower()
    if "sleep stage" not in d:
        return None
    tail = d.split("sleep stage")[-1].strip()
    if not tail:
        raise StageParseError(f"lege stadiumtekst: {desc!r}")
    if tail[0] == "w":
        return "W"
    if tail[0] == "r":
        return "R"
    cijfers = [c for c in tail if c.isdigit()]
    if not cijfers:
        raise StageParseError(f"onbekend stadium: {desc!r}")
    return {"1": "N1", "2": "N2", "3": "N3", "4": "N3"}[cijfers[0]]


def parse_scorer(txt_path: Path, dur_s: float):
    """(hypno, events, arousal_onsets) uit ÉÉN bestand — één tijdas."""
    n_ep = int(__import__("math").ceil(dur_s / EPOCH_S))
    hypno = ["W"] * n_ep
    events, arousals = [], []
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 5:
                continue
            try:
                onset, dur = float(p[2]), float(p[3])
            except ValueError:
                continue
            if not (0 <= onset < dur_s):
                continue
            st = _stage(p[4])
            if st is not None:
                ep0 = int(onset // EPOCH_S)
                for i in range(max(1, round(dur / EPOCH_S))):
                    if 0 <= ep0 + i < n_ep:
                        hypno[ep0 + i] = st
            elif RESP.search(p[4]):
                events.append((onset, dur))
            elif "arousal" in p[4].lower():
                arousals.append(onset)
    tst_h = sum(1 for s in hypno if s in SLEEP) * EPOCH_S / 3600.0
    if tst_h < 0.5:
        raise StageParseError(f"{txt_path.name}: {tst_h:.2f} u slaap")
    return hypno, events, arousals


def _frac(events, arousal_onsets, window=15.0):
    if not events:
        return None
    n = sum(1 for o, d in events
            if any(arousal_couples_to_event(x, o, o + d, window)
                   for x in arousal_onsets))
    return n / len(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path.home() / "PSG-IPA"))
    ap.add_argument("--profile", default="aasm_v3_breath")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.data_dir)
    import psgscoring

    print(f"{'opname':<8}{'mens x mens':>14}{'ons x ons':>12}"
          f"{'ons x mens':>13}{'onze ev':>9}{'mens ev':>9}{'onze ar':>9}")
    print("-" * 74)
    alles, uit = {"mm": [], "oo": [], "om": []}, {}
    for sn in RECS:
        psg = root / "Resp_events" / "PSG" / f"{sn}_Respiration.edf"
        scs = sorted((root / "Resp_events" / "Annotations" / "manual")
                     .glob(f"{sn}_Respiration_manual_scorer*.txt"))
        if not (psg.exists() and scs):
            continue
        hdr = mne.io.read_raw_edf(psg, preload=False, verbose=False)
        dur = hdr.n_times / hdr.info["sfreq"]
        per, hypnos = [], []
        for sc in scs:
            h, ev, ar = parse_scorer(sc, dur)
            per.append((ev, ar)); hypnos.append(h)
        n_ep = min(len(h) for h in hypnos)
        hypno = [max(set(x), key=x.count)
                 for x in ([h[i] for h in hypnos] for i in range(n_ep))]

        raw = mne.io.read_raw_edf(psg, preload=True, verbose=False)
        res = psgscoring.run_pneumo_analysis(
            raw, hypno=hypno, scoring_profile=a.profile)
        onze_ev = [(float(e["onset_s"]), float(e["duration_s"]))
                   for e in (res.get("respiratory") or {}).get("events", [])
                   if e.get("onset_s") is not None
                   and e.get("duration_s") is not None]
        onze_ar = [float(e["onset_s"])
                   for e in (res.get("arousal") or {}).get("events", [])
                   if e.get("onset_s") is not None]

        mm = [f for f in (_frac(ev, ar) for ev, ar in per) if f is not None]
        oo = _frac(onze_ev, onze_ar)
        om = [f for f in (_frac(ev, onze_ar) for ev, _ar in per)
              if f is not None]
        if not mm:
            continue
        uit[sn] = {"mens_x_mens": mm, "ons_x_ons": oo, "ons_x_mens": om,
                   "n_onze_ev": len(onze_ev), "n_mens_ev": median(
                       [len(ev) for ev, _a in per]),
                   "n_onze_ar": len(onze_ar)}
        alles["mm"] += mm
        if oo is not None:
            alles["oo"].append(oo)
        alles["om"] += om
        print(f"{sn:<8}{median(mm)*100:>13.1f}%"
              f"{(oo*100 if oo is not None else float('nan')):>11.1f}%"
              f"{median(om)*100:>12.1f}%"
              f"{len(onze_ev):>9}{median([len(ev) for ev,_a in per]):>9.0f}"
              f"{len(onze_ar):>9}")

    if not alles["mm"]:
        return
    print(f"\n{'mens x mens':<16}{median(alles['mm'])*100:>7.1f}%   "
          f"(n={len(alles['mm'])} scoorder-nachten)")
    print(f"{'ons x ons':<16}{median(alles['oo'])*100:>7.1f}%   "
          f"(n={len(alles['oo'])} opnames)")
    print(f"{'ons x mens':<16}{median(alles['om'])*100:>7.1f}%   "
          f"(n={len(alles['om'])})")
    print("\nOns x mens is de sleutel: laag => het ligt aan de arousals,\n"
          "hoog => onze respiratoire events liggen elders.")
    if a.out:
        Path(a.out).write_text(json.dumps(uit, indent=2, default=float))


if __name__ == "__main__":
    main()
