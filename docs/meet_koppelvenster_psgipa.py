"""Hoe koppelen mensen een hypopnee aan een arousal?

`RULE1B_AROUSAL_WINDOW_S = 15.0` is een gekozen getal. PSG-IPA `Resp_events/`
laat het meten: twaalf onafhankelijke respiratoire scoringen tegen EEN vaste
arousal-annotatie die de scoorders zagen tijdens het scoren. Voor elke
gescoorde hypopnee is de latentie tot de eerstvolgende arousal-onset dus de
koppeling die een mens werkelijk maakte.

Nul-verdeling: dezelfde events 120 s verschoven. Waar de waargenomen curve
boven de nulcurve uitkomt, zit echte koppeling; waar ze samenvallen, is het
toeval van de arousal-dichtheid.

Leest alleen annotaties -- geen signaalverwerking, dus verwaarloosbaar CPU.
"""
import statistics as st
from collections import defaultdict
from pathlib import Path
import mne; mne.set_log_level("ERROR")
import numpy as np

D = Path("/home/bart/PSG-IPA/Resp_events/Annotations/manual")
VENSTERS = [0, 2, 5, 10, 15, 20, 30, 45, 60]
SHIFT = 120.0

def latenties(events, arousals, shift=0.0):
    """Voor elk event: tijd van eventEINDE tot de eerstvolgende arousal-onset."""
    ao = np.array(sorted(arousals))
    out = []
    for o, d in events:
        end = o + d + shift
        nxt = ao[ao >= end]
        out.append(float(nxt[0] - end) if len(nxt) else float("inf"))
    return out

alle_obs, alle_null = [], []
per_type = defaultdict(list)
for sn in ("SN1", "SN2", "SN3", "SN4", "SN5"):
    fs = sorted(D.glob(f"{sn}_Respiration_manual_scorer*.edf"))
    if not fs:
        continue
    a0 = mne.read_annotations(str(fs[0]))
    arousals = [float(o) for o, x in zip(a0.onset, a0.description)
                if "eeg arousal" in str(x).lower()]
    obs, null = [], []
    for f in fs:
        a = mne.read_annotations(str(f))
        hyp = [(float(o), float(d)) for o, d, x in zip(a.onset, a.duration, a.description)
               if "hypopnea" in str(x).lower()]
        apn = [(float(o), float(d)) for o, d, x in zip(a.onset, a.duration, a.description)
               if "apnea" in str(x).lower() and "hypopnea" not in str(x).lower()]
        obs += latenties(hyp, arousals)
        null += latenties(hyp, arousals, SHIFT)
        per_type["hypopnee"] += latenties(hyp, arousals)
        per_type["apneu"] += latenties(apn, arousals)
    alle_obs += obs; alle_null += null
    f15 = sum(1 for x in obs if x <= 15) / len(obs) if obs else 0
    n15 = sum(1 for x in null if x <= 15) / len(null) if null else 0
    print(f"{sn}: {len(arousals):4d} arousals, {len(obs):5d} hypopneus over 12 scoorders "
          f"| binnen 15 s: {100*f15:5.1f}%  (nul {100*n15:5.1f}%)")

print(f"\nALLE: {len(alle_obs)} hypopnee-scoringen")
print(f"{'venster':>8} {'waargenomen':>12} {'nul (120s verschoven)':>22} {'overschot':>10}")
for w in VENSTERS:
    o = sum(1 for x in alle_obs if x <= w) / len(alle_obs)
    n = sum(1 for x in alle_null if x <= w) / len(alle_null)
    print(f"{w:6d} s {100*o:11.1f}% {100*n:21.1f}% {100*(o-n):9.1f}")
fin = [x for x in alle_obs if np.isfinite(x)]
print(f"\nmediane latentie {st.median(fin):.1f}s | "
      f"10e {np.percentile(fin,10):.1f} 25e {np.percentile(fin,25):.1f} "
      f"75e {np.percentile(fin,75):.1f} 90e {np.percentile(fin,90):.1f}")
for k, v in per_type.items():
    fv = [x for x in v if np.isfinite(x)]
    if fv:
        print(f"  {k:10s} n={len(v):5d} mediaan {st.median(fv):6.1f}s  "
              f"binnen 15 s {100*sum(1 for x in v if x<=15)/len(v):5.1f}%")
