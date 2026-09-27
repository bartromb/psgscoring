"""Gedeelde nabewerking voor de kandidaten: kans → events, slaappoort, CSV.

Alle kandidaten krijgen dezelfde, vooraf vastgelegde nabewerking, zodat een
verschil in F1 aan de detector ligt en niet aan de eventvorming:
  - prob_to_events: 1 s-gemiddelde van de kans, drempel, aaneengesloten runs,
    gaten ≤ merge_gap_s dichten, runs < min_dur_s (AASM: 3 s) weg.
  - gate_sleep: alleen events waarvan de onset in een slaapepoch (N1/N2/N3/R)
    van het scoorder-1-hypnogram valt — psgscoring scoort zelf ook geen
    arousals in W (score_wake_arousals=False), en de referentie-annotatie is
    op slaap gescoord.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

SLEEP = {"N1", "N2", "N3", "R"}


def moving_average(p: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return p
    c = np.cumsum(np.insert(p.astype(np.float64), 0, 0.0))
    out = (c[n:] - c[:-n]) / n
    pad_l = (n - 1) // 2
    pad_r = n - 1 - pad_l
    return np.concatenate([np.full(pad_l, out[0]), out, np.full(pad_r, out[-1])])


def runs_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def prob_to_events(p: np.ndarray, fs: float, thr: float, min_dur_s: float = 3.0,
                   merge_gap_s: float = 1.0, smooth_s: float = 1.0) -> list[tuple[float, float]]:
    ps = moving_average(p, int(round(smooth_s * fs))) if smooth_s > 0 else p
    runs = runs_from_mask(ps >= thr)
    merged: list[list[int]] = []
    gap = merge_gap_s * fs
    for a, b in runs:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a / fs, b / fs) for a, b in merged if (b - a) / fs >= min_dur_s]


def gate_sleep(events, hypno: list, epoch_s: float = 30.0):
    out = []
    for a, b in events:
        ep = int(a // epoch_s)
        if 0 <= ep < len(hypno) and hypno[ep] in SLEEP:
            out.append((a, b))
    return out


def write_pred(path: Path, events, typ: str = "arousal"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["onset_s", "offset_s", "type"])
        for a, b in sorted(events):
            w.writerow([f"{a:.3f}", f"{b:.3f}", typ])
