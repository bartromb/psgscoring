"""Nabewerking: kansen (apneu, hypopneu) @ fs → events (onset, offset, type)."""
from __future__ import annotations
import numpy as np
SLEEP = {"N1", "N2", "N3", "R"}


def moving_average(p, n):
    if n <= 1:
        return p
    c = np.cumsum(np.insert(p.astype(np.float64), 0, 0.0))
    out = (c[n:] - c[:-n]) / n
    pad_l = (n - 1) // 2; pad_r = n - 1 - pad_l
    return np.concatenate([np.full(pad_l, out[0]), out, np.full(pad_r, out[-1])])


def runs_from_mask(mask):
    m = np.concatenate([[0], mask.astype(np.int8), [0]]); d = np.diff(m)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def probs_to_events(p_ap, p_hy, fs, thr, min_dur_s=10.0, merge_gap_s=2.0, smooth_s=1.0):
    """p_event = max(kop); drempel; gaten ≤ merge_gap samengevoegd; ≥ min_dur; type per event."""
    p = np.maximum(p_ap, p_hy)
    ps = moving_average(p, int(round(smooth_s * fs))) if smooth_s > 0 else p
    merged = []
    gap = merge_gap_s * fs
    for a, b in runs_from_mask(ps >= thr):
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if (b - a) / fs < min_dur_s:
            continue
        typ = "apnea" if float(p_ap[a:b].mean()) >= float(p_hy[a:b].mean()) else "hypopnea"
        out.append((a / fs, b / fs, typ))
    return out


def gate_sleep(events, hypno, epoch_s=30.0):
    out = []
    for a, b, t in events:
        ep = int(a // epoch_s)
        if 0 <= ep < len(hypno) and hypno[ep] in SLEEP:
            out.append((a, b, t))
    return out
