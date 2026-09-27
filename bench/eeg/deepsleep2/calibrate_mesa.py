#!/usr/bin/env python3
"""DeepSleep2 eerlijk ijken op MESA in plaats van op PSG-IPA.

Het voorgetrainde model is op PhysioNet-2018-"target arousals" getraind (niet-
apneu-arousals, RERA-spannen inbegrepen; arousals bij apneus/hypopneus als
'ongescoord' gemaskeerd). Tegen AASM-EEG-arousalgrenzen kan het dus vroeg en
lang vuren. Hier worden op de 80 MESA-VALIDATIENACHTEN van bench/eeg/unet50
(al in het register) afgeleid: (1) de mediane onset- en offsetverschuiving van
overlappende voorspelde/menselijke paren, (2) de drempel met de hoogste
gepoolde event-F1 (IoU 0,20), met en zonder die verschuivingscorrectie. Die
waarden worden daarna ongewijzigd op de PSG-IPA-kansen (out/*_prob10hz.npz)
toegepast. PSG-IPA blijft schoon.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, sys, time
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import mne
mne.set_log_level("ERROR")
import torch

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER / "upstream")); sys.path.insert(0, str(HIER.parent / "common"))
sys.path.insert(0, str(HIER.parent / "unet50")); sys.path.insert(0, str(HIER.parents[1]))
from architectures.architecture_v1 import DeepSleepNet  # noqa: E402
from postproc import prob_to_events, gate_sleep, write_pred  # noqa: E402
from data import parse_mesa_xml, MESA_EDF, MESA_XML  # noqa: E402
from evaluate import match  # noqa: E402

REF = HIER.parent / "ref"; OUT = HIER / "out"
CKPT = HIER / "upstream" / "models" / "model_2" / "my_checkpoint_21.pth.tar"
FS = 200; MAXLEN = 2 ** 23; FS10 = 10
THRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# MESA: EEG1 = Fz-Cz, EEG2 = Cz-Oz, EEG3 = C4-M1; Flow = thermistor
MESA_SLOTS = ["EEG1", "EEG1", "EEG3", "EEG3", "EEG2", "EEG2", "EOG-L", "EMG", "Abdo", "Thor", "Flow", "SpO2", "EKG"]


def build_input_mesa(rec: str):
    names = sorted(set(MESA_SLOTS))
    raw = mne.io.read_raw_edf(str(MESA_EDF / f"{rec}.edf"), include=names, preload=True, verbose=False)
    if any(n not in raw.ch_names for n in names):
        raise ValueError(f"{rec}: kanalen ontbreken {[n for n in names if n not in raw.ch_names]}")
    dur = raw.n_times / raw.info["sfreq"]
    raw.resample(FS, verbose=False)
    X = np.stack([raw.get_data(picks=[nm])[0] for nm in MESA_SLOTS]).astype(np.float64)
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True, ddof=1)
    X = np.divide(X - mu, sd, out=np.zeros_like(X), where=sd != 0)
    n = X.shape[1]; pad = MAXLEN - n; lo = pad // 2 + pad % 2
    return np.pad(X, ((0, 0), (lo, pad // 2))).astype(np.float32), n, lo, dur


def pooled_f1(nights, thr, d_on=0.0, d_off=0.0):
    tp = fp = fn = 0
    for nt in nights:
        ev = [(a + d_on, b + d_off) for a, b in prob_to_events(nt["p"], FS10, thr)]
        ev = [(a, b) for a, b in ev if b - a >= 3.0]
        ev = gate_sleep(ev, nt["hypno"])
        t, f, n, _ = match([(a, b, None) for a, b in ev], [(a, b, None) for a, b in nt["arousals"]], "iou", 0.20)
        tp += t; fp += f; fn += n
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "sens": tp / (tp + fn) if tp + fn else None, "ppv": tp / (tp + fp) if tp + fp else None}


def pair_offsets(nights, thr):
    d_on, d_off = [], []
    for nt in nights:
        ev = gate_sleep(prob_to_events(nt["p"], FS10, thr), nt["hypno"])
        _, _, _, pairs = match([(a, b, None) for a, b in ev], [(a, b, None) for a, b in nt["arousals"]], "any", 0.0)
        for i, j in pairs:
            d_on.append(ev[i][0] - nt["arousals"][j][0]); d_off.append(ev[i][1] - nt["arousals"][j][1])
    return float(np.median(d_on)), float(np.median(d_off)), len(d_on)


def main():
    ids = [l.strip() for l in (HIER.parent / "unet50" / "ids_val.txt").read_text().splitlines() if l.strip()]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSleepNet(13, True).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model_state_dict"]); model.eval()
    nights = []; t0 = time.time()
    for i, rec in enumerate(ids):
        try:
            Xp, n, lo, dur = build_input_mesa(rec)
        except Exception as e:  # noqa: BLE001
            print(rec, "overgeslagen:", e, flush=True); continue
        with torch.no_grad():
            p = model(torch.from_numpy(Xp)[None].to(dev), True)[0, 0].float().cpu().numpy()[lo:lo + n]
        k = FS // FS10
        p10 = p[: (n // k) * k].reshape(-1, k).mean(1)
        hypno, ar = parse_mesa_xml(MESA_XML / f"{rec}-nsrr.xml", dur)
        nights.append({"rec": rec, "p": p10, "hypno": hypno, "arousals": ar})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(ids)} {time.time() - t0:.0f} s", flush=True)
    del model; torch.cuda.empty_cache()
    print(f"{len(nights)} MESA-validatienachten door het model", flush=True)
    raw = {f"{t:.2f}": pooled_f1(nights, t) for t in THRS}
    t_raw = max(raw, key=lambda k: raw[k]["f1"])
    d_on, d_off, n_pairs = pair_offsets(nights, float(t_raw))
    corr = {f"{t:.2f}": pooled_f1(nights, t, -d_on, -d_off) for t in THRS}
    t_corr = max(corr, key=lambda k: corr[k]["f1"])
    cal = {"n_nights": len(nights), "ids": [nt["rec"] for nt in nights], "raw": raw, "thr_raw": float(t_raw),
           "median_onset_shift_s": d_on, "median_offset_shift_s": d_off, "n_pairs": n_pairs,
           "corrected": corr, "thr_corr": float(t_corr)}
    (HIER / "calibratie_mesa.json").write_text(json.dumps(cal, indent=1))
    print(f"MESA: beste drempel ruw {t_raw} F1 {raw[t_raw]['f1']:.3f}; verschuiving onset {d_on:+.1f} s offset {d_off:+.1f} s; "
          f"gecorrigeerd beste drempel {t_corr} F1 {corr[t_corr]['f1']:.3f}", flush=True)
    # toepassen op PSG-IPA
    for tag, thr, dd in (("mesacal_raw", float(t_raw), (0.0, 0.0)), ("mesacal_corr", float(t_corr), (-d_on, -d_off))):
        o = HIER / f"out_{tag}"; o.mkdir(exist_ok=True); log = {"thr": thr, "shift": dd, "n": {}}
        for sn in ["SN1", "SN2", "SN3", "SN4", "SN5"]:
            p10 = np.load(OUT / f"{sn}_prob10hz.npz")["p"].astype(np.float32)
            hyp = json.loads((REF / f"{sn}_hypno.json").read_text())["hypno"]
            ev = [(a + dd[0], b + dd[1]) for a, b in prob_to_events(p10, FS10, thr)]
            ev = gate_sleep([(a, b) for a, b in ev if b - a >= 3.0 and a >= 0], hyp)
            write_pred(o / f"{sn}_pred.csv", ev); log["n"][sn] = len(ev)
        (o / "run_log.json").write_text(json.dumps(log, indent=1)); print(tag, log, flush=True)


if __name__ == "__main__":
    main()
