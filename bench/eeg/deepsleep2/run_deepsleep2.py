#!/usr/bin/env python3
"""DeepSleep 2.0 (Fonod 2022, MIT) — voorgetraind model_2, inferentie op PSG-IPA.

    bench/eeg/_venv/bin/python bench/eeg/deepsleep2/run_deepsleep2.py [--device cuda]

Invoer: 13 kanalen à 200 Hz in de PhysioNet-2018-volgorde
  0 F3-M2  1 F4-M1  2 C3-M2  3 C4-M1  4 O1-M2  5 O2-M1  6 E1-M2  7 Chin
  8 ABD  9 Chest  10 Airflow  11 SaO2  12 ECG
PSG-IPA (Resp_events) draagt F4-M1, C4-M1 (SN5: Cz-M1), O2-M1, E1-M2, chin,
abdomen, chest, nasale druk, SaO2, ECG. De ontbrekende linker homologen
(F3-M2, C3-M2, O1-M2) worden gevuld met het rechter kanaal; het model is met
RandShuffle over de zes EEG-kanalen getraind, dus die duplicatie is de minst
gewelddadige vulling. Z-normalisatie per kanaal over de hele opname (ddof=1),
gecentreerd nul-gevuld tot 2^23 samples — exact utils.preprocess van upstream.
Uitvoer: kans per sample → bench/eeg/common/postproc.prob_to_events op een
drempelraster; primair werkpunt 0,50 (vooraf gekozen: de natuurlijke drempel
van een sigmoïde segmentatie; het model levert geen eigen drempel), rest =
orakelcurve. Slaappoort via het scoorder-1-hypnogram.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
import numpy as np
import mne
mne.set_log_level("ERROR")
import torch

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER / "upstream"))
sys.path.insert(0, str(HIER.parent / "common"))
from architectures.architecture_v1 import DeepSleepNet  # noqa: E402
from postproc import prob_to_events, gate_sleep, write_pred  # noqa: E402

ROOT = Path("/srv/DATA/PSG-IPA/Resp_events/PSG")
REF = HIER.parent / "ref"
OUT = HIER / "out"
CKPT = HIER / "upstream" / "models" / "model_2" / "my_checkpoint_21.pth.tar"
FS = 200
MAXLEN = 2 ** 23
THRS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
PRIMARY = 0.50

# per slot: kandidaatnamen in volgorde van voorkeur
SLOTS = [
    ("F3-M2",  ["EEG F3-M2", "EEG F4-M1"]),
    ("F4-M1",  ["EEG F4-M1"]),
    ("C3-M2",  ["EEG C3-M2", "EEG C4-M1", "EEG Cz-M1"]),
    ("C4-M1",  ["EEG C4-M1", "EEG Cz-M1"]),
    ("O1-M2",  ["EEG O1-M2", "EEG O2-M1"]),
    ("O2-M1",  ["EEG O2-M1"]),
    ("E1-M2",  ["EOG E1-M2"]),
    ("Chin",   ["EMG chin"]),
    ("ABD",    ["Resp abdomen"]),
    ("Chest",  ["Resp chest"]),
    ("Airflow", ["Resp nasal"]),
    ("SaO2",   ["SaO2"]),
    ("ECG",    ["ECG"]),
]


def build_input(sn: str):
    raw = mne.io.read_raw_edf(str(ROOT / f"{sn}_Respiration.edf"), preload=True, verbose=False)
    chosen = {}
    for slot, cands in SLOTS:
        nm = next((c for c in cands if c in raw.ch_names), None)
        if nm is None:
            raise SystemExit(f"{sn}: geen kanaal voor slot {slot}")
        chosen[slot] = nm
    raw.pick(sorted(set(chosen.values()), key=raw.ch_names.index))
    raw.resample(FS, verbose=False)
    X = np.stack([raw.get_data(picks=[chosen[s]])[0] for s, _ in SLOTS]).astype(np.float64)
    mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True, ddof=1)
    X = np.divide(X - mu, sd, out=np.zeros_like(X), where=sd != 0)
    n = X.shape[1]
    pad = MAXLEN - n
    if pad < 0:
        raise SystemExit(f"{sn}: langer dan 2^23 samples")
    lo = pad // 2 + pad % 2
    Xp = np.pad(X, ((0, 0), (lo, pad // 2))).astype(np.float32)
    return Xp, n, lo, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sns", default="SN1,SN2,SN3,SN4,SN5")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    dev = torch.device(a.device)
    model = DeepSleepNet(in_channels=13, linear=True).to(dev)
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    log = {}
    for sn in a.sns.split(","):
        t0 = time.time()
        Xp, n, lo, chosen = build_input(sn)
        meta = json.loads((REF / f"{sn}_hypno.json").read_text())
        with torch.no_grad():
            x = torch.from_numpy(Xp)[None].to(dev)
            p = model(x, True)[0, 0].float().cpu().numpy()
        p = p[lo:lo + n]
        # bewaar de kans op 10 Hz (float16) voor herbeoordeling zonder GPU
        k = FS // 10
        p10 = p[: (n // k) * k].reshape(-1, k).mean(axis=1).astype(np.float16)
        np.savez_compressed(OUT / f"{sn}_prob10hz.npz", p=p10, fs=10)
        rij = {"channels": chosen, "n_samples": n, "runtime_s": round(time.time() - t0, 1),
               "p_mean": float(p.mean()), "p_p95": float(np.percentile(p, 95)), "n_events": {}}
        for thr in THRS:
            ev = prob_to_events(p, FS, thr)
            ev_g = gate_sleep(ev, meta["hypno"])
            write_pred(OUT / f"{sn}_pred_t{thr:.2f}.csv", ev_g)
            rij["n_events"][f"{thr:.2f}"] = {"raw": len(ev), "gated": len(ev_g)}
            if abs(thr - PRIMARY) < 1e-9:
                write_pred(OUT / f"{sn}_pred.csv", ev_g)
        log[sn] = rij
        print(sn, {k: v["gated"] for k, v in rij["n_events"].items()}, f"{rij['runtime_s']} s", flush=True)
    (OUT / "run_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
