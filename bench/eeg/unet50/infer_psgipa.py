#!/usr/bin/env python3
"""U-Net (MESA-getraind) op PSG-IPA: kans -> events met het MESA-validatiewerkpunt.

    bench/eeg/_venv/bin/python bench/eeg/unet50/infer_psgipa.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys, time
from pathlib import Path
import numpy as np
import torch

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER)); sys.path.insert(0, str(HIER.parent / "common"))
from data import load_psgipa, FS  # noqa: E402
from model import UNet1D  # noqa: E402
from postproc import prob_to_events, gate_sleep, write_pred  # noqa: E402
from train import predict_night  # noqa: E402

REF = HIER.parent / "ref"; OUT = HIER / "out"
THRS = [round(0.15 + 0.05 * i, 2) for i in range(15)]


def main():
    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(HIER / "model_best.pt", map_location=dev, weights_only=False)
    model = UNet1D(**ck["config"]).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    thr0 = float(ck["thr"])
    log = {"thr_primary": thr0, "epoch": ck["epoch"], "val_f1": ck["val"]["best_f1"], "per_sn": {}}
    for sn in ["SN1", "SN2", "SN3", "SN4", "SN5"]:
        t0 = time.time()
        X, dur, names = load_psgipa(sn)
        meta = json.loads((REF / f"{sn}_hypno.json").read_text())
        p = predict_night(model, X, dev)
        k = FS // 10
        np.savez_compressed(OUT / f"{sn}_prob10hz.npz", p=p[: (len(p) // k) * k].reshape(-1, k).mean(1).astype(np.float16), fs=10)
        rij = {"channels": names, "runtime_s": round(time.time() - t0, 1), "n_events": {}}
        for thr in sorted(set(THRS + [thr0])):
            ev = gate_sleep(prob_to_events(p, FS, thr), meta["hypno"])
            write_pred(OUT / f"{sn}_pred_t{thr:.2f}.csv", ev)
            rij["n_events"][f"{thr:.2f}"] = len(ev)
            if abs(thr - thr0) < 1e-9:
                write_pred(OUT / f"{sn}_pred.csv", ev)
        log["per_sn"][sn] = rij
        print(sn, names, "n@thr", rij["n_events"][f"{thr0:.2f}"], flush=True)
    (OUT / "run_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
