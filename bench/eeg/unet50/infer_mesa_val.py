#!/usr/bin/env python3
"""U-Net op de MESA-validatienachten: per nacht een pred-CSV op het gekozen werkpunt,
zodat ../common/evaluate_mesa.py hem naast psgscoring (../baseline_mesa) en MSED kan
leggen. Referentie-arousals (NSRR) worden als eventtijden meegeschreven."""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER)); sys.path.insert(0, str(HIER.parent / "common"))
from data import load_mesa_night, FS  # noqa: E402
from model import UNet1D  # noqa: E402
from postproc import prob_to_events, gate_sleep, write_pred  # noqa: E402
from train import predict_night, VAL_CH  # noqa: E402

OUT = HIER / "out_mesa"


def main():
    OUT.mkdir(exist_ok=True)
    ids = [l.strip() for l in (HIER / "ids_val.txt").read_text().splitlines() if l.strip()]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(HIER / "model_best.pt", map_location=dev, weights_only=False)
    model = UNet1D(**ck["config"]).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    thr = float(ck["thr"]); log = {"thr": thr, "per_rec": {}}; t0 = time.time()
    with get_context("spawn").Pool(6, maxtasksperchild=4) as pool:
        for nt in pool.imap_unordered(load_mesa_night, ids, chunksize=1):
            if nt is None or "error" in nt:
                log["per_rec"][nt["rec"] if nt else "?"] = {"error": nt.get("error") if nt else None}; continue
            p = predict_night(model, nt["x"][VAL_CH], dev)
            ev = gate_sleep(prob_to_events(p, FS, thr), nt["hypno"])
            write_pred(OUT / f"{nt['rec']}_pred.csv", ev); write_pred(OUT / f"{nt['rec']}_ref.csv", nt["arousals"])
            log["per_rec"][nt["rec"]] = {"n_pred": len(ev), "n_ref": len(nt["arousals"])}
    log["runtime_s"] = round(time.time() - t0)
    (OUT / "run_log.json").write_text(json.dumps(log, indent=1)); print("klaar", log["runtime_s"], "s")


if __name__ == "__main__":
    main()
