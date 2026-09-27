#!/usr/bin/env python3
"""Train de schone U-Net (Ehrlich 2024-recept) op ONGEBRUIKTE MESA-nachten.

    OMP_NUM_THREADS=1 bench/eeg/_venv/bin/python bench/eeg/unet50/train.py \
        --n-train 320 --n-val 80 --epochs 30 --steps 600 --workers 8

Selectie: alle MESA-nachten met EDF+XML die NIET in
/srv/DATA/MESA/gebruikte_mesa_ids.txt staan, geschud met vaste seed, eerste
n_train+n_val. Die ids worden in het register bijgeschreven onder
"## dsp-scout eeg 2026-09-27 (n)" -- de enige schrijfactie onder /srv/DATA.
PSG-IPA komt nergens in dit script voor.

Werkpunt: de drempel met de hoogste gepoolde event-F1 (IoU 0,20, slaappoort)
op de MESA-VALIDATIENACHTEN; vroegtijdig stoppen op dezelfde maat. Signalen
blijven in het geheugen; alleen gewichten, ids en metrieken gaan naar schijf.
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse, json, math, random, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER)); sys.path.insert(0, str(HIER.parent / "common")); sys.path.insert(0, str(HIER.parents[1]))
from data import (load_mesa_night, FS, MESA_EDF, MESA_XML,  # noqa: E402
                  MESA_EEG_IDX, MESA_EOG_IDX, MESA_EMG_IDX)
from model import UNet1D  # noqa: E402
from postproc import prob_to_events, gate_sleep  # noqa: E402
from evaluate import match  # noqa: E402  (bench/evaluate.py: gretige IoU-matcher)

REGISTER = Path("/srv/DATA/MESA/gebruikte_mesa_ids.txt")
HEADER = "## dsp-scout eeg 2026-09-27"
THRS = [round(0.15 + 0.05 * i, 2) for i in range(15)]      # 0,15 .. 0,85
VAL_CH = [0, 3, 5]                                          # EEG3 (C4-M1), EOG-L, EMG


def select_ids(n_total: int, seed: int) -> list[str]:
    edfs = {p.stem for p in MESA_EDF.glob("mesa-sleep-*.edf")}
    xmls = {p.stem.replace("-nsrr", "") for p in MESA_XML.glob("mesa-sleep-*-nsrr.xml")}
    used = {l.strip() for l in REGISTER.read_text().splitlines() if l.strip().startswith("mesa-sleep")}
    avail = sorted((edfs & xmls) - used)
    rng = random.Random(seed); rng.shuffle(avail)
    return avail[:n_total]


def register(ids: list[str]) -> bool:
    txt = REGISTER.read_text()
    if HEADER in txt:
        return False
    with REGISTER.open("a") as fh:
        fh.write(f"{HEADER} ({len(ids)})\n" + "\n".join(ids) + "\n")
    return True


def sample_batch(nights, bs, win, rng):
    xs = np.empty((bs, 3, win), dtype=np.float32); ys = np.empty((bs, win), dtype=np.float32)
    for i in range(bs):
        nt = nights[rng.integers(len(nights))]
        T = nt["x"].shape[1]; a, b = nt["sleep_span"]
        lo = max(0, int(a * FS) - 60 * FS); hi = min(T - win, int(b * FS) + 60 * FS - win)
        s = int(rng.integers(lo, max(lo + 1, hi)))
        eeg = rng.choice(MESA_EEG_IDX, p=[0.5, 0.25, 0.25]); eog = rng.choice(MESA_EOG_IDX)
        x = nt["x"][[eeg, eog, MESA_EMG_IDX], s:s + win].astype(np.float32)
        x *= rng.uniform(0.8, 1.3, size=(3, 1)).astype(np.float32)
        xs[i] = x; ys[i] = nt["y"][s:s + win]
    return torch.from_numpy(xs), torch.from_numpy(ys)


@torch.no_grad()
def predict_night(model, x_np, dev):
    x = torch.from_numpy(x_np.astype(np.float32))[None].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logit = model(x)
    return torch.sigmoid(logit.float())[0].cpu().numpy()


def validate(model, nights, dev):
    model.eval()
    agg = {t: [0, 0, 0] for t in THRS}; aps = []
    for nt in nights:
        p = predict_night(model, nt["x"][VAL_CH], dev)
        a, b = nt["sleep_span"]; sl = slice(int(a * FS), int(b * FS))
        n1 = (sl.stop - sl.start) // FS
        p1 = p[sl][: n1 * FS].reshape(n1, FS).mean(1); y1 = nt["y"][sl][: n1 * FS].reshape(n1, FS).max(1)
        if y1.any():
            aps.append(average_precision_score(y1, p1))
        ref = [(s, e, None) for s, e in nt["arousals"]]
        for t in THRS:
            ev = gate_sleep(prob_to_events(p, FS, t), nt["hypno"])
            tp, fp, fn, _ = match([(s, e, None) for s, e in ev], ref, "iou", 0.20)
            agg[t][0] += tp; agg[t][1] += fp; agg[t][2] += fn
    out = {}
    for t, (tp, fp, fn) in agg.items():
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        out[f"{t:.2f}"] = {"tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4),
                           "sens": round(tp / (tp + fn), 4) if tp + fn else None,
                           "ppv": round(tp / (tp + fp), 4) if tp + fp else None}
    best = max(out, key=lambda k: out[k]["f1"])
    return {"ap_mean": float(np.mean(aps)) if aps else None, "per_thr": out, "best_thr": float(best),
            "best_f1": out[best]["f1"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=320); ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--win-s", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=6)
    a = ap.parse_args()
    out = HIER; log_path = out / "train_log.json"
    ids = select_ids(a.n_train + a.n_val, a.seed)
    (out / "ids_train.txt").write_text("\n".join(ids[:a.n_train]) + "\n")
    (out / "ids_val.txt").write_text("\n".join(ids[a.n_train:]) + "\n")
    print(f"register bijgeschreven: {register(ids)} ({len(ids)} ids)", flush=True)

    t0 = time.time(); nights = {}; errors = []
    with get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(load_mesa_night, ids, chunksize=1)):
            if r is None or "error" in r:
                errors.append(r)
            else:
                nights[r["rec"]] = r
            if (i + 1) % 25 == 0:
                print(f"  geladen {i + 1}/{len(ids)} ({len(errors)} fouten) {time.time() - t0:.0f} s", flush=True)
    train_n = [nights[r] for r in ids[:a.n_train] if r in nights]
    val_n = [nights[r] for r in ids[a.n_train:] if r in nights]
    print(f"train {len(train_n)} val {len(val_n)} fouten {len(errors)} laadtijd {time.time() - t0:.0f} s", flush=True)
    frac = float(np.mean([nt["y"].mean() for nt in train_n]))
    print(f"arousal-fractie van de tijd (train): {frac:.4f}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    model = UNet1D().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    total = a.epochs * a.steps; warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    win = a.win_s * FS
    log = {"args": vars(a), "n_train": len(train_n), "n_val": len(val_n), "errors": errors,
           "n_params": n_par, "arousal_fraction_train": frac, "epochs": []}
    best_f1, best_ep, bad = -1.0, -1, 0
    for ep in range(a.epochs):
        model.train(); te = time.time(); losses = []
        for _ in range(a.steps):
            x, y = sample_batch(train_n, a.batch, win, rng)
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); losses.append(loss.item())
        tv = time.time(); val = validate(model, val_n, dev)
        rec = {"epoch": ep, "loss": float(np.mean(losses)), "train_s": round(tv - te), "val_s": round(time.time() - tv),
               "lr": sched.get_last_lr()[0], **val}
        log["epochs"].append(rec)
        print(f"ep {ep:2d} loss {rec['loss']:.4f} val AP {val['ap_mean']:.3f} best thr {val['best_thr']:.2f} "
              f"F1 {val['best_f1']:.3f} ({rec['train_s']} s + {rec['val_s']} s)", flush=True)
        if val["best_f1"] > best_f1:
            best_f1, best_ep, bad = val["best_f1"], ep, 0
            torch.save({"state_dict": model.state_dict(), "thr": val["best_thr"], "epoch": ep,
                        "val": val, "config": {"in_ch": 3, "base": 16, "pools": [2, 4, 4, 4], "k": 21}},
                       out / "model_best.pt")
        else:
            bad += 1
        log["best"] = {"epoch": best_ep, "f1": best_f1}
        log_path.write_text(json.dumps(log, indent=1))
        if bad >= a.patience:
            print("vroegtijdig gestopt", flush=True); break
    print(f"klaar: beste epoch {best_ep} val-F1 {best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
