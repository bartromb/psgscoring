#!/usr/bin/env python3
"""Train de respiratoire U-Net op MESA-nachten buiten de twee testsets (standaard-n150 en de 140 van 07-10).

    OMP_NUM_THREADS=1 ../eeg/_venv/bin/python train.py --n-train 400 --n-val 100 --epochs 30 --steps 400 --workers 6

Werkpunt en vroegtijdig stoppen op de gepoolde event-F1 (IoU 0,20, typeonbewust, slaappoort)
op de validatienachten; signalen blijven in het geheugen."""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse, json, math, random, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER)); sys.path.insert(0, str(HIER.parents[1]))
from data import load_mesa_night, FS, MESA_EDF, MESA_XML, EPOCH_S  # noqa: E402
from model import UNet1D  # noqa: E402
from postproc import probs_to_events, gate_sleep  # noqa: E402
from evaluate import match  # noqa: E402  (bench/evaluate.py)

REGISTER = Path("/srv/DATA/MESA/gebruikte_mesa_ids.txt")
HEADER = "## dsp-scout resp 2026-10-07"
THRS = [round(0.15 + 0.05 * i, 2) for i in range(15)]
STD150_SEED, STEP2_LIST = 20260801, HIER.parents[1] / "docs" / "breath_dual_mesa_20261007" / "opnames.txt"


def select_ids(n_total: int, seed: int) -> list[str]:
    edfs = {p.stem for p in MESA_EDF.glob("mesa-sleep-*.edf")}
    xmls = {p.stem.replace("-nsrr", "") for p in MESA_XML.glob("mesa-sleep-*-nsrr.xml")}
    ids = sorted(edfs & xmls)
    std150 = set(random.Random(STD150_SEED).sample(ids, 150))
    step2 = set(STEP2_LIST.read_text().split()) if STEP2_LIST.exists() else set()
    avail = [x for x in ids if x not in std150 and x not in step2]
    rng = random.Random(seed); rng.shuffle(avail)
    return avail[:n_total]


def register(ids: list[str]) -> bool:
    txt = REGISTER.read_text()
    if HEADER in txt:
        return False
    with REGISTER.open("a") as fh:
        fh.write(f"{HEADER} — buiten standaard-n150 en de 140 van 07-10; hergebruik ({len(ids)})\n" + "\n".join(ids) + "\n")
    return True


def sample_batch(nights, bs, win, rng):
    xs = np.empty((bs, 5, win), dtype=np.float32); ys = np.empty((bs, 2, win), dtype=np.float32)
    for i in range(bs):
        nt = nights[rng.integers(len(nights))]
        T = nt["x"].shape[1]; a, b = nt["sleep_span"]
        lo = max(0, int(a * FS) - 120 * FS); hi = min(T - win, int(b * FS) + 120 * FS - win)
        s = int(rng.integers(lo, max(lo + 1, hi)))
        x = nt["x"][:, s:s + win].astype(np.float32)
        # kanaal-uitval: druk p 0,30, thermistor p 0,20 (nooit beide), effortbanden p 0,10
        drop_p = rng.random() < 0.30 and nt["mask"][1]
        drop_t = (not drop_p) and rng.random() < 0.20 and nt["mask"][0]
        if drop_p: x[0] = 0.0
        if drop_t: x[1] = 0.0
        for k in (2, 3):
            if rng.random() < 0.10: x[k] = 0.0
        x[:4] *= rng.uniform(0.8, 1.3, size=(4, 1)).astype(np.float32)
        xs[i] = x; ys[i] = nt["y"][:, s:s + win]
    return torch.from_numpy(xs), torch.from_numpy(ys)


@torch.no_grad()
def predict_night(model, x_np, dev):
    x = torch.from_numpy(x_np.astype(np.float32))[None].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logit = model(x)
    return torch.sigmoid(logit.float())[0].cpu().numpy()


def validate(model, nights, dev):
    model.eval()
    agg = {t: [0, 0, 0] for t in THRS}
    for nt in nights:
        p = predict_night(model, nt["x"], dev)
        ref = [(a, b, None) for a, b, _t in nt["events"]]
        for t in THRS:
            ev = gate_sleep(probs_to_events(p[0], p[1], FS, t), nt["hypno"])
            tp, fp, fn, _ = match([(a, b, None) for a, b, _t in ev], ref, "iou", 0.20)
            agg[t][0] += tp; agg[t][1] += fp; agg[t][2] += fn
    out = {}
    for t, (tp, fp, fn) in agg.items():
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        out[f"{t:.2f}"] = {"tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4),
                           "sens": round(tp / (tp + fn), 4) if tp + fn else None,
                           "ppv": round(tp / (tp + fp), 4) if tp + fp else None}
    best = max(out, key=lambda k: out[k]["f1"])
    return {"per_thr": out, "best_thr": float(best), "best_f1": out[best]["f1"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400); ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--win-s", type=int, default=1800)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--seed", type=int, default=20261007)
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
    frac = [float(np.mean([nt["y"][k].mean() for nt in train_n])) for k in (0, 1)]
    print(f"tijdfractie apneu {frac[0]:.4f} hypopneu {frac[1]:.4f}", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    cfg = {"in_ch": 5, "out_ch": 2, "base": 16, "pools": [2, 4, 4, 4], "k": 21}
    model = UNet1D(**cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    total = a.epochs * a.steps; warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    win = a.win_s * FS
    log = {"args": vars(a), "n_train": len(train_n), "n_val": len(val_n), "errors": errors, "n_params": n_par,
           "fraction_train": frac, "config": cfg, "epochs": []}
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
        print(f"ep {ep:2d} loss {rec['loss']:.4f} best thr {val['best_thr']:.2f} F1 {val['best_f1']:.3f} "
              f"({rec['train_s']} s + {rec['val_s']} s)", flush=True)
        if val["best_f1"] > best_f1:
            best_f1, best_ep, bad = val["best_f1"], ep, 0
            torch.save({"state_dict": model.state_dict(), "thr": val["best_thr"], "epoch": ep, "val": val, "config": cfg},
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
