#!/usr/bin/env python3
"""Keten van de schaalmeting (docs/arousal_unet_schaal_preregistratie_20261025.md):

    laden (MESA 1278+40, SHHS1 1278+40, in het geheugen)
      -> bevroren model: τ op de validatieset van 80
      -> arm M (MESA, 3 kanalen) -> arm MS (MESA+SHHS1, 3) -> arm MSF (MESA+SHHS1, 4 met flow)
      -> inferentie van bevroren + M + MS + MSF op SHHS1-150b, MESA-76, PSG-IPA SN1-5

    OMP_NUM_THREADS=1 bench/eeg/_venv/bin/python bench/eeg/unet_scale/train_scale.py \
        --out /srv/CODE/docs/arousal_unet_schaal_20261025 --workers 8

Hervatbaar: een arm met `KLAAR` in zijn uitvoermap wordt overgeslagen. Importeert de
bibliotheek `psgscoring` NIET (de 10 s-samenvoeging komt pas in de evaluatie). Niets
onder /srv/DATA wordt geschreven; signalen blijven in het geheugen.
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse, gc, hashlib, json, math, resource, subprocess, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

HIER = Path(__file__).resolve().parent
for p in (HIER, HIER.parent / "unet50", HIER.parent / "common", HIER.parents[1]):
    sys.path.insert(0, str(p))
from data_scale import load_night, read_ids, FS  # noqa: E402
from model import UNet1D  # noqa: E402
from postproc import prob_to_events, gate_sleep, write_pred  # noqa: E402
from evaluate import match  # noqa: E402  (bench/evaluate.py: gretige IoU-matcher)

REPO = HIER.parents[2]
PREREG = REPO / "docs" / "arousal_unet_schaal_preregistratie_20261025.md"
FROZEN = Path("/srv/CODE/docs/arousal_unet_20260927/model_frozen_cc91bb86.pt")
FROZEN_SHA = "cc91bb867c167d7387cf619c8dd005ba294d30f640d0fe0a59c2eaeb3a798566"
THRS = [round(0.15 + 0.05 * i, 2) for i in range(15)]           # 0,15 .. 0,85
LIB_TAU = 0.35
# naam, cohorten in de training, flow als vierde ingang, max. epochs
ARMS = [("M", ("mesa",), False, 50), ("MS", ("mesa", "shhs"), False, 60), ("MSF", ("mesa", "shhs"), True, 60)]
SNS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blok in iter(lambda: fh.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ── laden ─────────────────────────────────────────────────────────────────────────────────
def load_set(specs: list[tuple], workers: int, label: str) -> tuple[dict, list]:
    t0 = time.time(); nights: dict = {}; errors: list = []
    with get_context("spawn").Pool(workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(load_night, specs, chunksize=1)):
            if r is None or "error" in r:
                errors.append(r)
            else:
                nights[(r["cohort"], r["rec"])] = r
            if (i + 1) % 100 == 0 or i + 1 == len(specs):
                log(f"  {label}: geladen {i + 1}/{len(specs)} ({len(errors)} fouten) {time.time() - t0:.0f} s, RSS {rss_gb():.1f} GB")
    return nights, errors


# ── training ──────────────────────────────────────────────────────────────────────────────
def sample_batch(nights, bs, win, rng, with_flow):
    n_in = 4 if with_flow else 3
    xs = np.empty((bs, n_in, win), dtype=np.float32); ys = np.empty((bs, win), dtype=np.float32)
    for i in range(bs):
        nt = nights[rng.integers(len(nights))]
        T = nt["x"].shape[1]; a, b = nt["sleep_span"]
        lo = max(0, int(a * FS) - 60 * FS); hi = min(T - win, int(b * FS) + 60 * FS - win)
        s = int(rng.integers(lo, max(lo + 1, hi)))
        ch = [int(rng.choice(nt["eeg_idx"], p=nt["eeg_w"])), int(rng.choice(nt["eog_idx"])), nt["emg_idx"]]
        if with_flow:
            ch.append(int(rng.choice(nt["flow_idx"])))
        x = nt["x"][ch, s:s + win].astype(np.float32)
        x *= rng.uniform(0.8, 1.3, size=(n_in, 1)).astype(np.float32)
        xs[i] = x; ys[i] = nt["y"][s:s + win]
    return torch.from_numpy(xs), torch.from_numpy(ys)


@torch.no_grad()
def predict_night(model, x_np, dev):
    x = torch.from_numpy(x_np.astype(np.float32))[None].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logit = model(x)
    return torch.sigmoid(logit.float())[0].cpu().numpy()


def validate(model, nights, dev, n_in):
    """Gepoolde event-F1 (IoU 0,20, slaappoort) per τ op de validatienachten; AP per nacht."""
    model.eval()
    agg = {t: [0, 0, 0] for t in THRS}; aps = []
    for nt in nights:
        p = predict_night(model, nt["x"][nt["val_ch"][:n_in]], dev)
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
                           "ppv": round(tp / (tp + fp), 4) if tp + fp else None,
                           "count_ratio": round((tp + fp) / (tp + fn), 4) if tp + fn else None}
    best = max(out, key=lambda k: out[k]["f1"])
    neutraal = min(out, key=lambda k: abs((out[k]["count_ratio"] or 9) - 1))
    return {"ap_mean": float(np.mean(aps)) if aps else None, "per_thr": out,
            "best_thr": float(best), "best_f1": out[best]["f1"], "neutral_thr": float(neutraal)}


def train_arm(name, train_n, val_n, with_flow, epochs, a, out_dir: Path, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "KLAAR").exists():
        log(f"arm {name}: KLAAR bestaat al, overgeslagen"); return
    n_in = 4 if with_flow else 3
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = UNet1D(in_ch=n_in).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    total = epochs * a.steps; warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    win = a.win_s * FS
    frac = float(np.mean([nt["y"].mean() for nt in train_n]))
    cohorten = {c: sum(1 for nt in train_n if nt["cohort"] == c) for c in ("mesa", "shhs")}
    log(f"arm {name}: train {len(train_n)} {cohorten} val {len(val_n)} in_ch {n_in} epochs ≤ {epochs} "
        f"arousal-fractie {frac:.4f}")
    lg = {"arm": name, "args": vars(a), "seed": seed, "n_train": len(train_n), "train_cohorten": cohorten,
          "n_val": len(val_n), "in_ch": n_in, "epochs_max": epochs, "arousal_fraction_train": frac,
          "n_params": sum(p.numel() for p in model.parameters()), "epochs": []}
    best_f1, best_ep, bad = -1.0, -1, 0
    for ep in range(epochs):
        model.train(); te = time.time(); losses = []
        for _ in range(a.steps):
            x, y = sample_batch(train_n, a.batch, win, rng, with_flow)
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); losses.append(loss.item())
        tv = time.time(); val = validate(model, val_n, dev, n_in)
        rec = {"epoch": ep, "loss": float(np.mean(losses)), "train_s": round(tv - te), "val_s": round(time.time() - tv),
               "lr": sched.get_last_lr()[0], **val}
        lg["epochs"].append(rec)
        log(f"arm {name} ep {ep:2d} loss {rec['loss']:.4f} val AP {val['ap_mean']:.3f} best thr {val['best_thr']:.2f} "
            f"F1 {val['best_f1']:.3f} neutraal {val['neutral_thr']:.2f} ({rec['train_s']} s + {rec['val_s']} s)")
        if val["best_f1"] > best_f1:
            best_f1, best_ep, bad = val["best_f1"], ep, 0
            torch.save({"state_dict": model.state_dict(), "thr": val["best_thr"], "neutral_thr": val["neutral_thr"],
                        "epoch": ep, "val": val, "arm": name,
                        "config": {"in_ch": n_in, "base": 16, "pools": [2, 4, 4, 4], "k": 21}},
                       out_dir / "model_best.pt")
        else:
            bad += 1
        lg["best"] = {"epoch": best_ep, "f1": best_f1}
        (out_dir / "train_log.json").write_text(json.dumps(lg, indent=1, default=str))
        if bad >= a.patience:
            log(f"arm {name}: vroegtijdig gestopt"); break
    ck = torch.load(out_dir / "model_best.pt", map_location="cpu", weights_only=False)
    (out_dir / "tau.json").write_text(json.dumps({"arm": name, "tau_val": ck["thr"], "tau_neutral": ck["neutral_thr"],
                                                  "epoch": ck["epoch"], "val_f1": ck["val"]["best_f1"],
                                                  "per_thr": ck["val"]["per_thr"]}, indent=1))
    (out_dir / "KLAAR").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} beste epoch {best_ep} val-F1 {best_f1:.4f} "
                                   f"sha256 {sha256(out_dir / 'model_best.pt')}\n")
    log(f"arm {name}: klaar, beste epoch {best_ep} val-F1 {best_f1:.3f}")


# ── inferentie ────────────────────────────────────────────────────────────────────────────
def load_model(path: Path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = UNet1D(**ck["config"]).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck


def jsonl_append(path: Path, rij: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rij, default=str) + "\n")


def infer(models: dict, nights: list, cohort: str, out: Path) -> None:
    d = out / "out" / cohort; d.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logp = d / "infer_log.jsonl"
    for nt in nights:
        rec = nt["rec"]
        if not (d / f"{rec}_ref.csv").exists():
            write_pred(d / f"{rec}_ref.csv", nt["arousals"])
        if not (d / f"{rec}_hypno.json").exists():
            tst_h = sum(1 for s in nt["hypno"] if s in ("N1", "N2", "N3", "R")) * 30.0 / 3600.0
            (d / f"{rec}_hypno.json").write_text(json.dumps({"hypno": nt["hypno"], "dur_s": nt["dur"], "tst_h": round(tst_h, 4)}))
        for name, (model, n_in, tau) in models.items():
            if (d / f"{rec}_{name}_grid.json").exists():
                continue
            if len(nt["val_ch"]) < n_in:
                jsonl_append(logp, {"rec": rec, "model": name, "error": "geen flowkanaal voor een 4-kanaalsmodel"})
                continue
            t0 = time.time()
            ch = nt["val_ch"][:n_in]
            p = predict_night(model, nt["x"][ch], dev)
            t_fwd = time.time() - t0
            k = FS // 10
            np.savez_compressed(d / f"{rec}_{name}_prob10hz.npz",
                                p=p[: (len(p) // k) * k].reshape(-1, k).mean(1).astype(np.float16), fs=10)
            grid = {}
            for t in sorted(set(THRS + [tau])):
                grid[f"{t:.2f}"] = [[round(a_, 3), round(b_, 3)] for a_, b_ in gate_sleep(prob_to_events(p, FS, t), nt["hypno"])]
            write_pred(d / f"{rec}_{name}_pred.csv", [tuple(e) for e in grid[f"{tau:.2f}"]])
            (d / f"{rec}_{name}_grid.json").write_text(json.dumps({"tau": tau, "events": grid}))
            jsonl_append(logp, {"rec": rec, "cohort": cohort, "model": name, "tau": tau,
                                "channels": [nt["names"][i] for i in ch],
                                "flow_kind": (nt["flow_pref_kind"] if n_in == 4 else None),
                                "n_ref": len(nt["arousals"]), "n_pred": len(grid[f"{tau:.2f}"]),
                                "t_forward_s": round(t_fwd, 2)})
        log(f"  {cohort} {rec}: {len(nt['arousals'])} ref, " +
            ", ".join(f"{n} {len(json.loads((d / f'{rec}_{n}_grid.json').read_text())['events'][f'{m[2]:.2f}'])}"
                      for n, m in models.items() if (d / f"{rec}_{n}_grid.json").exists()))


# ── keten ─────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--win-s", type=int, default=600); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="rookproef: enkele nachten, 1 epoch van 3 stappen")
    a = ap.parse_args()
    out: Path = a.out; out.mkdir(parents=True, exist_ok=True)
    try:
        git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        git = "?"
    log(f"start; preregistratie sha256 {sha256(PREREG)}; git {git}; torch {torch.__version__}; "
        f"cuda {torch.cuda.is_available()}; args {vars(a)}")
    ids = {k: read_ids(HIER / f"ids_{k}.txt") for k in ("mesa_train", "mesa_val", "shhs_train", "shhs_val", "shhs_test", "mesa76_test")}
    sns = list(SNS)
    arms = list(ARMS)
    if a.smoke:
        ids = {k: v[: {"mesa_train": 3, "mesa_val": 2, "shhs_train": 3, "shhs_val": 2, "shhs_test": 2, "mesa76_test": 2}[k]] for k, v in ids.items()}
        sns = sns[:1]; a.steps = 3
        arms = [(n, c, f, 1) for n, c, f, _ in arms]
    log("sets: " + ", ".join(f"{k} {len(v)}" for k, v in ids.items()) + f", psgipa {len(sns)}")

    specs = ([("mesa", r, True) for r in ids["mesa_train"] + ids["mesa_val"]]
             + [("shhs", r, True) for r in ids["shhs_train"] + ids["shhs_val"]])
    nights, errors = load_set(specs, a.workers, "train+val")
    (out / "laadfouten.json").write_text(json.dumps(errors, indent=1, default=str))
    def pick(cohort, lst):
        return [nights[(cohort, r)] for r in lst if (cohort, r) in nights]
    mesa_tr, mesa_va, shhs_tr, shhs_va = (pick("mesa", ids["mesa_train"]), pick("mesa", ids["mesa_val"]),
                                          pick("shhs", ids["shhs_train"]), pick("shhs", ids["shhs_val"]))
    val_n = mesa_va + shhs_va
    log(f"geladen: mesa train {len(mesa_tr)} val {len(mesa_va)}, shhs train {len(shhs_tr)} val {len(shhs_va)}, "
        f"fouten {len(errors)}, RSS {rss_gb():.1f} GB")
    flow_kinds = {}
    for nt in mesa_tr + shhs_tr:
        for k in nt["flow_kinds"]:
            flow_kinds[(nt["cohort"], k)] = flow_kinds.get((nt["cohort"], k), 0) + 1
    log(f"flowkanalen in de training (cohort, soort) -> nachten: { {f'{c}/{k}': v for (c, k), v in flow_kinds.items()} }")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bevroren model: τ op dezelfde validatieset van 80
    fz_dir = out / "frozen"; fz_dir.mkdir(exist_ok=True)
    if not (fz_dir / "tau.json").exists():
        sha = sha256(FROZEN)
        if sha != FROZEN_SHA:
            raise SystemExit(f"bevroren model sha256 {sha} != {FROZEN_SHA}")
        fz, ck = load_model(FROZEN, dev)
        val = validate(fz, val_n, dev, 3)
        (fz_dir / "tau.json").write_text(json.dumps({"arm": "frozen", "sha256": sha, "tau_val": val["best_thr"],
                                                     "tau_neutral": val["neutral_thr"], "val_f1": val["best_f1"],
                                                     "ck_thr": ck.get("thr"), "lib_tau": LIB_TAU, "per_thr": val["per_thr"]}, indent=1))
        log(f"bevroren model op val-80: F1 {val['best_f1']:.3f} op τ {val['best_thr']:.2f}, neutraal {val['neutral_thr']:.2f} "
            f"(ck-thr {ck.get('thr')})")
        del fz

    for name, cohorten, with_flow, epochs in arms:
        train_n = (mesa_tr if "mesa" in cohorten else []) + (shhs_tr if "shhs" in cohorten else [])
        train_arm(name, train_n, val_n, with_flow, epochs, a, out / name, a.seed)

    del nights, mesa_tr, shhs_tr, mesa_va, shhs_va, val_n; gc.collect()
    log(f"training klaar; geheugen vrijgegeven, RSS-piek {rss_gb():.1f} GB")

    models = {}
    fz, _ = load_model(FROZEN, dev)
    models["frozen"] = (fz, 3, float(json.loads((fz_dir / "tau.json").read_text())["tau_val"]))
    for name, _, with_flow, _ in arms:
        m, _ck = load_model(out / name / "model_best.pt", dev)
        models[name] = (m, 4 if with_flow else 3, float(json.loads((out / name / "tau.json").read_text())["tau_val"]))
    log("inferentie met τ per model: " + ", ".join(f"{n} {m[2]:.2f}" for n, m in models.items()))
    for cohort, specs in (("shhs150b", [("shhs", r, True) for r in ids["shhs_test"]]),
                          ("mesa76", [("mesa", r, True) for r in ids["mesa76_test"]]),
                          ("psgipa", [("psgipa", s, True) for s in sns])):
        tn, terr = load_set(specs, min(a.workers, max(1, len(specs))), cohort)
        (out / "out" / cohort).mkdir(parents=True, exist_ok=True)
        (out / "out" / cohort / "laadfouten.json").write_text(json.dumps(terr, indent=1, default=str))
        orde = [r for _, r, _ in specs]
        infer(models, [tn[k] for r in orde for k in tn if k[1] == r], cohort, out)
        del tn; gc.collect()
    (out / "ALLES_KLAAR").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("ALLES KLAAR")


if __name__ == "__main__":
    main()
