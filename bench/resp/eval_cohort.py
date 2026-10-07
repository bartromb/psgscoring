#!/usr/bin/env python3
"""Evaluatie van het bevroren respiratoire U-Net op een cohort (preregistratie
docs/resp_unet_preregistratie_20261007.md): per nacht events (CSV), F1/precisie/recall
(IoU 0,20, typeonbewust én typebewust), count-ratio, AHI tegen NSRR-AHI, per type.

    ../eeg/_venv/bin/python eval_cohort.py --cohort shhs1 --n 150 --seed 20261007
    ../eeg/_venv/bin/python eval_cohort.py --cohort mesa_val
    ../eeg/_venv/bin/python eval_cohort.py --cohort psgipa
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER)); sys.path.insert(0, str(HIER.parent)); sys.path.insert(0, str(HIER.parents[1]))
from data import load_mesa_night, load_shhs_night, load_psgipa_signals, FS, SHHS  # noqa: E402
from model import UNet1D  # noqa: E402
from postproc import probs_to_events, gate_sleep  # noqa: E402
from evaluate import match  # noqa: E402

MODEL = HIER / "model_best.pt"
SHHS_REG = Path("/srv/DATA/SHHS/gebruikte_shhs_ids.txt")
OUT = HIER / "out"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shhs1_fresh(n: int, seed: int) -> list[str]:
    ids = sorted(p.stem for p in (SHHS / "edfs" / "shhs1").glob("shhs1-*.edf")
                 if (SHHS / "annotations-events-nsrr" / "shhs1" / f"{p.stem}-nsrr.xml").exists())
    used = {l.strip() for l in SHHS_REG.read_text().splitlines() if l.strip().startswith("shhs")}
    pool = [x for x in ids if x not in used]
    return sorted(random.Random(seed).sample(pool, n))


def register_shhs(ids: list[str], header: str) -> bool:
    txt = SHHS_REG.read_text()
    if header in txt:
        return False
    with SHHS_REG.open("a") as fh:
        fh.write(f"{header} ({len(ids)})\n" + "\n".join(ids) + "\n")
    return True


@torch.no_grad()
def predict(model, x, dev, zero=()):
    x = x.astype(np.float32).copy()
    for i in zero:
        x[i] = 0.0
    t = torch.from_numpy(x)[None].to(dev)
    return torch.sigmoid(model(t).float())[0].cpu().numpy()


def schrijf(path: Path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["onset_s", "offset_s", "type"])
        for a, b, t in sorted(events):
            w.writerow([f"{a:.3f}", f"{b:.3f}", t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["shhs1", "mesa_val", "psgipa"])
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--thr", type=float, default=None); ap.add_argument("--zero", default="",
                    help="ablatie: kanaalindexen op nul, bv. '0' (druk), '1' (thermistor), '4' (spo2), '2,3' (effort)")
    ap.add_argument("--tag", default=""); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    if a.cpu:
        torch.set_num_threads(a.threads)
    ck = torch.load(MODEL, map_location=dev, weights_only=False)
    model = UNet1D(**ck["config"]).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    thr = float(a.thr) if a.thr is not None else float(ck["thr"])
    zero = tuple(int(z) for z in a.zero.split(",") if z.strip())
    out = OUT / a.cohort; out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or (f"_zero{a.zero.replace(',', '')}" if zero else "")
    log = out / f"log{tag}.jsonl"
    with log.open("a") as fh:
        fh.write(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "model_sha256": sha256(MODEL), "thr": thr,
                             "ck_thr": ck.get("thr"), "ck_epoch": ck.get("epoch"), "zero": zero, "device": str(dev)}) + "\n")
    if a.cohort == "shhs1":
        ids = shhs1_fresh(a.n, a.seed)
        (out / "ids.txt").write_text("\n".join(ids) + "\n")
        print("register:", register_shhs(ids, "## resp-U-Net replicatie 2026-10-07 — 150 verse shhs1-nachten, seed 20261007"), flush=True)
        loader = load_shhs_night
    elif a.cohort == "mesa_val":
        ids = (HIER / "ids_val.txt").read_text().split(); loader = load_mesa_night
    else:
        ids = ["SN1", "SN2", "SN3", "SN4", "SN5"]; loader = None
    rows = []
    for rec in ids:
        t0 = time.time()
        if a.cohort == "psgipa":
            d = load_psgipa_signals(rec)
            hyp = json.load(open(f"/srv/CODE/docs/arousal_unet_20260927/out/psgipa/{rec}_hypno.json"))
            d["hypno"] = hyp["hypno"] if isinstance(hyp, dict) and "hypno" in hyp else hyp
            d["events"] = None; d["tst_h"] = sum(1 for s in d["hypno"] if s in ("N1", "N2", "N3", "R")) * 30 / 3600
        else:
            d = loader(rec)
        if d.get("error"):
            rows.append({"rec": rec, "error": d["error"]}); print(rec, "FOUT", d["error"][:100], flush=True); continue
        t1 = time.time(); p = predict(model, d["x"], dev, zero); t_fwd = time.time() - t1
        ev = gate_sleep(probs_to_events(p[0], p[1], FS, thr), d["hypno"])
        schrijf(out / f"{rec}{tag}.csv", ev)
        rij = {"rec": rec, "n_pred": len(ev), "n_apnea_pred": sum(1 for e in ev if e[2] == "apnea"), "tst_h": d["tst_h"],
               "ahi_pred": len(ev) / d["tst_h"] if d["tst_h"] else None, "mask": d["mask"].tolist(), "t_fwd_s": round(t_fwd, 2),
               "t_total_s": round(time.time() - t0, 1), "channels": d["channels"]}
        if d.get("events") is not None:
            ref = [(x, y, t) for x, y, t in d["events"]]
            tp, fp, fn, _ = match([(x, y, None) for x, y, _ in ev], [(x, y, None) for x, y, _ in ref], "iou", 0.20)
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
            ref_t = [(x, y, "apnea" if "hypopnea" not in t.lower() else "hypopnea") for x, y, t in ref]
            tpt, fpt, fnt, _ = match(ev, ref_t, "iou", 0.20, type_aware=True)
            f1t = 2 * tpt / (2 * tpt + fpt + fnt) if (2 * tpt + fpt + fnt) else None
            rij.update(n_ref=len(ref), n_apnea_ref=sum(1 for e in ref_t if e[2] == "apnea"), ahi_ref=len(ref) / d["tst_h"],
                       tp=tp, fp=fp, fn=fn, f1=f1, precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
                       f1_typebewust=f1t, count_ratio=len(ev) / len(ref) if ref else None)
            schrijf(out / f"{rec}_ref.csv", ref_t)
        json.dump(d["hypno"], open(out / f"{rec}_hypno.json", "w"))
        rows.append(rij)
        with log.open("a") as fh:
            fh.write(json.dumps(rij, default=str) + "\n")
        print(rec, {k: rij.get(k) for k in ("n_pred", "n_ref", "f1", "ahi_pred", "ahi_ref", "t_fwd_s")}, flush=True)
    ok = [r for r in rows if "f1" in r and r["f1"] is not None]
    if ok:
        f1 = [r["f1"] for r in ok]; cr = [r["count_ratio"] for r in ok]; bias = [r["ahi_pred"] - r["ahi_ref"] for r in ok]
        print(f"\n{a.cohort}{tag}: n={len(ok)} F1 mediaan {np.median(f1):.3f} (p25 {np.percentile(f1,25):.3f}) gepoold "
              f"{2*sum(r['tp'] for r in ok)/(2*sum(r['tp'] for r in ok)+sum(r['fp'] for r in ok)+sum(r['fn'] for r in ok)):.3f}; "
              f"count-ratio mediaan {np.median(cr):.2f}; AHI-bias gemiddeld {np.mean(bias):+.2f}", flush=True)
    json.dump(rows, open(out / f"rows{tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
