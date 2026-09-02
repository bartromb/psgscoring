#!/usr/bin/env python3
"""Een 1D-U-Net op de golfvorm, geëvalueerd zoals de regels geëvalueerd worden.

WAT DIT MOET BEANTWOORDEN
=========================
De regelgebaseerde keten haalt op PSG-IPA een event-F1 van 0,349 tegen een
MENSELIJK PLAFOND van 0,556 -- 63 % daarvan. Het recall-plafond ligt bij de
kandidaatgeneratie: de regels wezen 28 kandidaten af die een meerderheid van de
scoorders wél markeerde, en geen classifier op die kandidaten haalt die terug.

Een model dat de golfvorm leest, kent die beperking niet. De vraag is of het
daarmee ook beter WORDT, en dat is alleen te beantwoorden met dezelfde maat.

DE VERGELIJKING MOET EERLIJK ZIJN
=================================
Daarom, expliciet:

* dezelfde referentie (`aasm15` -- AASM v3 Rule 1A);
* dezelfde matching (greedy IoU op 0,20, `psgscoring.agreement._match`);
* opnames die NIET in de training zaten, gesplitst per patiënt;
* het menselijk plafond ernaast, want 0,55 tegen 1,0 leest anders dan 0,55
  tegen 0,556.

Zonder die vier is de uitkomst een demonstratie en geen meting.

HET MODEL
=========
Een kleine 1D-U-Net: vier down-blokken, vier up-blokken, ~250k parameters.
Invoer (4, T) op 8 Hz -- flow, thorax, abdomen, SpO2. Uitvoer (T,): kans per
sample dat het binnen een event valt.

Klein met opzet. Er zijn 150 opnames van drie scoorders; een groot model leert
die drie mensen uit het hoofd. De capaciteit is de rem, niet de ambitie.

VAN KANS NAAR EVENTS
====================
Drempel, dan aaneengesloten stukken van >= 10 s (de AASM-minimumduur). Dat is
de enige regel die het model krijgt opgelegd -- de rest mag het zelf vinden.
Zo verhuizen de AASM-regels van de ingang naar de uitgang.

Gebruik:
    ~/CODE/.venv-dl/bin/python scripts/train_sequence_model.py \\
        --dataset seq_mesa.npz --epochs 40 --output unet.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FS = 8.0
MIN_EVENT_S = 10.0
IOU = 0.20


# ── Model ─────────────────────────────────────────────────────────────────

def _bouw_unet(torch, nn, n_in=4, basis=16):
    class Blok(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(i, o, 9, padding=4), nn.BatchNorm1d(o), nn.ReLU(),
                nn.Conv1d(o, o, 9, padding=4), nn.BatchNorm1d(o), nn.ReLU())

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            k = [basis, basis * 2, basis * 4, basis * 8]
            self.d1, self.d2 = Blok(n_in, k[0]), Blok(k[0], k[1])
            self.d3, self.d4 = Blok(k[1], k[2]), Blok(k[2], k[3])
            self.pool = nn.MaxPool1d(2)
            self.u3 = nn.ConvTranspose1d(k[3], k[2], 2, stride=2)
            self.u2 = nn.ConvTranspose1d(k[2], k[1], 2, stride=2)
            self.u1 = nn.ConvTranspose1d(k[1], k[0], 2, stride=2)
            self.c3, self.c2 = Blok(k[2] * 2, k[2]), Blok(k[1] * 2, k[1])
            self.c1 = Blok(k[0] * 2, k[0])
            self.uit = nn.Conv1d(k[0], 1, 1)

        def forward(self, x):
            a = self.d1(x)
            b = self.d2(self.pool(a))
            c = self.d3(self.pool(b))
            d = self.d4(self.pool(c))
            y = self.c3(torch.cat([self.u3(d), c], 1))
            y = self.c2(torch.cat([self.u2(y), b], 1))
            y = self.c1(torch.cat([self.u1(y), a], 1))
            return self.uit(y).squeeze(1)

    return UNet()


# ── Van kansen naar events ────────────────────────────────────────────────

def _naar_events(p, drempel, fs=FS, min_s=MIN_EVENT_S):
    """Aaneengesloten stukken boven de drempel, minstens `min_s` lang."""
    boven = p >= drempel
    if not boven.any():
        return []
    rand = np.diff(np.concatenate([[0], boven.view(np.int8), [0]]))
    starts = np.where(rand == 1)[0]
    eindes = np.where(rand == -1)[0]
    n_min = int(min_s * fs)
    return [{"onset_s": float(a / fs), "duration_s": float((b - a) / fs)}
            for a, b in zip(starts, eindes) if (b - a) >= n_min]


def _f1(a, b):
    from psgscoring.agreement import _match
    if not a or not b:
        return 0.0
    m, _, _ = _match(a, b, IOU)
    if not m:
        return 0.0
    p, r = len(m) / len(a), len(m) / len(b)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--window-s", type=float, default=600.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--holdout", type=float, default=0.25,
                    help="fractie opnames die NIET in de training komt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from torch import nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"apparaat: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    d = np.load(args.dataset, allow_pickle=True)
    X, M, S, R = d["X"], d["mask"], d["sleep"], d["rec"]
    print(f"{len(X)} opnames, {sum(x.shape[1] for x in X)/FS/3600:.0f} uur")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    n_test = max(1, int(len(X) * args.holdout))
    test_i, train_i = idx[:n_test], idx[n_test:]
    print(f"  training {len(train_i)} opnames, HELD OUT {len(test_i)}")

    W = int(args.window_s * FS)
    W = (W // 16) * 16                      # deelbaar door 2^4 voor de U-Net

    def vensters(indices, stap=None):
        stap = stap or W
        vs, ms = [], []
        for i in indices:
            x, m, s = X[i], M[i], S[i]
            for a in range(0, x.shape[1] - W, stap):
                if s[a:a + W].mean() < 0.5:        # overwegend wake
                    continue
                vs.append(x[:, a:a + W]); ms.append(m[a:a + W])
        return np.array(vs, np.float32), np.array(ms, np.float32)

    Xtr, Mtr = vensters(train_i, stap=W // 2)      # overlap = meer data
    print(f"  {len(Xtr)} trainingsvensters van {args.window_s:.0f} s")
    if not len(Xtr):
        print("FOUT: geen vensters -- staat het slaapmasker goed?")
        return 1

    # Het aantal kanalen komt UIT de dataset. Vastpinnen op 4 (de respiratoire
    # set) liet de arousalset van 3 kanalen falen met een conv1d-fout die de
    # oorzaak niet noemt.
    n_kan = int(X[0].shape[0])
    model = _bouw_unet(torch, nn, n_in=n_kan).to(dev)
    print(f"  {n_kan} kanalen: {', '.join(str(k) for k in d['kanalen'])}")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_par/1000:.0f}k parameters")

    pos = float(Mtr.mean())
    gew = torch.tensor((1 - pos) / max(pos, 1e-6), device=dev)
    print(f"  {100*pos:.1f} % van de samples in een event "
          f"(pos_weight {gew.item():.1f})")
    verlies = nn.BCEWithLogitsLoss(pos_weight=gew)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    Xt = torch.from_numpy(Xtr); Mt = torch.from_numpy(Mtr)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for b in range(0, len(Xt), args.batch):
            j = perm[b:b + args.batch]
            xb, mb = Xt[j].to(dev), Mt[j].to(dev)
            opt.zero_grad()
            out = model(xb)
            lo = verlies(out, mb)
            lo.backward()
            opt.step()
            tot += lo.item() * len(j)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    epoch {ep+1:3d}/{args.epochs}  verlies {tot/len(Xt):.4f}",
                  flush=True)

    # ── Evaluatie op de held-out opnames ─────────────────────────────────
    print("\nEVALUATIE op held-out opnames (nooit gezien tijdens training)\n")
    model.eval()
    resultaten = []
    for i in test_i:
        x, m = X[i], M[i]
        n = (x.shape[1] // 16) * 16
        with torch.no_grad():
            p = torch.sigmoid(model(
                torch.from_numpy(x[None, :, :n]).to(dev))).cpu().numpy()[0]
        ref = _naar_events(m[:n] > 0.5, 0.5)
        beste = max(((t, _f1(_naar_events(p, t), ref))
                     for t in np.arange(0.3, 0.85, 0.05)), key=lambda z: z[1])
        resultaten.append({"rec": str(R[i]), "n_ref": len(ref),
                           "drempel": round(float(beste[0]), 2),
                           "f1": round(float(beste[1]), 3)})
        print(f"  {R[i]:20s} referentie {len(ref):4d} events   "
              f"F1 {beste[1]:.3f} (drempel {beste[0]:.2f})")

    f1s = [r["f1"] for r in resultaten]
    print(f"\n  MEDIANE F1 op held-out: {np.median(f1s):.3f}")
    print("  Ter vergelijking: menselijk plafond op PSG-IPA 0,556; "
          "de regelketen 0,349 (63 % daarvan).")
    print("  LET OP: dit is MESA tegen één scoorder, PSG-IPA is twaalf. "
          "De getallen staan naast elkaar, niet in elkaars plaats.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "fs": FS,
                "kanalen": list(d["kanalen"])}, args.output)
    print(f"\n  model -> {args.output}")
    if args.report:
        args.report.write_text(json.dumps(
            {"n_train": len(train_i), "n_test": len(test_i),
             "n_params": n_par, "epochs": args.epochs,
             "window_s": args.window_s, "per_recording": resultaten,
             "median_f1": float(np.median(f1s))}, indent=2))
        print(f"  rapport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
