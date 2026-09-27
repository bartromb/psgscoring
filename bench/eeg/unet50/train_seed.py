#!/usr/bin/env python3
"""Bewaker (a) van docs/arousal_unet_preregistratie_20260927.md: hertraining met een ANDERE seed op
DEZELFDE 400 MESA-ids (320 train / 80 val, in de volgorde van ids_train.txt + ids_val.txt), uitvoer
in een aparte map zodat model_best.pt van de bevroren run onaangeroerd blijft; het register wordt
niet bijgeschreven (de ids staan er al).

    bench/eeg/_venv/bin/python bench/eeg/unet50/train_seed.py --seed 20260928 --out <map> [train.py-opties]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER)); sys.path.insert(0, str(HIER.parent / "common"))
import train  # noqa: E402

IDS = [l.strip() for f in ("ids_train.txt", "ids_val.txt")
       for l in (HIER / f).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a, rest = ap.parse_known_args()
    assert len(IDS) == 400, len(IDS)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train.HIER = out
    train.select_ids = lambda n, seed: IDS[:n]
    train.register = lambda ids: False
    sys.argv = [sys.argv[0], "--seed", str(a.seed)] + rest
    train.main()


if __name__ == "__main__":
    main()
