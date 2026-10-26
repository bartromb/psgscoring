#!/usr/bin/env python3
"""Trekt de sets van de schaalmeting (seed 20261025) en schrijft de registerregels.

    bench/eeg/_venv/bin/python bench/eeg/unet_scale/select_ids.py [--dry-run]

Pools exact als in docs/arousal_unet_schaal_preregistratie_20261025.md §Nachten. Idempotent:
bestaande id-lijsten worden niet overschreven en een registerkop die er al staat wordt niet
opnieuw geschreven. De enige schrijfacties onder /srv/DATA zijn de twee registerbestanden.
"""
from __future__ import annotations
import argparse, random, re, sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
SEED = 20261025
MESA_REG = Path("/srv/DATA/MESA/gebruikte_mesa_ids.txt")
SHHS_REG = Path("/srv/DATA/SHHS/gebruikte_shhs_ids.txt")
MESA_EDF = Path("/srv/DATA/MESA/mesa/polysomnography/edfs")
MESA_XML = Path("/srv/DATA/MESA/mesa/polysomnography/annotations-events-nsrr")
SHHS_EDF = Path("/srv/DATA/SHHS/shhs/polysomnography/edfs/shhs1")
SHHS_XML = Path("/srv/DATA/SHHS/shhs/polysomnography/annotations-events-nsrr/shhs1")
RESP_VAL = HIER.parents[1] / "resp" / "ids_val.txt"
MESA76 = Path("/srv/CODE/docs/arousal_unet_20260927/ids_mesa76.txt")
N_MESA_VAL, N_SHHS_TEST, N_SHHS_VAL, N_SHHS_TRAIN = 40, 150, 40, 1278
TAG = "U-Net-schaal arousal 2026-10-25"


def sections(reg: Path, prefix: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}; cur = "(geen kop)"
    for l in reg.read_text().splitlines():
        l = l.strip()
        if l.startswith("## "):
            cur = l; continue
        if l.startswith(prefix):
            out.setdefault(cur, set()).add(l.split()[0])
    return out


def mesa_pool() -> tuple[list[str], dict[str, int]]:
    secs = sections(MESA_REG, "mesa-sleep")
    def sec(pat):
        return set().union(set(), *[v for k, v in secs.items() if re.search(pat, k)])
    edfs = {p.stem for p in MESA_EDF.glob("mesa-sleep-*.edf")}
    xmls = {p.stem.replace("-nsrr", "") for p in MESA_XML.glob("mesa-sleep-*-nsrr.xml")}
    excl = {"n150": sec(r"orakel-decompositie"), "resp_val100": set(l.strip() for l in RESP_VAL.read_text().splitlines() if l.strip()),
            "dsp_eeg400": sec(r"dsp-scout eeg"), "unet76": sec(r"U-Net-replicatie"), "kand40": sec(r"kandidaatpoort")}
    E = set().union(*excl.values())
    pool = sorted((edfs & xmls) - E)
    m76 = {l.strip() for l in MESA76.read_text().splitlines() if l.strip()}
    assert not (set(pool) & m76), "MESA-76 mag niet in de pool"
    return pool, {k: len(v) for k, v in excl.items()} | {"uitsluiting_uniek": len(E), "pool": len(pool)}


def shhs_pool() -> tuple[list[str], dict[str, int]]:
    used = {l.strip() for l in SHHS_REG.read_text().splitlines() if l.strip().startswith("shhs1")}
    edfs = {p.stem for p in SHHS_EDF.glob("shhs1-*.edf")}
    xmls = {p.stem.replace("-nsrr", "") for p in SHHS_XML.glob("shhs1-*-nsrr.xml")}
    pool = sorted((edfs & xmls) - used)
    return pool, {"gebruikt": len(used), "pool": len(pool)}


def write_ids(name: str, ids: list[str], dry: bool) -> None:
    p = HIER / name
    if p.exists():
        bestaand = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        assert bestaand == ids, f"{name} bestaat al met een andere inhoud"
        return
    if not dry:
        p.write_text("\n".join(ids) + "\n")


def register(reg: Path, kop: str, ids: list[str], dry: bool) -> bool:
    if kop in reg.read_text():
        return False
    if not dry:
        with reg.open("a") as fh:
            fh.write(f"{kop} ({len(ids)})\n" + "\n".join(ids) + "\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    mp, minfo = mesa_pool(); sp, sinfo = shhs_pool()
    print("MESA", minfo); print("SHHS1", sinfo)
    rng = random.Random(SEED); mp = list(mp); rng.shuffle(mp)
    rng = random.Random(SEED); sp = list(sp); rng.shuffle(sp)
    mesa_val, mesa_train = mp[:N_MESA_VAL], mp[N_MESA_VAL:]
    shhs_test = sp[:N_SHHS_TEST]; shhs_val = sp[N_SHHS_TEST:N_SHHS_TEST + N_SHHS_VAL]
    shhs_train = sp[N_SHHS_TEST + N_SHHS_VAL:N_SHHS_TEST + N_SHHS_VAL + N_SHHS_TRAIN]
    assert len(set(mesa_val + mesa_train)) == len(mp) and len(set(shhs_test + shhs_val + shhs_train)) == N_SHHS_TEST + N_SHHS_VAL + N_SHHS_TRAIN
    for name, ids in [("ids_mesa_val.txt", mesa_val), ("ids_mesa_train.txt", mesa_train),
                      ("ids_shhs_test.txt", shhs_test), ("ids_shhs_val.txt", shhs_val), ("ids_shhs_train.txt", shhs_train)]:
        write_ids(name, ids, a.dry_run); print(name, len(ids))
    write_ids("ids_mesa76_test.txt", [l.strip() for l in MESA76.read_text().splitlines() if l.strip()], a.dry_run)
    kant = ("pool = alle nachten buiten n150, resp-val-100, dsp-scout-eeg-400, U-Net-76 en kandidaatpoort-40; "
            "de 474 arousalsets zijn in dit register niet van de respiratoire secties te scheiden en zitten dus in de pool; "
            "deze nachten zijn voortaan AROUSAL-GETRAIND")
    r1 = register(MESA_REG, f"## {TAG} — validatie 40, seed {SEED}; {kant}", mesa_val, a.dry_run)
    r2 = register(MESA_REG, f"## {TAG} — training 1278, seed {SEED}; {kant}", mesa_train, a.dry_run)
    r3 = register(SHHS_REG, f"## {TAG} — test SHHS1-150b, 150 verse nachten, seed {SEED}, 458 eerdere uitgesloten", shhs_test, a.dry_run)
    r4 = register(SHHS_REG, f"## {TAG} — validatie 40, vers, seed {SEED}", shhs_val, a.dry_run)
    r5 = register(SHHS_REG, f"## {TAG} — training 1278, vers, seed {SEED}", shhs_train, a.dry_run)
    print("register geschreven:", {"mesa_val": r1, "mesa_train": r2, "shhs_test": r3, "shhs_val": r4, "shhs_train": r5},
          "(dry-run)" if a.dry_run else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
