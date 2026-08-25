#!/usr/bin/env python3
"""
sweep_arousal_threshold_psgipa.py — bestaat er één werkpunt dat ons binnen de
menselijke spreiding brengt?

WAAROM PSG-IPA EN NIET MESA
---------------------------
MESA heeft één scoorder. PSG-IPA heeft er twaalf per opname, en die verschillen
onderling een factor 2 tot 4,5 in arousaltelling (SN2: 21-94; SN3: 101-234).
Tegen één scoorder meet je jezelf af aan een puntschatting die zelf ergens in
die spreiding ligt; tegen twaalf meet je of je binnen het menselijke bereik
valt. Dat laatste is de vraag die telt voor een klinische index.

De prijs is n = 5: een CORRELATIE tussen arousallast en count-ratio kan dit
cohort niet dragen. Daarvoor blijft MESA nodig. Dit script beantwoordt de
andere helft: het NIVEAU, tegen een referentie die zijn eigen onzekerheid
meelevert.

UITKOMSTMAAT (vooraf)
---------------------
Per werkpunt: op hoeveel van de vijf opnames ligt onze telling binnen
[min, max] van de twaalf scoorders. Een werkpunt dat op 5/5 komt maakt een
vaste drempel houdbaar; verschuift het benodigde werkpunt systematisch met de
arousallast, dan is een vaste drempel dat per definitie niet.

Gebruik:
    python scripts/sweep_arousal_threshold_psgipa.py --data-dir ~/PSG-IPA
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mne.set_log_level("ERROR")

from psgscoring.arousal import detect_arousals, detect_arousals_multi
from psgscoring.pipeline import _pick_eeg_multi

RECS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
EPOCH_S = 30.0
SLEEP = {"N1", "N2", "N3", "R"}


class StageParseError(RuntimeError):
    """Zie validate_plm_psgipa.py: een parser die niets herkent hoort te
    stoppen, niet door te tellen."""


def _stage(desc: str) -> str | None:
    d = desc.strip().lower()
    if "sleep stage" not in d:
        return None
    tail = d.split("sleep stage")[-1].strip()
    if not tail:
        raise StageParseError(f"lege stadiumtekst: {desc!r}")
    if tail[0] == "w":
        return "W"
    if tail[0] == "r":
        return "R"
    cijfers = [c for c in tail if c.isdigit()]
    if not cijfers:
        raise StageParseError(f"onbekend stadium: {desc!r}")
    n = cijfers[0]
    return {"1": "N1", "2": "N2", "3": "N3", "4": "N3"}[n]


def parse_scorer(txt_path: Path, dur_s: float):
    """(hypno, arousal_onsets) van één scoorder."""
    n_ep = int(np.ceil(dur_s / EPOCH_S))
    hypno = ["W"] * n_ep
    onsets = []
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 5:
                continue
            try:
                onset, dur = float(p[2]), float(p[3])
            except ValueError:
                continue
            if not (0 <= onset < dur_s):
                continue
            st = _stage(p[4])
            if st is not None:
                ep0 = int(onset // EPOCH_S)
                for i in range(max(1, round(dur / EPOCH_S))):
                    if 0 <= ep0 + i < n_ep:
                        hypno[ep0 + i] = st
            elif "arousal" in p[4].lower():
                onsets.append(onset)
    tst_h = sum(1 for s in hypno if s in SLEEP) * EPOCH_S / 3600.0
    if tst_h < 0.5:
        raise StageParseError(
            f"{txt_path.name}: {tst_h:.2f} u slaap — parser leest dit niet")
    return hypno, onsets, tst_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path.home() / "PSG-IPA"))
    # Geen VASTE kanaalnaam: SN4 draagt "EEG Cz-M1" waar de rest "EEG C4-M1"
    # heeft, en daar brak een hard gecodeerde naam op. De centrale afleiding
    # wordt per opname gekozen en meegerapporteerd, zodat achteraf leesbaar is
    # waarop gemeten is.
    ap.add_argument("--eeg-prefer", default="C4,Cz,C3,CZ")
    ap.add_argument("--emg", default="EMG chin")
    ap.add_argument(
        "--multi", action="store_true",
        help="Draai de PRODUCTIECONFIGURATIE: de afleidingsset zoals "
             "_pick_eeg_multi hem kiest, door detect_arousals_multi. De "
             "single-arm meet een configuratie die de klinische profielen "
             "niet draaien.")
    ap.add_argument("--sweep", default="0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.data_dir) / "EEG_arousals"
    drempels = [float(x) for x in a.sweep.split(",")]

    resultaat = {}
    for sn in RECS:
        psg = root / "PSG" / f"{sn}_EEGarousals.edf"
        if not psg.exists():
            continue
        hdr = mne.io.read_raw_edf(psg, preload=False, verbose=False)
        dur = hdr.n_times / hdr.info["sfreq"]
        scs = sorted((root / "Annotations" / "manual")
                     .glob(f"{sn}_EEGarousals_manual_scorer*.txt"))
        hypnos, tellingen, tsts = [], [], []
        for sc in scs:
            h, ons, tst = parse_scorer(sc, dur)
            hypnos.append(h); tellingen.append(len(ons)); tsts.append(tst)
        if not tellingen:
            continue
        n_ep = min(len(h) for h in hypnos)
        hypno = []
        for i in range(n_ep):
            stem = [h[i] for h in hypnos]
            hypno.append(max(set(stem), key=stem.count))

        if a.multi:
            # Alle EEG-kanalen meeladen; de picker kiest zelf, precies zoals
            # de pijplijn dat doet.
            houden = {c for c in hdr.ch_names if "EEG" in c.upper()} | {a.emg}
        else:
            eeg_naam0 = None
            for voorkeur in a.eeg_prefer.split(","):
                eeg_naam0 = next((c for c in hdr.ch_names
                                  if voorkeur.upper() in c.upper()), None)
                if eeg_naam0:
                    break
            if eeg_naam0 is None:
                raise SystemExit(f"{sn}: geen centrale afleiding in "
                                 f"{hdr.ch_names}")
            houden = {eeg_naam0, a.emg}
        raw = mne.io.read_raw_edf(
            psg, exclude=[c for c in hdr.ch_names if c not in houden],
            preload=True, verbose=False)
        sf = raw.info["sfreq"]
        emg = raw.get_data(picks=[a.emg])[0]
        derivs = None
        if a.multi:
            derivs = _pick_eeg_multi(raw, {})
            if not derivs:
                raise SystemExit(f"{sn}: geen EEG-afleiding gevonden")
            eeg_naam = " u ".join(n for n, _d, _s in derivs)
            eeg = derivs[0][1]
        else:
            eeg_naam = eeg_naam0
            eeg = raw.get_data(picks=[eeg_naam0])[0]

        rij = {"mens": sorted(tellingen), "mens_med": median(tellingen),
               "tst_h": round(median(tsts), 2), "eeg": eeg_naam, "armen": {}}
        def _n(lgbm, thr, _d=derivs, _e=eeg, _m=emg, _h=hypno, _s=sf):
            if _d is not None:
                r = detect_arousals_multi(_d, _s, _h, emg_data=_m,
                                          lgbm=lgbm, lgbm_threshold=thr)
            else:
                r = detect_arousals(_e, _s, _h, emg_data=_m,
                                    lgbm=lgbm, lgbm_threshold=thr)
            return len(r.get("events") or [])

        rij["armen"]["regels"] = _n(False, None)
        for t in drempels:
            rij["armen"][f"{t:.2f}"] = _n(True, t)
        resultaat[sn] = rij
        lo, hi = min(tellingen), max(tellingen)
        binnen = [k for k, v in rij["armen"].items() if lo <= v <= hi]
        print(f"{sn} ({eeg_naam}): mens {median(tellingen):.0f} [{lo}-{hi}] AI "
              f"{median(tellingen)/rij['tst_h']:.1f}/u | "
              + " ".join(f"{k}={v}" for k, v in rij["armen"].items())
              + f" | binnen: {','.join(binnen) or '—'}", flush=True)

    if not resultaat:
        return
    print("\n── op hoeveel van de vijf ligt de telling binnen de "
          "scoordersspreiding ──")
    armen = list(next(iter(resultaat.values()))["armen"])
    for arm in armen:
        n = sum(1 for r in resultaat.values()
                if min(r["mens"]) <= r["armen"][arm] <= max(r["mens"]))
        ratios = [r["armen"][arm] / r["mens_med"] for r in resultaat.values()]
        print(f"   {arm:>7} : {n}/{len(resultaat)}   "
              f"count-ratio mediaan {median(ratios):.2f}")
    print("\nVooraf: een werkpunt dat 5/5 haalt maakt een vaste drempel "
          "houdbaar.\nVerschuift het benodigde werkpunt met de arousallast, "
          "dan niet.")
    if a.out:
        Path(a.out).write_text(json.dumps(resultaat, indent=2, default=float))


if __name__ == "__main__":
    main()
