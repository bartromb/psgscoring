#!/usr/bin/env python3
"""Waarom mist de kandidaatstap de helft van SN3?

Het duur-orakel liet zien dat SN3 als ENIGE opname slechter wordt van een
duurbewuste regel. De verklaring die daarbij hoort: SN3's gemiste arousals
zitten niet IN de pool, dus geen enkele beslisregel kan ze terughalen. Deze
meting test dat rechtstreeks.

Maat: per scoorder de fractie van zijn arousals die door minstens EEN kandidaat
gedekt wordt (IoU >= 0,20); daarvan de mediaan over de twaalf scoorders. Dat is
de BOVENGRENS van wat welk model dan ook op die opname kan halen.

Daarnaast de poolgrootte, want dekking kopen met een grotere pool verplaatst het
probleem naar de selectie -- en dat is precies waar we al vastliepen.

Knoppen: de twee kandidaatdrempels, als expliciete argumenten meegegeven, dus
zonder de module te muteren.
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
from statistics import median
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, mne
mne.set_log_level("ERROR")

# Uitvoermap: PSGSCORING_MEETUITVOER, anders de werkmap. Nooit een
# tijdelijke map -- een harnas in de repo dat naar /tmp schrijft is bij
# de volgende sessie stuk.
_UIT = Path(os.environ.get("PSGSCORING_MEETUITVOER", "."))
sys.path.insert(0, "/home/bart/CODE/psgscoring")
sys.path.insert(0, "/home/bart/CODE/psgscoring/scripts")

ROOT = Path("/home/bart/PSG-IPA/EEG_arousals")
EPOCH_S = 30.0
SNS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
# (ratio, abrupt); (1.20, 1.00) is wat er nu draait.
COMBOS = [(1.20, 1.00), (1.10, 1.00), (1.20, 0.85), (1.10, 0.85), (1.00, 0.70)]


def een(sn):
    from sweep_event_locked_window import _psgipa_spans
    from sweep_arousal_threshold_psgipa import _stage
    from psgscoring.arousal import _union_arousals, detect_arousals
    from psgscoring.pipeline import arousal_derivation_channels
    from psgscoring.agreement import _match

    psg = ROOT / "PSG" / f"{sn}_EEGarousals.edf"
    hdr = mne.io.read_raw_edf(psg, preload=False, verbose=False)
    dur = hdr.n_times / hdr.info["sfreq"]
    scs = sorted((ROOT / "Annotations" / "manual").glob(
        f"{sn}_EEGarousals_manual_scorer*.txt"))
    hyps, spans = [], []
    for sc in scs:
        n_ep = int(np.ceil(dur / EPOCH_S)); h = ["W"] * n_ep
        with open(sc, encoding="utf-8", errors="replace") as f:
            next(f, None)
            for line in f:
                q = [x.strip() for x in line.split(",")]
                if len(q) < 5: continue
                try: o, d = float(q[2]), float(q[3])
                except ValueError: continue
                st = _stage(q[4])
                if st is None or not (0 <= o < dur): continue
                e0 = int(o // EPOCH_S)
                for i in range(max(1, round(d / EPOCH_S))):
                    if 0 <= e0 + i < n_ep: h[e0 + i] = st
        hyps.append(h); spans.append(_psgipa_spans(sc, dur))
    n_ep = min(len(h) for h in hyps)
    hyp = [max(set(x), key=x.count)
           for x in ([h[i] for h in hyps] for i in range(n_ep))]

    wil = arousal_derivation_channels(hdr.ch_names)
    raw = mne.io.read_raw_edf(psg, exclude=[c for c in hdr.ch_names
                              if c not in set(wil) | {"EMG chin"}],
                              preload=True, verbose=False)
    sf = raw.info["sfreq"]; emg = raw.get_data(picks=["EMG chin"])[0]
    sig = {nm: raw.get_data(picks=[nm])[0] for nm in wil}

    uit = {}
    for ratio, abrupt in COMBOS:
        per = []
        for nm in wil:
            r = detect_arousals(sig[nm], sf, hyp, emg_data=emg,
                                ratio_thresh=ratio, abrupt_thresh=abrupt)
            per.append(r.get("events") or [])
        pool = _union_arousals(per); pool.sort(key=lambda e: e["onset_s"])
        a = [{"onset_s": e["onset_s"],
              "duration_s": max(e.get("duration_s", 3.0), 0.1), "type": "a"}
             for e in pool]
        dek = []
        for h in spans:
            b = [{"onset_s": o, "duration_s": max(d, 0.1), "type": "a"}
                 for o, d in h]
            if not b: continue
            pairs, _oa, _ob = _match(a, b, 0.20) if a else ([], [], [])
            gedekt = len({j for _i, j, _v in pairs})
            dek.append(gedekt / len(b))
        uit[f"{ratio}/{abrupt}"] = {"pool": len(pool),
                                    "dekking": median(dek) if dek else None,
                                    "mens": median([len(h) for h in spans])}
    return sn, uit


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(een, sn): sn for sn in SNS}
        for f in as_completed(futs):
            try:
                sn, u = f.result(); res[sn] = u
                print(f"{sn} klaar", flush=True)
            except Exception as e:
                print(f"{futs[f]}: FOUT {type(e).__name__}: {e}", flush=True)

    keys = [f"{r}/{a}" for r, a in COMBOS]
    print(f"\nDEKKING — mediane fractie van de arousals van een scoorder die "
          f"minstens één kandidaat raakt\n")
    print(f"{'opname':<8}{'mens':>6}" + "".join(f"{k:>14}" for k in keys))
    print("-" * (14 + 14 * len(keys)))
    for sn in SNS:
        if sn not in res: continue
        r = res[sn]
        print(f"{sn:<8}{r[keys[0]]['mens']:>6.0f}" +
              "".join(f"{r[k]['dekking']*100:>13.0f}%" if r[k]['dekking'] is not None
                      else f"{'—':>14}" for k in keys))
    print(f"\nPOOLGROOTTE — de prijs van die dekking\n")
    print(f"{'opname':<8}{'':>6}" + "".join(f"{k:>14}" for k in keys))
    print("-" * (14 + 14 * len(keys)))
    for sn in SNS:
        if sn not in res: continue
        r = res[sn]
        print(f"{sn:<8}{'':>6}" + "".join(f"{r[k]['pool']:>14}" for k in keys))
    json.dump(res, open(str(_UIT / "arousal_kandidaatdekking.json"), "w"),
              indent=1, ensure_ascii=False)
    print("\nwegschreven: docs/arousal_kandidaatdekking.json")
    print("\nLees dit als een BOVENGRENS per opname, niet als een werkpunt: een "
          "grotere\npool verplaatst het probleem naar de selectie, en daar liep "
          "het al vast.")
