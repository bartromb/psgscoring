#!/usr/bin/env python3
"""Staan SN3's annotaties en signaal op dezelfde tijdas?

SN3 krijgt MEER kandidaten dan SN1 (1453 tegen 1163) en dekt er de HELFT mee
(50 % tegen 87 %). Lossere drempels tillen dat maar naar 60 %. Dat past niet bij
"te weinig kandidaten" en ook niet bij "verkeerde drempel"; het past wel bij een
referentie die niet op dezelfde tijdas staat.

Toets: verschuif de kandidaten kunstmatig over -60..+60 s en meet de dekking per
verschuiving. Piek bij 0 => de tijdas klopt en het is een echt signaalprobleem.
Piek elders => ik meet al die tijd tegen een verschoven referentie, en elke F1
in dit dossier voor die opname is te laag.

De vier andere opnames zijn de controle: die horen bij 0 te pieken.
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
LAGS = [round(x, 1) for x in np.arange(-60.0, 60.01, 2.0)]


def een(sn):
    from sweep_event_locked_window import _psgipa_spans
    from sweep_arousal_threshold_psgipa import _stage
    from psgscoring.arousal import (AROUSAL_LGBM_CAND_ABRUPT,
                                    AROUSAL_LGBM_CAND_RATIO,
                                    _union_arousals, detect_arousals)
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
    per = []
    for nm in wil:
        r = detect_arousals(raw.get_data(picks=[nm])[0], sf, hyp, emg_data=emg,
                            ratio_thresh=AROUSAL_LGBM_CAND_RATIO,
                            abrupt_thresh=AROUSAL_LGBM_CAND_ABRUPT)
        per.append(r.get("events") or [])
    pool = _union_arousals(per); pool.sort(key=lambda e: e["onset_s"])

    uit = {}
    for lag in LAGS:
        a = [{"onset_s": e["onset_s"] + lag,
              "duration_s": max(e.get("duration_s", 3.0), 0.1), "type": "a"}
             for e in pool]
        dek = []
        for h in spans:
            b = [{"onset_s": o, "duration_s": max(d, 0.1), "type": "a"}
                 for o, d in h]
            if not b: continue
            pairs, _oa, _ob = _match(a, b, 0.20)
            dek.append(len({j for _i, j, _v in pairs}) / len(b))
        uit[lag] = median(dek) if dek else 0.0
    return sn, uit, len(pool)


if __name__ == "__main__":
    res, pools = {}, {}
    with ProcessPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(een, sn): sn for sn in SNS}
        for f in as_completed(futs):
            try:
                sn, u, n = f.result(); res[sn] = u; pools[sn] = n
                print(f"{sn} klaar (pool {n})", flush=True)
            except Exception as e:
                print(f"{futs[f]}: FOUT {type(e).__name__}: {e}", flush=True)

    print("\nDEKKING PER KUNSTMATIGE VERSCHUIVING (%)\n")
    toon = [l for l in LAGS if abs(l) <= 30]
    print(f"{'verschuiving':<14}" + "".join(f"{sn:>7}" for sn in SNS))
    print("-" * (14 + 7 * len(SNS)))
    for l in toon:
        merk = " <-" if l == 0 else ""
        print(f"{l:>+8.0f} s     " +
              "".join(f"{res[sn][l]*100:>6.0f}" if sn in res else f"{'—':>6}"
                      for sn in SNS) + merk)
    print()
    for sn in SNS:
        if sn not in res: continue
        top = max(res[sn].items(), key=lambda kv: kv[1])
        print(f"{sn}: piek bij {top[0]:+.0f} s ({top[1]*100:.0f} %), "
              f"bij 0 s {res[sn][0.0]*100:.0f} %")
    json.dump({k: {str(a): b for a, b in v.items()} for k, v in res.items()},
              open(str(_UIT / "arousal_uitlijning_lag.json"), "w"),
              indent=1)
    print("\nwegschreven: docs/arousal_uitlijning_lag.json")
