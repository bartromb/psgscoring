#!/usr/bin/env python3
"""Levert de +2 s-verschuiving F1 op, of alleen dekking?

De detectie draait EEN keer per opname; de verschuivingen worden daarna op
dezelfde eventlijst toegepast. Elk verschil in de tabel is dus de verschuiving
en niets anders -- en de telling hoort per definitie gelijk te blijven, wat
meteen de controle op het harnas is.

Criteria vooraf: docs/arousal_onsetverschuiving_preregistratie.md
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
from statistics import median, mean
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
SHIFTS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0]


def een(sn):
    from sweep_event_locked_window import _psgipa_spans, _f1
    from sweep_arousal_threshold_psgipa import _stage
    from psgscoring.arousal import detect_arousals_multi
    from psgscoring.pipeline import arousal_derivation_channels

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
    derivs = [(n, raw.get_data(picks=[n])[0], sf) for n in wil]
    r = detect_arousals_multi(derivs, sf, hyp, emg_data=emg,
                              lgbm=True, lgbm_threshold=0.80)
    ev = r.get("events") or []
    base = [(float(e["onset_s"]), float(e.get("duration_s") or 3.0)) for e in ev]

    uit = {}
    for d in SHIFTS:
        sp = [(o + d, du) for o, du in base]
        f1s = [x for x in (_f1(sp, h) for h in spans) if x is not None]
        uit[d] = median(f1s) if f1s else None
    return sn, uit, len(ev), median([len(h) for h in spans])


if __name__ == "__main__":
    res, n_ev, mens = {}, {}, {}
    with ProcessPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(een, sn): sn for sn in SNS}
        for f in as_completed(futs):
            try:
                sn, u, n, m = f.result()
                res[sn] = u; n_ev[sn] = n; mens[sn] = m
                print(f"{sn} klaar ({n} events)", flush=True)
            except Exception as e:
                print(f"{futs[f]}: FOUT {type(e).__name__}: {e}", flush=True)

    print("\nEVENT-F1 PER ONSETVERSCHUIVING (mediaan over 12 scoorders)\n")
    print(f"{'verschuiving':<14}" + "".join(f"{sn:>8}" for sn in SNS) +
          f"{'gem. Δ':>10}{'beter':>7}")
    print("-" * (14 + 8 * len(SNS) + 17))
    for d in SHIFTS:
        rij = "".join(f"{res[sn][d]:>8.3f}" if sn in res else f"{'—':>8}"
                      for sn in SNS)
        dd = [res[sn][d] - res[sn][0.0] for sn in SNS if sn in res]
        beter = sum(1 for x in dd if x > 0)
        merk = "  <- nu" if d == 0.0 else ("  <- besluit" if d == 2.0 else "")
        print(f"{d:>+8.0f} s     {rij}{mean(dd):>+10.4f}{beter:>5}/5{merk}")

    print(f"\n{'opname':<8}{'events':>8}{'mens':>7}   (telling moet gelijk zijn "
          f"over alle verschuivingen — de detectie draaide één keer)")
    for sn in SNS:
        if sn in n_ev:
            print(f"{sn:<8}{n_ev[sn]:>8}{mens[sn]:>7.0f}")

    top = max(SHIFTS, key=lambda d: mean([res[sn][d] - res[sn][0.0]
                                          for sn in SNS if sn in res]))
    dd2 = [res[sn][2.0] - res[sn][0.0] for sn in SNS if sn in res]
    print(f"\nVooraf: +2 s moet gemiddelde Δ > 0 én ≥4/5 beter halen, én het "
          f"maximum\nvan de reeks moet op +1/+2/+3 liggen.")
    print(f"  gemeten: +2 s geeft Δ {mean(dd2):+.4f} op {sum(1 for x in dd2 if x>0)}/5; "
          f"maximum van de reeks ligt op {top:+.0f} s")
    json.dump({k: {str(a): b for a, b in v.items()} for k, v in res.items()},
              open(str(_UIT / "arousal_onsetverschuiving_f1.json"), "w"),
              indent=1)
    print("wegschreven: docs/arousal_onsetverschuiving_f1.json")
