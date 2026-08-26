#!/usr/bin/env python3
"""v3 tegen de v6-ladder op PSG-IPA, op IDENTIEKE kandidaten.

v6 gebruikt EXACT dezelfde features als v3 -- het enige verschil is dat
`duration_s` bij het bouwen van de bomen een zwaardere split-gain kreeg
(LightGBM `feature_contri`). De patch is daarom alleen het verwisselen van de
booster; geen featurebouwer wordt aangeraakt. Dat is precies de bedoeling: elk
verschil in de tabel is de WEGING en niets anders.

Werkpunt per arm: de drempel die op MESA OOF dezelfde precisie haalt als v3 bij
0,80 (0,6345). Afgeleid op MESA, zodat PSG-IPA schoon blijft.

Criteria vooraf, zie docs/arousal_v6_preregistratie.md:
  1. poort  : holdout-AP q=7 >= 0,7229           (in het trainingsrapport)
  2. primair: mediane F1 > 0,514 EN beter op >= 4 van 5 opnames
  3. doel   : SN3 moet stijgen
  bewaker  : mediane telratio in [0,85; 1,15]
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
from statistics import median
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, mne
mne.set_log_level("ERROR")
sys.path.insert(0, "/home/bart/CODE/psgscoring")
sys.path.insert(0, "/home/bart/CODE/psgscoring/scripts")

AB = Path("/home/bart/MESA-ab-test")
ROOT = Path("/home/bart/PSG-IPA/EEG_arousals")
EPOCH_S = 30.0
SNS = ["SN1", "SN2", "SN3", "SN4", "SN5"]


def armen():
    """(naam, modelpad, werkpunt) -- v3 op 0,80, elk v6-arm op zijn eigen punt."""
    out = [("v3", None, 0.80)]
    for f in (2, 4, 8):
        rp = AB / f"arousal_classifier_v6f{f}_report.json"
        mp = AB / f"arousal_classifier_v6f{f}.txt"
        if not (rp.exists() and mp.exists()):
            continue
        r = json.loads(rp.read_text())
        m = r.get("matched_operating_point")
        if not m:
            print(f"v6f{f}: GEEN werkpunt op gelijke precisie -- arm valt af", flush=True)
            continue
        out.append((f"v6f{f}", str(mp), float(m["threshold"])))
    return out


def _zet_model(pad):
    import lightgbm as lgb
    import psgscoring.arousal as A
    boost = lgb.Booster(model_file=pad)
    assert boost.num_feature() == len(A._AROUSAL_LGBM_FEATURE_ORDER), (
        boost.num_feature(), len(A._AROUSAL_LGBM_FEATURE_ORDER))
    A._AROUSAL_LGBM_BOOSTER = boost
    A._load_arousal_lgbm_booster = lambda: boost


def een(sn, naam, pad, thr):
    # ALTIJD expliciet zetten, ook voor v3. De workers worden hergebruikt: een
    # worker die eerder een v6-arm draaide houdt die booster in zijn module-
    # globals, en een v3-taak zou dan stilzwijgend met v6 gemeten worden. Dat
    # levert een tabel op die er goed uitziet en niets meet.
    if pad is None:
        from psgscoring.arousal import AROUSAL_LGBM_MODEL_PATH as pad
    _zet_model(pad)
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
                              lgbm=True, lgbm_threshold=thr)
    ev = r.get("events") or []
    sp = [(float(e["onset_s"]), float(e.get("duration_s") or 3.0)) for e in ev]
    f1s = [x for x in (_f1(sp, h) for h in spans) if x is not None]
    mens = median([len(h) for h in spans])
    return {"sn": sn, "arm": naam, "n": len(ev), "mens": mens,
            "ratio": len(ev) / mens if mens else None,
            "f1": median(f1s) if f1s else None}


if __name__ == "__main__":
    AR = armen()
    print("armen:", ", ".join(f"{n}@{t:.3f}" for n, _, t in AR), flush=True)
    taken = [(sn, n, p, t) for (n, p, t) in AR for sn in SNS]
    uit = {}
    with ProcessPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(een, sn, n, p, t): (sn, n) for sn, n, p, t in taken}
        for f in as_completed(futs):
            sn, n = futs[f]
            try: uit[(sn, n)] = f.result()
            except Exception as e:
                print(f"{sn}/{n}: FOUT {type(e).__name__}: {e}", flush=True)

    namen = [n for n, _, _ in AR]
    print(f"\n{'opname':<7}{'mens':>6}" + "".join(f"{n:>24}" for n in namen))
    print(f"{'':<7}{'':>6}" + "".join(f"{'n / ratio / F1':>24}" for _ in namen))
    print("-" * (13 + 24 * len(namen)))
    for sn in SNS:
        cel = ""
        for n in namen:
            r = uit.get((sn, n))
            cel += (f"{r['n']:>9}/{r['ratio']:>5.2f}/{r['f1']:>6.3f}"
                    if r and r["f1"] is not None else f"{'—':>24}")
        print(f"{sn:<7}{uit.get((sn,'v3'),{}).get('mens',0):>6.0f}{cel}")

    print()
    base = {sn: uit[(sn, "v3")]["f1"] for sn in SNS if (sn, "v3") in uit}
    for n in namen:
        rr = [uit[(sn, n)]["ratio"] for sn in SNS if (sn, n) in uit]
        ff = [uit[(sn, n)]["f1"] for sn in SNS if (sn, n) in uit]
        if not ff: continue
        beter = sum(1 for sn in SNS if (sn, n) in uit and sn in base
                    and uit[(sn, n)]["f1"] > base[sn])
        sn3 = uit.get(("SN3", n), {}).get("f1")
        sn3_txt = f"{sn3:.3f}" if sn3 is not None else "—"
        print(f"{n:<7} mediane F1 {median(ff):.3f} | telratio mediaan "
              f"{median(rr):.2f} ({min(rr):.2f}-{max(rr):.2f}) | beter dan v3 op "
              f"{beter}/5 | SN3 {sn3_txt}")

    print("\nVooraf (docs/arousal_v6_preregistratie.md): mediane F1 > 0,514 EN "
          "beter op >= 4/5 EN SN3 stijgt; telratio-mediaan in [0,85; 1,15].")
