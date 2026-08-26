#!/usr/bin/env python3
"""Foutanalyse: wat onderscheidt een TEN ONRECHTE verworpen kandidaat van een
TERECHT verworpene?

Het pool-orakel gaf 0,896 tegen onze 0,514: de arousals zitten in de pool en de
selectie laat ze liggen. Deze analyse kijkt naar de kandidaten die het model
verwerpt (p < 0,80) en splitst ze op wat de scoorders ervan vonden.

DE LABELING GEBRUIKT ALLE TWAALF SCOORDERS, niet één:
    ECHT      >= 6 van de 12 scoorders hebben hier een arousal   (meerderheid)
    ONECHT    0 van de 12
    dubieus   1-5           -> UITGESLOTEN uit het contrast

Die middengroep weglaten is geen data weggooien maar het contrast scherpstellen:
kandidaten waar de mens zelf verdeeld over is, zeggen niets over wat het model
had moeten doen.

Uitkomst per feature: Cohen's d tussen de verworpen-ECHTE en de
verworpen-ONECHTE. Is die overal klein, dan zit het onderscheid niet in deze
features en helpt geen enkel model dat er alleen op leert.
"""
import os, sys
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, mne
mne.set_log_level("ERROR")

# Uitvoermap: PSGSCORING_MEETUITVOER, anders de werkmap. Nooit een
# tijdelijke map -- een harnas in de repo dat naar /tmp schrijft is bij
# de volgende sessie stuk.
_UIT = Path(os.environ.get("PSGSCORING_MEETUITVOER", "."))
sys.path.insert(0,"/home/bart/CODE/psgscoring"); sys.path.insert(0,"/home/bart/CODE/psgscoring/scripts")
ROOT=Path("/home/bart/PSG-IPA/EEG_arousals"); EPOCH_S=30.0; THR=0.80

def _uv(x):
    x=np.asarray(x,dtype=float).copy()
    return x*1e6 if np.max(np.abs(x))<0.01 else x

def een(sn):
    from sweep_event_locked_window import _psgipa_spans
    from sweep_arousal_threshold_psgipa import _stage
    from psgscoring.arousal import (AROUSAL_LGBM_CAND_ABRUPT, AROUSAL_LGBM_CAND_RATIO,
        _AROUSAL_LGBM_FEATURE_ORDER, _arousal_lgbm_features,
        _load_arousal_lgbm_booster, _union_arousals, detect_arousals)
    from psgscoring.pipeline import arousal_derivation_channels
    from psgscoring.agreement import _match

    psg=ROOT/"PSG"/f"{sn}_EEGarousals.edf"
    hdr=mne.io.read_raw_edf(psg,preload=False,verbose=False)
    dur=hdr.n_times/hdr.info["sfreq"]
    scs=sorted((ROOT/"Annotations"/"manual").glob(f"{sn}_EEGarousals_manual_scorer*.txt"))
    hyps,spans=[],[]
    for sc in scs:
        n_ep=int(np.ceil(dur/EPOCH_S)); h=["W"]*n_ep
        with open(sc,encoding="utf-8",errors="replace") as f:
            next(f,None)
            for line in f:
                q=[x.strip() for x in line.split(",")]
                if len(q)<5: continue
                try: o,d=float(q[2]),float(q[3])
                except ValueError: continue
                st=_stage(q[4])
                if st is None or not (0<=o<dur): continue
                e0=int(o//EPOCH_S)
                for i in range(max(1,round(d/EPOCH_S))):
                    if 0<=e0+i<n_ep: h[e0+i]=st
        hyps.append(h); spans.append(_psgipa_spans(sc,dur))
    n_ep=min(len(h) for h in hyps)
    hyp=[max(set(x),key=x.count) for x in ([h[i] for h in hyps] for i in range(n_ep))]

    wil=arousal_derivation_channels(hdr.ch_names)
    raw=mne.io.read_raw_edf(psg,exclude=[c for c in hdr.ch_names
        if c not in set(wil)|{"EMG chin"}],preload=True,verbose=False)
    sf=raw.info["sfreq"]; emg=raw.get_data(picks=["EMG chin"])[0]

    per=[]
    for nm in wil:
        r=detect_arousals(raw.get_data(picks=[nm])[0],sf,hyp,emg_data=emg,
                          ratio_thresh=AROUSAL_LGBM_CAND_RATIO,
                          abrupt_thresh=AROUSAL_LGBM_CAND_ABRUPT)
        per.append(r.get("events") or [])
    pool=_union_arousals(per); pool.sort(key=lambda e:e["onset_s"])

    # hoeveel scoorders zien hier een arousal?
    stemmen=np.zeros(len(pool),dtype=int)
    a=[{"onset_s":e["onset_s"],"duration_s":max(e.get("duration_s",3.0),0.1),"type":"a"} for e in pool]
    for h in spans:
        b=[{"onset_s":o,"duration_s":max(d,0.1),"type":"a"} for o,d in h]
        if not b: continue
        pairs,_oa,_ob=_match(a,b,0.20)
        for i,_j,_v in pairs: stemmen[i]+=1

    eeg0=_uv(raw.get_data(picks=[wil[0]])[0])
    rows=[_arousal_lgbm_features(c,eeg0,sf,_uv(emg[:len(eeg0)]),len(hyp)) for c in pool]
    X=np.array([[r[c] for c in _AROUSAL_LGBM_FEATURE_ORDER] for r in rows],dtype=float)
    p=np.asarray(_load_arousal_lgbm_booster().predict(X),dtype=float)
    return {"sn":sn,"X":X,"p":p,"stemmen":stemmen,"n_pool":len(pool)}

if __name__=="__main__":
    from psgscoring.arousal import _AROUSAL_LGBM_FEATURE_ORDER as ORDER
    res=[]
    with ProcessPoolExecutor(max_workers=5) as pool:
        futs={pool.submit(een,sn):sn for sn in ["SN1","SN2","SN3","SN4","SN5"]}
        for f in as_completed(futs):
            try: res.append(f.result())
            except Exception as e: print(f"{futs[f]}: FOUT {e}",flush=True)
    res.sort(key=lambda r:r["sn"])
    X=np.vstack([r["X"] for r in res]); p=np.concatenate([r["p"] for r in res])
    st=np.concatenate([r["stemmen"] for r in res])
    echt, onecht = st>=6, st==0
    verworpen = p<THR
    print(f"\n{'opname':<7}{'pool':>7}{'echt':>7}{'dubieus':>9}{'onecht':>8}"
          f"{'TP':>6}{'FN':>6}{'FP':>6}")
    print("-"*56)
    for r in res:
        s,pp=r["stemmen"],r["p"]
        e,o=s>=6,s==0
        print(f"{r['sn']:<7}{r['n_pool']:>7}{e.sum():>7}{((s>0)&(s<6)).sum():>9}"
              f"{o.sum():>8}{(e&(pp>=THR)).sum():>6}{(e&(pp<THR)).sum():>6}"
              f"{(o&(pp>=THR)).sum():>6}")
    fn = echt & verworpen
    tn = onecht & verworpen
    print(f"\ngepoold: {fn.sum()} ten onrechte verworpen, {tn.sum()} terecht verworpen")
    print(f"kans van de ten onrechte verworpenen: mediaan {np.median(p[fn]):.3f}  "
          f"p75 {np.percentile(p[fn],75):.3f}  aandeel > 0,50: {np.mean(p[fn]>0.5)*100:.1f} %")
    print(f"kans van de terecht verworpenen    : mediaan {np.median(p[tn]):.3f}  "
          f"p75 {np.percentile(p[tn],75):.3f}")
    d=[]
    for i,naam in enumerate(ORDER):
        a_,b_=X[fn,i],X[tn,i]
        sp=np.sqrt((a_.var()+b_.var())/2)
        d.append((abs(a_.mean()-b_.mean())/sp if sp>0 else 0.0, naam,
                  np.median(a_), np.median(b_)))
    d.sort(reverse=True)
    print(f"\n{'feature':<24}{'Cohen d':>9}{'mediaan FN':>13}{'mediaan TN':>13}")
    print("-"*59)
    # Volledige lijst, plus de gain-rangorde van v3 ernaast. De vraag is niet
    # alleen WELK feature onderscheidt, maar of het model dat feature al zwaar
    # weegt -- een groot verschil dat het model al zwaar weegt is geen
    # aangrijpingspunt, een groot verschil dat het licht weegt wel.
    import lightgbm as lgb, json as _json
    from psgscoring.arousal import AROUSAL_LGBM_MODEL_PATH
    bst = lgb.Booster(model_file=AROUSAL_LGBM_MODEL_PATH)
    gain = bst.feature_importance("gain")
    rang = {ORDER[i]: r+1 for r, i in enumerate(sorted(range(len(ORDER)),
            key=lambda j: -gain[j]))}
    print(f"\n{'feature':<24}{'Cohen d':>9}{'mediaan FN':>13}{'mediaan TN':>13}"
          f"{'gain-rang':>11}")
    print("-"*70)
    for dd,naam,ma,mb in d:
        print(f"{naam:<24}{dd:>9.3f}{ma:>13.3f}{mb:>13.3f}{rang.get(naam,0):>11}")
    print(f"\ngrootste d = {d[0][0]:.3f}")
    uit = [{"feature": n, "d": round(float(dd),4), "med_fn": float(ma),
            "med_tn": float(mb), "gain_rang": rang.get(n)} for dd,n,ma,mb in d]
    _json.dump(uit, open(str(_UIT / "arousal_foutanalyse_features.json"),"w"),
               indent=1, ensure_ascii=False)
    print("wegschreven: docs/arousal_foutanalyse_features.json")

    # -- Duur-orakel ---------------------------------------------------------
    # Voor ik een model hertrain: hoeveel valt er MAXIMAAL te halen met een
    # duurbewuste beslisregel op de BESTAANDE kansen? Familie:
    #     houd als  p >= t_kort   OF  (duur >= D en p >= t_lang)
    # met t_lang <= t_kort. Het orakel kiest (t_kort, D, t_lang) op precies
    # dezelfde set waarop het gemeten wordt -- dat is een BOVENGRENS, geen
    # werkpunt. Valt die bovengrens tegen, dan is v6 het ook niet waard.
    di = ORDER.index("duration_s")
    dur = X[:, di]
    y = echt.astype(bool)
    keuze = echt | onecht          # dubieuze kandidaten tellen niet mee
    def f1_van(mask):
        tp = int((mask & y & keuze).sum()); fp = int((mask & ~y & keuze).sum())
        fn = int((~mask & y & keuze).sum())
        return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
    basis = f1_van(p >= THR)
    beste = (basis, THR, None, None)
    for tk in np.arange(0.50, 0.96, 0.02):
        for D in (0.0, 5, 6, 7, 8, 9, 10, 12, 15):
            for tl in np.arange(0.20, tk + 1e-9, 0.02):
                m = (p >= tk) | ((dur >= D) & (p >= tl))
                f = f1_van(m)
                if f > beste[0]:
                    beste = (f, tk, D, tl)
    print("\nDUUR-ORAKEL op de gepoolde kandidaten (echt = >=6 stemmen)")
    print(f"  v3 zoals hij draait (p >= {THR}):   F1 {basis:.4f}")
    if beste[2] is None:
        print("  geen duurbewuste regel verslaat dat.")
    else:
        f, tk, D, tl = beste
        m = (p >= tk) | ((dur >= D) & (p >= tl))
        tp = int((m & y & keuze).sum()); fp = int((m & ~y & keuze).sum())
        fn = int((~m & y & keuze).sum())
        print(f"  beste duurbewuste regel:          F1 {f:.4f}  (+{f-basis:.4f})")
        print(f"    houd p >= {tk:.2f}  OF  (duur >= {D:g}s en p >= {tl:.2f})")
        print(f"    TP {tp}  FP {fp}  FN {fn}")
    print("\n  Dit is een ORAKEL: parameters gekozen op de meetset zelf. Wat een")
    print("  hertraind model haalt ligt hieronder, nooit erboven.")
    np.savez(str(_UIT / "arousal_foutanalyse_pool.npz"),
             p=p, stemmen=st, duration_s=dur)
    print("wegschreven: docs/arousal_foutanalyse_pool.npz")
