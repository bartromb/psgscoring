"""Single versus multi-derivatie, beide met de hybride aan.

docs/arousal_derivatie_preregistratie.md. Meet recall apart, want de
verwachting is dat multi dekking koopt met precisie.
"""
import os, statistics as st, sys, json
from pathlib import Path
sys.path.insert(0, "/home/bart/CODE/psgscoring")
import mne; mne.set_log_level("ERROR")
import numpy as np
import psgscoring.arousal as A
from psgscoring.arousal import detect_arousals, detect_arousals_multi

D = Path("/home/bart/PSG-IPA/EEG_arousals")
def iou(a0,a1,b0,b1):
    i=max(0.0,min(a1,b1)-max(a0,b0)); u=max(a1,b1)-min(a0,b0)
    return i/u if u>0 else 0.0
def prf(pred, ref, thr=0.20):
    used=set(); tp=0
    for p0,p1 in pred:
        best,bi=0.0,None
        for i,(r0,r1) in enumerate(ref):
            if i in used: continue
            v=iou(p0,p1,r0,r1)
            if v>best: best,bi=v,i
        if bi is not None and best>=thr: used.add(bi); tp+=1
    fp,fn=len(pred)-tp,len(ref)-tp
    if tp==0: return 0.0,0.0,0.0
    p,r=tp/(tp+fp),tp/(tp+fn)
    return 2*p*r/(p+r), p, r

res={}
for sn in ("SN1","SN2","SN3","SN4","SN5"):
    raw=mne.io.read_raw_edf(D/"PSG"/f"{sn}_EEGarousals.edf",preload=True,verbose=False)
    sf=float(raw.info["sfreq"]); n_ep=int(np.ceil(raw.times[-1]/30))
    fs=sorted((D/"Annotations"/"manual").glob(f"{sn}_EEGarousals_manual_scorer*.edf"))
    a0=mne.read_annotations(str(fs[0])); hyp={}
    for o,d,x in zip(a0.onset,a0.duration,a0.description):
        t=str(x).replace("Sleep stage ","").strip()
        if t in ("W","N1","N2","N3","R"):
            for k in range(max(1,int(round(float(d)/30)))): hyp[int(o//30)+k]=t
    hypno=[hyp.get(e,"W") for e in range(n_ep)]
    tst=sum(1 for s in hypno if s in ("N1","N2","N3","R"))*30/3600
    sets=[]
    for f in fs:
        a=mne.read_annotations(str(f))
        sets.append([(float(o),float(o)+float(d)) for o,d,x in
                     zip(a.onset,a.duration,a.description) if "eeg arousal" in str(x).lower()])
    idx_s=st.median(len(s)/tst for s in sets)
    def ch(*names):
        for n in names:
            if n in raw.ch_names: return n, raw.get_data(picks=[n])[0]
        return None, None
    cn,cd = ch("EEG C4-M1","EEG Cz-M1","EEG C3-M2")
    on,od = ch("EEG O2-M1","EEG O1-M2")
    fn_,fd = ch("EEG F4-M1","EEG F3-M2")
    emg = raw.get_data(picks=["EMG chin"])[0] if "EMG chin" in raw.ch_names else None
    eog = raw.get_data(picks=["EOG E1-M2"])[0] if "EOG E1-M2" in raw.ch_names else None
    derivs=[(n,d,sf) for n,d in ((cn,cd),(on,od),(fn_,fd)) if n]
    arms={}
    # de OUDE productiestand: multi + regelgebaseerd
    os.environ["PSGSCORING_AROUSAL_LGBM"]="0"
    arms["multi+regels"]=detect_arousals_multi(derivs,sf,hypno,emg_data=emg)
    os.environ["PSGSCORING_AROUSAL_LGBM"]="1"; A.AROUSAL_LGBM_THRESHOLD=0.60
    arms["single"]=detect_arousals(cd,sf,hypno,emg_data=emg)
    arms["multi"]=detect_arousals_multi(derivs,sf,hypno,emg_data=emg)
    arms["multi+eog"]=detect_arousals_multi(derivs,sf,hypno,emg_data=emg,
                                            eog_data=eog,eog_reject=eog is not None)
    os.environ["PSGSCORING_AROUSAL_LGBM"]="0"
    e=res[sn]={"idx_scorer":round(idx_s,2),"afleidingen":[d[0] for d in derivs],"armen":{}}
    for k,v in arms.items():
        ev=[(x["onset_s"],x["end_s"]) for x in (v.get("events") or [])]
        m=[prf(ev,s) for s in sets]
        e["armen"][k]={"n":len(ev),"idx":round(len(ev)/tst,2),
                       "f1":round(st.median(x[0] for x in m),3),
                       "prec":round(st.median(x[1] for x in m),3),
                       "rec":round(st.median(x[2] for x in m),3),
                       "dur":round(float(np.median([b-a for a,b in ev])),2) if ev else 0.0}
    print(f"  {sn} [{len(derivs)} afl.] scoorder idx {idx_s:5.1f} | " +
          " | ".join(f"{k} F1 {e['armen'][k]['f1']:.3f} rec {e['armen'][k]['rec']:.3f} "
                     f"prec {e['armen'][k]['prec']:.3f} idx {e['armen'][k]['idx']:5.1f}"
                     for k in ("multi+regels","single","multi","multi+eog")), flush=True)
    del raw
json.dump(res,open("/tmp/claude-1000/-home-bart-CODE/1b7d6371-10c0-44b6-8046-19b4e23ebdc4/scratchpad/derivatie2.json","w"),indent=1)
print("\n"+"="*70)
for k in ("multi+regels","single","multi","multi+eog"):
    f=st.median(v["armen"][k]["f1"] for v in res.values())
    r=st.median(v["armen"][k]["rec"] for v in res.values())
    p=st.median(v["armen"][k]["prec"] for v in res.values())
    q=[v["armen"][k]["idx"]/v["idx_scorer"] for v in res.values()]
    print(f"{k:10s} F1 {f:.3f}  recall {r:.3f}  precisie {p:.3f}  "
          f"q {min(q):.2f}-{max(q):.2f} (spreiding {max(q)/min(q):.2f})")
print("\n=== wat de uitrol van gisteren deed in de ECHTE configuratie ===")
for k in ("multi+regels","multi"):
    f=st.median(v["armen"][k]["f1"] for v in res.values())
    q=[v["armen"][k]["idx"]/v["idx_scorer"] for v in res.values()]
    print(f"  {k:13s} F1 {f:.3f}  q mediaan {st.median(q):.2f}  bereik {min(q):.2f}-{max(q):.2f}")
fs_=st.median(v["armen"]["single"]["f1"] for v in res.values())
fm=st.median(v["armen"]["multi"]["f1"] for v in res.values())
rs=st.median(v["armen"]["single"]["rec"] for v in res.values())
rm=st.median(v["armen"]["multi"]["rec"] for v in res.values())
print("="*70)
print(f"PRIMAIR   F1(multi) > F1(single) : {fm:.3f} vs {fs_:.3f} -> "
      f"{'GEHAALD' if fm>fs_ else 'NIET GEHAALD'}")
print(f"SECUNDAIR recall stijgt          : {rm:.3f} vs {rs:.3f} -> "
      f"{'GEHAALD' if rm>rs else 'NIET GEHAALD'}")
