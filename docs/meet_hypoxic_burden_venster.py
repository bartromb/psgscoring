"""Voorspelt de vensterlengte de afwijking t.o.v. de specificatie?

Als de burden 'gewoon' een oppervlakte is, hoort een andere basislijn de
SCHAAL te verzetten en de VOLGORDE te bewaren. Spearman 0,69 zegt dat de
volgorde wel degelijk verschuift. Kandidaat: de hersteldrempel. Stopt de
integratie bij basislijn - 1 %, dan is de vensterlengte een SPRONGfunctie van
waar de basislijn toevallig ligt -- herstelt de saturatie net wel, dan zijn de
vensters kort; net niet, dan loopt elk event de volle 120 s.

Voorspelling: de ratio default/specificatie loopt mee met het aandeel events
dat de 120 s-grens haalt.
"""
import sys
sys.path.insert(0,"/home/bart/CODE/psgscoring-dev"); sys.path.insert(0,"/home/bart/CODE/docs")
import mne; mne.set_log_level("ERROR")
import numpy as np, xml.etree.ElementTree as ET, statistics as st
from mesa_arousal_harness import read_annotations, ANNS, EDFS

RATIO={"mesa-sleep-1374":2.34,"mesa-sleep-2149":1.33,"mesa-sleep-2747":0.77,
       "mesa-sleep-3135":0.29,"mesa-sleep-3743":1.31,"mesa-sleep-3823":0.47,
       "mesa-sleep-6157":1.49,"mesa-sleep-1020":2.22}
print(f"{'opname':>18} {'ratio':>6} {'med.venster':>12} {'%aan de cap':>12} {'%overlap':>9}")
rows=[]
for rid,ratio in RATIO.items():
    root=ET.parse(ANNS/f"{rid}-nsrr.xml").getroot()
    ev=[(float(e.findtext("Start")),float(e.findtext("Duration")))
        for e in root.iter("ScoredEvent")
        if any(k in (e.findtext("EventConcept") or "") for k in ("Hypopnea","apnea","Apnea"))
        and e.findtext("Start") and e.findtext("Duration")]
    ev.sort()
    hyp,_=read_annotations(ANNS/f"{rid}-nsrr.xml")
    raw=mne.io.read_raw_edf(EDFS/f"{rid}.edf",preload=False,verbose=False)
    sf=float(raw.info["sfreq"]); spo2=raw.get_data(picks=["SpO2"])[0]; del raw
    n=len(spo2); spe=int(sf*30)
    sm=np.zeros(n,bool)
    for i,s in enumerate(hyp):
        if s in ("N1","N2","N3","R"): sm[i*spe:min((i+1)*spe,n)]=True
    ss=spo2[sm]; ss=ss[(~np.isnan(ss))&(ss>=50)]
    gbl=float(np.percentile(ss,95))
    lens=[]; cap=0; prev=-1e9; ov=0
    for o,d in ev:
        end=o+d
        ps=max(0,int((o-120)*sf)); pe=max(0,int(o*sf))
        pre=spo2[ps:pe]; pre=pre[(~np.isnan(pre))&(pre>=50)]
        bl=max(float(np.percentile(pre,90)) if len(pre)>3 else gbl, gbl)
        thr=bl-1.0
        i0=int(o*sf); i1=min(n,int((end+120)*sf)); seg=spo2[i0:i1]
        ee=int(d*sf); rec=len(seg)
        for k in range(min(ee,len(seg)),len(seg)):
            if not np.isnan(seg[k]) and seg[k]>=thr: rec=k+1; break
        w=rec/sf; lens.append(w)
        if rec>=len(seg)-1: cap+=1
        if o<prev: ov+=1
        prev=o+w
    pc=100*cap/len(lens); po=100*ov/len(lens)
    rows.append((ratio,pc,po))
    print(f"{rid:>18} {ratio:6.2f} {st.median(lens):11.1f}s {pc:11.0f}% {po:8.0f}%")
def pear(a,b):
    ma,mb=st.mean(a),st.mean(b)
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    den=(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**0.5
    return num/den if den else float("nan")
r=[x[0] for x in rows]
print(f"\nratio vs %aan-de-cap : r = {pear(r,[x[1] for x in rows]):.2f}")
print(f"ratio vs %overlap    : r = {pear(r,[x[2] for x in rows]):.2f}")
