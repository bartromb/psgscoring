"""Is de driftgevoeligheid van onze hypoxic burden een eigenschap van ONZE
implementatie of van de gepubliceerde definitie? (W31b)

Onze default `percentile` neemt `baseline = max(lokaal, nachtbreed)`. Azarbarzin
neemt de basislijn aan de linkerflank van een per-opname uit het
ensemble-gemiddelde afgeleid zoekvenster -- zuiver lokaal, geen plafond. Als
het plafond de oorzaak is, hoort `ensemble` de drift niet te vertonen.

Invoer: de MENSELIJK gescoorde NSRR-events, zodat onze eventdetector geen rol
speelt. Alleen het SpO2-kanaal wordt gelezen.

Drift = 95e percentiel SpO2 in het eerste uur slaap minus dat in het laatste uur.
"""
import sys
sys.path.insert(0, "/home/bart/CODE/psgscoring-dev")
sys.path.insert(0, "/home/bart/CODE/docs")
import mne; mne.set_log_level("ERROR")
import numpy as np
from mesa_arousal_harness import read_annotations, ANNS, EDFS
from psgscoring.spo2 import compute_hypoxic_burden
import xml.etree.ElementTree as ET

def resp_events(xml):
    root = ET.parse(xml).getroot()
    out = []
    for ev in root.iter("ScoredEvent"):
        c = (ev.findtext("EventConcept") or "")
        if not any(k in c for k in ("Hypopnea", "apnea", "Apnea")):
            continue
        s, d = ev.findtext("Start"), ev.findtext("Duration")
        if s is None or d is None:
            continue
        out.append({"onset_s": float(s), "duration_s": float(d)})
    return sorted(out, key=lambda e: e["onset_s"])

ids = sys.argv[1:] if len(sys.argv) > 1 else [
    "mesa-sleep-1374","mesa-sleep-2149","mesa-sleep-2747",
    "mesa-sleep-3135","mesa-sleep-3743","mesa-sleep-3823",
    "mesa-sleep-6157","mesa-sleep-1020"]
print(f"{'opname':>18} {'drift':>6} {'n_ev':>5} {'max(l,g)':>9} {'lokaal':>8} "
      f"{'ensemble':>9} {'azarbarzin':>11}")
for rid in ids:
    xml = ANNS / f"{rid}-nsrr.xml"; edf = EDFS / f"{rid}.edf"
    if not xml.exists() or not edf.exists():
        continue
    hyp, _ = read_annotations(xml)
    ev = resp_events(xml)
    if not hyp or not ev:
        continue
    raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
    if "SpO2" not in raw.ch_names:
        continue
    sf = float(raw.info["sfreq"])
    spo2 = raw.get_data(picks=["SpO2"])[0]
    del raw
    # drift: 95e pct eerste versus laatste uur slaap
    n = len(spo2)
    sm = np.zeros(n, bool)
    spe = int(sf*30)
    for i, s in enumerate(hyp):
        if s in ("N1","N2","N3","R"):
            sm[i*spe:min((i+1)*spe, n)] = True
    idx = np.where(sm)[0]
    if len(idx) < int(sf*7200):
        drift = float("nan")
    else:
        h = int(sf*3600)
        a = spo2[idx[:h]]; b = spo2[idx[-h:]]
        a = a[(a>=50)&(a<=100)]; b = b[(b>=50)&(b<=100)]
        drift = float(np.percentile(a,95)-np.percentile(b,95)) if len(a) and len(b) else float("nan")
    r = {}
    for tag, kw in (("maxlg", {}), ("lokaal", {"local_baseline_only": True}),
                    ("ens", {"baseline_method": "ensemble"}), ("azb", {"baseline_method": "azarbarzin"})):
        try:
            r[tag] = compute_hypoxic_burden(spo2, sf, ev, hyp, **kw).get("hypoxic_burden")
        except Exception as e:
            r[tag] = None
    print(f"{rid:>18} {drift:6.1f} {len(ev):5d} {r['maxlg'] or 0:9.2f} "
          f"{r['lokaal'] or 0:8.2f} {r['ens'] or 0:9.2f} {r['azb'] or 0:11.2f}", flush=True)
