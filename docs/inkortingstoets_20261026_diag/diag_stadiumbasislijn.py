#!/usr/bin/env python
"""Post-hoc diagnose (beschrijvend) van de stadiumbasislijn op MESA-nachten uit de inkortingstoets:
per kanaal (Pres, Therm) en per stadium de verhouding stadium-P90 / dynamische basislijn, en het
aandeel slaapsamples onder de apneudrempel (env < 0,1 × basislijn) met en zonder stadiummenging."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/srv/CODE/psgscoring"); sys.path.insert(0, "/srv/CODE/psgscoring/scripts")
import mne; mne.set_log_level("ERROR")
from psgscoring.signal import preprocess_flow, compute_dynamic_baseline, compute_stage_baseline
from validate_mesa import parse_nsrr
EDF = Path("/srv/DATA/MESA/mesa/polysomnography/edfs"); XML = Path("/srv/DATA/MESA/mesa/polysomnography/annotations-events-nsrr")
NACHTEN = {"mesa-sleep-0111": "fout (3→183 apneus, alle thermistor)", "mesa-sleep-0225": "fout (17→220)", "mesa-sleep-1305": "fout (13→98)", "mesa-sleep-0001": "controle (71→80)", "mesa-sleep-0468": "controle (257→258)"}
uit = []
for rid, rol in NACHTEN.items():
    raw = mne.io.read_raw_edf(str(EDF / f"{rid}.edf"), preload=False, verbose=False)
    kan = [c for c in raw.ch_names if c in ("Pres", "Therm")]
    raw = mne.io.read_raw_edf(str(EDF / f"{rid}.edf"), include=kan, preload=True, verbose=False)
    sf = float(raw.info["sfreq"]); dur = float(raw.times[-1]); hypno, refs, tst_h = parse_nsrr(XML / f"{rid}-nsrr.xml", dur)
    rij = {"recording": rid, "rol": rol, "sf": sf, "tst_h": round(tst_h, 2), "n_ref_aasm15": len(refs["aasm15"]), "kanalen": {}}
    spe = int(sf * 30); n_ep = len(hypno); hyp_arr = np.array(hypno)
    for ch in kan:
        x = raw.get_data(picks=[ch])[0]
        env = preprocess_flow(x, sf, is_nasal_pressure=(ch == "Pres"))
        n = len(env); dyn = compute_dynamic_baseline(env, sf); st = compute_stage_baseline(env, sf, hypno, dynamic_baseline=dyn); meng = (dyn + st) / 2
        # stadiummasker op samplelengte
        rep = np.repeat(hyp_arr[: min(n_ep, -(-n // spe))], spe)[:n]; mask_len = rep.size
        slaap = np.zeros(n, bool); slaap[:mask_len] = np.isin(rep, ["N1", "N2", "N3", "R"])
        per_st = {}
        for s in ("N1", "N2", "N3", "R"):
            m = np.zeros(n, bool); m[:mask_len] = rep == s
            if m.sum() < sf * 60: continue
            per_st[s] = {"min": round(m.sum() / sf / 60, 1), "stadium/dyn_mediaan": round(float(np.median(st[m] / np.maximum(dyn[m], 1e-9))), 3),
                         "onder_drempel_dyn_%": round(100 * float(np.mean(env[m] < 0.1 * dyn[m])), 2), "onder_drempel_meng_%": round(100 * float(np.mean(env[m] < 0.1 * meng[m])), 2),
                         "env_p50/dyn": round(float(np.median(env[m] / np.maximum(dyn[m], 1e-9))), 3)}
        rij["kanalen"][ch] = {"onder_drempel_slaap_dyn_%": round(100 * float(np.mean(env[slaap] < 0.1 * dyn[slaap])), 2), "onder_drempel_slaap_meng_%": round(100 * float(np.mean(env[slaap] < 0.1 * meng[slaap])), 2),
                              "stadium/dyn_mediaan_slaap": round(float(np.median(st[slaap] / np.maximum(dyn[slaap], 1e-9))), 3), "per_stadium": per_st}
    uit.append(rij); print(json.dumps(rij), flush=True)
json.dump(uit, open("/srv/CODE/docs/inkortingstoets_20261026_diag/diag.json", "w"), indent=1)
print("KLAAR")
