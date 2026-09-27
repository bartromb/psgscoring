#!/usr/bin/env python3
"""MSED: één voorwaartse pas, meerdere classificatiedrempels; PSG-IPA én MESA.

    bench/eeg/_venv/bin/python bench/eeg/msed/run_msed_sweep.py --cohort psgipa
    bench/eeg/_venv/bin/python bench/eeg/msed/run_msed_sweep.py --cohort mesa

PSG-IPA: kanaalvariant `dup` (zie run_msed.py). MESA (de 80 validatienachten van
../unet50/ids_val.txt, al in het register): C3 := C4 := EEG3 (C4-M1), EOG-L/R,
EMG, LegL := LegR := Leg (MESA heeft één ongezijderd beenkanaal), NasalP := Pres,
Thor, Abdo. Per drempel alleen de arousalklasse; slaappoort met het hypnogram
van de referentie (scoorder 1 resp. NSRR). MESA-referentie-arousals worden als
eventtijden naast de voorspellingen geschreven (geen signaal).
De drempelveeg op PSG-IPA is een ORAKEL; de MESA-veeg dient om een drempel te
kiezen ZONDER PSG-IPA te zien.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import torch, einops

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent / "common")); sys.path.insert(0, str(HIER.parent / "unet50")); sys.path.insert(0, str(HIER))
from postproc import gate_sleep, write_pred  # noqa: E402
from msed.preprocessing import process_file, FS  # noqa: E402
from msed.functions import binary_to_array  # noqa: E402
from msed.predict_events import initialize_model  # noqa: E402
from msed.utils.config import load_config  # noqa: E402
from run_msed import BASE_MAP, C3_VARIANT, MODEL  # noqa: E402
from data import parse_mesa_xml, MESA_EDF, MESA_XML  # noqa: E402

REF = HIER.parent / "ref"
THRS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.64, 0.70, 0.75, 0.80, 0.85, 0.90]
MESA_MAP = {"A1": [], "A2": [], "C3": ["EEG3"], "C4": ["EEG3"], "EOGL": ["EOG-L"], "EOGR": ["EOG-R"], "EOGRef": [],
            "Chin": ["EMG"], "ChinRef": [], "LegL": ["Leg"], "LegR": ["Leg"], "NasalP": ["Pres"], "Thor": ["Thor"], "Abdo": ["Abdo"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=("psgipa", "mesa"), required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop-sleep", action="store_true",
                    help="MESA: alleen [eerste slaapepoch - 5 min, laatste + 5 min] voeden en daarbinnen "
                         "opnieuw z-scoren (diagnose: MESA-nachten zijn 10-12 u met lange wakkere randen, "
                         "PSG-IPA is op lights-off/on geknipt; MSED z-scoort over de hele opname)")
    a = ap.parse_args()
    out = HIER / f"out_sweep_{a.cohort}{'_crop' if a.crop_sleep else ''}"; out.mkdir(exist_ok=True)
    if a.cohort == "psgipa":
        recs = [(sn, Path("/srv/DATA/PSG-IPA/Resp_events/PSG") / f"{sn}_Respiration.edf") for sn in ["SN1", "SN2", "SN3", "SN4", "SN5"]]
        cmap = dict(BASE_MAP); cmap["C3"] = C3_VARIANT["dup"]
    else:
        ids = [l.strip() for l in (HIER.parent / "unet50" / "ids_val.txt").read_text().splitlines() if l.strip()]
        recs = [(r, MESA_EDF / f"{r}.edf") for r in ids]; cmap = MESA_MAP
    config = load_config(MODEL); dev = torch.device(a.device)
    model = initialize_model(config)
    model.load_state_dict(torch.load(MODEL / "weights.pth", map_location="cpu", weights_only=True))
    model.to(dev); model.device = dev; model.eval()
    ld = torch.tensor(model.localizations_default_expanded).to(dev)
    log = {"cohort": a.cohort, "channel_map": cmap, "thresholds": THRS, "per_rec": {}}
    for rec, edf in recs:
        t0 = time.time()
        try:
            data = process_file(edf, cmap)
        except Exception as e:  # noqa: BLE001
            print(rec, "overgeslagen:", e, flush=True); log["per_rec"][rec] = {"error": repr(e)}; continue
        if a.cohort == "psgipa":
            hypno = json.loads((REF / f"{rec}_hypno.json").read_text())["hypno"]
        else:
            hypno, ar = parse_mesa_xml(MESA_XML / f"{rec}-nsrr.xml", data.shape[-1] / FS)
            write_pred(out / f"{rec}_ref.csv", ar)
        t_off = 0.0
        if a.crop_sleep:
            sl = [i for i, st in enumerate(hypno) if st in ("N1", "N2", "N3", "R")]
            t0s = max(0.0, sl[0] * 30.0 - 300.0); t1s = min(data.shape[-1] / FS, (sl[-1] + 1) * 30.0 + 300.0)
            data = data[:, int(t0s * FS):int(t1s * FS)]
            data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
            t_off = t0s
        ws = model.window_size; stride = ws // 2
        x_all = einops.rearrange(torch.tensor(data, dtype=torch.float32).unfold(-1, ws, stride), "C N T -> N C T")
        masks = {t: np.zeros(data.shape[-1], dtype=np.uint8) for t in THRS}
        with torch.no_grad():
            for i in range(0, x_all.shape[0], a.batch):
                loc, clf = model(x_all[i:i + a.batch].to(dev))
                for t in THRS:
                    model.detector.classification_threshold = t
                    for w_off, w_events in enumerate(model.detector(loc, clf, ld)):
                        w_idx = i + w_off
                        for ev in w_events:
                            if ev[-1] != 0:      # alleen arousal (klasse-index 0, zie run_msed.py)
                                continue
                            s = int(ev[0] * ws) + w_idx * stride; e = int(ev[1] * ws) + w_idx * stride
                            masks[t][max(0, s):e] = 1
        rij = {"runtime_s": round(time.time() - t0, 1)}
        for t in THRS:
            evs = gate_sleep([(s / FS + t_off, e / FS + t_off) for s, e in binary_to_array(masks[t])], hypno)
            write_pred(out / f"{rec}_t{t:.2f}.csv", evs); rij[f"{t:.2f}"] = len(evs)
        log["per_rec"][rec] = rij
        print(rec, rij, flush=True)
    (out / "run_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
