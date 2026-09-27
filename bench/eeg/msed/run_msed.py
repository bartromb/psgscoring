#!/usr/bin/env python3
"""MSED (Zahid e.a., IEEE TBME 2023, MIT) — voorgetraind `splitstream`, PSG-IPA.

    bench/eeg/_venv/bin/python bench/eeg/msed/run_msed.py [--variant dup|f4] [--device cuda]

Gebruikt upstream `msed.preprocessing.process_file` (resample 128 Hz, filters,
z-score per kanaal) en `predict_events.initialize_model` ongewijzigd; alleen de
kanaalkeuze en de inferentielus staan hier, omdat upstream zijn channel_map.json
NAAST de EDF's schrijft (dat zou onder /srv/DATA zijn) en interactief vraagt.

MSED wil tien kanalen: C3, C4, EOGL, EOGR, Chin, LegL, LegR, NasalP, Thor, Abdo
(A1/A2/EOGRef/ChinRef optioneel). PSG-IPA heeft één centrale afleiding:
  variant dup (PRIMAIR, vooraf gekozen): C3 := C4-M1 (Cz-M1 op SN5) — twee
      centrale kanalen zoals MrOS (C3-A2/C4-A1), zonder nieuwe informatie;
  variant f4 (gevoeligheid): C3 := F4-M1 — extra informatie, andere regio.
Drempel: die van upstream zelf (results_eval.json: arousal 0,64, op hún
evaluatieset geoptimaliseerd), dus géén drempelkeuze op PSG-IPA. Daarna
alleen de slaappoort van bench/eeg/common/postproc.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import torch
import einops

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent / "common"))
from postproc import gate_sleep, write_pred  # noqa: E402
from msed.preprocessing import process_file, FS  # noqa: E402
from msed.functions import binary_to_array  # noqa: E402
from msed.predict_events import initialize_model  # noqa: E402
from msed.utils.config import load_config  # noqa: E402

ROOT = Path("/srv/DATA/PSG-IPA/Resp_events/PSG")
REF = HIER.parent / "ref"
MODEL = HIER / "upstream" / "models" / "splitstream"

BASE_MAP = {
    "A1": [], "A2": [],
    "C4": ["EEG C4-M1", "EEG Cz-M1"],
    "EOGL": ["EOG E1-M2"], "EOGR": ["EOG E2-M2"], "EOGRef": [],
    "Chin": ["EMG chin"], "ChinRef": [],
    "LegL": ["EMG LAT"], "LegR": ["EMG RAT"],
    "NasalP": ["Resp nasal"], "Thor": ["Resp chest"], "Abdo": ["Resp abdomen"],
}
C3_VARIANT = {"dup": ["EEG C4-M1", "EEG Cz-M1"], "f4": ["EEG F4-M1"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dup", choices=sorted(C3_VARIANT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sns", default="SN1,SN2,SN3,SN4,SN5")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    out = HIER / f"out_{a.variant}"; out.mkdir(exist_ok=True)
    cmap = dict(BASE_MAP); cmap["C3"] = C3_VARIANT[a.variant]
    config = load_config(MODEL)
    dev = torch.device(a.device)
    model = initialize_model(config)
    model.load_state_dict(torch.load(MODEL / "weights.pth", map_location="cpu", weights_only=True))
    model.to(dev); model.device = dev; model.eval()
    thr = config.threshold
    class_names = list(thr.keys())
    log = {"variant": a.variant, "threshold": thr, "channel_map": cmap, "per_sn": {}}
    for sn in a.sns.split(","):
        t0 = time.time()
        data = process_file(ROOT / f"{sn}_Respiration.edf", cmap)   # (10, T) @128 Hz
        meta = json.loads((REF / f"{sn}_hypno.json").read_text())
        window_size = model.window_size; stride = int(0.5 * window_size)
        x_all = einops.rearrange(torch.tensor(data, dtype=torch.float32).unfold(-1, window_size, stride), "C N T -> N C T")
        preds = []
        with torch.no_grad():
            for i in range(0, x_all.shape[0], a.batch):
                preds.extend(model.predict(x_all[i:i + a.batch].to(dev)))
        mask = np.zeros((len(thr), data.shape[-1]), dtype=np.uint8)
        for w_idx, w_events in enumerate(preds):
            for ev in w_events:
                s = int(ev[0] * window_size) + w_idx * stride
                e = int(ev[1] * window_size) + w_idx * stride
                # Detection.forward levert class_index-1 (arousal=0, lm=1, sdb=2);
                # upstream predict_events.py schrijft ev[-1]-1 en zet arousals zo in
                # de LAATSTE rij (sdb). Gecontroleerd op SN5: klasse-0-events vallen
                # op de referentie-arousals (bv. 515-528 s vs 517-533 s).
                mask[ev[-1], s:e] = 1
        rij = {"runtime_s": round(time.time() - t0, 1), "n_windows": int(x_all.shape[0])}
        for ci, ev_name in enumerate(class_names):
            evs = [(s / FS, e / FS) for s, e in binary_to_array(mask[ci])]
            rij[ev_name] = {"raw": len(evs)}
            if ev_name == "arousal":
                gated = gate_sleep(evs, meta["hypno"])
                write_pred(out / f"{sn}_pred.csv", gated)
                write_pred(out / f"{sn}_pred_ungated.csv", evs)
                rij[ev_name]["gated"] = len(gated)
                durs = [e - s for s, e in gated]
                rij[ev_name]["dur_median_s"] = float(np.median(durs)) if durs else None
            else:
                write_pred(out / f"{sn}_{ev_name}.csv", evs, typ=ev_name)
        log["per_sn"][sn] = rij
        print(sn, rij, flush=True)
    (out / "run_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
