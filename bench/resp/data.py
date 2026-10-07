"""MESA-nachten voor de respiratoire U-Net: 5 kanalen @ 8 Hz + NSRR-aasm15-labels. Signalen alleen in het geheugen."""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import mne
from scipy.ndimage import uniform_filter1d, maximum_filter1d
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parents[1] / "scripts")); sys.path.insert(0, str(HIER.parents[1]))
mne.set_log_level("ERROR")
FS = 8
EPOCH_S = 30.0
MESA_EDF = Path("/srv/DATA/MESA/mesa/polysomnography/edfs")
MESA_XML = Path("/srv/DATA/MESA/mesa/polysomnography/annotations-events-nsrr")
SLEEP = {"N1", "N2", "N3", "R"}
ROLES = ["flow_pressure", "flow_thermistor", "thorax", "abdomen", "spo2"]
REF = "aasm15"
APNEA_TYPES = {"obstructive", "central", "mixed", "apnea", "uncertain"}


def normalize_flow(x, fs=FS, win_s=18 * 60):
    n = int(win_s * fs); x = np.nan_to_num(x.astype(np.float32))
    x = x - uniform_filter1d(x, n, mode="nearest")
    rms = np.sqrt(uniform_filter1d(x * x, n, mode="nearest")) + 1e-7
    return np.clip(x / rms, -20, 20)


def normalize_spo2(x, fs=FS, win_s=10 * 60):
    x = x.astype(np.float32)
    if np.nanmedian(x) <= 1.5:
        x = x * 100.0
    bad = ~np.isfinite(x) | (x < 50.0)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():                       # voorwaarts vullen vanaf de eerste geldige waarde
        idx = np.where(~bad, np.arange(x.size), 0); np.maximum.accumulate(idx, out=idx); x = x[idx]
    top = maximum_filter1d(x, int(win_s * fs), mode="nearest")
    return np.clip((x - top) / 3.0, -10.0, 1.0)


def load_night(edf: Path, xml: Path):
    from psgscoring.utils import detect_channels
    from validate_mesa import parse_nsrr
    hdr = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
    ch = detect_channels(hdr.ch_names)
    names = {r: ch.get(r) for r in ROLES}
    present = [names[r] for r in ROLES if names[r] in hdr.ch_names]
    if not any(names[r] in hdr.ch_names for r in ("flow_pressure", "flow_thermistor")):
        raise ValueError(f"{edf.name}: geen flowkanaal ({hdr.ch_names})")
    raw = mne.io.read_raw_edf(str(edf), include=present, preload=True, verbose=False)
    dur = raw.n_times / raw.info["sfreq"]
    raw.resample(FS, verbose=False)
    n = raw.n_times
    X = np.zeros((len(ROLES), n), dtype=np.float32); mask = np.zeros(len(ROLES), dtype=bool)
    for i, r in enumerate(ROLES):
        nm = names[r]
        if nm in raw.ch_names:
            x = raw.get_data(picks=[nm])[0]
            X[i] = normalize_spo2(x) if r == "spo2" else normalize_flow(x); mask[i] = True
    hypno, refs, tst_h = parse_nsrr(xml, dur)
    ev = [(a, b, str(t)) for a, b, t in refs.get(REF, [])]
    y = np.zeros((2, n), dtype=np.int8)
    for a, b, t in ev:
        k = 0 if (t.split("|")[0].strip().lower() in APNEA_TYPES or ("apnea" in t.lower() and "hypopnea" not in t.lower())) else 1
        y[k, int(a * FS): max(int(a * FS) + 1, int(round(b * FS)))] = 1
    return {"x": X.astype(np.float16), "mask": mask, "y": y, "hypno": hypno, "events": ev, "dur": dur, "tst_h": tst_h,
            "channels": names}


def load_mesa_night(rec: str):
    try:
        d = load_night(MESA_EDF / f"{rec}.edf", MESA_XML / f"{rec}-nsrr.xml")
    except Exception as e:  # noqa: BLE001
        return {"rec": rec, "error": repr(e)}
    sl = [i for i, s in enumerate(d["hypno"]) if s in SLEEP]
    if len(sl) < 120:
        return {"rec": rec, "error": f"te weinig slaap ({len(sl)} epochs)"}
    d.update(rec=rec, sleep_span=(sl[0] * EPOCH_S, (sl[-1] + 1) * EPOCH_S))
    return d
