"""Data-laag voor de schone U-Net-herimplementatie (Ehrlich e.a. 2024, Sci Rep).

Drie kanalen à 50 Hz: EEG, EOG, kin-EMG. Voorbewerking exact zoals het artikel
beschrijft: anti-alias-FIR + decimatie naar 50 Hz (mne.resample), daarna per
kanaal het gemiddelde en de RMS over een lopend venster van 18 min
verwijderen. Labels: NSRR-arousals als masker per sample. MESA wordt TER
PLEKKE gelezen en alleen in het geheugen gehouden -- er wordt niets onder
bench/ of /srv/DATA geschreven.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import mne
mne.set_log_level("ERROR")
from scipy.ndimage import uniform_filter1d

FS = 50
EPOCH_S = 30.0
MESA_EDF = Path("/srv/DATA/MESA/mesa/polysomnography/edfs")
MESA_XML = Path("/srv/DATA/MESA/mesa/polysomnography/annotations-events-nsrr")
PSGIPA = Path("/srv/DATA/PSG-IPA/Resp_events/PSG")
STAGE_MAP = {"wake": "W", "stage 1 sleep": "N1", "stage 2 sleep": "N2",
             "stage 3 sleep": "N3", "stage 4 sleep": "N3", "rem sleep": "R"}
SLEEP = {"N1", "N2", "N3", "R"}
# MESA: EEG1 = Fz-Cz, EEG2 = Cz-Oz, EEG3 = C4-M1 (NSRR-montagedocumentatie).
MESA_CH = ["EEG3", "EEG1", "EEG2", "EOG-L", "EOG-R", "EMG"]
MESA_EEG_IDX = [0, 1, 2]     # kanaalkeuze-augmentatie: 0 (C4-M1) krijgt gewicht 0,5
MESA_EOG_IDX = [3, 4]
MESA_EMG_IDX = 5


def normalize(x: np.ndarray, fs: float = FS, win_s: float = 18 * 60) -> np.ndarray:
    n = int(win_s * fs)
    x = x.astype(np.float32)
    mu = uniform_filter1d(x, n, mode="nearest")
    x = x - mu
    rms = np.sqrt(uniform_filter1d(x * x, n, mode="nearest")) + 1e-7
    return np.clip(x / rms, -20, 20)


def load_channels(edf: Path, names: list[str]) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(str(edf), include=names, preload=True, verbose=False)
    missing = [n for n in names if n not in raw.ch_names]
    if missing:
        raise ValueError(f"{edf.name}: ontbrekende kanalen {missing}")
    dur = raw.n_times / raw.info["sfreq"]
    raw.resample(FS, verbose=False)
    X = np.stack([normalize(raw.get_data(picks=[n])[0]) for n in names]).astype(np.float16)
    return X, dur


def parse_mesa_xml(xml: Path, dur_s: float):
    root = ET.parse(str(xml)).getroot()
    n_ep = int(math.ceil(dur_s / EPOCH_S))
    hypno = ["W"] * n_ep
    arousals = []
    for ev in root.iter("ScoredEvent"):
        concept = (ev.findtext("EventConcept") or "").split("|")[0].strip().lower()
        et = (ev.findtext("EventType") or "").lower()
        try:
            start = float(ev.findtext("Start")); d = float(ev.findtext("Duration"))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(start) and np.isfinite(d)) or start < 0 or start >= dur_s:
            continue
        st = STAGE_MAP.get(concept)
        if st is not None:
            ep0 = int(start // EPOCH_S)
            for i in range(max(1, int(round(d / EPOCH_S)))):
                if 0 <= ep0 + i < n_ep:
                    hypno[ep0 + i] = st
        elif "arousal" in et or "arousal" in concept:
            arousals.append((start, min(start + d, dur_s)))
    arousals.sort()
    return hypno, arousals


def mask_from_events(events, n: int, fs: float = FS) -> np.ndarray:
    y = np.zeros(n, dtype=np.int8)
    for a, b in events:
        y[int(a * fs): max(int(a * fs) + 1, int(round(b * fs)))] = 1
    return y


def load_mesa_night(rec: str) -> dict | None:
    try:
        X, dur = load_channels(MESA_EDF / f"{rec}.edf", MESA_CH)
    except Exception as e:  # noqa: BLE001
        return {"rec": rec, "error": repr(e)}
    hypno, ar = parse_mesa_xml(MESA_XML / f"{rec}-nsrr.xml", dur)
    sl = [i for i, s in enumerate(hypno) if s in SLEEP]
    if len(sl) < 60 or len(ar) == 0:
        return {"rec": rec, "error": f"te weinig slaap ({len(sl)} epochs) of geen arousals ({len(ar)})"}
    return {"rec": rec, "x": X, "y": mask_from_events(ar, X.shape[1]), "hypno": hypno,
            "arousals": ar, "sleep_span": (sl[0] * EPOCH_S, (sl[-1] + 1) * EPOCH_S), "dur": dur}


def load_psgipa(sn: str) -> tuple[np.ndarray, float, list[str]]:
    hdr = mne.io.read_raw_edf(str(PSGIPA / f"{sn}_Respiration.edf"), preload=False, verbose=False)
    eeg = next(c for c in ("EEG C4-M1", "EEG Cz-M1", "EEG C3-M2") if c in hdr.ch_names)
    names = [eeg, "EOG E1-M2", "EMG chin"]
    X, dur = load_channels(PSGIPA / f"{sn}_Respiration.edf", names)
    return X, dur, names
