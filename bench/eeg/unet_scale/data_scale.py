"""Data-laag voor de schaalmeting (preregistratie docs/arousal_unet_schaal_preregistratie_20261025.md).

Zelfde voorbewerking als `../unet50/data.py` (anti-alias + resample 50 Hz, lopend
gemiddelde en RMS over 18 min weg, clip ±20, float16), nu voor MESA én SHHS1 en met een
optioneel flowkanaal. Per nacht blijven ALLE kandidaat-kanalen in het geheugen
(EEG-varianten, EOG L/R, EMG, flowvarianten) met een indexkaart, zodat de
kanaalkeuze-augmentatie per venster kan kiezen. Niets van MESA/SHHS gaat naar schijf.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import csv, json, sys, warnings
from pathlib import Path
import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent / "unet50")); sys.path.insert(0, str(HIER.parent / "common"))
from data import (normalize, parse_mesa_xml, mask_from_events, FS, EPOCH_S, SLEEP,  # noqa: E402
                  MESA_EDF, MESA_XML, PSGIPA)
warnings.filterwarnings("ignore")
import mne  # noqa: E402
mne.set_log_level("ERROR")

SHHS_EDF = Path("/srv/DATA/SHHS/shhs/polysomnography/edfs/shhs1")
SHHS_XML = Path("/srv/DATA/SHHS/shhs/polysomnography/annotations-events-nsrr/shhs1")
BENCH_REF = HIER.parent / "ref"

# (kanaalnaam, gewicht) voor EEG; (kanaalnaam, soort) voor flow. Volgorde = voorkeur bij
# validatie/inferentie (eerste aanwezige). Flow: druk vóór thermistor.
MONTAGE = {
    "mesa": {"eeg": [("EEG3", 0.5), ("EEG1", 0.25), ("EEG2", 0.25)], "eog": ["EOG-L", "EOG-R"],
             "emg": "EMG", "flow": [("Pres", "pressure"), ("Therm", "thermistor")]},
    "shhs": {"eeg": [("EEG", 0.5), ("EEG(sec)", 0.5)], "eog": ["EOG(L)", "EOG(R)"],
             "emg": "EMG", "flow": [("NEW AIR", "thermistor"), ("AIRFLOW", "thermistor")]},
}


def _resolve(name: str, ch_names: list[str]) -> str | None:
    """Exacte naam, of de eerste MNE-hernoemde duplicaat ('AIRFLOW' -> 'AIRFLOW-0')."""
    if name in ch_names:
        return name
    for c in ch_names:
        if c.startswith(name + "-") and c[len(name) + 1:].isdigit():
            return c
    return None


def _edf_xml(cohort: str, rec: str) -> tuple[Path, Path]:
    if cohort == "mesa":
        return MESA_EDF / f"{rec}.edf", MESA_XML / f"{rec}-nsrr.xml"
    if cohort == "shhs":
        return SHHS_EDF / f"{rec}.edf", SHHS_XML / f"{rec}-nsrr.xml"
    raise ValueError(cohort)


def _read(edf: Path, names: list[str]) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
    raw.pick(names)
    raw.load_data()
    dur = raw.n_times / raw.info["sfreq"]
    raw.resample(FS, verbose=False)
    X = np.stack([normalize(raw.get_data(picks=[n])[0]) for n in names]).astype(np.float16)
    return X, dur


def _finish(rec: str, cohort: str, X, dur, names, groups, hypno, arousals) -> dict:
    sl = [i for i, s in enumerate(hypno) if s in SLEEP]
    if len(sl) < 60 or len(arousals) == 0:
        return {"rec": rec, "cohort": cohort,
                "error": f"te weinig slaap ({len(sl)} epochs) of geen arousals ({len(arousals)})"}
    flow_pref = None
    if groups["flow"]:
        pr = [i for i, k in groups["flow"] if k == "pressure"]
        flow_pref = pr[0] if pr else groups["flow"][0][0]
    val_ch = [groups["eeg"][0][0], groups["eog"][0], groups["emg"]] + ([flow_pref] if flow_pref is not None else [])
    return {"rec": rec, "cohort": cohort, "x": X, "y": mask_from_events(arousals, X.shape[1]),
            "hypno": hypno, "arousals": arousals, "dur": dur, "names": names,
            "sleep_span": (sl[0] * EPOCH_S, (sl[-1] + 1) * EPOCH_S),
            "eeg_idx": np.array([i for i, _ in groups["eeg"]]),
            "eeg_w": np.array([w for _, w in groups["eeg"]]) / sum(w for _, w in groups["eeg"]),
            "eog_idx": np.array(groups["eog"]), "emg_idx": int(groups["emg"]),
            "flow_idx": np.array([i for i, _ in groups["flow"]], dtype=int),
            "flow_kinds": [k for _, k in groups["flow"]], "val_ch": val_ch,
            "flow_pref_kind": (dict(groups["flow"]).get(flow_pref) if flow_pref is not None else None)}


def load_night(spec: tuple) -> dict:
    """spec = (cohort, rec, with_flow). cohort in {mesa, shhs, psgipa}."""
    cohort, rec, with_flow = spec
    try:
        if cohort == "psgipa":
            return load_psgipa_night(rec, with_flow)
        edf, xml = _edf_xml(cohort, rec)
        m = MONTAGE[cohort]
        hdr = mne.io.read_raw_edf(str(edf), preload=False, verbose=False).ch_names
        names: list[str] = []; groups = {"eeg": [], "eog": [], "emg": None, "flow": []}
        for n, w in m["eeg"]:
            r = _resolve(n, hdr)
            if r:
                groups["eeg"].append((len(names), w)); names.append(r)
        for n in m["eog"]:
            r = _resolve(n, hdr)
            if r:
                groups["eog"].append(len(names)); names.append(r)
        r = _resolve(m["emg"], hdr)
        if not groups["eeg"] or not groups["eog"] or r is None:
            return {"rec": rec, "cohort": cohort, "error": f"ontbrekend EEG/EOG/EMG in {hdr}"}
        groups["emg"] = len(names); names.append(r)
        if with_flow:
            for n, kind in m["flow"]:
                r = _resolve(n, hdr)
                if r:
                    groups["flow"].append((len(names), kind)); names.append(r)
            if not groups["flow"]:
                return {"rec": rec, "cohort": cohort, "error": f"geen flowkanaal in {hdr}"}
        X, dur = _read(edf, names)
        hypno, ar = parse_mesa_xml(xml, dur)
        return _finish(rec, cohort, X, dur, names, groups, hypno, ar)
    except Exception as e:  # noqa: BLE001
        return {"rec": rec, "cohort": cohort, "error": repr(e)}


def load_psgipa_night(sn: str, with_flow: bool) -> dict:
    edf = PSGIPA / f"{sn}_Respiration.edf"
    hdr = mne.io.read_raw_edf(str(edf), preload=False, verbose=False).ch_names
    eeg = next(c for c in ("EEG C4-M1", "EEG Cz-M1", "EEG C3-M2") if c in hdr)
    names = [eeg, "EOG E1-M2", "EMG chin"]
    groups = {"eeg": [(0, 1.0)], "eog": [1], "emg": 2, "flow": []}
    if with_flow:
        groups["flow"] = [(3, "pressure")]; names.append("Resp nasal")
    X, dur = _read(edf, names)
    meta = json.loads((BENCH_REF / f"{sn}_hypno.json").read_text())
    ar = []
    with (BENCH_REF / f"{sn}_ref_arousal.csv").open() as fh:
        for row in csv.DictReader(fh):
            ar.append((float(row["onset_s"]), float(row["offset_s"])))
    ar.sort()
    return _finish(sn, "psgipa", X, dur, names, groups, meta["hypno"], ar)


def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
