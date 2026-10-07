"""Bevroren 1D-U-Net-arousaldetector ``unet_v1`` (opt-in, v0.35.0).

Eigen BSD-schone herimplementatie naar Ehrlich e.a. 2024 (Sci Rep 14), getraind op
317 verse MESA-nachten (seed 20260927) en op 27-09-2026 volgens preregistratie
gerepliceerd: SHHS1 150 verse nachten arousal-F1 0,543 → 0,676 (134/147,
p = 2e-22), MESA-76 0,556 → 0,687, PSG-IPA 0,555 → 0,745 op 5/5
(`docs/arousal_unet_replicatie_20260927.md`). Dit bestand bevat ALLEEN de
inferentieketen; het gewicht staat als ONNX in ``psgscoring/data/arousal_unet_v1.onnx``
met de checksums in ``arousal_unet_v1.json`` (bron `bench/eeg/unet50/model_best.pt`,
sha256 cc91bb86…). Geen torch in de bibliotheek: ``onnxruntime`` onder de
``[ml]``-extra.

Keten (identiek aan de replicatie, bevroren):
  1. elk kanaal (EEG, EOG, kin-EMG) naar 50 Hz met ``mne.filter.resample``
     (dezelfde FFT-resampler als ``Raw.resample``);
  2. per kanaal lopend gemiddelde en RMS over 18 min verwijderd
     (``uniform_filter1d``, mode "nearest"), clip ±20, float16-rondgang;
  3. U-Net → kans per sample;
  4. 1 s lopend gemiddelde, drempel τ (default 0,35), gaten ≤ 1 s samengevoegd,
     events ≥ 3 s, slaappoort (onset in een slaap-epoch);
  5. de 10 s-samenvoeging van de huidige keten (``enforce_min_arousal_interval``).
Daarna neemt ``run_arousal_respiratory_analysis`` het over (onsetverschuiving,
koppeling, RERA). Zonder EOG of kin-EMG valt de keten terug op de LGBM-detector
(alleen-EEG was in de replicatie slechter dan de baseline: 0,513 < 0,568).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FS = 50
PAD_MULTIPLE = 128
NORM_WIN_S = 18 * 60
DEFAULT_THRESHOLD = 0.35
DATA_DIR = Path(__file__).parent / "data"
ONNX_PATH = DATA_DIR / "arousal_unet_v1.onnx"
META_PATH = DATA_DIR / "arousal_unet_v1.json"
SLEEP = {"N1", "N2", "N3", "R"}

_SESSION = None
_SESSION_THREADS = None


def meta() -> dict:
    return json.loads(META_PATH.read_text())


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for blok in iter(lambda: fh.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def available() -> tuple[bool, str | None]:
    """(beschikbaar, reden-indien-niet). Controleert onnxruntime, het bestand en
    de checksum uit de metadata — een stil ander gewicht mag niet."""
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"onnxruntime niet beschikbaar ({e.__class__.__name__}); installeer psgscoring[ml]"
    if not ONNX_PATH.exists() or not META_PATH.exists():
        return False, "gewicht of metadata ontbreekt in psgscoring/data"
    if os.environ.get("PSGSCORING_UNET_SKIP_SHA") != "1":
        verwacht = meta().get("onnx_sha256")
        if sha256_of(ONNX_PATH) != verwacht:
            return False, "checksum van arousal_unet_v1.onnx klopt niet met de metadata"
    return True, None


def _session(threads: int | None = None):
    global _SESSION, _SESSION_THREADS
    import onnxruntime as ort
    n = int(threads or os.environ.get("PSGSCORING_UNET_THREADS") or 4)
    if _SESSION is None or _SESSION_THREADS != n:
        so = ort.SessionOptions()
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
        _SESSION = ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])
        _SESSION_THREADS = n
    return _SESSION


def normalize(x: np.ndarray, fs: float = FS, win_s: float = NORM_WIN_S) -> np.ndarray:
    """Lopend gemiddelde en RMS over ``win_s`` verwijderd, clip ±20, float16-rondgang
    (de bench sloeg de genormaliseerde invoer als float16 op)."""
    from scipy.ndimage import uniform_filter1d
    n = int(win_s * fs)
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    mu = uniform_filter1d(x, n, mode="nearest")
    x = x - mu
    rms = np.sqrt(uniform_filter1d(x * x, n, mode="nearest")) + 1e-7
    return np.clip(x / rms, -20, 20).astype(np.float16).astype(np.float32)


def to_50hz(x: np.ndarray, sf: float) -> np.ndarray:
    """Naar 50 Hz met dezelfde resampler als ``mne.io.Raw.resample``."""
    x = np.asarray(x, dtype=np.float64)
    if abs(float(sf) - FS) < 1e-9:
        return x
    from mne.filter import resample
    return resample(x, up=float(FS), down=float(sf), npad="auto", window="auto",
                    pad="auto", method="fft", verbose=False)


def prepare_input(eeg, sf_eeg, eog, sf_eog, emg, sf_emg) -> np.ndarray:
    """(3, n) float32 op 50 Hz: EEG, EOG, kin-EMG — in die volgorde."""
    kan = [normalize(to_50hz(eeg, sf_eeg)), normalize(to_50hz(eog, sf_eog)),
           normalize(to_50hz(emg, sf_emg))]
    n = min(k.size for k in kan)
    return np.stack([k[:n] for k in kan]).astype(np.float32)


def predict(X: np.ndarray, threads: int | None = None) -> np.ndarray:
    """Kans per sample (50 Hz) voor X van vorm (3, n); de nacht in één pas."""
    n = X.shape[-1]
    pad = (-n) % PAD_MULTIPLE
    if pad:
        X = np.pad(X, ((0, 0), (0, pad)))
    p = _session(threads).run(None, {"x": X[None].astype(np.float32)})[0][0]
    return p[:n].astype(np.float32)


def moving_average(p: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return p
    c = np.cumsum(np.insert(p.astype(np.float64), 0, 0.0))
    out = (c[n:] - c[:-n]) / n
    pad_l = (n - 1) // 2
    pad_r = n - 1 - pad_l
    return np.concatenate([np.full(pad_l, out[0]), out, np.full(pad_r, out[-1])])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def prob_to_events(p: np.ndarray, fs: float, thr: float, min_dur_s: float = 3.0,
                   merge_gap_s: float = 1.0, smooth_s: float = 1.0) -> list[tuple[float, float]]:
    ps = moving_average(p, int(round(smooth_s * fs))) if smooth_s > 0 else p
    merged: list[list[int]] = []
    gap = merge_gap_s * fs
    for a, b in _runs(ps >= thr):
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a / fs, b / fs) for a, b in merged if (b - a) / fs >= min_dur_s]


def gate_sleep(events, hypno: list, epoch_s: float = 30.0):
    out = []
    for a, b in events:
        ep = int(a // epoch_s)
        if 0 <= ep < len(hypno) and hypno[ep] in SLEEP:
            out.append((a, b))
    return out


def detect_arousals_unet(eeg, sf_eeg, eog, sf_eog, emg, sf_emg, hypno: list, *,
                         threshold: float = DEFAULT_THRESHOLD,
                         min_interval_s: float = 0.0,
                         artifact_epochs=None,
                         threads: int | None = None) -> dict:
    """Zelfde vorm als ``detect_arousals``: {"success", "events", "summary", "error"}."""
    from .arousal import _recompute_arousal_summary, enforce_min_arousal_interval
    t0 = time.time()
    try:
        X = prepare_input(eeg, sf_eeg, eog, sf_eog, emg, sf_emg)
        t1 = time.time()
        p = predict(X, threads=threads)
        t_fwd = time.time() - t1
        raw_ev = gate_sleep(prob_to_events(p, FS, float(threshold)), hypno)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "events": [], "summary": {}, "error": f"unet_v1: {e!r}"}
    events = []
    for a, b in raw_ev:
        ep = int(a // 30.0)
        i0, i1 = int(a * FS), max(int(b * FS), int(a * FS) + 1)
        events.append({
            "onset_s": round(float(a), 3), "end_s": round(float(b), 3),
            "duration_s": round(float(b - a), 3),
            "stage": hypno[ep] if 0 <= ep < len(hypno) else "W",
            "derivation": "unet_v1",
            "confidence": round(float(np.max(p[i0:i1])), 3),
        })
    stats: dict = {}
    if min_interval_s and min_interval_s > 0:
        events = enforce_min_arousal_interval(events, float(min_interval_s), stats)
    art = set(int(i) for i in (artifact_epochs or []))
    summary = _recompute_arousal_summary(events, hypno, art)
    summary.update({
        "detector": "unet_v1", "unet_threshold": float(threshold),
        "unet_n_raw": len(raw_ev), "min_interval_s": float(min_interval_s or 0.0),
        "n_interval_merged": int(stats.get("n_merged", 0)),
        "unet_onnx_sha256": meta().get("onnx_sha256"),
        "unet_t_total_s": round(time.time() - t0, 2), "unet_t_forward_s": round(t_fwd, 2),
        "unet_n_samples": int(X.shape[-1]),
    })
    logger.info("[arousal] unet_v1: %d events (%d vóór samenvoeging) op τ %.2f, "
                "%.1f s (voorwaarts %.1f s)", len(events), len(raw_ev),
                float(threshold), time.time() - t0, t_fwd)
    return {"success": True, "events": events, "summary": summary, "error": None}
