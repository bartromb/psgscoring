#!/usr/bin/env python3
"""
diag_emg_feature_distribution.py — D2, meetstap: schuift `emg_var_ratio` op
een REFERENTIEEL kin-EMG weg van de trainingsverdeling?

WAAROM
------
Het gebundelde model splitst 486 keer op `emg_var_ratio`, in 279 van de 500
bomen, met alle drempels boven nul. Het is getraind op MESA, waar de chin-EMG
**bipolair** is. Een klinische export met een **referentieel** kanaal (EMG1
tegen een mastoïd) heeft een andere variantiekarakteristiek en draagt
ECG-contaminatie mee. Schuift de verdeling van dat ene feature systematisch,
dan verschuift de hele kansverdeling en stapelt dat op een vast werkpunt.

Dit script MEET dat, meer niet. Het verandert niets en beveelt niets aan.

Per opname: de kandidatenlijst zoals de hybride hem opbouwt (ruime drempels),
daarna dezelfde featurebouwer als de filter gebruikt -- niet een nabouw, want
dan meet je je eigen nabouw.

Gebruik:
    python scripts/diag_emg_feature_distribution.py \
        --clinical REC.edf --ctx ctx.json --mesa-n 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
mne.set_log_level("ERROR")

from psgscoring.arousal import (
    _AROUSAL_LGBM_FEATURE_ORDER,
    AROUSAL_LGBM_CAND_ABRUPT,
    AROUSAL_LGBM_CAND_RATIO,
    _arousal_lgbm_features,
    _load_arousal_lgbm_booster,
    detect_arousals,
)


def _uv(x):
    x = np.asarray(x, dtype=float).copy()
    return x * 1e6 if np.max(np.abs(x)) < 0.01 else x


def features_for(eeg, emg, sf, hypno, artifact_epochs=None):
    """Kandidaten + hun featurerijen, langs exact het pad van de filter."""
    res = detect_arousals(
        eeg, sf, hypno, emg_data=emg, artifact_epochs=artifact_epochs,
        ratio_thresh=AROUSAL_LGBM_CAND_RATIO,
        abrupt_thresh=AROUSAL_LGBM_CAND_ABRUPT)
    cands = res.get("events") or []
    if not cands:
        return [], None, None
    eeg_uv = _uv(eeg)
    emg_uv = _uv(emg[:len(eeg)]) if emg is not None else None
    rows = [_arousal_lgbm_features(c, eeg_uv, sf, emg_uv, len(hypno))
            for c in cands]
    vr = np.array([r["emg_var_ratio"] for r in rows], dtype=float)
    booster = _load_arousal_lgbm_booster()
    X = np.array([[r[c] for c in _AROUSAL_LGBM_FEATURE_ORDER] for r in rows],
                 dtype=float)
    proba = np.asarray(booster.predict(X), dtype=float)
    return cands, vr, proba


def _summarise(naam, vr, proba):
    q = np.percentile(vr, [10, 25, 50, 75, 90])
    print(f"{naam:<22}{len(vr):>6}"
          f"{q[0]:>9.2f}{q[1]:>8.2f}{q[2]:>8.2f}{q[3]:>8.2f}{q[4]:>9.2f}"
          f"{np.mean(vr <= 1.0)*100:>9.1f}"
          f"{np.median(proba):>9.3f}{np.mean(proba >= 0.80)*100:>8.1f}",
          flush=True)
    return {"n": len(vr), "p10": q[0], "p25": q[1], "p50": q[2],
            "p75": q[3], "p90": q[4],
            "pct_le_1": float(np.mean(vr <= 1.0) * 100),
            "proba_med": float(np.median(proba)),
            "pct_keep_080": float(np.mean(proba >= 0.80) * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--mesa-dir", default="/home/bart/MESA/mesa")
    ap.add_argument("--mesa-n", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    kop = (f"{'opname':<22}{'kand':>6}{'p10':>9}{'p25':>8}{'p50':>8}"
           f"{'p75':>8}{'p90':>9}{'%<=1':>9}{'p50 proba':>9}{'%>=.8':>8}")
    print(kop); print("-" * len(kop), flush=True)
    uit = {}

    # ── klinisch, referentieel EMG ────────────────────────────────────
    ctx = json.loads(Path(a.ctx).read_text())
    hypno = ctx["hypno"]
    hdr = mne.io.read_raw_edf(a.clinical, preload=False, verbose=False)
    eeg_naam = ctx.get("eeg_ch")
    emg_naam = ctx.get("emg_ch")
    raw = mne.io.read_raw_edf(
        a.clinical,
        exclude=[c for c in hdr.ch_names if c not in {eeg_naam, emg_naam}],
        preload=True, verbose=False)
    sf = raw.info["sfreq"]
    _c, vr, pr = features_for(raw.get_data(picks=[eeg_naam])[0],
                              raw.get_data(picks=[emg_naam])[0],
                              sf, hypno, ctx.get("art_epochs"))
    if vr is not None:
        uit["klinisch (referentieel)"] = _summarise("klinisch (ref.)", vr, pr)

    # ── MESA, bipolair EMG ────────────────────────────────────────────
    from validate_mesa import parse_nsrr
    edfs = sorted((Path(a.mesa_dir) / "polysomnography" / "edfs")
                  .glob("mesa-sleep-*.edf"))[:a.mesa_n]
    xmls = Path(a.mesa_dir) / "polysomnography" / "annotations-events-nsrr"
    alle_vr, alle_pr = [], []
    for edf in edfs:
        xml = xmls / f"{edf.stem}-nsrr.xml"
        if not xml.exists():
            continue
        h = mne.io.read_raw_edf(edf, preload=False, verbose=False)
        if "EMG" not in h.ch_names or "EEG3" not in h.ch_names:
            continue
        dur = h.n_times / h.info["sfreq"]
        hyp, _r, _t = parse_nsrr(xml, dur)
        r2 = mne.io.read_raw_edf(
            edf, exclude=[c for c in h.ch_names if c not in {"EEG3", "EMG"}],
            preload=True, verbose=False)
        _c, vr2, pr2 = features_for(r2.get_data(picks=["EEG3"])[0],
                                    r2.get_data(picks=["EMG"])[0],
                                    r2.info["sfreq"], hyp)
        if vr2 is None:
            continue
        alle_vr.append(vr2); alle_pr.append(pr2)
        _summarise(edf.stem.replace("mesa-sleep-", "MESA "), vr2, pr2)
    if alle_vr:
        uit["MESA (bipolair, gepoold)"] = _summarise(
            "MESA gepoold (bip.)", np.concatenate(alle_vr),
            np.concatenate(alle_pr))

    print("\nDe kolom die telt is %<=1: het aandeel kandidaten waar het EMG "
          "tijdens\nde arousal NIET meer varieert dan ervoor. Het model splitst "
          "486 keer op\ndit feature en al zijn drempels liggen boven nul.",
          flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(uit, indent=2, default=float))


if __name__ == "__main__":
    main()
