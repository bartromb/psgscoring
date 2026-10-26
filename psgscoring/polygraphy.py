"""Analysevenster op signaalgeldigheid voor de polygrafieroute (v0.37.0, waakpoort stap 1).

Een polygrafie heeft geen EEG, dus geen hypnogram: de app gaf elke epoch "N2" en deelde door de
registratietijd. De hPG-simulatie op MESA (`docs/hpg_simulatie_20261026.md`) liet zien wat dat
kost: 56 % van de events van die route begint in een waak-epoch, twee derde daarvan ná het laatste
slaapepoch — de recorder loopt door nadat de slaap en vaak de sensor voorbij zijn. Stap 0 van de
waakpoort (`docs/hpg_simulatie_20261025/waakpoort_stap0.py`) telde: 59 % van die waak-events ligt
op los signaal (SpO2 buiten bereik, flow vlak). Deze module bouwt daar het venster voor; de
preregistratie met de meting op verse nachten staat in `docs/waakpoort_preregistratie_20261026.md`.

Regel (vastgelegd vóór de meting, drempels uit stap 0, niet bijgesteld):
  - per 30 s-epoch: ``flow_dead`` = IQR (p75 − p25) van elk bruikbaar flowkanaal ≤ 2 % van de
    mediane epoch-IQR van dat kanaal over de hele opname, of exact 0; ``spo2_invalid`` =
    epochmediaan buiten 50–100 % of minder dan de helft van de samples in dat bereik;
    ``sensor_off = flow_dead ∨ spo2_invalid``; een flowkanaal waarvan de mediane IQR 0 is
    (de hele nacht dood) telt niet mee;
  - venster = van het begin van de eerste tot het einde van de laatste run van ≥ 4 geldige
    epochs (2 min); binnen het venster vallen runs van ≥ 6 opeenvolgende ``sensor_off``-epochs
    (3 min) eruit — een centrale apneu haalt nooit 3 min;
  - hypnogram: "N2" voor geanalyseerde epochs, "W" voor de rest; een handmatig venster
    (lights off/on) zet epochs daarbuiten óók op "W". De bestaande slaappoort en TST-noemer van
    de keten doen de rest: dit is geen tweede poortmechanisme.

Wat dit NIET doet: waak met geldig signaal herkennen (inslapen, WASO) — daarvoor is een
slaap/waak-proxy nodig, apart te pre-registreren.
"""
from __future__ import annotations

import numpy as np

EPOCH_S = 30.0
DEAD_FRAC = 0.02
MIN_VALID_RUN = 4
MIN_DEAD_RUN = 6
SPO2_RANGE = (50.0, 100.0)
SPO2_MIN_FRAC = 0.5
FLOW_ROLES = ("flow_pressure", "flow_thermistor")
EFFORT_ROLES = ("thorax", "abdomen")

__all__ = ["epoch_validity", "analysis_window", "polygraphy_hypnogram",
           "polygraphy_hypnogram_from_raw", "EPOCH_S", "DEAD_FRAC", "MIN_VALID_RUN",
           "MIN_DEAD_RUN", "SPO2_RANGE", "SPO2_MIN_FRAC"]


def _epochs(x: np.ndarray, fs: float, n_epochs: int, epoch_s: float) -> np.ndarray:
    """(n_epochs, k) met NaN waar de opname korter is."""
    k = int(round(float(fs) * epoch_s))
    x = np.asarray(x, dtype=np.float64).ravel()
    need = n_epochs * k
    if x.size < need:
        x = np.concatenate([x, np.full(need - x.size, np.nan)])
    return x[:need].reshape(n_epochs, k)


def _epoch_iqr(x: np.ndarray, fs: float, n_epochs: int, epoch_s: float) -> np.ndarray:
    m = _epochs(x, fs, n_epochs, epoch_s)
    k = m.shape[1]
    with np.errstate(all="ignore"):
        q = np.nanpercentile(m, [25, 75], axis=1)
    iqr = q[1] - q[0]
    te_kort = np.sum(~np.isnan(m), axis=1) < max(1, k // 2)
    iqr[te_kort] = np.nan
    return iqr


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def epoch_validity(signals: dict, n_epochs: int, epoch_s: float = EPOCH_S,
                   dead_frac: float = DEAD_FRAC, spo2_range=SPO2_RANGE,
                   spo2_min_frac: float = SPO2_MIN_FRAC) -> dict:
    """Per epoch: ``flow_dead``, ``spo2_invalid``, ``effort_dead``, ``sensor_off`` (bool-arrays),
    plus ``reference_iqr`` per kanaal en ``used``/``ignored`` kanaalrollen.

    ``signals``: rol → (array, fs) met rollen uit ``utils.detect_channels``
    (``flow_pressure``, ``flow_thermistor``, ``spo2``, ``thorax``, ``abdomen``); ontbrekende
    rollen zijn toegestaan. Een epoch zonder data (NaN) geldt als dood/ongeldig.
    """
    n = int(n_epochs)
    ref: dict = {}
    dead: dict = {}
    ignored: list[str] = []
    for rol in FLOW_ROLES + EFFORT_ROLES:
        if rol not in signals:
            continue
        x, fs = signals[rol]
        iqr = _epoch_iqr(x, fs, n, epoch_s)
        r = float(np.nanmedian(iqr)) if np.isfinite(iqr).any() else 0.0
        ref[rol] = r
        if not (r > 0):
            ignored.append(rol)
            continue
        dead[rol] = np.isnan(iqr) | (iqr <= dead_frac * r) | (iqr == 0)
    flow_used = [r for r in FLOW_ROLES if r in dead]
    effort_used = [r for r in EFFORT_ROLES if r in dead]
    flow_dead = (np.logical_and.reduce([dead[r] for r in flow_used]) if flow_used
                 else np.zeros(n, dtype=bool))
    effort_dead = (np.logical_and.reduce([dead[r] for r in effort_used]) if effort_used
                   else np.zeros(n, dtype=bool))
    spo2_used = "spo2" in signals
    if spo2_used:
        m = _epochs(*signals["spo2"], n, epoch_s)
        lo, hi = spo2_range
        with np.errstate(all="ignore"):
            med = np.nanmedian(m, axis=1)
            ok = np.nanmean(((m >= lo) & (m <= hi)).astype(float), axis=1)
        spo2_invalid = np.isnan(med) | (med < lo) | (med > hi) | (ok < spo2_min_frac)
    else:
        spo2_invalid = np.zeros(n, dtype=bool)
    return {"flow_dead": flow_dead.astype(bool), "spo2_invalid": spo2_invalid.astype(bool),
            "effort_dead": effort_dead.astype(bool),
            "sensor_off": (flow_dead | spo2_invalid).astype(bool),
            "reference_iqr": ref, "flow_used": flow_used, "effort_used": effort_used,
            "spo2_used": spo2_used, "ignored": ignored}


def analysis_window(sensor_off: np.ndarray, min_valid_run: int = MIN_VALID_RUN,
                    min_dead_run: int = MIN_DEAD_RUN) -> dict:
    """Regel B: venster van de eerste tot de laatste run van ≥ ``min_valid_run`` geldige epochs;
    binnen het venster vallen runs van ≥ ``min_dead_run`` dode epochs eruit."""
    off = np.asarray(sensor_off, dtype=bool)
    n = off.size
    valid_runs = [r for r in _runs(~off) if r[1] - r[0] >= min_valid_run]
    analysed = np.zeros(n, dtype=bool)
    if not valid_runs:
        return {"analysed": analysed, "start_epoch": 0, "end_epoch": 0, "excluded_runs": [],
                "reason": "geen run van geldige epochs gevonden — niets te analyseren"}
    a0, a1 = valid_runs[0][0], valid_runs[-1][1]
    analysed[a0:a1] = True
    excluded = []
    for r in _runs(off & analysed):
        if r[1] - r[0] >= min_dead_run:
            analysed[r[0]:r[1]] = False
            excluded.append((int(r[0]), int(r[1])))
    return {"analysed": analysed, "start_epoch": int(a0), "end_epoch": int(a1),
            "excluded_runs": excluded, "reason": None}


def polygraphy_hypnogram(signals: dict, n_epochs: int, *, manual_window_s=None,
                         epoch_s: float = EPOCH_S, dead_frac: float = DEAD_FRAC,
                         min_valid_run: int = MIN_VALID_RUN, min_dead_run: int = MIN_DEAD_RUN,
                         spo2_range=SPO2_RANGE, spo2_min_frac: float = SPO2_MIN_FRAC):
    """(hypnogram, provenance): "N2" voor geanalyseerde epochs, "W" daarbuiten.

    Zonder bruikbaar flow- én SpO2-kanaal is er geen venster (alles "N2", ``method`` "none").
    ``manual_window_s=(t0, t1)`` (lights off/on, seconden vanaf het begin van de opname) zet
    epochs daarbuiten op "W", bovenop het automatische venster.
    """
    n = int(n_epochs)
    thresholds = {"dead_frac": dead_frac, "min_valid_run": min_valid_run,
                  "min_dead_run": min_dead_run, "spo2_range": list(spo2_range),
                  "spo2_min_frac": spo2_min_frac, "epoch_s": epoch_s}
    prov: dict = {"method": "signal_validity_window", "reason": None, "n_epochs": n,
                  "epoch_s": epoch_s, "thresholds": thresholds, "manual_window_s": None,
                  "channels": {"flow": [], "spo2": False, "effort": [], "ignored": []},
                  "window_start_s": None, "window_end_s": None, "excluded_runs_s": [],
                  "analysed_min": None,
                  "excluded_min": {"before_window": 0.0, "after_window": 0.0,
                                   "inside_window": 0.0, "manual": 0.0, "total": 0.0}}
    v = epoch_validity(signals or {}, n, epoch_s, dead_frac, spo2_range, spo2_min_frac)
    prov["channels"] = {"flow": v["flow_used"], "spo2": v["spo2_used"],
                        "effort": v["effort_used"], "ignored": v["ignored"]}
    analysed = np.ones(n, dtype=bool)
    if not v["flow_used"] and not v["spo2_used"]:
        prov["method"] = "none"
        prov["reason"] = ("geen bruikbaar flow- of SpO2-kanaal — geen analysevenster, "
                          "alle epochs geanalyseerd (registratietijd als noemer)")
    else:
        w = analysis_window(v["sensor_off"], min_valid_run, min_dead_run)
        analysed = w["analysed"]
        prov["reason"] = w["reason"]
        prov["window_start_s"] = w["start_epoch"] * epoch_s
        prov["window_end_s"] = w["end_epoch"] * epoch_s
        prov["excluded_runs_s"] = [[a * epoch_s, b * epoch_s] for a, b in w["excluded_runs"]]
        mins = epoch_s / 60.0
        prov["excluded_min"]["before_window"] = w["start_epoch"] * mins
        prov["excluded_min"]["after_window"] = (n - w["end_epoch"]) * mins
        prov["excluded_min"]["inside_window"] = sum(b - a for a, b in w["excluded_runs"]) * mins
    if manual_window_s is not None:
        t0, t1 = float(manual_window_s[0]), float(manual_window_s[1])
        e0 = max(0, int(np.floor(t0 / epoch_s)))
        e1 = min(n, int(np.ceil(t1 / epoch_s)))
        buiten = np.ones(n, dtype=bool)
        buiten[e0:e1] = False
        prov["manual_window_s"] = [t0, t1]
        prov["excluded_min"]["manual"] = float(buiten.sum()) * epoch_s / 60.0
        analysed = analysed & ~buiten
    hypno = ["N2" if a else "W" for a in analysed]
    prov["analysed_min"] = float(analysed.sum()) * epoch_s / 60.0
    prov["excluded_min"]["total"] = float((~analysed).sum()) * epoch_s / 60.0
    return hypno, prov


def polygraphy_hypnogram_from_raw(raw, channel_map: dict | None = None, *, n_epochs: int | None = None,
                                  manual_window_s=None, epoch_s: float = EPOCH_S, **kw):
    """Zelfde als ``polygraphy_hypnogram`` op een MNE-Raw: kanaalrollen uit ``channel_map``
    (rol → kanaalnaam, zoals ``run_pneumo_analysis``) of anders ``utils.detect_channels``."""
    from .utils import detect_channels
    roles = dict(channel_map or {})
    if not roles:
        roles = detect_channels(list(raw.ch_names))
    sf = float(raw.info["sfreq"])
    signals = {}
    for rol in FLOW_ROLES + ("spo2",) + EFFORT_ROLES:
        naam = roles.get(rol)
        if naam and naam in raw.ch_names:
            signals[rol] = (raw.get_data(picks=[naam])[0], sf)
    if n_epochs is None:
        n_epochs = int((raw.n_times / sf) // epoch_s)
    hypno, prov = polygraphy_hypnogram(signals, n_epochs, manual_window_s=manual_window_s,
                                       epoch_s=epoch_s, **kw)
    prov["channel_names"] = {rol: roles.get(rol) for rol in signals}
    return hypno, prov
