"""
arousal.py — Arousal- & RERA-detectie (psgscoring)

Ported into psgscoring from YASAFlaskified `arousal_analysis.py` so the library
is self-contained. Single-channel detection is byte-identical to the source;
multi-derivation is layered on top. Pure numpy/scipy — no Flask, no i18n.

Conform AASM Adult Scoring Manual, Chapter 5 (Arousals):
  - Arousal: abrupte EEG-frequentieverandering ≥3s (α/θ/β in NREM; α in REM)
    voorafgegaan door ≥10s stabiele slaap
  - Respiratoire arousal: arousal binnen 15s na einde apnea/hypopnea
  - RERA: flow-limitatie + arousal ZONDER apnea/hypopnea drempel te bereiken
  - Arousal-index: aantal arousals per uur slaap (normaal < 10–15/u)

Klinisch verband:
  apnea/hypopnea → hypoxie/hypercapnie/mechanische load → cortical arousal
  → slaapfragmentatie → overmatige slaperigheid overdag (EDS)
  → cardiovasculaire stress (nachtelijke bloeddrukpieken)
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from .indices import per_hour
from scipy.signal import find_peaks  # v0.8.40: consolidated imports

logger = logging.getLogger("psgscoring.arousal")


# ═══════════════════════════════════════════════════════════════
# v0.9.8: LightGBM candidate-level re-classifier
# Enabled per request via env var YASAFLASKIFIED_AROUSAL_LGBM=1.
# When enabled, detect_arousals runs with permissive thresholds
# (ratio=1.2, abrupt=1.0) to maximise candidate recall, then
# filters surviving candidates with a LightGBM model trained on
# MESA q∈{5,6} (n_subj=653, ~562k candidates) at probability
# threshold AROUSAL_LGBM_THRESHOLD (default 0.60).
# Backward-compat: when the env var is absent or "0", behaviour is
# bit-identical to the v0.8.40 rule-based detector.
# ═══════════════════════════════════════════════════════════════

AROUSAL_LGBM_MODEL_PATH = os.environ.get(
    "PSGSCORING_AROUSAL_LGBM_MODEL",
    str(Path(__file__).with_name("data") / "arousal_classifier_v3.txt"),
)
AROUSAL_LGBM_THRESHOLD = float(
    os.environ.get(
        "PSGSCORING_AROUSAL_LGBM_THRESHOLD",
        os.environ.get("YASAFLASKIFIED_AROUSAL_LGBM_THRESHOLD", "0.60"),
    )
)
# Permissive candidate-stage thresholds used only in hybrid mode.
# v0.23.1: minimale samplefrequentie voor het hybride pad.
#
# Het model gebruikt bandvermogens tot en met beta (16-30 Hz) en een spectrale
# randfrequentie. Ligt Nyquist onder 30 Hz, dan bestaan die kenmerken niet in
# het signaal en krijgt een model dat op 256 Hz getraind is een gedegenereerde
# vector. Het verwerpt dan ALLES: op de golden-fixture met EEG op 32 Hz gingen
# 23 kandidaten naar 0, met kansen van 0,012 tot 0,066 tegen een drempel van
# 0,60. Op een profiel waar hypopneus alleen via een arousal kwalificeren
# betekent dat AHI 0 -- stil, zonder fout.
#
# 64 Hz is de ondergrens waarbij de betaband nog volledig representeerbaar is.
AROUSAL_LGBM_MIN_SF      = 64.0

AROUSAL_LGBM_CAND_RATIO  = 1.2
AROUSAL_LGBM_CAND_ABRUPT = 1.0

# Feature column order — must match arousal_classifier_v3_report.json
# feature_names list. Changing this list invalidates the bundled model.
_AROUSAL_LGBM_FEATURE_ORDER: list[str] = [
    "duration_s", "stage_code", "stage_n1", "stage_n2", "stage_n3",
    "stage_rem", "dom_band_code", "alpha_ratio", "beta_ratio",
    "onset_ratio", "emg_confirmed", "cvr_boost",
    "bp_pre_delta", "bp_pre_theta", "bp_pre_alpha", "bp_pre_sigma", "bp_pre_beta",
    "bp_cand_delta", "bp_cand_theta", "bp_cand_alpha", "bp_cand_sigma", "bp_cand_beta",
    "bp_post_delta", "bp_post_theta", "bp_post_alpha", "bp_post_sigma", "bp_post_beta",
    "ratio_alpha_to_sigma", "ratio_beta_to_sigma", "ratio_arousal_to_sigma",
    "delta_alpha_pre_cand", "delta_beta_pre_cand", "delta_total_pre_cand",
    "hj_pre_act", "hj_pre_mob", "hj_pre_comp",
    "hj_cand_act", "hj_cand_mob", "hj_cand_comp",
    "hj_post_act", "hj_post_mob", "hj_post_comp",
    "spec_edge_95", "spec_edge_50",
    "td_cand_var", "td_cand_kurt", "td_cand_zcr",
    "emg_var_ratio", "hr_shift_rel", "pos_in_night",
]
_AROUSAL_LGBM_BOOSTER = None  # cached after first use


def _emg_usable_for_lgbm(emg_data, n_eeg: int) -> bool:
    """Kan het model iets met dit EMG-kanaal, of degenereert het?

    Het gebundelde model splitst 486 keer op ``emg_var_ratio``, verdeeld over
    279 van de 500 bomen, en ALLE drempels liggen boven nul (min 0,0157,
    mediaan 1,86, max 884). Op gain is het feature nummer vier.

    ``_arousal_lgbm_features()`` zet dat feature op een constante 0,0 zodra
    het EMG ontbreekt, korter is dan het EEG, of geen variantie heeft. Elke
    kandidaat gaat dan in alle 486 splits dezelfde kant op; de kansverdeling
    schuift als geheel omlaag en op een VAST werkpunt blijft er een fractie
    over van wat er met EMG overblijft. De combinatie die daarbij ontstaat --
    ``emg_confirmed`` gezet, ``emg_var_ratio`` nul -- komt in de trainingsdata
    niet voor.

    Dit is dezelfde faalwijze als de lage-samplefrequentie-degeneratie die
    ``AROUSAL_LGBM_MIN_SF`` afvangt; op de EMG-as ontbrak de guard.
    """
    if emg_data is None:
        return False
    arr = np.asarray(emg_data)
    if arr.size < n_eeg:
        # _arousal_lgbm_features() eist emg_uv.size >= eeg_uv.size en valt
        # anders terug op 0.0 -- een kort kanaal is hier geen kanaal.
        return False
    return float(np.var(arr)) > 0.0


def _is_arousal_lgbm_enabled() -> bool:
    return (
        os.environ.get(
            "PSGSCORING_AROUSAL_LGBM",
            os.environ.get("YASAFLASKIFIED_AROUSAL_LGBM", "0"),
        )
        == "1"
    )


def _load_arousal_lgbm_booster():
    global _AROUSAL_LGBM_BOOSTER
    if _AROUSAL_LGBM_BOOSTER is not None:
        return _AROUSAL_LGBM_BOOSTER
    if not Path(AROUSAL_LGBM_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Arousal LGBM model not found at {AROUSAL_LGBM_MODEL_PATH}; "
            "ship arousal_classifier_v3.txt into myproject/data/."
        )
    import lightgbm as lgb  # noqa: WPS433
    _AROUSAL_LGBM_BOOSTER = lgb.Booster(model_file=AROUSAL_LGBM_MODEL_PATH)
    return _AROUSAL_LGBM_BOOSTER


def _bandpower_window(seg: np.ndarray, sf: float, lo: float, hi: float) -> float:
    """Welch-free FFT bandpower estimate over `seg` (µV)."""
    if seg.size < int(sf):
        return 0.0
    n = seg.size
    win = np.hanning(n)
    fx = np.fft.rfft(seg * win)
    f = np.fft.rfftfreq(n, 1.0 / sf)
    psd = (np.abs(fx) ** 2) / (sf * np.sum(win ** 2))
    if n % 2:
        psd[1:] *= 2
    else:
        psd[1:-1] *= 2
    band = (f >= lo) & (f < hi)
    if not np.any(band):
        return 0.0
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrate(psd[band], f[band]))


def _hjorth_window(seg: np.ndarray) -> tuple[float, float, float]:
    if seg.size < 3:
        return (0.0, 0.0, 0.0)
    var0 = float(np.var(seg))
    if var0 <= 0:
        return (0.0, 0.0, 0.0)
    d1 = np.diff(seg)
    var1 = float(np.var(d1))
    mob = float(np.sqrt(var1 / var0)) if var1 > 0 else 0.0
    if d1.size < 3 or mob <= 0:
        return (var0, mob, 0.0)
    d2 = np.diff(d1)
    var2 = float(np.var(d2))
    mob1 = float(np.sqrt(var2 / var1)) if var1 > 0 else 0.0
    return (var0, mob, mob1 / mob if mob > 0 else 0.0)


def _spectral_edge_window(seg: np.ndarray, sf: float, edge_pct: float) -> float:
    if seg.size < int(sf):
        return 0.0
    n = seg.size
    win = np.hanning(n)
    fx = np.fft.rfft(seg * win)
    f = np.fft.rfftfreq(n, 1.0 / sf)
    psd = (np.abs(fx) ** 2)
    if n % 2:
        psd[1:] *= 2
    else:
        psd[1:-1] *= 2
    band = (f >= 0.5) & (f <= 30.0)
    if not np.any(band):
        return 0.0
    p = psd[band]
    fb = f[band]
    cum = np.cumsum(p)
    if cum[-1] <= 0:
        return 0.0
    idx = int(np.searchsorted(cum, edge_pct * cum[-1]))
    return float(fb[min(idx, fb.size - 1)])


def _arousal_lgbm_features(
    cand: dict,
    eeg_uv: np.ndarray, sf: float,
    emg_uv: np.ndarray | None,
    n_epochs: int,
) -> dict:
    """Build the 50-feature vector for one candidate. Mirrors exactly
    the layout in build_arousal_dataset.py used to train v3."""
    onset_s = float(cand["onset_s"])
    end_s   = float(cand["end_s"])
    onset_i = int(onset_s * sf)
    end_i   = int(end_s * sf)
    pre_i   = max(0, onset_i - int(5.0 * sf))
    post_i  = min(eeg_uv.size, end_i + int(5.0 * sf))

    pre  = eeg_uv[pre_i:onset_i]
    cand_seg = eeg_uv[onset_i:end_i]
    post = eeg_uv[end_i:post_i]

    stage = cand.get("stage", "W")
    stage_codes = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
    band_codes  = {"alpha": 0, "theta": 1, "beta": 2}
    ep_idx = int(cand.get("epoch", int(onset_s // 30)))

    eps = 1e-12
    out: dict[str, float] = {
        "duration_s":     float(cand.get("duration_s", end_s - onset_s)),
        "stage_code":     float(stage_codes.get(stage, 0)),
        "stage_n1":       float(stage == "N1"),
        "stage_n2":       float(stage == "N2"),
        "stage_n3":       float(stage == "N3"),
        "stage_rem":      float(stage == "R"),
        "dom_band_code":  float(band_codes.get(cand.get("dominant_band", "alpha"), 0)),
        "alpha_ratio":    float(cand.get("alpha_ratio", 0.0)),
        "beta_ratio":     float(cand.get("beta_ratio", 0.0)),
        "onset_ratio":    float(cand.get("onset_ratio", 0.0)),
        "emg_confirmed":  float(bool(cand.get("emg_confirmed", False))),
        "cvr_boost":      float(cand.get("cvr_boost", 0.0)),
    }

    for label_, seg in (("pre", pre), ("cand", cand_seg), ("post", post)):
        if seg.size < int(sf):
            for b in ("delta", "theta", "alpha", "sigma", "beta"):
                out[f"bp_{label_}_{b}"] = 0.0
            continue
        out[f"bp_{label_}_delta"] = _bandpower_window(seg, sf, 0.5, 4.0)
        out[f"bp_{label_}_theta"] = _bandpower_window(seg, sf, 4.0, 8.0)
        out[f"bp_{label_}_alpha"] = _bandpower_window(seg, sf, 8.0, 11.0)
        out[f"bp_{label_}_sigma"] = _bandpower_window(seg, sf, 12.0, 15.0)
        out[f"bp_{label_}_beta"]  = _bandpower_window(seg, sf, 16.0, 30.0)

    out["ratio_alpha_to_sigma"]   = out["bp_cand_alpha"] / (out["bp_cand_sigma"] + eps)
    out["ratio_beta_to_sigma"]    = out["bp_cand_beta"]  / (out["bp_cand_sigma"] + eps)
    out["ratio_arousal_to_sigma"] = (out["bp_cand_alpha"] + out["bp_cand_theta"]
                                     + out["bp_cand_beta"]) / (out["bp_cand_sigma"] + eps)
    out["delta_alpha_pre_cand"]   = (out["bp_cand_alpha"] - out["bp_pre_alpha"]) \
                                    / (out["bp_pre_alpha"] + eps)
    out["delta_beta_pre_cand"]    = (out["bp_cand_beta"]  - out["bp_pre_beta"]) \
                                    / (out["bp_pre_beta"]  + eps)
    out["delta_total_pre_cand"]   = ((out["bp_cand_alpha"] + out["bp_cand_theta"]
                                      + out["bp_cand_beta"])
                                     - (out["bp_pre_alpha"] + out["bp_pre_theta"]
                                        + out["bp_pre_beta"])) \
                                    / (out["bp_pre_alpha"] + out["bp_pre_theta"]
                                       + out["bp_pre_beta"] + eps)

    for label_, seg in (("pre", pre), ("cand", cand_seg), ("post", post)):
        a, m, c = _hjorth_window(seg)
        out[f"hj_{label_}_act"]  = a
        out[f"hj_{label_}_mob"]  = m
        out[f"hj_{label_}_comp"] = c

    out["spec_edge_95"] = _spectral_edge_window(cand_seg, sf, 0.95)
    out["spec_edge_50"] = _spectral_edge_window(cand_seg, sf, 0.50)

    if cand_seg.size >= 3:
        out["td_cand_var"] = float(np.var(cand_seg))
        m_  = float(np.mean(cand_seg))
        sd_ = float(np.std(cand_seg))
        out["td_cand_kurt"] = float(np.mean(((cand_seg - m_) / (sd_ + eps)) ** 4) - 3) if sd_ > 0 else 0.0
        out["td_cand_zcr"]  = float(np.mean(np.diff(np.signbit(cand_seg)) != 0))
    else:
        out["td_cand_var"]  = 0.0
        out["td_cand_kurt"] = 0.0
        out["td_cand_zcr"]  = 0.0

    if emg_uv is not None and emg_uv.size >= eeg_uv.size:
        emg_pre  = emg_uv[pre_i:onset_i]
        emg_cand = emg_uv[onset_i:end_i]
        v_pre  = float(np.var(emg_pre))  if emg_pre.size  >= 3 else 0.0
        v_cand = float(np.var(emg_cand)) if emg_cand.size >= 3 else 0.0
        out["emg_var_ratio"] = v_cand / (v_pre + eps)
    else:
        out["emg_var_ratio"] = 0.0

    out["hr_shift_rel"] = 0.0  # v3 trained without HR feature populated
    out["pos_in_night"] = ep_idx / max(1, n_epochs - 1)

    return out


def _filter_candidates_with_lgbm(
    events: list[dict],
    eeg_uv: np.ndarray, sf: float,
    emg_uv: np.ndarray | None,
    n_epochs: int,
    threshold: float = AROUSAL_LGBM_THRESHOLD,
    thresholds: list[float] | None = None,
) -> tuple[list[dict], list[float]]:
    """Score each candidate with the LGBM model and return the events
    with ``proba >= threshold``, plus the per-candidate probabilities.

    ``thresholds`` geeft een drempel PER kandidaat en wint van ``threshold``.
    Dat is wat het event-locked werkpunt nodig heeft: het model levert een
    kans, en de drempel waarop je die afkapt hoort van de prior af te hangen.
    Vlak na een respiratoir event-einde is die prior aantoonbaar anders.
    """
    if not events:
        return [], []
    booster = _load_arousal_lgbm_booster()
    feat_rows = [_arousal_lgbm_features(c, eeg_uv, sf, emg_uv, n_epochs) for c in events]
    X = np.array([[r[c] for c in _AROUSAL_LGBM_FEATURE_ORDER] for r in feat_rows],
                 dtype=float)
    proba = booster.predict(X)
    thr_per = thresholds if thresholds is not None else [threshold] * len(events)
    kept = []
    for ev, p, thr in zip(events, proba, thr_per):
        if p >= thr:
            ev = dict(ev)
            ev["lgbm_proba"] = round(float(p), 4)
            ev["lgbm_threshold_used"] = round(float(thr), 4)
            ev["event_locked"] = bool(thr < threshold)
            kept.append(ev)
    return kept, [float(p) for p in proba]


def _recompute_arousal_summary(
    arousals: list[dict], hypno: list, artifact_set: set,
) -> dict:
    """Rebuild the summary fields after LGBM filtering."""
    total_sleep_s = sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                        if _is_sleep(s) and i not in artifact_set)
    # Geen ondergrens op de noemer: zie psgscoring/indices.py. Een index
    # zonder slaaptijd bestaat niet en wordt None, niet aantal x 1000.
    total_sleep_h = total_sleep_s / 3600
    rem_h  = sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                 if _is_rem(s) and i not in artifact_set) / 3600
    nrem_h = sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                 if _is_nrem(s) and i not in artifact_set) / 3600
    nrem_ar = [a for a in arousals if _is_nrem(a.get("stage", "W"))]
    rem_ar  = [a for a in arousals if _is_rem(a.get("stage", "W"))]
    # Eén decimaal. Deze index staat in het rapport, en een arousal-index is
    # niet tot op 0,001/u te kennen: hij hangt af van waar een scorer de grens
    # van een arousal legt. In 0.14.7 stond hier per ongeluk 3 (de vervanging
    # van _safe, dat op 1 afrondde), waardoor het rapport 57,906/u toonde naast
    # buurwaarden met één decimaal. Niet terugzetten zonder reden.
    ai = per_hour(len(arousals), total_sleep_h)
    return {
        "n_arousals":         len(arousals),
        "arousal_index":      ai,
        "nrem_arousal_index": per_hour(len(nrem_ar), nrem_h),
        "rem_arousal_index":  per_hour(len(rem_ar), rem_h),
        "avg_duration_s":     _safe(float(np.mean([a["duration_s"]
                                       for a in arousals]))) if arousals else None,
        "severity":           _classify_arousal_index(ai),
        "n_theta_dominant":   sum(1 for a in arousals if a.get("dominant_band") == "theta"),
        "n_alpha_dominant":   sum(1 for a in arousals if a.get("dominant_band") == "alpha"),
        "n_beta_dominant":    sum(1 for a in arousals if a.get("dominant_band") == "beta"),
        "n_emg_confirmed":    sum(1 for a in arousals if a.get("emg_confirmed")),
    }


# ═══════════════════════════════════════════════════════════════
# CONSTANTEN  (AASM, v0.8.11 — verbeterd)
# ═══════════════════════════════════════════════════════════════

AROUSAL_MIN_DUR_S     = 3.0     # ≥3s EEG-frequentieverandering
AROUSAL_MAX_DUR_S     = 30.0    # >30s = waarschijnlijk wakker
PRESLEEP_MIN_S        = 10.0    # ≥10s slaap vóór arousal vereist
#: AASM: twee arousals moeten door >=10 s slaap gescheiden zijn. Dezelfde 10 s
#: als PRESLEEP_MIN_S en niet toevallig: het IS die regel, alleen toegepast op
#: de vorige arousal in plaats van op het hypnogram. `PRESLEEP_MIN_S` toetst of
#: er slaap-EPOCHS voorafgaan, en een epoch waarin net een arousal zat heet nog
#: steeds N2 -- die check ziet een vorige arousal dus niet.
AROUSAL_MIN_INTERVAL_S = 10.0
POST_RESP_WINDOW_S    = 15.0    # arousal binnen 15s na resp. event = respiratoir
# Zoveel VOOR het einde van een event begint het koppelvenster. Komt uit
# correlate_arousals_to_respiratory (window_pre_s); staat hier als constante
# zodat het event-locked werkpunt dezelfde geometrie gebruikt in plaats van
# een eigen getal.
AROUSAL_PRE_RESP_WINDOW_S = 5.0
RERA_FLOW_LIMIT_THR   = 0.80    # flow 80–100% = flow-limitatie (plateau)
RERA_MIN_DUR_S        = 10.0    # ≥10s flow-limitatie voor RERA
#: Koppelvenster voor RERA's. Stond hardgecodeerd op 10,0 in `detect_reras` en
#: op 15,0 in `_compute_rera_rdi` -- twee getallen voor dezelfde vraag, die
#: onafhankelijk konden verschuiven. Zie PostProcessingRules.rera_arousal_window_s.
RERA_AROUSAL_WINDOW_S = 15.0

# v0.8.11: Correcte frequentiebanden conform AASM
ALPHA_NARROW_BAND     = (8, 11)    # Alpha ZONDER spindle-overlap (was 8-13)
SIGMA_BAND            = (12, 15)   # Slaapspindels — UITSLUITEN uit arousal
THETA_BAND            = (4, 8)     # v0.8.11: NIEUW — theta-shift arousals
BETA_BAND             = (16, 30)   # >16 Hz (AASM definitie)
DELTA_BAND            = (0.5, 4)
ALPHA_BAND            = (8, 13)    # Breed alpha (voor backward compat in stats)

# v0.8.11: Drempels
AROUSAL_RATIO_THRESH  = 2.0     # v0.8.11: verlaagd van 3.0 → 2.0 (v0.8.11: verder verlaagd)
ABRUPT_RATIO_THRESH   = 1.5     # v0.8.11: verlaagd van 2.0 → 1.5 (2s FFT-vensters smoothen te veel)
EPOCH_LEN_S           = 30

# v0.23.0: spectrale-verschuivingscriterium (opt-in, `arousal_spectral_shift`).
#
# De regels hierboven vergelijken VERMOGEN in de snelle banden met een
# basislijn uit de opname zelf. De AASM beschrijft een verschuiving van de
# FREQUENTIE. Vermogen is onbegrensd en amplitude-gevoelig, dus betekent een
# vaste verhouding op de ene opname iets anders dan op de andere — gemeten op
# PSG-IPA: de drempel die de scoordermediaan reproduceert loopt van 1,2 tot
# 4,0 over vijf nachten.
#
# `r = (alpha + theta + beta) / (delta + alpha + theta + beta + sigma)` is de
# fractie van het spectrale vermogen in de snelle banden: dimensieloos,
# begrensd op [0,1] en invariant onder een amplitudeschaling van het EEG. Op
# een begrensde grootheid is een ABSOLUUT increment tussen opnames
# vergelijkbaar; op een onbegrensde vermogensmaat is het dat nooit.
#
# Waarden vastgelegd in docs/arousal_spectral_shift_preregistratie.md vóór
# enige meting, gekozen uit de grootheid en niet uit de data.
AROUSAL_SHIFT_DELTA   = 0.15    # r moet 0,15 absoluut boven de lokale basislijn
AROUSAL_SHIFT_ABRUPT  = 0.10    # r in de eerste 1 s ligt 0,10 boven de 3 s ervoor

# v0.23.0: hysterese (opt-in, `arousal_hysteresis`).
#
# Fase 1 bouwt de mask per sample en labelt die direct -- er wordt geen enkel
# gat gedicht. Bandvermogen fluctueert op subseconde-schaal, dus de mask
# flikkert en één arousal valt uiteen in scherven. Gemeten op MESA: 1897 ruwe
# regio's waarvan er 65 de 3 s-eis halen; de rest verdwijnt. Wat overblijft is
# niet de sterkste maar de toevallig langste aaneengesloten scherf, en de
# mediane eventduur (3,6 s) ligt daardoor op de ondergrens zelf -- tegen 8,6 s
# (PSG-IPA) en 11,0 s (MESA) bij menselijke scoorders.
#
# Hysterese is de standaardvorm voor een eventdetector en ligt dichter bij wat
# een scoorder doet: binnenkomen bij duidelijk verhoogde activiteit, doorlopen
# zolang ze verhoogd blijft. De INSTAPDREMPEL blijft exact `ratio_thresh`, dus
# deze vlag bepaalt alleen waar een event eindigt, niet of het begint.
#
# Waarde vastgelegd in docs/arousal_duration_preregistratie.md vóór de meting.
AROUSAL_EXIT_RATIO    = 1.2     # doorlopen zolang het vermogen 20% boven de vloer blijft


# ═══════════════════════════════════════════════════════════════
# HULPFUNCTIES
# ═══════════════════════════════════════════════════════════════

def _safe(val, dec=1):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), dec)
    except Exception:
        return None


def _bandpower_instant(eeg: np.ndarray, sf: float,
                        band: tuple, win_s: float = 2.0) -> np.ndarray:
    """
    Bereken instantaan bandvermogen via glijdend Welch-venster.
    Geeft een tijdreeks terug (één waarde per sample via interpolatie).
    """
    win   = int(win_s * sf)
    step  = max(1, win // 4)
    freqs = np.fft.rfftfreq(win, 1 / sf)
    lo, hi = band
    band_idx = (freqs >= lo) & (freqs <= hi)

    n_steps  = (len(eeg) - win) // step + 1
    powers   = np.zeros(n_steps)
    centers  = np.zeros(n_steps, dtype=int)

    for i in range(n_steps):
        s   = i * step
        e   = s + win
        seg = eeg[s:e] * np.hanning(win)
        psd = np.abs(np.fft.rfft(seg)) ** 2 / win
        powers[i]  = float(np.sum(psd[band_idx]))
        centers[i] = s + win // 2

    # Interpoleer terug naar sample-resolutie
    t_full  = np.arange(len(eeg))
    power_full = np.interp(t_full, centers, powers)
    return power_full


def _is_nrem(stage) -> bool:
    return stage in (1, 2, 3, "N1", "N2", "N3")


def _is_rem(stage) -> bool:
    return stage in (4, "R")


def _is_sleep(stage) -> bool:
    return stage not in (0, -1, "W")


def _build_stage_mask(hypno: list, sf: float,
                       total_samples: int, stages) -> np.ndarray:
    spe  = int(sf * EPOCH_LEN_S)
    mask = np.zeros(total_samples, dtype=bool)
    for ep_i, stage in enumerate(hypno):
        if stage in stages:
            s = ep_i * spe
            e = min(s + spe, total_samples)
            mask[s:e] = True
    return mask


# ═══════════════════════════════════════════════════════════════
# AROUSAL DETECTIE  (AASM spectrale methode)
# ═══════════════════════════════════════════════════════════════


def _is_kcomplex(
    eeg_uv: np.ndarray,
    onset_idx: int,
    sf: float,
    neg_thresh_uv: float = 75.0,
    window_s: float = 1.0,
) -> bool:
    """
    v0.8.11 — K-complex morfologische check.

    Een K-complex is een bipolaire golf: grote negatieve piek (<-75 µV)
    gevolgd door een positieve piek, alles binnen ~1 seconde.
    Als een arousal-kandidaat begint met zo\'n morfologie, is het
    waarschijnlijk een K-complex, geen echte arousal.

    Verhoog de min-duur lokaal naar 5.0 s om false positives te vermijden.
    Returns True als K-complex morfologie aanwezig is.
    """
    win = int(sf * window_s)
    end = min(onset_idx + win, len(eeg_uv))
    seg = eeg_uv[onset_idx:end]
    if len(seg) < int(sf * 0.3):
        return False
    min_val = float(np.min(seg))
    max_val = float(np.max(seg))
    min_idx = int(np.argmin(seg))
    max_idx = int(np.argmax(seg))
    # Typisch K-complex: negatieve piek gevolgd door positieve piek
    bipolaire_vorm = (
        min_val < -neg_thresh_uv and
        max_val > 30.0 and
        min_idx < max_idx       # negatief VOOR positief
    )
    return bipolaire_vorm


def _detect_cvr_confidence_boost(
    hr_data: np.ndarray | None,
    sf_hr: float,
    onset_s: float,
    pre_window_s: float = 10.0,
    post_window_s: float = 15.0,
    brady_delta_bpm: float = 5.0,
    tachy_delta_bpm: float = 10.0,
) -> float:
    """
    v0.8.11 — Autonome arousal confidence boost via Cyclical Variation of
    Heart Rate (CVR).

    Bij een respiratoir of corticaal event:
    - bradycardie tijdens het event (parasympathische activatie)
    - gevolgd door plotse tachycardie bij het einde (sympathische rebound)

    Als dit patroon aanwezig is naast een borderline EEG-arousal, verhogen
    we de confidence met 0.10–0.20.

    Returns
    -------
    boost : float (0.0 = geen patroon, 0.10–0.20 = aanwezig)
    """
    if hr_data is None or len(hr_data) == 0:
        return 0.0
    try:
        pre_start  = max(0, int((onset_s - pre_window_s) * sf_hr))
        pre_end    = max(0, int(onset_s * sf_hr))
        post_start = int(onset_s * sf_hr)
        post_end   = min(len(hr_data), int((onset_s + post_window_s) * sf_hr))

        if pre_end <= pre_start or post_end <= post_start:
            return 0.0

        hr_pre  = hr_data[pre_start:pre_end]
        hr_post = hr_data[post_start:post_end]

        # Verwijder fysiologisch onmogelijke waarden
        hr_pre  = hr_pre[(hr_pre > 20) & (hr_pre < 250)]
        hr_post = hr_post[(hr_post > 20) & (hr_post < 250)]

        if len(hr_pre) < 3 or len(hr_post) < 3:
            return 0.0

        mean_pre  = float(np.mean(hr_pre))
        mean_post = float(np.max(hr_post[:max(1, len(hr_post)//3)]))
        # Minimum van pre-venster (bradycardie)
        min_pre   = float(np.min(hr_pre))

        brady_present = (mean_pre - min_pre) >= brady_delta_bpm
        tachy_present = (mean_post - mean_pre) >= tachy_delta_bpm

        if brady_present and tachy_present:
            strength = min(1.0, (mean_post - min_pre) / 30.0)
            return round(0.10 + 0.10 * strength, 2)
        if tachy_present:
            return 0.10
        return 0.0
    except Exception:
        return 0.0

def detect_arousals(eeg_data: np.ndarray, sf: float,
                    hypno: list,
                    emg_data: np.ndarray = None,
                    artifact_epochs: list = None,
                    hr_data: np.ndarray = None,
                    sf_hr: float = 1.0,
                    ratio_thresh: float | None = None,
                    abrupt_thresh: float | None = None,
                    spectral_shift: bool = False,
                    shift_delta: float | None = None,
                    shift_abrupt: float | None = None,
                    hysteresis: bool = False,
                    exit_ratio: float | None = None,
                    lgbm: bool | None = None,
                    lgbm_threshold: float | None = None,
                    resp_event_ends: list | None = None,
                    event_locked_threshold: float | None = None,
                    min_interval_s: float = 0.0,
                    rem_alpha_baseline: bool = False,
                    _no_hybrid: bool = False) -> dict:
    """
    Detecteer EEG-arousals conform AASM, Sectie 5.

    v0.8.11 verbeteringen:
    1. Theta band (4-8 Hz) toegevoegd — veel arousals bij ouderen
    2. Alpha ingeperkt tot 8-11 Hz — voorkomt spindle vals-positieven
    3. Sigma band (12-15 Hz) apart gedetecteerd en UITGESLOTEN
    4. Abruptheid-criterium: vermogen moet >2× toenemen t.o.v. 3s ervoor
    5. Pre-sleep check valideert hypnogram (niet alleen arousal-vrij)
    6. Robuustere basislijn (mediaan van laagste 50% periodes)

    AASM definitie: "abrupte verschuiving van EEG-frequentie met inbegrip
    van alpha, theta en/of frequenties >16 Hz (maar niet slaapspindels),
    gedurende ≥3s, met ≥10s stabiele slaap voorafgaand."

    In REM: + EMG toename ≥1s.

    v0.9.8: when YASAFLASKIFIED_AROUSAL_LGBM=1 is set, the candidate
    stage runs with permissive thresholds (ratio=1.2, abrupt=1.0) and
    surviving candidates are filtered post-hoc by a LightGBM
    re-classifier at threshold AROUSAL_LGBM_THRESHOLD (default 0.60).
    Backward-compat: with the env var unset the function is
    bit-identical to the rule-based v0.8.40 detector.

    v0.23.0 (``spectral_shift=True``, opt-in): the power criterion is
    replaced by a criterion on the FAST-BAND FRACTION
    ``r = (alpha+theta+beta) / (delta+alpha+theta+beta+sigma)``, which is
    bounded and invariant under an amplitude scaling of the EEG — see the
    module constants and docs/arousal_spectral_shift_preregistratie.md.
    With ``spectral_shift=False`` (the default) this function is
    byte-identical to v0.22.0.

    v0.23.0 (``hysteresis=True``, opt-in): an event keeps running while the
    power stays above ``exit_ratio`` times the local baseline, instead of
    ending at the first sample that drops below the entry threshold. The
    entry threshold is unchanged, so this only moves event ENDS. See
    docs/arousal_duration_preregistratie.md.
    """
    # v0.9.8: hybrid mode dispatch — swap the candidate thresholds in
    # the module globals while the rule-based body runs, then filter
    # via LGBM after the function completes. The swap is restored in a
    # try/finally below so concurrent callers see the original values.
    # `lgbm=True` komt uit het profiel; de env-variabele blijft werken en
    # wint, zodat een installatie hem kan forceren of uitzetten.
    _env = os.environ.get("PSGSCORING_AROUSAL_LGBM",
                          os.environ.get("YASAFLASKIFIED_AROUSAL_LGBM"))
    if _env is not None:
        _want = _env == "1"
    else:
        _want = bool(lgbm) if lgbm is not None else False
    if event_locked_threshold is not None:
        _base = (float(lgbm_threshold) if lgbm_threshold is not None
                 else AROUSAL_LGBM_THRESHOLD)
        if float(event_locked_threshold) > _base:
            raise ValueError(
                f"event_locked_threshold ({event_locked_threshold}) is "
                f"strenger dan het gewone werkpunt ({_base}); het venster mag "
                f"alleen versoepelen -- een hogere drempel daar past de prior "
                f"omgekeerd toe. Kies een lager getal.")
    _hybrid_requested = _want and not _no_hybrid
    _hybrid = _hybrid_requested
    # v0.23.0: verruim de kandidaatdrempels alleen als de classifier ook
    # werkelijk kan draaien. Lukte dat niet -- model ontbreekt, lightgbm niet
    # geinstalleerd, corrupte booster -- dan bleef `result["events"]` de
    # KANDIDATENLIJST op ratio 1,2 staan, terwijl het log "falling back to
    # rule-based output" meldde. Gemeten op PSG-IPA, single derivatie:
    #   SN2  regels 203 ev (37,1/u) | met model  60 (11,0) | zonder model 777 (142,1)
    #   SN4  regels  94 ev (12,8/u) | met model  99 (13,5) | zonder model 979 (133,8)
    # tegen scoordermedianen van 8,5 en 14,3/u.
    _lgbm_ok = _hybrid_requested
    _lgbm_reason = None
    if _hybrid and sf < AROUSAL_LGBM_MIN_SF:
        logger.warning(
            "[arousal] EEG op %.0f Hz ligt onder %.0f Hz: de betaband "
            "(16-30 Hz) is niet representeerbaar en de classifier zou alle "
            "kandidaten verwerpen. Regelgebaseerd pad.",
            sf, AROUSAL_LGBM_MIN_SF,
        )
        _hybrid = False
        _lgbm_ok = False
        _lgbm_reason = "sample_rate_below_%.0f" % AROUSAL_LGBM_MIN_SF
    if _hybrid and not _emg_usable_for_lgbm(emg_data, len(eeg_data)):
        # v0.27.1. Het werkpunt 0,80 is gekozen op MESA-runs waar de chin-EMG
        # WEL werd meegeladen; de klinische keten leverde het kanaal nooit aan.
        # Zonder guard is het gevolg niet "iets minder gevoelig" maar een
        # arousal-index die de klinische werkelijkheid tegenspreekt: twee
        # AZORG-opnames gingen van 23,0 naar 4,9 en van 11,0 naar 3,5 /u, die
        # laatste bij AHI 42 en 217 respiratoire events.
        logger.warning(
            "[arousal] geen bruikbaar kin-EMG: het model splitst 486 keer op "
            "emg_var_ratio en dat feature is dan constant 0. Regelgebaseerd "
            "pad; kandidaatdrempels NIET verruimd.",
        )
        _hybrid = False
        _lgbm_ok = False
        _lgbm_reason = (
            "no_emg_channel: model v3 leunt op emg_var_ratio (486 splits, "
            "alle drempels > 0); zonder EMG is het feature constant 0 en "
            "degenereert de kansverdeling"
        )
    if _hybrid:
        try:
            _load_arousal_lgbm_booster()
        except Exception as _e:  # noqa: BLE001 -- elke laadfout telt hier gelijk
            logger.warning(
                "[arousal] LGBM-model niet beschikbaar (%s); regelgebaseerd "
                "pad, kandidaatdrempels NIET verruimd", _e,
            )
            _hybrid = False
            _lgbm_ok = False
            _lgbm_reason = "model_unavailable"
    # v0.8.1: effective thresholds are LOCAL (concurrency-safe — no module-global
    # mutation, which was fragile under the 8 parallel workers + the multi-derivation
    # loop). Explicit caller values win; else LGBM-candidate values in hybrid mode;
    # else the module defaults. Byte-identical to the old global-swap behaviour.
    if ratio_thresh is None:
        ratio_thresh = AROUSAL_LGBM_CAND_RATIO if _hybrid else AROUSAL_RATIO_THRESH
    if abrupt_thresh is None:
        abrupt_thresh = AROUSAL_LGBM_CAND_ABRUPT if _hybrid else ABRUPT_RATIO_THRESH
    if _hybrid:
        logger.info(
            "[arousal] LGBM hybrid mode enabled "
            "(candidate ratio=%.2f, abrupt=%.2f, threshold=%.2f)",
            ratio_thresh, abrupt_thresh, AROUSAL_LGBM_THRESHOLD,
        )

    result = {"success": False, "events": [], "summary": {}, "error": None}
    try:
        n_samples = len(eeg_data)
        spe       = int(sf * EPOCH_LEN_S)

        # v0.8.11 FIX: Converteer EEG naar µV als het in Volt lijkt te zijn
        # raw.get_data() geeft Volt (bijv. 50 µV = 5e-5 V)
        # Bandpower in Volt² geeft ~1e-10 waarden → numerieke problemen
        eeg_uv = eeg_data.copy()
        if np.max(np.abs(eeg_uv)) < 0.01:  # max < 10 mV → waarschijnlijk Volt
            eeg_uv = eeg_uv * 1e6
            logger.debug("Arousal EEG: V→µV conversie (max=%.1f µV)", np.max(np.abs(eeg_uv)))

        # ── Bandvermogen tijdreeksen (v0.8.11: theta + alpha_narrow + beta) ──
        alpha_pow = _bandpower_instant(eeg_uv, sf, ALPHA_NARROW_BAND, win_s=2.0)
        theta_pow = _bandpower_instant(eeg_uv, sf, THETA_BAND, win_s=2.0)
        beta_pow  = _bandpower_instant(eeg_uv, sf, BETA_BAND,  win_s=2.0)
        sigma_pow = _bandpower_instant(eeg_uv, sf, SIGMA_BAND, win_s=2.0)
        delta_pow = _bandpower_instant(eeg_uv, sf, DELTA_BAND, win_s=2.0)

        # Gecombineerd arousal-vermogen: alpha_narrow + theta + beta
        # (AASM: "alpha, theta en/of >16 Hz")
        arousal_pow = alpha_pow + theta_pow + beta_pow

        # v0.23.0: schaalvrije variant — de FRACTIE van het spectrale vermogen
        # in de snelle banden. delta_pow werd tot nu toe berekend en alleen als
        # rapportagewaarde gebruikt; het is precies de noemer die van een
        # vermogensmaat een frequentiemaat maakt.
        if spectral_shift:
            if shift_delta is None:
                shift_delta = AROUSAL_SHIFT_DELTA
            if shift_abrupt is None:
                shift_abrupt = AROUSAL_SHIFT_ABRUPT
            _total_pow = (delta_pow + alpha_pow + theta_pow
                          + beta_pow + sigma_pow)
            _total_pow = np.maximum(_total_pow, 1e-12)
            fast_frac  = arousal_pow / _total_pow      # r(t), NREM
            alpha_frac = alpha_pow / _total_pow        # REM: theta is achtergrond
            # De detectiegrootheden zelf worden vervangen; de rapportagevelden
            # (alpha_ratio, beta_ratio, dominant_band) blijven op vermogen.
            _detect_nrem = fast_frac
            _detect_rem  = alpha_frac
        else:
            _detect_nrem = arousal_pow
            _detect_rem  = alpha_pow

        # ── Baseline per slaapfase (v0.8.11: rolling 2-min venster) ────
        nrem_mask = _build_stage_mask(hypno, sf, n_samples,
                                       {"N1","N2","N3",1,2,3})
        rem_mask  = _build_stage_mask(hypno, sf, n_samples, {"R",4})

        def _robust_baseline(power_arr, mask):
            """Robuuste basislijn: mediaan van laagste 50% periodes.
            Voorkomt dat arousals zelf de basislijn verhogen."""
            seg = power_arr[mask]
            if len(seg) < int(sf * 60):
                seg = power_arr[power_arr > 0] if np.any(power_arr > 0) else power_arr
            if len(seg) == 0:
                return 1.0
            cutoff = np.percentile(seg, 50)
            quiet = seg[seg <= cutoff]
            if len(quiet) > 10:
                return max(float(np.median(quiet)), 1e-9)
            return max(float(np.percentile(seg, 25)), 1e-9)

        def _rolling_baseline(power_arr, stage_mask, window_s=120):
            """v0.8.11: Rolling basislijn over 2 min stabiele slaap.

            Bij ernstig gefragmenteerde slaap is een nacht-gemiddelde misleidend
            — de basislijn is al verhoogd door de vele arousals zelf.
            Een rolling venster van 120s stabiele slaap adapteert lokaal.

            Ref: Gemini review — "rolling baseline of the preceding 2 minutes
            of stable sleep to prevent habituation to the average power."
            """
            n = len(power_arr)
            win = int(window_s * sf)
            step = max(1, int(10 * sf))  # Bereken elke 10s
            anchors_x = []
            anchors_y = []

            for pos in range(0, n, step):
                start = max(0, pos - win)
                seg = power_arr[start:pos]
                seg_mask = stage_mask[start:pos]
                stable = seg[seg_mask]
                if len(stable) > int(sf * 10):
                    # Laagste 50% = stabiele slaap (excl. arousals)
                    cutoff = np.percentile(stable, 50)
                    quiet = stable[stable <= cutoff]
                    bl = float(np.median(quiet)) if len(quiet) > 5 else float(np.median(stable))
                else:
                    bl = None  # Niet genoeg data — wordt geïnterpoleerd
                if bl is not None and bl > 1e-9:
                    anchors_x.append(pos)
                    anchors_y.append(bl)

            if len(anchors_x) < 2:
                # v0.8.40: Fallback naar globale basislijn, maar met
                # noise-floor veiligheidsvloer om te voorkomen dat
                # arousal-inflated globale basislijn de drempel opdrijft.
                global_bl = _robust_baseline(power_arr, stage_mask)
                positive = power_arr[power_arr > 0]
                noise_floor = (float(np.percentile(positive, 5))
                               if positive.size > 0 else 1e-9)
                # Gebruik max van globale baseline en 2× noise floor:
                # bij zware fragmentatie trekt noise floor de drempel
                # realistisch omlaag.
                safe_bl = max(global_bl, noise_floor * 2.0)
                return np.full(n, safe_bl)

            baseline = np.interp(np.arange(n), anchors_x, anchors_y)
            return np.maximum(baseline, 1e-9)

        # Gebruik rolling baseline per sample (v0.8.11)
        # v0.23.0: bij spectral_shift draait dezelfde rolling machinerie op de
        # fractie i.p.v. het vermogen — de basislijn is dan de lokale rustige
        # spectrale balans, niet de lokale rustige amplitude.
        #
        # LET OP bij de else-tak: de oude code bouwt de REM-basislijn op
        # `arousal_pow` (alpha+theta+beta) terwijl fase 1 in REM alleen
        # `alpha_pow` toetst. Die asymmetrie is bestaand gedrag en blijft
        # ongemoeid; ze rechtzetten verandert de REM-arousals stil (gemeten op
        # PSG-IPA SN3: 61 -> 73 events). Alleen onder de vlag zijn teller en
        # noemer dezelfde grootheid.
        if spectral_shift:
            arousal_bl_nrem_arr = _rolling_baseline(_detect_nrem, nrem_mask)
            arousal_bl_rem_arr  = _rolling_baseline(_detect_rem, rem_mask)
        else:
            arousal_bl_nrem_arr = _rolling_baseline(arousal_pow, nrem_mask)
            # `rem_alpha_baseline` maakt teller en noemer in REM dezelfde
            # grootheid. Fase 1 toetst daar `alpha_pow`; de basislijn stond op
            # `alpha+theta+beta`. Een REM-arousal werd dus afgezet tegen een
            # noemer waar theta zwaar in meeweegt, en theta IS de
            # REM-achtergrond -- de drempel ligt daardoor te hoog.
            # Default False = bestaand gedrag. Aanzetten verandert de
            # REM-telling (op PSG-IPA SN3 gemeten: 61 -> 73 events) en vraagt
            # dus een meting, geen aanname.
            arousal_bl_rem_arr = _rolling_baseline(
                alpha_pow if rem_alpha_baseline else arousal_pow, rem_mask)
        sigma_bl_nrem_arr   = _rolling_baseline(sigma_pow, nrem_mask)

        # Globale baselines voor statistiek (backward compat)
        arousal_bl_nrem = _robust_baseline(arousal_pow, nrem_mask)
        arousal_bl_rem  = _robust_baseline(arousal_pow, rem_mask)
        sigma_bl_nrem   = _robust_baseline(sigma_pow, nrem_mask)

        # Per-band baselines (voor detail-stats)
        alpha_bl_nrem = _robust_baseline(alpha_pow, nrem_mask)
        alpha_bl_rem  = _robust_baseline(alpha_pow, rem_mask)
        beta_bl_nrem  = _robust_baseline(beta_pow, nrem_mask)

        # ── EMG verwerking voor REM arousal criterium (AASM) ─────
        emg_rms = None
        emg_bl_rem = None
        EMG_WINDOW_S = 0.25
        EMG_RISE_FACTOR = 2.0
        EMG_MIN_DUR_S = 1.0

        if emg_data is not None and len(emg_data) >= n_samples:
            from scipy.signal import butter, filtfilt  # local import for backward compat
            try:
                # v0.8.11 FIX: converteer EMG naar µV (zelfde issue als EEG/PLM)
                emg_work = emg_data[:n_samples].copy()
                if np.max(np.abs(emg_work)) < 0.01:
                    emg_work = emg_work * 1e6
                    logger.debug("Arousal EMG: V→µV conversie")
                # v0.8.40: Guard tegen ongeldige bandpass bij lage sf
                nyq = sf / 2.0
                high = min(100.0, nyq - 1.0)
                if nyq > 10.0 and high > 10.0:
                    b, a = butter(4, [10, high], btype="band", fs=sf)
                    emg_filt = filtfilt(b, a, emg_work)
                else:
                    logger.warning(
                        "EMG: sample rate %.1f Hz te laag voor 10-100 Hz "
                        "bandpass — ongefilterd gebruikt", sf
                    )
                    emg_filt = emg_work
            except Exception:
                emg_filt = emg_data[:n_samples]

            win = max(int(sf * EMG_WINDOW_S), 1)
            emg_sq = emg_filt ** 2
            kernel = np.ones(win) / win
            emg_rms = np.sqrt(np.convolve(emg_sq, kernel, mode="same"))

            rem_emg = emg_rms[rem_mask] if np.any(rem_mask) else emg_rms
            emg_bl_rem = max(float(np.percentile(rem_emg, 25)), 1e-9) if len(rem_emg) > 0 else 1e-9

        # ── v0.8.11: TWO-PHASE AROUSAL DETECTION ─────────────────
        # PROBLEEM v14.7-14.8: per-sample conjunctie (elevated & abrupt
        # & ~sigma) vereist ALLE voorwaarden gelijktijdig True op elke
        # sample voor >=3s. De abruptheidsratio (rolling 3s pre-average)
        # volgt het signaal → na ~1s stijgt pre-average mee → nooit 3s True.
        #
        # OPLOSSING v0.8.11: twee fasen (zoals een menselijke scorer):
        #   Fase 1: vind regio's met verhoogd vermogen (>=3s, enkel power)
        #   Fase 2: valideer elk event op onset-abruptheid en spindle
        #           (event-niveau, niet per-sample)

        arousal_mask = np.zeros(n_samples, dtype=bool)
        artifact_set = set(artifact_epochs or [])
        n_te_lang = 0          # regio's boven AROUSAL_MAX_DUR_S
        te_lang_s = 0.0        # en hoeveel slaaptijd daarin zat

        # Slaap-mask per sample
        sleep_sample_mask = np.zeros(n_samples, dtype=bool)
        for ep_i, stage in enumerate(hypno):
            if _is_sleep(stage) and ep_i not in artifact_set:
                s2 = ep_i * spe
                e2 = min(s2 + spe, n_samples)
                sleep_sample_mask[s2:e2] = True

        # ── FASE 1: Vind verhoogd-vermogen regio's (rolling baseline) ──
        for ep_i, stage in enumerate(hypno):
            if ep_i in artifact_set:
                continue
            s = ep_i * spe
            e = min(s + spe, n_samples)
            if _is_nrem(stage):
                # v0.8.11: vergelijk met rolling baseline i.p.v. globaal
                local_bl = arousal_bl_nrem_arr[s:e]
                if spectral_shift:
                    # ABSOLUUT increment op een begrensde grootheid
                    arousal_mask[s:e] = _detect_nrem[s:e] > local_bl + shift_delta
                else:
                    arousal_mask[s:e] = arousal_pow[s:e] > ratio_thresh * local_bl
            elif _is_rem(stage):
                local_bl = arousal_bl_rem_arr[s:e]
                if spectral_shift:
                    arousal_mask[s:e] = _detect_rem[s:e] > local_bl + shift_delta
                else:
                    arousal_mask[s:e] = alpha_pow[s:e] > ratio_thresh * local_bl

        # v0.23.0: hysterese — een event loopt door zolang het vermogen boven
        # `exit_ratio` blijft. `arousal_mask` bevat de INSTAPpunten; hieronder
        # wordt elk aaneengesloten stuk van de ruimere `sustain_mask` behouden
        # dat minstens één instappunt bevat. Zonder de vlag verandert er niets.
        if hysteresis:
            if exit_ratio is None:
                exit_ratio = AROUSAL_EXIT_RATIO
            # Bij spectral_shift is de instap een ABSOLUUT increment, geen
            # verhouding. De uitstap ligt dan op dezelfde fractie van de
            # instap als hier: exit_ratio / ratio_thresh = 1,2/2,0 = 0,6.
            sustain_mask = np.zeros(n_samples, dtype=bool)
            for ep_i, stage in enumerate(hypno):
                if ep_i in artifact_set:
                    continue
                s = ep_i * spe
                e = min(s + spe, n_samples)
                if _is_nrem(stage):
                    local_bl = arousal_bl_nrem_arr[s:e]
                    if spectral_shift:
                        sustain_mask[s:e] = (_detect_nrem[s:e]
                                             > local_bl + shift_delta * exit_ratio
                                             / max(ratio_thresh, 1e-9))
                    else:
                        sustain_mask[s:e] = arousal_pow[s:e] > exit_ratio * local_bl
                elif _is_rem(stage):
                    local_bl = arousal_bl_rem_arr[s:e]
                    if spectral_shift:
                        sustain_mask[s:e] = (_detect_rem[s:e]
                                             > local_bl + shift_delta * exit_ratio
                                             / max(ratio_thresh, 1e-9))
                    else:
                        sustain_mask[s:e] = alpha_pow[s:e] > exit_ratio * local_bl
            sus_lab, n_sus = label(sustain_mask)
            if n_sus > 0:
                # welke sustain-regio's raken een instappunt?
                touched = np.unique(sus_lab[arousal_mask])
                touched = touched[touched > 0]
                arousal_mask = np.isin(sus_lab, touched)

        # ── FASE 2: Label, valideer per event ──
        labeled, n_events = label(arousal_mask)
        arousals = []

        for i in range(1, n_events + 1):
            indices = np.where(labeled == i)[0]
            dur_s   = len(indices) / sf

            # Te kort of te lang. De BOVENgrens verdient een telling: een
            # regio langer dan 30 s is vermoedelijk een ontwaking, en die
            # verdween hier zonder spoor. Een lezer kon niet zien of een lage
            # arousal-index betekende dat er weinig gebeurde of dat er veel
            # weggegooid was. De AASM kent geen bovengrens voor een arousal;
            # deze is een pragmatische wachtregel, en dan hoort er een teller
            # bij. De ACCEPTATIE verandert niet -- alleen de zichtbaarheid.
            if dur_s > AROUSAL_MAX_DUR_S:
                n_te_lang += 1
                te_lang_s += dur_s
                continue
            if dur_s < AROUSAL_MIN_DUR_S:
                continue

            onset_s = float(indices[0]) / sf
            end_s   = float(indices[-1]) / sf
            ep_idx  = int(onset_s // EPOCH_LEN_S)
            stage   = hypno[ep_idx] if ep_idx < len(hypno) else "W"

            # Check A: Pre-sleep (>=10s slaap, >=60%)
            pre_start = max(0, indices[0] - int(PRESLEEP_MIN_S * sf))
            pre_end   = indices[0]
            if pre_end > pre_start:
                sleep_frac = np.sum(sleep_sample_mask[pre_start:pre_end]) / (pre_end - pre_start)
                if sleep_frac < 0.6:
                    continue

            # Check B: Onset-abruptheid (event-niveau)
            onset_idx = indices[0]
            pre_3s_start = max(0, onset_idx - int(3.0 * sf))
            onset_1s_end = min(onset_idx + int(1.0 * sf), indices[-1] + 1)
            # LET OP: de oude regel gebruikte hier ALTIJD arousal_pow, ook in
            # REM (waar fase 1 op alpha draait). Dat gedrag blijft ongemoeid als
            # de vlag uit staat; alleen bij spectral_shift volgt de abruptheid
            # dezelfde grootheid als fase 1.
            # Bij `rem_alpha_baseline` volgt ook de ABRUPTHEID in REM de alpha.
            # Half repareren zou een derde asymmetrie maken: fase 1 op alpha,
            # de basislijn op alpha, en de abruptheid nog op alpha+theta+beta.
            if spectral_shift:
                _abr_src = _detect_rem if _is_rem(stage) else _detect_nrem
            elif rem_alpha_baseline and _is_rem(stage):
                _abr_src = alpha_pow
            else:
                _abr_src = arousal_pow
            pre_power  = float(np.mean(_abr_src[pre_3s_start:onset_idx])) if onset_idx > pre_3s_start else 1e-12
            onset_power = float(np.mean(_abr_src[onset_idx:onset_1s_end]))
            if spectral_shift:
                # Verschil i.p.v. verhouding: op een fractie is een verhouding
                # opnieuw afhankelijk van waar de basislijn toevallig ligt.
                onset_ratio = onset_power - pre_power
                if onset_ratio < shift_abrupt:
                    continue
            else:
                onset_ratio = onset_power / max(pre_power, 1e-12)
                if onset_ratio < abrupt_thresh:
                    continue

            # Check C: Spindle-exclusie (v0.8.11: ratio-check)
            # Bij het ontwaken uit N2 valt de arousal-burst vaak samen met een
            # afbrekende spindle. De oude logica (arousal < sigma) verwierp dit
            # onterecht. Nu: reject ENKEL als sigma >2× rolling baseline EN
            # alpha+beta samen MINDER dan 50% van sigma zijn.
            # → Een arousal met bijkomende spindle-activiteit wordt geaccepteerd
            #   zolang alpha+beta de event domineren.
            if _is_nrem(stage):
                ev_sigma = float(np.mean(sigma_pow[indices[0]:indices[-1]+1]))
                ev_alpha_beta = float(np.mean(
                    alpha_pow[indices[0]:indices[-1]+1] +
                    beta_pow[indices[0]:indices[-1]+1]))
                local_sigma_bl = float(np.mean(sigma_bl_nrem_arr[indices[0]:indices[-1]+1]))
                sigma_elevated = ev_sigma > 2.0 * local_sigma_bl
                # Reject: sigma dominant EN alpha+beta < 50% van sigma
                if sigma_elevated and ev_alpha_beta < 0.5 * ev_sigma:
                    continue

            # Check D: REM EMG
            #
            # `emg_confirmed` beschrijft precies één ding: heeft de door de
            # AASM geeiste EMG-stijging in REM daadwerkelijk plaatsgevonden.
            # Het stond hier als DEFAULT op True -- ook op NREM-events, waar
            # de test niet van toepassing is, en op montages zonder kin-EMG,
            # waar hij niet kan draaien. `n_emg_confirmed` in de samenvatting
            # telde daardoor elke arousal mee als bevestigd. Dat is geen
            # conservatieve default maar een onwaarheid in een rapportveld.
            #
            # De ACCEPTATIE verandert hier niet: zonder EMG blijft het event
            # staan op alleen alpha+abrupt, precies zoals hiervoor. Het model
            # splitst nergens op dit feature (0 splits in alle 500 bomen), dus
            # de classifier ziet het verschil niet.
            emg_confirmed = False
            if _is_rem(stage):
                if emg_rms is not None and emg_bl_rem:
                    emg_seg = emg_rms[indices[0]:indices[-1]+1]
                    emg_dur = np.sum(emg_seg > EMG_RISE_FACTOR * emg_bl_rem) / sf
                    emg_confirmed = emg_dur >= EMG_MIN_DUR_S
                    if not emg_confirmed:
                        continue
                # Geen EMG → accepteer toch (alleen alpha+abrupt)

            # Check E (v0.8.11): K-complex morfologische check
            # Bipolaire golf (>75 µV neg + pos) in eerste 1s → verhoog min-duur
            kcomplex_min_dur_s = AROUSAL_MIN_DUR_S  # standaard 3.0 s
            if _is_nrem(stage):
                if _is_kcomplex(eeg_uv, indices[0], sf):
                    kcomplex_min_dur_s = 5.0  # conservatiever bij K-complex morfologie
                if dur_s < kcomplex_min_dur_s:
                    continue

            # Check F (v0.8.11): CVR confidence boost
            cvr_boost = 0.0
            if hr_data is not None and len(hr_data) > 0:
                cvr_boost = _detect_cvr_confidence_boost(
                    hr_data, sf_hr, onset_s
                )

            seg_alpha = float(np.mean(alpha_pow[indices[0]:indices[-1]+1]))
            seg_theta = float(np.mean(theta_pow[indices[0]:indices[-1]+1]))
            seg_beta  = float(np.mean(beta_pow[indices[0]:indices[-1]+1]))
            seg_delta = float(np.mean(delta_pow[indices[0]:indices[-1]+1]))
            alpha_ratio = seg_alpha / alpha_bl_nrem if alpha_bl_nrem > 0 else 0
            beta_ratio  = seg_beta  / beta_bl_nrem  if beta_bl_nrem  > 0 else 0
            band_powers = {"alpha": seg_alpha, "theta": seg_theta, "beta": seg_beta}
            dominant_band = max(band_powers, key=band_powers.get)
            if _is_rem(stage):
                dominant_band = "alpha"

            arousals.append({
                "onset_s":       _safe(onset_s),
                "end_s":         _safe(end_s),
                "duration_s":    _safe(dur_s),
                "stage":         stage,
                "epoch":         ep_idx,
                "dominant_band": dominant_band,
                "alpha_ratio":   _safe(alpha_ratio, 2),
                "beta_ratio":    _safe(beta_ratio, 2),
                "onset_ratio":   _safe(onset_ratio, 2),
                "emg_confirmed": emg_confirmed,
                "cvr_boost":     _safe(cvr_boost, 2),
                "type":          "spontaneous",
            })

        # ── Statistieken ─────────────────────────────────────────
        total_sleep_s = sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                            if _is_sleep(s) and i not in artifact_set)
        # Zie psgscoring/indices.py — geen ondergrens op de noemer.
        total_sleep_h = total_sleep_s / 3600
        rem_h  = sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                     if _is_rem(s) and i not in artifact_set) / 3600
        nrem_h = sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                     if _is_nrem(s) and i not in artifact_set) / 3600

        nrem_ar = [a for a in arousals if _is_nrem(a["stage"])]
        rem_ar  = [a for a in arousals if _is_rem(a["stage"])]

        result["events"]  = arousals
        result["summary"] = {
            "n_arousals":          len(arousals),
            "arousal_index":       per_hour(len(arousals), total_sleep_h),
            "nrem_arousal_index":  per_hour(len(nrem_ar), nrem_h),
            "rem_arousal_index":   per_hour(len(rem_ar), rem_h),
            "avg_duration_s":      _safe(float(np.mean([a["duration_s"]
                                          for a in arousals]))) if arousals else None,
            "severity":            _classify_arousal_index(
                                       per_hour(len(arousals), total_sleep_h)),
            # v0.8.11: extra stats
            "n_theta_dominant":    sum(1 for a in arousals if a["dominant_band"] == "theta"),
            "n_alpha_dominant":    sum(1 for a in arousals if a["dominant_band"] == "alpha"),
            "n_beta_dominant":     sum(1 for a in arousals if a["dominant_band"] == "beta"),
            "n_emg_confirmed":     sum(1 for a in arousals if a["emg_confirmed"]),
            # Zie de duurcheck in fase 2. Nul hoort ook zichtbaar te zijn:
            # het verschil tussen "niets weggegooid" en "niet gekeken" is
            # precies wat hier jarenlang niet af te lezen was.
            "n_too_long_discarded": n_te_lang,
            "too_long_discarded_s": _safe(te_lang_s),
            "max_duration_s":       AROUSAL_MAX_DUR_S,
        }
        result["success"] = True

        # v0.9.8: LGBM filter applied after the rule-based body has
        # populated result["events"] / result["summary"]. We post-filter
        # with the model and recompute the summary; the rule-based output
        # is still available under result["pre_lgbm_n_arousals"] for
        # diagnostics.
        if _hybrid and result.get("events"):
            try:
                eeg_uv_for_lgbm = eeg_data.copy()
                if np.max(np.abs(eeg_uv_for_lgbm)) < 0.01:
                    eeg_uv_for_lgbm = eeg_uv_for_lgbm * 1e6
                emg_uv_for_lgbm = None
                if emg_data is not None:
                    emg_uv_for_lgbm = emg_data[:n_samples].copy()
                    if np.max(np.abs(emg_uv_for_lgbm)) < 0.01:
                        emg_uv_for_lgbm = emg_uv_for_lgbm * 1e6
                # Werkpunt: profielwaarde als die er is, anders de
                # moduleconstante (die zelf al een env-override kent).
                _thr = (float(lgbm_threshold) if lgbm_threshold is not None
                        else AROUSAL_LGBM_THRESHOLD)
                # Event-locked werkpunt. Het venster is EXACT dat van
                # correlate_arousals_to_respiratory (event-onset tot
                # POST_RESP_WINDOW_S na het einde), zodat detectie en koppeling
                # dezelfde geometrie delen -- een event dat het venster toelaat
                # maar de koppeling niet erkent, komt nergens in terug.
                _thr_per = None
                if event_locked_threshold is not None and resp_event_ends:
                    _lo = float(event_locked_threshold)
                    _ends = [float(x) for x in resp_event_ends]
                    _thr_per = [
                        _lo if any(
                            (e - AROUSAL_PRE_RESP_WINDOW_S) <= c["onset_s"]
                            <= (e + POST_RESP_WINDOW_S) for e in _ends
                        ) else _thr
                        for c in result["events"]
                    ]
                kept, proba = _filter_candidates_with_lgbm(
                    result["events"], eeg_uv_for_lgbm, sf,
                    emg_uv_for_lgbm, len(hypno),
                    threshold=_thr, thresholds=_thr_per,
                )
                n_pre = len(result["events"])
                result["pre_lgbm_n_arousals"] = n_pre
                result["events"] = kept
                result["summary"] = _recompute_arousal_summary(
                    kept, hypno, set(artifact_epochs or []),
                )
                # `_recompute_arousal_summary` bouwt een VERSE dict, dus de
                # duurtellers uit fase 2 vielen hier weg zodra de classifier
                # draaide -- en dat is juist het pad dat de klinische profielen
                # nemen. Zelfde val als bij n_interval_merged.
                result["summary"]["n_too_long_discarded"] = n_te_lang
                result["summary"]["too_long_discarded_s"] = _safe(te_lang_s)
                result["summary"]["max_duration_s"] = AROUSAL_MAX_DUR_S
                result["summary"]["lgbm_threshold"] = _thr
                result["summary"]["lgbm_n_pre"]     = n_pre
                result["summary"]["lgbm_n_post"]    = len(kept)
                if _thr_per is not None:
                    result["summary"]["n_event_locked"] = sum(
                        1 for e in kept if e.get("event_locked"))
                    result["summary"]["event_locked_threshold"] = float(
                        event_locked_threshold)
                logger.info(
                    "[arousal] LGBM filter: %d candidates → %d kept "
                    "(threshold %.2f)", n_pre, len(kept), _thr,
                )
            except Exception as e:  # noqa: BLE001
                # De drempels staan hier al ruim, dus teruggeven wat er ligt
                # zou de KANDIDATEN opleveren. Opnieuw detecteren op de
                # regelgebaseerde drempels; `_no_hybrid` stopt de recursie.
                logger.warning(
                    "[arousal] LGBM-filter mislukt na het laden (%s); "
                    "opnieuw detecteren op de regelgebaseerde drempels", e,
                )
                _lgbm_ok = False
                result = detect_arousals(
                    eeg_data, sf, hypno, emg_data=emg_data,
                    artifact_epochs=artifact_epochs, hr_data=hr_data,
                    sf_hr=sf_hr,
                    spectral_shift=spectral_shift,
                    shift_delta=shift_delta, shift_abrupt=shift_abrupt,
                    hysteresis=hysteresis, exit_ratio=exit_ratio,
                    _no_hybrid=True,
                )
                result["lgbm_error"] = str(e)

    except Exception as e:
        result["error"]     = str(e)
        result["traceback"] = traceback.format_exc()

    # AASM: >=10 s slaap tussen twee arousals. HIER en niet eerder: dit is een
    # regel over de UITEINDELIJKE eventlijst, dus hij hoort na het
    # classifierfilter -- dat kan van een paar te dichte kandidaten er al een
    # verwijderd hebben, en dan valt er niets samen te voegen.
    min_interval_s = _min_interval_from_env(min_interval_s)
    if min_interval_s and result.get("success") and result.get("events"):
        _mi_stats: dict = {}
        result["events"] = enforce_min_arousal_interval(
            result["events"], min_interval_s, stats=_mi_stats)
        if _mi_stats["n_merged"]:
            _bewaar = {k: result["summary"].get(k)
                       for k in ("n_too_long_discarded", "too_long_discarded_s",
                                 "max_duration_s", "lgbm_threshold",
                                 "lgbm_n_pre", "lgbm_n_post")
                       if isinstance(result.get("summary"), dict)
                       and k in result["summary"]}
            result["summary"] = _recompute_arousal_summary(
                result["events"], hypno, set(artifact_epochs or []))
            result["summary"].update(_bewaar)
            logger.info("[arousal] %d paren samengevoegd op de %.0f s-regel "
                        "(%d -> %d)", _mi_stats["n_merged"], min_interval_s,
                        _mi_stats["n_before"], _mi_stats["n_after"])
        if isinstance(result.get("summary"), dict):
            result["summary"]["min_interval_s"] = float(min_interval_s)
            result["summary"]["n_interval_merged"] = _mi_stats["n_merged"]

    if _hybrid_requested and isinstance(result.get("summary"), dict):
        # Een consument moet kunnen zien DAT de hybride gevraagd was en of hij
        # gedraaid heeft. Zonder dit is een regelgebaseerd resultaat niet te
        # onderscheiden van een gefilterd resultaat.
        result["summary"]["lgbm_available"] = _lgbm_ok
        if _lgbm_reason:
            result["summary"]["lgbm_skipped_reason"] = _lgbm_reason
    return result


# ═══════════════════════════════════════════════════════════════
# MULTI-DERIVATIE  (v0.8.1 — event-level union over EEG-afleidingen)
# ═══════════════════════════════════════════════════════════════

def _min_interval_from_env(explicit: float) -> float:
    """De 10 s-regel ook leesbaar voor wie de pipeline NIET gebruikt.

    `PSGSCORING_AROUSAL_MIN_INTERVAL_S` wordt in `pipeline.py` gelezen, maar de
    meetharnassen roepen `detect_arousals`/`detect_arousals_multi` rechtstreeks
    aan -- `sweep_arousal_threshold_psgipa.py` bijvoorbeeld. Die zagen de vlag
    dus niet, en een arm die niets doet is vandaag al twee keer als meting
    gerapporteerd (zie docs/rule1a_arousal_20260829.md).

    De env WINT van het argument, net als bij `PSGSCORING_AROUSAL_LGBM` een
    paar regels verderop: een installatie of een meting moet hem kunnen
    forceren. Onleesbaar -> het argument, met waarschuwing.
    """
    env = os.environ.get("PSGSCORING_AROUSAL_MIN_INTERVAL_S")
    if env is None:
        return float(explicit or 0.0)
    try:
        return float(env)
    except ValueError:
        logger.warning(
            "[arousal] PSGSCORING_AROUSAL_MIN_INTERVAL_S=%r is geen getal; "
            "doorgegeven waarde aangehouden", env)
        return float(explicit or 0.0)


def enforce_min_arousal_interval(
    events: list[dict],
    min_interval_s: float,
    stats: dict | None = None,
) -> list[dict]:
    """Twee arousals binnen ``min_interval_s`` van elkaar zijn er EEN.

    De AASM eist dat een arousal wordt voorafgegaan door ten minste 10 s
    stabiele slaap. Check A in `detect_arousals` toetst daarvan alleen de
    hypnogramkant -- of de voorgaande 10 s als slaap gescoord staat -- en een
    epoch waarin net een arousal zat heet nog steeds N2. Twee bursts van 4 s uit
    elkaar leverden daardoor twee arousals waar er een hoort te staan.

    In multi-derivatie telt dit dubbel: `_union_arousals` fuseert alleen bij
    TEMPORELE OVERLAP, dus twee afleidingen die 2 s na elkaar vuren leveren twee
    events op.

    Deze functie kan alleen events WEGNEMEN, nooit toevoegen: de richting van
    het effect staat vast voor de meting. Dat is het spiegelbeeld van
    `bridge_event_gaps` in respiratory.py, en om dezelfde reden nuttig.

    Samengevoegd betekent: onset van de eerste, einde van de laatste. Band en
    stadium komen van de langste bijdrager, net als in `_union_arousals`, zodat
    de twee samenvoegingen dezelfde regel volgen. Het resultaat kan langer
    duren dan ``AROUSAL_MAX_DUR_S`` -- dat is bewust, want die grens is een
    KANDIDAATfilter en geen regel over de uiteindelijke eventlijst.

    ``min_interval_s <= 0`` geeft de lijst ONGEWIJZIGD terug, hetzelfde object.
    """
    if stats is not None:
        stats.setdefault("min_interval_s", float(min_interval_s))
        stats.setdefault("n_merged", 0)
        stats.setdefault("n_before", len(events or []))
        stats.setdefault("n_after", len(events or []))
    if min_interval_s <= 0 or not events or len(events) < 2:
        return events

    def _span(e):
        o = float(e.get("onset_s") or 0.0)
        end = float(e["end_s"] if e.get("end_s") is not None
                    else o + float(e.get("duration_s") or 0.0))
        return o, end

    geordend = sorted(events, key=lambda e: _span(e)[0])
    uit: list[dict] = [dict(geordend[0])]
    n_merged = 0
    for ev in geordend[1:]:
        o, end = _span(ev)
        huidig = uit[-1]
        ho, hend = _span(huidig)
        if (o - hend) < float(min_interval_s):
            if (end - o) > (hend - ho):          # langste wint band/stadium
                huidig["dominant_band"] = ev.get("dominant_band",
                                                 huidig.get("dominant_band"))
                huidig["stage"] = ev.get("stage", huidig.get("stage"))
            nieuw_end = max(end, hend)
            huidig["onset_s"] = _safe(ho)
            huidig["end_s"] = _safe(nieuw_end)
            huidig["duration_s"] = _safe(nieuw_end - ho)
            huidig["merged_from"] = int(huidig.get("merged_from", 1)) + 1
            _d = set(huidig.get("derivations") or []) | set(ev.get("derivations") or [])
            if _d:
                huidig["derivations"] = sorted(_d)
            n_merged += 1
        else:
            uit.append(dict(ev))

    if stats is not None:
        stats["n_merged"] = n_merged
        stats["n_after"] = len(uit)
    return uit


def _union_arousals(event_lists: list[list[dict]]) -> list[dict]:
    """Voeg arousals van meerdere afleidingen samen tot één event-lijst.

    Twee per-afleiding-events zijn dezelfde arousal wanneer ze in de tijd
    overlappen; ze worden gefuseerd (onset = vroegste, end = laatste) en hun
    ``derivations`` worden verenigd. Niet-overlappende events (bv. een
    occipitaal-only arousal die het centrale kanaal miste) blijven apart — dat
    is de sensitiviteitswinst. Band-info/stage wordt overgenomen van het
    langste bijdragende event.
    """
    flat = [e for lst in event_lists for e in lst]
    if not flat:
        return []
    flat = sorted(flat, key=lambda e: (e.get("onset_s") or 0.0))
    merged: list[dict] = []
    for e in flat:
        o = float(e.get("onset_s") or 0.0)
        end = float(e["end_s"] if e.get("end_s") is not None
                    else o + float(e.get("duration_s") or 0.0))
        deriv = e.get("derivation")
        for m in merged:
            mo = float(m.get("onset_s") or 0.0)
            mend = float(m["end_s"] if m.get("end_s") is not None
                         else mo + float(m.get("duration_s") or 0.0))
            if min(end, mend) > max(o, mo):                 # temporele overlap
                if (end - o) > (mend - mo):                 # langste wint band/stage
                    m["dominant_band"] = e.get("dominant_band", m.get("dominant_band"))
                    m["stage"] = e.get("stage", m.get("stage"))
                new_o, new_end = min(o, mo), max(end, mend)
                m["onset_s"] = _safe(new_o)
                m["end_s"] = _safe(new_end)
                m["duration_s"] = _safe(new_end - new_o)
                if deriv and deriv not in m["derivations"]:
                    m["derivations"] = sorted(m["derivations"] + [deriv])
                break
        else:
            ne = dict(e)
            ne["derivations"] = [deriv] if deriv else []
            merged.append(ne)
    merged.sort(key=lambda e: (e.get("onset_s") or 0.0))
    return merged


def _is_occipital(name) -> bool:
    n = (name or "").upper()
    return ("O1" in n) or ("O2" in n) or ("OZ" in n)


def _eog_reject_occipital(events, eog_data, sf, factor: float = 3.0):
    """Verwerp occipitaal-ONLY arousals die samenvallen met een grote oogbeweging
    (EOG-doorslag naar de occipitale elektroden) — zoals een humane scorer die de
    EOG kruist en zo'n 'arousal' als oogbeweging herkent. Raakt cross-kanaal
    bevestigde events niet aan. Retourneert (behouden_events, n_verworpen)."""
    if eog_data is None or len(eog_data) == 0:
        return events, 0
    eog = np.abs(np.asarray(eog_data, dtype=float))
    win = max(1, int(sf))
    n = len(eog) // win
    if n < 2:
        return events, 0
    rms = np.array([np.sqrt(np.mean(eog[i * win:(i + 1) * win] ** 2)) for i in range(n)])
    pos = rms[rms > 0]
    base = float(np.median(pos)) if pos.size else 0.0
    if base <= 0:
        return events, 0
    kept, dropped = [], 0
    for e in events:
        derivs = e.get("derivations") or ([e["derivation"]] if e.get("derivation") else [])
        occ_only = bool(derivs) and all(_is_occipital(d) for d in derivs)
        if occ_only:
            o = int((e.get("onset_s") or 0.0) * sf)
            end = int(((e.get("onset_s") or 0.0) + (e.get("duration_s") or 0.0)) * sf)
            seg = eog[o:max(end, o + win)]
            if len(seg) and np.sqrt(np.mean(seg ** 2)) > factor * base:
                dropped += 1
                continue
        kept.append(e)
    return kept, dropped


def detect_arousals_multi(derivations, sf: float, hypno: list,
                          emg_data: np.ndarray | None = None,
                          artifact_epochs: list | None = None,
                          hr_data: np.ndarray | None = None,
                          sf_hr: float = 1.0,
                          per_channel_thresh: dict | None = None,
                          eog_data: np.ndarray | None = None,
                          eog_reject: bool = False,
                          spectral_shift: bool = False,
                          hysteresis: bool = False,
                          lgbm: bool | None = None,
                          lgbm_threshold: float | None = None,
                          resp_event_ends: list | None = None,
                          event_locked_threshold: float | None = None,
                          min_interval_s: float = 0.0,
                          rem_alpha_baseline: bool = False) -> dict:
    """Multi-derivatie arousal-detectie via event-level union.

    ``derivations``: geordende lijst ``[(naam, eeg_data[, sf]), ...]`` — element 0
    is de centrale single-channel-pick. Elke afleiding draait door de rijpe
    single-channel ``detect_arousals`` (abruptheid, spindle-exclusie, K-complex,
    REM-EMG blijven per kanaal behouden); de events worden daarna samengevoegd.

    **Invariant:** met precies één afleiding is het resultaat byte-identiek aan
    ``detect_arousals`` (geen union, geen provenance-tag, geen her-summary).
    """
    if not derivations:
        return {"success": False, "events": [], "summary": {}, "error": "no_derivations"}
    pct = per_channel_thresh or {}
    per = []
    for item in derivations:
        name, eeg = item[0], item[1]
        rt, at = pct.get(name, (None, None))
        res = detect_arousals(eeg, sf, hypno, emg_data=emg_data,
                              artifact_epochs=artifact_epochs,
                              hr_data=hr_data, sf_hr=sf_hr,
                              ratio_thresh=rt, abrupt_thresh=at,
                              spectral_shift=spectral_shift,
                              hysteresis=hysteresis, lgbm=lgbm,
                              lgbm_threshold=lgbm_threshold,
                              resp_event_ends=resp_event_ends,
                              event_locked_threshold=event_locked_threshold,
                              min_interval_s=min_interval_s,
                              rem_alpha_baseline=rem_alpha_baseline)
        if res.get("success"):
            per.append((name, res))
    if not per:
        return {"success": False, "events": [], "summary": {},
                "error": "all_derivations_failed"}
    if len(per) == 1:
        return per[0][1]                     # byte-identiek aan single-channel
    for name, res in per:
        for ev in res.get("events", []):
            ev["derivation"] = name
    merged = _union_arousals([res.get("events", []) for _, res in per])
    n_eog_rejected = 0
    if eog_reject and eog_data is not None:
        merged, n_eog_rejected = _eog_reject_occipital(merged, eog_data, sf)
    # Nogmaals NA de union, en dat is geen dubbelop: `_union_arousals` fuseert
    # alleen bij temporele OVERLAP, dus twee afleidingen die 2 s na elkaar
    # vuren leveren hier twee events op die de 10 s-regel nog niet gezien
    # hebben.
    _mi_stats: dict = {}
    min_interval_s = _min_interval_from_env(min_interval_s)
    merged = enforce_min_arousal_interval(merged, min_interval_s, stats=_mi_stats)
    summ = _recompute_arousal_summary(merged, hypno, set(artifact_epochs or []))
    if min_interval_s:
        summ["min_interval_s"] = float(min_interval_s)
        # De regel draait TWEE keer: per afleiding binnen `detect_arousals`, en
        # nog eens na de union. Hier stond alleen die tweede, en dat is de
        # kleinste van de twee: op PSG-IPA SN3 meldde het veld 4 terwijl de
        # telling 173 -> 159 ging. Een lezer die dat veld gebruikt om te zien
        # hoeveel de regel deed, kreeg een factor drie te weinig.
        #
        # Zelfde vorm als de lgbm-tellingen hierboven: per afleiding optellen.
        # De uitsplitsing blijft leesbaar, want de twee stappen betekenen niet
        # hetzelfde -- de eerste voegt buren binnen EEN kanaal samen, de tweede
        # buren die pas door de union naast elkaar kwamen te liggen.
        _per_deriv = sum((r.get("summary") or {}).get("n_interval_merged", 0)
                         for _, r in per)
        summ["n_interval_merged"] = _per_deriv + _mi_stats["n_merged"]
        summ["n_interval_merged_per_derivation"] = _per_deriv
        summ["n_interval_merged_after_union"] = _mi_stats["n_merged"]
    summ["n_derivations"] = len(per)
    summ["derivations"] = [n for n, _ in per]
    summ["n_per_derivation"] = {n: len(res.get("events", [])) for n, res in per}
    summ["n_eog_rejected"] = n_eog_rejected
    # LGBM-provenance overnemen uit de per-afleiding-resultaten. Deze functie
    # bouwde de samenvatting van nul op en liet `lgbm_available`,
    # `lgbm_skipped_reason` en de voor/na-tellingen achter -- en multi is de
    # DEFAULT op de klinische profielen. Op precies het pad waar de classifier
    # draait was dus niet af te lezen OF hij gedraaid had. Alle afleidingen
    # krijgen dezelfde emg_data en dezelfde vlaggen, dus de status is uniform;
    # de tellingen zijn per afleiding en worden opgeteld.
    # Duurtellers optellen over de afleidingen: elk kanaal gooit zijn eigen
    # te lange regio's weg, en de union ziet die nooit.
    if any("n_too_long_discarded" in (r.get("summary") or {}) for _, r in per):
        summ["n_too_long_discarded"] = sum(
            (r.get("summary") or {}).get("n_too_long_discarded", 0) for _, r in per)
        summ["too_long_discarded_s"] = _safe(sum(
            (r.get("summary") or {}).get("too_long_discarded_s", 0.0) or 0.0
            for _, r in per))
        summ["max_duration_s"] = AROUSAL_MAX_DUR_S

    _first = per[0][1].get("summary", {}) or {}
    for _k in ("lgbm_available", "lgbm_skipped_reason", "lgbm_threshold",
               "event_locked_threshold"):
        if _k in _first:
            summ[_k] = _first[_k]
    if any("lgbm_n_pre" in (r.get("summary") or {}) for _, r in per):
        summ["lgbm_n_pre"] = sum((r.get("summary") or {}).get("lgbm_n_pre", 0)
                                 for _, r in per)
        summ["lgbm_n_post"] = sum((r.get("summary") or {}).get("lgbm_n_post", 0)
                                  for _, r in per)
    if any("n_event_locked" in (r.get("summary") or {}) for _, r in per):
        # Na de union is een telling per afleiding niet meer wat de lezer
        # wil; tellen op de samengevoegde lijst.
        summ["n_event_locked"] = sum(1 for e in merged if e.get("event_locked"))
    out = {"success": True, "events": merged, "summary": summ, "error": None}
    _pre = [r["pre_lgbm_n_arousals"] for _, r in per
            if r.get("pre_lgbm_n_arousals") is not None]
    if _pre:
        out["pre_lgbm_n_arousals"] = sum(_pre)
    return out


def _classify_arousal_index(ai: float) -> str:
    if ai is None:
        return "unknown"
    if ai < 10:   return "normal"
    if ai < 20:   return "mildly_elevated"
    if ai < 40:   return "moderately_elevated"
    return "severely_elevated"


# ═══════════════════════════════════════════════════════════════
# RESPIRATOIR-AROUSAL KOPPELING
# ═══════════════════════════════════════════════════════════════

def arousal_couples_to_event(arousal_onset_s: float,
                             event_onset_s: float,
                             event_end_s: float,
                             window_post_s: float = 15.0) -> bool:
    """Hoort deze arousal bij dit respiratoire event?

    EEN definitie, gebruikt door zowel de scoring als de rapportage. Tot
    v0.23.0 hanteerden die twee paden verschillende regels:

      scoring  (breath_scoring.py)  event-ONSET tot 15 s na het einde
      rapport  (deze module)        latentie t.o.v. het EINDE, -5 tot +15 s

    Een arousal twee seconden na de onset van een event van dertig seconden
    bevestigde dat event dus in de index en heette in hetzelfde rapport
    spontaan. De scoringsregel is hier normatief omdat die de gepubliceerde
    index produceert; de rapportage roept nu dezelfde functie aan.

    De ondergrens is de event-onset en niet `onset - 5`: een arousal die
    begint voordat het event begint, kan er niet door zijn veroorzaakt.

    GEMETEN OMVANG. Over zes MESA-opnames onder `aasm_v3_rec`: van 796
    arousal-eventkoppelingen voldeden er 446 (56 %) aan beide regels, 350
    (44 %) alleen aan de scoringsregel, en nul alleen aan de rapportageregel.
    Die asymmetrie is structureel: voor elk event langer dan vijf seconden
    omvat het scoringsvenster [t0, t1+15] het rapportagevenster [t1-5, t1+15].
    Het rapportagepad kon dus alleen ondertellen, en telde bijna de helft van
    de gekoppelde arousals als spontaan.
    """
    return event_onset_s <= arousal_onset_s <= event_end_s + window_post_s


def correlate_arousals_to_respiratory(
    arousals:       list,
    resp_events:    list,
    window_pre_s:   float = 5.0,
    window_post_s:  float = POST_RESP_WINDOW_S,
) -> dict:
    """
    Koppel arousals aan voorafgaande respiratoire events.

    AASM definitie respiratoire arousal:
      Arousal die optreedt binnen 15s na het EINDE van een apnea of hypopnea.

    Geeft terug:
      - Respiratoire arousals (gekoppeld aan event)
      - Spontane arousals (geen respiratoir verband)
      - Per respiratoir event: had het een arousal? Latentie?
      - Statistieken: % events met arousal, gemiddelde arousal-latentie
    """
    result = {
        "success":            False,
        "respiratory_arousals": [],
        "spontaneous_arousals": [],
        "resp_events_with_arousal": [],
        "summary":            {},
        "error":              None,
    }
    try:
        if not arousals or not resp_events:
            result["success"] = True
            result["summary"] = _empty_correlation_summary()
            return result

        # Markeer arousals als respiratoir indien ze binnen venster vallen
        ar_annotated = []
        for ar in arousals:
            ar_copy = dict(ar)
            ar_copy["linked_event"] = None
            ar_copy["arousal_latency_s"] = None

            ar_onset = ar["onset_s"] or 0

            # Zoek respiratoir event dat eindigde in [onset - window_pre ... onset + window_post]
            best_match = None
            best_latency = float("inf")

            for ev in resp_events:
                ev_start = ev.get("onset_s") or 0
                ev_end = ev_start + (ev.get("duration_s") or 0)
                # Latentie = arousal_onset - event_end (positief = na event)
                latency = ar_onset - ev_end

                # v0.23.0: dezelfde koppelregel als de scoring. `window_pre_s`
                # blijft in de signatuur voor aanroepers die hem meegeven,
                # maar bepaalt de koppeling niet meer.
                if arousal_couples_to_event(ar_onset, ev_start, ev_end,
                                            window_post_s):
                    if abs(latency) < abs(best_latency):
                        best_latency = latency
                        best_match   = ev

            if best_match is not None:
                ar_copy["type"]               = "respiratory"
                ar_copy["linked_event_type"]  = best_match.get("type")
                ar_copy["linked_event_onset"] = best_match.get("onset_s")
                ar_copy["linked_event_dur"]   = best_match.get("duration_s")
                ar_copy["arousal_latency_s"]  = _safe(best_latency)
                result["respiratory_arousals"].append(ar_copy)
            else:
                ar_copy["type"] = "spontaneous"
                result["spontaneous_arousals"].append(ar_copy)

            ar_annotated.append(ar_copy)

        # Update originele events: had elk respiratoir event een arousal?
        for ev in resp_events:
            ev_end     = (ev.get("onset_s") or 0) + (ev.get("duration_s") or 0)
            ev_annotated = dict(ev)
            ev_annotated["had_arousal"] = False
            ev_annotated["arousal_latency_s"] = None

            for ar in result["respiratory_arousals"]:
                if ar.get("linked_event_onset") == ev.get("onset_s"):
                    ev_annotated["had_arousal"]        = True
                    ev_annotated["arousal_latency_s"]  = ar.get("arousal_latency_s")
                    break

            result["resp_events_with_arousal"].append(ev_annotated)

        # ── Statistieken ─────────────────────────────────────────
        n_resp_ar = len(result["respiratory_arousals"])
        n_spont   = len(result["spontaneous_arousals"])
        n_total   = n_resp_ar + n_spont
        n_resp_ev = len(resp_events)
        n_ev_with_ar = sum(1 for ev in result["resp_events_with_arousal"]
                           if ev["had_arousal"])

        latencies = [ar["arousal_latency_s"]
                     for ar in result["respiratory_arousals"]
                     if ar.get("arousal_latency_s") is not None]

        # Per event-type
        type_stats = {}
        for ev_type in ("obstructive","central","mixed","hypopnea",
                        "hypopnea_central"):
            ev_of_type = [e for e in resp_events if e.get("type") == ev_type]
            ar_for_type = [e for e in result["resp_events_with_arousal"]
                           if e.get("type") == ev_type and e.get("had_arousal")]
            if ev_of_type:
                type_stats[ev_type] = {
                    "n_events":      len(ev_of_type),
                    "n_with_arousal": len(ar_for_type),
                    "arousal_rate":   _safe(len(ar_for_type) /
                                           len(ev_of_type) * 100),
                }

        result["summary"] = {
            "n_respiratory_arousals":   n_resp_ar,
            "n_spontaneous_arousals":   n_spont,
            "n_total_arousals":         n_total,
            "pct_respiratory":          _safe(n_resp_ar / n_total * 100) if n_total > 0 else 0,
            "pct_spontaneous":          _safe(n_spont   / n_total * 100) if n_total > 0 else 0,
            # Koppeling met respiratoire events
            "n_resp_events_total":      n_resp_ev,
            "n_resp_events_with_arousal": n_ev_with_ar,
            "pct_events_with_arousal":  _safe(n_ev_with_ar / n_resp_ev * 100) if n_resp_ev > 0 else 0,
            # Latentie (seconden na event-einde)
            "avg_arousal_latency_s":    _safe(float(np.mean(latencies))) if latencies else None,
            "min_arousal_latency_s":    _safe(float(np.min(latencies)))  if latencies else None,
            "max_arousal_latency_s":    _safe(float(np.max(latencies)))  if latencies else None,
            # Per event-type
            "by_event_type":            type_stats,
            # Klinische interpretatie
            "clinical_interpretation":  _interpret_arousal_coupling(
                n_resp_ar, n_spont, n_ev_with_ar, n_resp_ev,
                float(np.mean(latencies)) if latencies else None),
        }
        result["arousals_annotated"] = ar_annotated
        result["success"] = True

    except Exception as e:
        result["error"]     = str(e)
        result["traceback"] = traceback.format_exc()
    return result


def _empty_correlation_summary() -> dict:
    return {
        "n_respiratory_arousals": 0, "n_spontaneous_arousals": 0,
        "n_total_arousals": 0, "pct_respiratory": 0,
        "n_resp_events_total": 0, "n_resp_events_with_arousal": 0,
        "pct_events_with_arousal": 0, "by_event_type": {},
        "clinical_interpretation": [],
    }


def _interpret_arousal_coupling(
    n_resp: int, n_spont: int,
    n_ev_with_ar: int, n_ev_total: int,
    avg_latency: float | None,
) -> list:
    """
    Genereer klinische interpretatie van het arousal-respiratoir verband.
    """
    msgs = []
    total = n_resp + n_spont

    if total == 0:
        return [{"level":"info","msg":"Geen arousals gedetecteerd."}]

    resp_pct   = n_resp / total * 100 if total > 0 else 0
    ev_ar_pct  = n_ev_with_ar / n_ev_total * 100 if n_ev_total > 0 else 0

    # Overheersend respiratoir
    if resp_pct >= 70:
        msgs.append({
            "level": "warning",
            "code":  "PREDOMINANTLY_RESPIRATORY",
            "msg":   f"{resp_pct:.0f}% van alle arousals is respiratoir van origine. "
                     "Slaapfragmentatie wordt gedomineerd door apnea/hypopnea-gerelateerde "
                     "ontwakingen — sterk argument voor nCPAP-therapie.",
        })
    elif resp_pct >= 40:
        msgs.append({
            "level": "info",
            "code":  "SIGNIFICANT_RESPIRATORY",
            "msg":   f"{resp_pct:.0f}% van de arousals is respiratoir. "
                     "Respiratoire events dragen significant bij aan slaapfragmentatie.",
        })
    else:
        msgs.append({
            "level": "info",
            "code":  "PREDOMINANTLY_SPONTANEOUS",
            "msg":   f"Meerderheid arousals ({100-resp_pct:.0f}%) is spontaan "
                     "(niet direct respiratoir). Overweeg andere oorzaken: "
                     "PLM, pijn, licht/geluid, medicatie.",
        })

    # Hoog percentage events met arousal
    if ev_ar_pct >= 60:
        msgs.append({
            "level": "warning",
            "code":  "HIGH_EVENT_AROUSAL_RATE",
            "msg":   f"{ev_ar_pct:.0f}% van de respiratoire events gaat gepaard met "
                     "een arousal. Ernstige slaapfragmentatie — hoge kans op "
                     "overmatige slaperigheid overdag (EDS/ESS).",
        })

    # Korte latentie = directe arousal = goede corticale respons
    if avg_latency is not None:
        if avg_latency < 5:
            msgs.append({
                "level": "info",
                "code":  "SHORT_AROUSAL_LATENCY",
                "msg":   f"Gemiddelde arousal-latentie {avg_latency:.1f}s — "
                         "snelle corticale respons op respiratoire stress. "
                         "Typisch bij mild-tot-matig OSAS.",
            })
        elif avg_latency > 12:
            msgs.append({
                "level": "warning",
                "code":  "LONG_AROUSAL_LATENCY",
                "msg":   f"Verlengde arousal-latentie ({avg_latency:.1f}s). "
                         "Vertraagde corticale arousal kan wijzen op verminderd "
                         "arousability — risicofactor bij ernstig OSAS.",
            })

    return msgs


# ═══════════════════════════════════════════════════════════════
# RERA DETECTIE  (Respiratory Effort Related Arousals)
# ═══════════════════════════════════════════════════════════════

def detect_reras(
    flow_data:    np.ndarray,
    flow_norm:    np.ndarray,
    sf_flow:      float,
    arousals:     list,
    resp_events:  list,
    hypno:        list,
    artifact_epochs: list = None,
    arousal_window_s: float = RERA_AROUSAL_WINDOW_S,
) -> dict:
    """
    Detecteer RERAs conform AASM, Sectie 3E.

    RERA = sequentie van ademhalingen met:
      1. Toenemende inspiratoire inspanning (flow plateau of crescendo-effort)
      2. ZONDER apnea of hypopnea drempel te bereiken (flow > 70% basislijn)
      3. Eindigend met een arousal
      4. Duur ≥10s

    Flow-limitatie criterium (plateau):
      Normaal: sinusvormige inspiratoire flow
      Gelimiteerd: afgeplatte top (plateau) = hogere bovenste luchtweg-weerstand
      Detectie: top-flatness ratio < 0.85 (verhouding piek/gemiddelde van inspiratie)
    """
    result = {"success": False, "events": [], "summary": {}, "error": None}
    try:
        # ── Detecteer flow-limitatie periodes ──
        flow_limited_mask = _detect_flow_limitation(flow_norm, sf_flow)

        # ── Verbind met slaap ──
        sleep_stages = {"N1","N2","N3","R",1,2,3,4}
        spe = int(sf_flow * EPOCH_LEN_S)
        sleep_mask = np.zeros(len(flow_norm), dtype=bool)
        for ep_i, stage in enumerate(hypno):
            if stage in sleep_stages:
                s = ep_i * spe
                e = min(s + spe, len(sleep_mask))
                sleep_mask[s:e] = True

        # ── Label flow-limitatie segmenten ──
        labeled, n_seg = label(flow_limited_mask & sleep_mask)
        rera_candidates = []

        for i in range(1, n_seg + 1):
            indices = np.where(labeled == i)[0]
            dur_s   = len(indices) / sf_flow
            if dur_s < RERA_MIN_DUR_S:
                continue

            onset_s = float(indices[0])  / sf_flow
            end_s   = float(indices[-1]) / sf_flow
            ep_idx  = int(onset_s // EPOCH_LEN_S)
            stage   = hypno[ep_idx] if ep_idx < len(hypno) else "W"

            rera_candidates.append({
                "onset_s":   _safe(onset_s),
                "end_s":     _safe(end_s),
                "duration_s": _safe(dur_s),
                "stage":     stage,
                "epoch":     ep_idx,
            })

        # ── Filter: alleen kandidaten die NIET overlappen met apnea/hypopnea ──
        confirmed_reras = []
        for cand in rera_candidates:
            overlap = False
            for ev in resp_events:
                ev_start = ev.get("onset_s", 0)
                ev_end   = ev_start + (ev.get("duration_s", 0))
                c_start  = cand["onset_s"] or 0
                c_end    = cand["end_s"]   or 0
                # Overlapping check
                if c_start < ev_end and c_end > ev_start:
                    overlap = True
                    break
            if overlap:
                continue

            # ── Vereiste: eindigend met arousal (binnen 10s) ──
            c_end = cand["end_s"] or 0
            has_arousal = False
            linked_arousal = None
            for ar in arousals:
                ar_onset = ar.get("onset_s", 0)
                if 0 <= ar_onset - c_end <= arousal_window_s:
                    has_arousal    = True
                    linked_arousal = ar_onset
                    break

            if has_arousal:
                cand["linked_arousal_onset"] = linked_arousal
                confirmed_reras.append(cand)

        # ── Statistieken ──
        _art_set = set(artifact_epochs or [])
        total_sleep_s = sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                            if _is_sleep(s) and i not in _art_set)
        total_sleep_h = total_sleep_s / 3600   # zie psgscoring/indices.py

        result["events"]  = confirmed_reras
        result["summary"] = {
            # NIET het getal dat gerapporteerd wordt. Er bestaan twee
            # RERA-definities naast elkaar: deze (flow-limitatie op de
            # envelope) en `pipeline._compute_rera_rdi` (FRI + flattening).
            # Alleen de tweede voedt `respiratory.summary.n_rera`, de RDI en
            # het PDF-rapport. Deze telling is diagnostisch.
            #
            # Ze stonden onder bijna dezelfde naam -- `n_reras` hier,
            # `n_rera` daar -- en een consument die de verkeerde pakt krijgt
            # een ander getal zonder dat iets dat meldt. `generate_psg_report.py`
            # in YASAFlaskified leest deze; die module staat in tasks.py
            # uitgecommentarieerd en is dus dood, maar wie hem terughaalt
            # rapporteert stilzwijgend de andere definitie.
            "authoritative": False,
            "reported_by":   "diagnostic only; see respiratory.summary.n_rera",
            "n_reras":     len(confirmed_reras),
            "rera_index":  per_hour(len(confirmed_reras), total_sleep_h),
            "rdi":         per_hour(len(resp_events) + len(confirmed_reras),
                                    total_sleep_h),  # RDI = AHI + RERA-index
        }
        result["success"] = True

    except Exception as e:
        result["error"]     = str(e)
        result["traceback"] = traceback.format_exc()
    return result


def _detect_flow_limitation(flow_norm: np.ndarray, sf: float) -> np.ndarray:
    """
    Detecteer flow-limitatie (plateau-vormige inspiratoire flow).

    Methode:
      Per ademhaling (0.5–4s periodes):
        - Piek flow bepalen
        - Top-flatness: verhouding gemiddelde van bovenste 50% / piek
        - Als flatness > 0.85 = normale top
        - Als flatness < 0.75 = afgeplatte top = flow-limitatie
    """
    limited = np.zeros(len(flow_norm), dtype=bool)

    # Splits in vermoedelijke inspiratoire cycli via pieken
    min_cycle_samples = int(0.5 * sf)
    max_cycle_samples = int(4.0 * sf)

    # Smooth signaal voor piekdetectie
    win = max(1, int(sf * 0.5))
    smooth = np.convolve(flow_norm, np.ones(win)/win, mode="same")

    # Vind lokale maxima (inspiratoire pieken)
    # (find_peaks geïmporteerd bovenaan module — v0.8.40)
    peaks, _ = find_peaks(smooth,
                          distance=min_cycle_samples,
                          height=0.40)  # minimale flow = 40% basislijn

    for pk in peaks:
        # Zoek het omringende inspiratoire segment (van dal naar dal)
        # v0.8.40: Bounded search voorkomt oneindige lus bij
        # monotoon stijgend/dalend signaal (slecht gekalibreerde sensor)
        max_extend = int(2.0 * sf)  # max 2s aan elke kant zoeken
        left  = pk
        right = pk
        steps = 0
        while left > 0 and smooth[left-1] < smooth[left] and steps < max_extend:
            left -= 1
            steps += 1
        steps = 0
        while right < len(smooth)-1 and smooth[right+1] < smooth[right] and steps < max_extend:
            right += 1
            steps += 1

        seg_len = right - left
        if seg_len < min_cycle_samples or seg_len > max_cycle_samples:
            continue

        seg = smooth[left:right+1]
        piek = float(np.max(seg))
        if piek < 0.40:
            continue

        # Top-flatness: gemiddelde van stalen > 75% van piek
        top_mask = seg > 0.75 * piek
        if not np.any(top_mask):
            continue

        flatness = float(np.mean(seg[top_mask])) / piek

        # Plateau = flatness < 0.80 (top is afgeplat)
        if flatness < 0.80:
            limited[left:right+1] = True

    # Smooth de mask (verwijder korte artefacten)
    from scipy.ndimage import binary_closing, binary_opening
    limited = binary_closing(limited,  structure=np.ones(int(sf)))
    limited = binary_opening(limited, structure=np.ones(int(sf * 2)))

    # Zorg dat flow-limitatie niet optreedt tijdens diepe apnea
    limited[flow_norm < 0.40] = False

    return limited


# ═══════════════════════════════════════════════════════════════
# GECOMBINEERDE ANALYSE + SAMENVATTING
# ═══════════════════════════════════════════════════════════════

def run_arousal_respiratory_analysis(
    eeg_data:    np.ndarray,
    sf_eeg:      float,
    flow_data:   np.ndarray | None,
    flow_norm:   np.ndarray | None,
    sf_flow:     float | None,
    resp_events: list,
    hypno:       list,
    emg_data:    np.ndarray | None = None,
    artifact_epochs: list = None,
    hr_data:     np.ndarray | None = None,
    sf_hr:       float = 1.0,
    derivations: list | None = None,
    per_channel_thresh: dict | None = None,
    eog_data:    np.ndarray | None = None,
    eog_reject:  bool = False,
    spectral_shift: bool = False,
    hysteresis:  bool = False,
    lgbm:        bool | None = None,
    lgbm_threshold: float | None = None,
    event_locked_threshold: float | None = None,
    onset_offset_s: float = 0.0,
    min_interval_s: float = 0.0,
    rem_alpha_baseline: bool = False,
) -> dict:
    """
    Master-functie: detecteer arousals, RERAs en koppel aan respiratoire events.

    ``derivations`` (optioneel): lijst ``[(naam, eeg_data[, sf]), ...]`` voor
    multi-derivatie-detectie (event-level union). Als None → exact het huidige
    single-channel gedrag op ``eeg_data`` (byte-identiek).

    Parameters
    ----------
    eeg_data    : primair EEG-kanaal (al geselecteerd, in μV)
    sf_eeg      : samplefrequentie EEG
    flow_data   : luchtstroom-signaal (voor RERA)
    flow_norm   : genormaliseerde luchtstroom (0–1, voor RERA)
    sf_flow     : samplefrequentie luchtstroom
    resp_events : lijst van respiratoire events uit detect_respiratory_events()
    hypno       : slaapfase-lijst
    emg_data    : optioneel chin-EMG signaal voor REM arousal criterium
    artifact_epochs : epoch-indices met artefacten (uitgesloten uit detectie + indices)

    Returns
    -------
    dict met arousals, koppeling, RERAs en samenvatting
    """
    output = {"success": False, "error": None}

    # ── Stap 1: Arousals detecteren ──────────────────────────────
    logger.info("[arousal 1/3] EEG-arousal detectie...")
    # De event-eindes komen uit de respiratoire scoring die HIERVOOR gedraaid
    # heeft -- `resp_events` staat al in de signatuur. Het diagnosedocument
    # ging uit van de omgekeerde volgorde en stelde een two-pass voor; dat is
    # niet nodig, de events zijn er al.
    _ends = None
    if event_locked_threshold is not None:
        _ends = [float(e["onset_s"]) + float(e["duration_s"])
                 for e in (resp_events or [])
                 if e.get("onset_s") is not None
                 and e.get("duration_s") is not None]
        if _ends:
            logger.info("[arousal] event-locked werkpunt %.2f rond %d "
                        "respiratoire event-eindes",
                        event_locked_threshold, len(_ends))
    if derivations:
        ar_result = detect_arousals_multi(derivations, sf_eeg, hypno, emg_data=emg_data,
                                          artifact_epochs=artifact_epochs,
                                          hr_data=hr_data, sf_hr=sf_hr,
                                          per_channel_thresh=per_channel_thresh,
                                          eog_data=eog_data, eog_reject=eog_reject,
                                          spectral_shift=spectral_shift,
                                          hysteresis=hysteresis, lgbm=lgbm,
                                          lgbm_threshold=lgbm_threshold,
                                          resp_event_ends=_ends,
                                          event_locked_threshold=event_locked_threshold,
                                          min_interval_s=min_interval_s,
                                          rem_alpha_baseline=rem_alpha_baseline)
    else:
        ar_result = detect_arousals(eeg_data, sf_eeg, hypno, emg_data=emg_data,
                                    artifact_epochs=artifact_epochs,
                                    hr_data=hr_data, sf_hr=sf_hr,
                                    spectral_shift=spectral_shift,
                                    hysteresis=hysteresis, lgbm=lgbm,
                                    lgbm_threshold=lgbm_threshold,
                                    resp_event_ends=_ends,
                                    event_locked_threshold=event_locked_threshold,
                                    min_interval_s=min_interval_s,
                                    rem_alpha_baseline=rem_alpha_baseline)
    # -- Onsetverschuiving ------------------------------------------
    # HIER en niet later: stap 2 koppelt deze arousals aan de respiratoire
    # events en stap 3 draait er de RERA-detectie op. Na afloop schuiven zou
    # alleen veranderen wat er GERAPPORTEERD wordt, niet wat de AHI en de RDI
    # voedt -- en juist dat onderscheid is het hele punt van de vlag.
    if onset_offset_s:
        for _e in (ar_result.get("events") or []):
            for _k in ("onset_s", "end_s"):
                if _e.get(_k) is not None:
                    _e[_k] = round(float(_e[_k]) + float(onset_offset_s), 3)
        logger.info("[arousal] onsets %+.2f s verschoven (profielvlag)",
                    onset_offset_s)
        ar_result.setdefault("summary", {})["onset_offset_s"] = float(onset_offset_s)

    output["arousals"] = ar_result

    arousals = ar_result.get("events", [])

    # ── Stap 2: Koppeling met respiratoire events ─────────────────
    logger.info("[arousal 2/3] Respiratoir-arousal koppeling...")
    corr_result = correlate_arousals_to_respiratory(arousals, resp_events)
    output["coupling"] = corr_result

    # Update arousal-events met type-annotaties
    if corr_result.get("arousals_annotated"):
        output["arousals"]["events"] = corr_result["arousals_annotated"]

    # ── Stap 3: RERA detectie ─────────────────────────────────────
    logger.info("[arousal 3/3] RERA detectie...")
    if flow_data is not None and flow_norm is not None and sf_flow is not None:
        rera_result = detect_reras(
            flow_data, flow_norm, sf_flow,
            arousals, resp_events, hypno,
            artifact_epochs=artifact_epochs)
        output["reras"] = rera_result
    else:
        output["reras"] = {
            "success": False,
            "error": "Geen luchtstroom-data voor RERA",
            "events": [], "summary": {"n_reras": 0, "rera_index": 0, "rdi": None},
        }

    # ── Gecombineerde samenvatting ────────────────────────────────
    ar_sum   = ar_result.get("summary",   {})
    cor_sum  = corr_result.get("summary", {})
    rera_sum = output["reras"].get("summary", {})

    output["summary"] = {
        # Arousal totalen
        "arousal_index":              ar_sum.get("arousal_index"),
        "nrem_arousal_index":         ar_sum.get("nrem_arousal_index"),
        "rem_arousal_index":          ar_sum.get("rem_arousal_index"),
        "arousal_severity":           ar_sum.get("severity"),

        # Respiratoir-arousal verband
        "n_respiratory_arousals":     cor_sum.get("n_respiratory_arousals"),
        "n_spontaneous_arousals":     cor_sum.get("n_spontaneous_arousals"),
        "pct_respiratory_arousals":   cor_sum.get("pct_respiratory"),
        "pct_events_with_arousal":    cor_sum.get("pct_events_with_arousal"),
        "avg_arousal_latency_s":      cor_sum.get("avg_arousal_latency_s"),
        "by_event_type":              cor_sum.get("by_event_type", {}),

        # RERAs
        "n_reras":                    rera_sum.get("n_reras", 0),
        "rera_index":                 rera_sum.get("rera_index", 0),
        "rdi":                        rera_sum.get("rdi"),

        # Klinische interpretatie
        "clinical_interpretation":    cor_sum.get("clinical_interpretation", []),
    }

    # Herkomst van de arousal-index zelf. Dit stond alleen in de GENESTE
    # samenvatting (`output["arousals"]["summary"]`), en de rapportlaag leest
    # de platte. Gevolg: een index uit het regelgebaseerde pad en een index
    # uit het gefilterde pad zagen er in het rapport identiek uit, terwijl ze
    # een factor kunnen schelen. Alleen doorgeven wat er is -- op een profiel
    # zonder classifier blijven de sleutels weg.
    for _k in ("lgbm_available", "lgbm_skipped_reason", "lgbm_threshold",
               "lgbm_n_pre", "lgbm_n_post", "n_event_locked",
               "event_locked_threshold", "min_interval_s", "n_interval_merged",
               "n_too_long_discarded", "too_long_discarded_s",
               "max_duration_s"):
        if _k in ar_sum:
            output["summary"][_k] = ar_sum[_k]

    output["success"] = True
    logger.info("Arousal-analyse voltooid.")
    return output
