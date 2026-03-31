"""
pneumo_analysis.py — Automatische pneumologische scoring voor YASAFlaskified v0.8.0
Verwerkt respiratoire EDF-kanalen conform AASM 2.6 richtlijnen.

v14 verbeteringen:
  - Breath-by-breath amplitude detectie (AASM-conform piek-tot-dal)
  - Per-ademhaling event boundaries
  - Flattening index (inspiratoire flow-limitatie)
  - Stadium-specifieke basislijn (REM vs NREM)
  - Positie-reset basislijn (bij positieverandering)
  - Cheyne-Stokes respiratie detectie (autocorrelatie)
  - Dual-sensor: thermistor voor apneu, nasale druk voor hypopneu
  - Paradoxale ademhaling = obstructief (ruwe signaalvariabiliteit)
"""

import numpy as np
import pandas as pd
try:
    import mne
except ImportError:
    mne = None
import traceback
from collections import Counter
from scipy import signal
from scipy.ndimage import label, maximum_filter1d
from scipy.stats import pearsonr
import logging
logger = logging.getLogger("psgscoring.respiratory")

# Arousal-analyse module (v7.2)
try:
    from ._arousal import run_arousal_respiratory_analysis
    _AROUSAL_AVAILABLE = True
except ImportError:
    _AROUSAL_AVAILABLE = False

# ─────────────────────────────────────────────
# KANAALNAAM-PATRONEN
# ─────────────────────────────────────────────

CHANNEL_PATTERNS = {
    # AASM 2.6: Nasale druk transducer → hypopnea scoring (gevoeliger)
    "flow_pressure": ["nasal pressure", "nasalpressure", "ptaf", "pnasale",
                      "cannula", "npt", "nasal pres", "pflow", "np ",
                      "naf", "nasal flow"],
    # AASM 2.6: Oronasale thermistor → apnea scoring (detecteert cessatie)
    "flow_thermistor": ["thermistor", "therm", "thermist", "oronasal",
                        "oro-nasal", "airflow", "air flow"],
    # Fallback: generiek "flow" kanaal (als geen specifiek kanaal)
    "flow":     ["flow", "nasal", "resp flow"],
    "thorax":   ["thorax", "thor", "chest", "thoracic", "ribcage",
                 "effort thor", "rc", "rib", "chest belt", "rcg"],
    "abdomen":  ["abdom", "abd", "belly", "effort abd", "ab",
                 "abdominal", "effort abdom", "abd belt", "abdo"],
    "spo2":     ["spo2", "sao2", "saturation", "o2", "oximetry",
                 "puls spo2", "pulse ox"],
    "pulse":    ["pulse", "pr", "heart rate", "hr", "puls rate"],
    "ecg":      ["ecg", "ekg", "cardiac", "einthoven", "ii", "ecg ii"],
    "position": ["position", "positie", "pos", "body pos", "lage",
                 "body position", "bpos"],
    "snore":    ["snore", "snoring", "ronfle", "ronchus", "micro",
                 "snurk", "snore mic", "microphone"],
    "leg_l":    ["leg l", "lleg", "emg leg l", "tibial l", "left leg",
                 "plo", "pla", "plg l", "lat l", "lats l",
                 "bein l", "bein li", "jambe g", "tib ant l",
                 "emg tib l", "emg tibant l", "plm l", "plm-l",
                 "emg la", "ta l", "ta li"],
    "leg_r":    ["leg r", "rleg", "emg leg r", "tibial r", "right leg",
                 "pro", "pra", "plg r", "lat r", "lats r",
                 "bein r", "bein re", "jambe d", "tib ant r",
                 "emg tib r", "emg tibant r", "plm r", "plm-r",
                 "emg ra", "ta r", "ta re"],
    "eeg":      ["eeg", "c3", "c4", "f3", "f4", "o1", "o2"],
}

# ── AASM 2.6 drempelwaarden ──
APNEA_THRESHOLD          = 0.10    # <10% basislijn = apnea
HYPOPNEA_THRESHOLD       = 0.70    # <70% basislijn = hypopnea
APNEA_MIN_DUR_S          = 10.0
HYPOPNEA_MIN_DUR_S       = 10.0
DESATURATION_DROP_PCT    = 3.0     # ≥3% SpO2-daling voor hypopnea-bevestiging
EFFORT_ABSENT_RATIO      = 0.20    # <20% van baseline effort = afwezig
EFFORT_PRESENT_RATIO     = 0.40    # >40% = aanwezig
MIXED_SPLIT_FRACTION     = 0.50    # eerste 50% = centraal deel bij gemengd
EPOCH_LEN_S              = 30
BASELINE_WINDOW_S        = 300     # 5 min venster voor dynamische basislijn

POSITION_MAP = {0: "Prone", 1: "Left", 2: "Supine", 3: "Right", 4: "Upright"}


# ═══════════════════════════════════════════════════════════════
# KANAALDETECTIE
# ═══════════════════════════════════════════════════════════════

def detect_channels(ch_names: list) -> dict:
    """Detecteer EDF-kanalen automatisch op basis van labelnamen (case-insensitive matching)."""
    ch_lower = {ch.lower(): ch for ch in ch_names}
    found = {}
    for ch_type, patterns in CHANNEL_PATTERNS.items():
        for pat in patterns:
            match = next(
                (orig for lc, orig in ch_lower.items() if pat in lc), None)
            if match:
                found[ch_type] = match
                break
    return found


def channel_map_from_user(user_map: dict, ch_names: list) -> dict:
    """Bouw een kanaal-mapping vanuit gebruikersinput (overschrijft auto-detectie)."""
    auto = detect_channels(ch_names)
    merged = {**auto}
    for k, v in (user_map or {}).items():
        if v and v in ch_names:
            merged[k] = v
    return merged


# ═══════════════════════════════════════════════════════════════
# HULPFUNCTIES
# ═══════════════════════════════════════════════════════════════

def safe_r(val, dec=1):
    """Veilige Pearson-correlatie: geeft 0.0 terug bij constante of te korte arrays."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), dec)
    except Exception:
        return None


def hypno_to_numeric(hypno: list) -> np.ndarray:
    """Converteer slaapstadia (W/N1/N2/N3/R) naar numerieke waarden (0/1/2/3/4)."""
    mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
    return np.array([mapping.get(s, -1) for s in hypno])


def is_nrem(stage) -> bool:
    """Controleer of een slaapstadium NREM is (N1, N2 of N3)."""
    return stage in (1, 2, 3, "N1", "N2", "N3")


def is_rem(stage) -> bool:
    """Controleer of een slaapstadium REM is."""
    return stage in (4, "R")


def is_sleep(stage) -> bool:
    """Controleer of een slaapstadium slaap is (niet Wake)."""
    return stage not in (0, -1, "W")


def _fmt_time(seconds: float) -> str:
    """Formateer een tijdstip in seconden naar HH:MM:SS notatie."""
    if seconds is None:
        return "—"
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def build_sleep_mask(hypno: list, sf: float,
                     total_samples: int,
                     artifact_epochs: list = None) -> np.ndarray:
    """
    Bouw een sample-level mask: True = geldige slaap.
    Excludeert Wake (stage 0) EN artefact-epochs.
    
    artifact_epochs: lijst van epoch-nummers die artefact bevatten
                     (uit yasa_analysis.run_artifact_detection)
    """
    hypno_num = hypno_to_numeric(hypno)
    artifact_set = set(artifact_epochs or [])
    spe = int(sf * EPOCH_LEN_S)
    mask = np.zeros(total_samples, dtype=bool)
    for ep_i, stage in enumerate(hypno_num):
        s = ep_i * spe
        e = min(s + spe, total_samples)
        if stage > 0 and ep_i not in artifact_set:
            mask[s:e] = True
    return mask


# ═══════════════════════════════════════════════════════════════
# LUCHTSTROOM PREPROCESSING
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# v0.8.0: GEVALIDEERDE SIGNAALVERWERKING
# ═══════════════════════════════════════════════════════════════

def linearize_nasal_pressure(data: np.ndarray) -> np.ndarray:
    """
    Vierkantswortel-transformatie van nasale druksignaal.

    AASM Rule 3: nasale druk wordt aanbevolen voor hypopnea-detectie,
    maar het signaal is proportioneel aan flow² (Bernoulli).
    Zonder linearisatie: 50% flowdaling → 75% amplitudereductie → overschatting.

    Referenties:
      - Monserrat et al. (AJRCCM 2001): √-transformatie lineariseert
        nasale druk/flow-relatie (r²=0.88-0.96 vs pneumotachografie)
      - Thurnheer et al.: bevestigt linearisatie over korte periodes
      - AASM Scoring Manual 2.6 Rule 3: "nasal pressure transducer
        (with or without square root transformation of the signal)"

    Formule: sign(x) × √|x|  (behoudt richting van inspiratie/expiratie)
    """
    return np.sign(data) * np.sqrt(np.abs(data))


def compute_mmsd(flow_data: np.ndarray, sf: float,
                 window_s: float = 1.0) -> np.ndarray:
    """
    Mean Magnitude of Second Derivative (MMSD) van het flowsignaal.

    Robuuste maat voor ademhalingsactiviteit die ONAFHANKELIJK is van
    absolute amplitude en baseline-drift. Detecteert de "scherpte" van
    de golfvorm.

    Referentie:
      - Lee et al. (Physiol Meas 2008): MMSD-gebaseerde apnea-detectie
        behaalde 92% overeenstemming met manuele scoring (κ=0.78)
        op 24 PSG-opnames.

    Parameters
    ----------
    flow_data : gefilterd flowsignaal (na bandpass, VOOR envelope)
    sf        : samplefrequentie
    window_s  : venstergrootte voor gemiddelde (default 1s)

    Returns
    -------
    mmsd : per-sample MMSD waarde (hoge waarde = actieve ademhaling)
    """
    # Tweede afgeleide (versnelling van flow)
    d2 = np.diff(flow_data, n=2)
    # Pad terug naar originele lengte
    d2 = np.concatenate([[0], d2, [0]])
    # Gemiddelde magnitude in glijdend venster
    abs_d2 = np.abs(d2)
    win = max(1, int(sf * window_s))
    kernel = np.ones(win) / win
    mmsd = np.convolve(abs_d2, kernel, mode="same")
    return mmsd


def preprocess_flow(flow_data: np.ndarray, sf: float,
                    is_nasal_pressure: bool = False) -> np.ndarray:
    """
    Band-pass 0.1–3 Hz → Hilbert envelope → 1s smooth.

    v0.8.0: optionele √-linearisatie voor nasale druksignaal.
    """
    # v0.8.0: Lineariseer nasale druk VOOR filtering
    if is_nasal_pressure:
        flow_data = linearize_nasal_pressure(flow_data)

    nyq = sf / 2
    lo  = max(0.05 / nyq, 0.001)
    hi  = min(3.0  / nyq, 0.99)
    b, a = signal.butter(3, [lo, hi], btype="band")
    filtered = signal.filtfilt(b, a, flow_data)
    envelope = np.abs(signal.hilbert(filtered))
    win = max(1, int(sf))
    smooth = np.convolve(envelope, np.ones(win) / win, mode="same")
    return smooth


def compute_dynamic_baseline(flow_env: np.ndarray, sf: float,
                              window_s: int = BASELINE_WINDOW_S) -> np.ndarray:
    """
    Dynamische basislijn: glijdend 5-minuten venster, 95e percentiel.
    Geeft per sample een basislijn terug — adapteert aan langzame drift.

    v7.1 FIX: geoptimaliseerd — berekent elke 10s ipv per sample.
    Bij 256 Hz + 8u opname: ~2.880 iteraties ipv ~7.400.000.
    """
    win  = int(window_s * sf)
    n    = len(flow_env)
    step = max(1, int(sf * 10))   # elke 10 seconden

    sample_points   = np.arange(0, n, step)
    baseline_sparse = np.empty(len(sample_points))

    for idx, center in enumerate(sample_points):
        start = max(0, center - win // 2)
        end   = min(n, center + win // 2)
        seg   = flow_env[start:end]
        p95   = np.percentile(seg, 95)
        stable = seg[seg > 0.30 * p95]
        if len(stable) > 10:
            baseline_sparse[idx] = np.percentile(stable, 95)
        else:
            baseline_sparse[idx] = p95

    # Interpoleer terug naar volledige resolutie
    baseline = np.interp(np.arange(n), sample_points, baseline_sparse)
    return np.maximum(baseline, 1e-6)


def preprocess_effort(effort_data: np.ndarray, sf: float) -> np.ndarray:
    """
    Effort-signaal (thorax/abdomen) preprocessen:
    Band-pass 0.05–2 Hz → amplitude envelope.
    """
    nyq = sf / 2
    lo  = max(0.03 / nyq, 0.001)
    hi  = min(2.0  / nyq, 0.99)
    b, a = signal.butter(3, [lo, hi], btype="band")
    filtered = signal.filtfilt(b, a, effort_data)
    envelope = np.abs(signal.hilbert(filtered))
    win = max(1, int(sf * 2))
    return np.convolve(envelope, np.ones(win) / win, mode="same")


def bandpass_flow(flow_data: np.ndarray, sf: float) -> np.ndarray:
    """Band-pass filteren (0.05–3 Hz) zonder envelope — behoudt waveform."""
    nyq = sf / 2
    lo  = max(0.05 / nyq, 0.001)
    hi  = min(3.0  / nyq, 0.99)
    b, a = signal.butter(3, [lo, hi], btype="band")
    return signal.filtfilt(b, a, flow_data)


# ═══════════════════════════════════════════════════════════════
# BREATH-BY-BREATH ANALYSE (v14 — AASM-conform)
# ═══════════════════════════════════════════════════════════════

def detect_breaths(flow_filtered: np.ndarray, sf: float,
                   min_breath_s: float = 1.0,
                   max_breath_s: float = 15.0) -> list[dict]:
    """
    Detecteer individuele ademhalingen via zero-crossings op het
    gefilterde flow-signaal. Elke ademhaling = inspiratie + expiratie.

    Returns
    -------
    list of dict met per ademhaling:
      - start, mid, end (sample indices)
      - onset_s, duration_s (in seconden)
      - peak_insp: maximale inspiratoire flow
      - trough_exp: minimale expiratoire flow
      - amplitude: piek-tot-dal (AASM-definitie)
      - insp_segment: numpy array van inspiratoire fase
    """
    if len(flow_filtered) < int(sf * 2):
        return []

    # Zero-crossings: signaalwisselingen
    sign = np.sign(flow_filtered)
    sign[sign == 0] = 1  # vermijd nul
    crossings = np.where(np.diff(sign))[0]

    if len(crossings) < 3:
        return []

    breaths = []
    min_samp = int(min_breath_s * sf)
    max_samp = int(max_breath_s * sf)

    i = 0
    while i < len(crossings) - 1:
        start = crossings[i]
        end   = crossings[i + 1] if i + 1 < len(crossings) else len(flow_filtered) - 1

        # Zoek volgende crossing voor complete ademcyclus (insp+exp)
        if i + 2 < len(crossings):
            cycle_end = crossings[i + 2]
        else:
            cycle_end = end
            i += 1
            continue

        dur_samp = cycle_end - start
        if dur_samp < min_samp or dur_samp > max_samp:
            i += 1
            continue

        seg = flow_filtered[start:cycle_end]
        mid = start + np.argmax(np.abs(seg[:len(seg)//2+1]))  # midden ~inspiratie-piek

        # Bepaal of inspiratie positief of negatief is
        first_half = flow_filtered[start:start + dur_samp//2]
        second_half = flow_filtered[start + dur_samp//2:cycle_end]

        if np.mean(first_half) > np.mean(second_half):
            # Inspiratie = positief, expiratie = negatief
            peak_insp  = float(np.max(first_half))
            trough_exp = float(np.min(second_half))
            insp_seg   = first_half
        else:
            peak_insp  = float(np.max(second_half))
            trough_exp = float(np.min(first_half))
            insp_seg   = second_half

        amplitude = abs(peak_insp - trough_exp)

        breaths.append({
            "start":       start,
            "mid":         mid,
            "end":         cycle_end,
            "onset_s":     start / sf,
            "duration_s":  dur_samp / sf,
            "peak_insp":   peak_insp,
            "trough_exp":  trough_exp,
            "amplitude":   amplitude,
            "insp_segment": insp_seg,
        })
        i += 2  # spring naar volgende ademcyclus

    return breaths


def compute_breath_amplitudes(breaths: list[dict], sf: float,
                              window_breaths: int = 10) -> np.ndarray:
    """
    Per ademhaling: amplitude als fractie van lokale basislijn.

    Basislijn = mediaan van de voorgaande `window_breaths` ademhalingen.

    Returns: array van length len(breaths), waarden 0–2+.
             1.0 = normaal, 0.5 = 50% reductie, etc.
    """
    n = len(breaths)
    if n == 0:
        return np.array([])

    amps = np.array([b["amplitude"] for b in breaths])
    ratios = np.ones(n)

    for i in range(n):
        # Lokale basislijn: mediaan van voorgaande ademhalingen
        start = max(0, i - window_breaths)
        baseline_amps = amps[start:i] if i > 0 else amps[:1]
        # Gebruik alleen ademhalingen met redelijke amplitude
        good = baseline_amps[baseline_amps > np.percentile(baseline_amps, 25)]
        if len(good) > 2:
            bl = float(np.median(good))
        elif len(baseline_amps) > 0:
            bl = float(np.median(baseline_amps))
        else:
            bl = float(amps[i])
        ratios[i] = amps[i] / bl if bl > 1e-9 else 1.0

    return ratios


def compute_flattening_index(insp_segment: np.ndarray) -> float:
    """
    Bereken de flattening index van een inspiratoir segment.

    Flattening = mate van plateau in de inspiratoire flow:
      - Normaal profiel: driehoekig/rond → index laag (~0.1)
      - Flow-limitatie: plateau → index hoog (>0.3)
      - Compleet plat: index ~1.0

    Methode: fractie van het inspiratoire segment dat >80% van de
    piekamplitude bedraagt.
    """
    if len(insp_segment) < 5:
        return 0.0
    peak = np.max(np.abs(insp_segment))
    if peak < 1e-9:
        return 1.0  # geen flow = maximaal flat
    threshold = 0.80 * peak
    n_above = np.sum(np.abs(insp_segment) > threshold)
    return float(n_above / len(insp_segment))


def detect_breath_events(breaths: list[dict], breath_ratios: np.ndarray,
                         sf: float, hypno: list,
                         apnea_thresh: float = 0.10,
                         hypopnea_thresh: float = 0.70,
                         min_dur_s: float = 10.0) -> tuple[list, list]:
    """
    Detecteer apneu- en hypopneu-events op basis van per-ademhaling
    amplitude-reductie (AASM-conform).

    Een event begint bij de eerste ademhaling met voldoende reductie
    en eindigt bij de eerste ademhaling die weer boven de drempel komt.

    Returns: (apnea_events, hypopnea_candidates)
    """
    if len(breaths) == 0:
        return [], []

    apneas = []
    hypopneas = []

    n = len(breaths)
    i = 0

    while i < n:
        ratio = breath_ratios[i]

        if ratio > hypopnea_thresh:
            i += 1
            continue

        # Start van een event
        event_start_idx = i
        event_start_s = breaths[i]["onset_s"]
        is_apnea = ratio < apnea_thresh
        min_ratio = ratio

        # Zoek einde van het event
        j = i + 1
        while j < n and breath_ratios[j] < hypopnea_thresh:
            if breath_ratios[j] < apnea_thresh:
                is_apnea = True
            min_ratio = min(min_ratio, breath_ratios[j])
            j += 1

        event_end_s = breaths[j - 1]["onset_s"] + breaths[j - 1]["duration_s"]
        event_dur = event_end_s - event_start_s

        if event_dur >= min_dur_s:
            # Bereken gemiddelde flattening over event
            flat_indices = []
            for k in range(event_start_idx, j):
                if breaths[k].get("insp_segment") is not None and len(breaths[k]["insp_segment"]) > 3:
                    flat_indices.append(compute_flattening_index(breaths[k]["insp_segment"]))
            avg_flattening = float(np.mean(flat_indices)) if flat_indices else None

            ep_idx = int(event_start_s // EPOCH_LEN_S)
            stage = hypno[ep_idx] if ep_idx < len(hypno) else "W"

            event = {
                "onset_s":       safe_r(event_start_s),
                "duration_s":    safe_r(event_dur),
                "stage":         stage,
                "epoch":         ep_idx,
                "min_ratio":     safe_r(min_ratio, 3),
                "n_breaths":     j - event_start_idx,
                "breath_start":  event_start_idx,
                "breath_end":    j,
                "avg_flattening": safe_r(avg_flattening, 3),
                "sample_start":  breaths[event_start_idx]["start"],
                "sample_end":    breaths[j - 1]["end"],
            }

            if is_apnea:
                apneas.append(event)
            else:
                hypopneas.append(event)

        i = j  # spring voorbij het event

    return apneas, hypopneas


def compute_stage_baseline(flow_env: np.ndarray, sf: float, hypno: list,
                           artifact_epochs: list = None) -> np.ndarray:
    """
    Stadium-specifieke basislijn.

    NREM en REM hebben verschillende ademhalingspatronen:
      - NREM: stabiel, regulair
      - REM: variabeler, onregelmatiger (fysiologisch)

    Berekent per-stadium baseline en interpoleert per sample.
    """
    artifact_set = set(artifact_epochs or [])
    spe = int(sf * EPOCH_LEN_S)
    n = len(flow_env)

    # Bereken per-stadium baseline
    stage_baselines = {}
    for stage in ["N1", "N2", "N3", "R"]:
        samples = []
        for ep_i, s in enumerate(hypno):
            if s == stage and ep_i not in artifact_set:
                start = ep_i * spe
                end = min(start + spe, n)
                samples.extend(flow_env[start:end].tolist())
        if len(samples) > int(sf * 30):
            arr = np.array(samples)
            stable = arr[arr > np.percentile(arr, 30)]
            if len(stable) > 10:
                stage_baselines[stage] = float(np.percentile(stable, 90))

    if not stage_baselines:
        # Fallback: globale dynamische basislijn
        return compute_dynamic_baseline(flow_env, sf)

    # Interpoleer per epoch
    baseline = np.zeros(n)
    global_bl = compute_dynamic_baseline(flow_env, sf)

    for ep_i, stage in enumerate(hypno):
        start = ep_i * spe
        end = min(start + spe, n)
        if stage in stage_baselines:
            baseline[start:end] = stage_baselines[stage]
        else:
            # Wake of onbekend: gebruik globale basislijn
            baseline[start:end] = global_bl[start:end]

    # Smooth de overgangen (5s ramp) om abrupte sprongen te voorkomen
    win = max(1, int(sf * 5))
    baseline = np.convolve(baseline, np.ones(win) / win, mode="same")

    return np.maximum(baseline, 1e-6)


def detect_position_changes(pos_data: np.ndarray, sf: float,
                            min_stable_s: float = 30.0) -> list[dict]:
    """
    Detecteer positieveranderingen in het position-kanaal.

    Returns: lijst van {"sample": int, "time_s": float, "from": int, "to": int}
    """
    if pos_data is None or len(pos_data) < int(sf * 60):
        return []

    # Quantize positie (typisch 0-4 discrete waarden)
    pos_q = np.round(pos_data).astype(int)

    # Smooth om ruis te verwijderen (mediaan filter)
    from scipy.ndimage import median_filter
    win = max(3, int(sf * 5))
    if win % 2 == 0:
        win += 1
    pos_smooth = median_filter(pos_q, size=win)

    changes = []
    prev_pos = pos_smooth[0]
    prev_change_sample = 0

    for i in range(1, len(pos_smooth)):
        if pos_smooth[i] != prev_pos:
            # Check of positie stabiel is (niet flicker)
            check_end = min(i + int(min_stable_s * sf), len(pos_smooth))
            if check_end - i > int(sf * 10):
                stable_seg = pos_smooth[i:check_end]
                if np.sum(stable_seg == pos_smooth[i]) > 0.8 * len(stable_seg):
                    changes.append({
                        "sample": i,
                        "time_s": i / sf,
                        "from": int(prev_pos),
                        "to": int(pos_smooth[i]),
                    })
                    prev_pos = pos_smooth[i]
                    prev_change_sample = i

    return changes


def reset_baseline_at_position_changes(baseline: np.ndarray, flow_env: np.ndarray,
                                        sf: float, pos_changes: list[dict],
                                        recalc_window_s: float = 60.0) -> np.ndarray:
    """
    Na een positieverandering: herbereken de basislijn over de eerste
    60s in de nieuwe positie (de oude basislijn is niet meer relevant).
    """
    if not pos_changes:
        return baseline

    result = baseline.copy()
    n = len(flow_env)

    for change in pos_changes:
        sample = change["sample"]
        recalc_end = min(sample + int(recalc_window_s * sf), n)

        # Bereken nieuwe basislijn over eerste 60s na positieverandering
        seg = flow_env[sample:recalc_end]
        if len(seg) > int(sf * 10):
            stable = seg[seg > np.percentile(seg, 30)]
            if len(stable) > 10:
                new_bl = float(np.percentile(stable, 90))
            else:
                new_bl = float(np.percentile(seg, 90))

            # Ramp: geleidelijke overgang (10s)
            ramp_samp = min(int(sf * 10), recalc_end - sample)
            for i in range(ramp_samp):
                alpha = i / ramp_samp
                idx = sample + i
                if idx < n:
                    result[idx] = (1 - alpha) * result[idx] + alpha * new_bl

            # Rest van herberekende zone: nieuwe basislijn
            result[sample + ramp_samp:recalc_end] = new_bl

    return result


def detect_cheyne_stokes(flow_env: np.ndarray, sf: float,
                         hypno: list,
                         min_cycle_s: float = 40.0,
                         max_cycle_s: float = 120.0) -> dict:
    """
    Detecteer Cheyne-Stokes respiratie (CSR) patroon.

    CSR = cyclisch crescendo-decrescendo ademhalingspatroon met
    centrale apneus. Typisch bij hartfalen.

    Methode:
    1. Bereken amplitude-envelope op lage frequentie (periodiciteit)
    2. Autocorrelatie: zoek periodiek patroon in 40-120s range
    3. Als consistent periodiek: markeer als CSR

    Returns: dict met success, csr_detected, cycles, periodicity_s, etc.
    """
    result = {"success": False, "csr_detected": False, "cycles": [],
              "periodicity_s": None, "csr_minutes": 0}
    try:
        if len(flow_env) < int(sf * min_cycle_s * 3):
            result["success"] = True
            return result

        # Zeer lage frequentie envelope (0.005-0.05 Hz = 20-200s periodes)
        nyq = sf / 2
        lo = max(0.005 / nyq, 0.0001)
        hi = min(0.05 / nyq, 0.49)
        if lo >= hi:
            result["success"] = True
            return result

        b, a = signal.butter(2, [lo, hi], btype="band")
        slow_env = signal.filtfilt(b, a, flow_env)
        slow_env = np.abs(slow_env)

        # Autocorrelatie
        n = len(slow_env)
        min_lag = int(min_cycle_s * sf)
        max_lag = min(int(max_cycle_s * sf), n // 2)

        if max_lag <= min_lag:
            result["success"] = True
            return result

        # Genormaliseerde autocorrelatie
        slow_centered = slow_env - np.mean(slow_env)
        var = np.var(slow_centered)
        if var < 1e-12:
            result["success"] = True
            return result

        lags = np.arange(min_lag, max_lag, max(1, int(sf * 2)))
        acorr = np.array([
            np.mean(slow_centered[:n-lag] * slow_centered[lag:]) / var
            for lag in lags
        ])

        # Zoek piek in autocorrelatie
        if len(acorr) > 2:
            peak_idx = np.argmax(acorr)
            peak_val = acorr[peak_idx]

            if peak_val > 0.3:  # significante periodiciteit
                period_s = lags[peak_idx] / sf
                result["periodicity_s"] = safe_r(period_s)
                result["csr_detected"] = True

                # Tel CSR-cycli: zoek segmenten met consistent patroon
                # (vereenvoudigd: kijk per 2-minuten segment)
                seg_len = int(120 * sf)
                csr_minutes = 0
                for seg_start in range(0, n - seg_len, seg_len):
                    seg = slow_centered[seg_start:seg_start + seg_len]
                    seg_var = np.var(seg)
                    if seg_var > 0.5 * var:
                        csr_minutes += 2

                result["csr_minutes"] = csr_minutes
                result["csr_pct_sleep"] = safe_r(
                    csr_minutes / max(1, sum(EPOCH_LEN_S for s in hypno if s != "W") / 60) * 100)

        result["success"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════
# AASM APNEA CLASSIFICATIE — KERN
# ═══════════════════════════════════════════════════════════════

def classify_apnea_type(
    onset_idx: int,
    end_idx:   int,
    thorax_env:   np.ndarray | None,
    abdomen_env:  np.ndarray | None,
    thorax_raw:   np.ndarray | None,
    abdomen_raw:  np.ndarray | None,
    effort_baseline: float,
    sf: float,
) -> tuple[str, float, dict]:
    """
    Classificeer een apnea als obstructief, centraal of gemengd
    conform AASM 2.6 Adult Scoring Rules (sectie 3B).

    AASM criteria:
    ─────────────────────────────────────────────────────────────
    OBSTRUCTIEF apnea:
      - Flow ≥90% reductie (al gedetecteerd door aanroeper)
      - Aanhoudende of toenemende inspiratoire inspanning
        gedurende de volledige apneaduur
      - Vaak: paradoxale thoracoabdominale beweging

    CENTRAAL apnea:
      - Flow ≥90% reductie
      - AFWEZIGHEID van inspiratoire inspanning gedurende de
        volledige apneaduur
      - Thorax en abdomen beide vlak (geen beweging)

    GEMENGD apnea:
      - Flow ≥90% reductie
      - Afwezige inspanning in het EERSTE deel
      - Hervatting van inspanning in het TWEEDE deel
      - Klassiek beeld: centraal begin → obstructief einde

    Returns
    -------
    (type_str, confidence_0_to_1, detail_dict)
    """
    seg_len  = end_idx - onset_idx
    if seg_len < 2:
        return "obstructive", 0.5, {}

    # ── Effort-segmenten extraheren ──
    effort_segs = {}
    if thorax_env is not None:
        effort_segs["thorax"] = thorax_env[onset_idx:end_idx]
    if abdomen_env is not None:
        effort_segs["abdomen"] = abdomen_env[onset_idx:end_idx]

    if not effort_segs:
        return "obstructive", 0.3, {"note": "geen effort-kanalen"}

    # ── Stap 1: Effort-amplitude (envelope) ──
    mean_efforts = []
    for name, seg in effort_segs.items():
        mean_efforts.append(float(np.mean(seg)))
    event_effort = float(np.mean(mean_efforts))
    effort_ratio = event_effort / effort_baseline if effort_baseline > 1e-9 else 0.0

    # ── Stap 2: Effort-variabiliteit (ruwe signalen) ──
    # Cruciaal: bij paradoxale ademhaling is de envelope laag maar
    # het RUWE signaal toont WEL beweging (hoge std dev).
    raw_variability = 0.0
    raw_var_ratio = 0.0
    if thorax_raw is not None and abdomen_raw is not None:
        t_std = float(np.std(thorax_raw[onset_idx:end_idx]))
        a_std = float(np.std(abdomen_raw[onset_idx:end_idx]))
        raw_variability = (t_std + a_std) / 2

        # Baseline variabiliteit (stabiel ademen)
        pre_start = max(0, onset_idx - int(120 * sf))
        pre_end   = max(0, onset_idx - int(5 * sf))
        if pre_end > pre_start:
            t_bl_std = float(np.std(thorax_raw[pre_start:pre_end]))
            a_bl_std = float(np.std(abdomen_raw[pre_start:pre_end]))
            bl_var = max((t_bl_std + a_bl_std) / 2, 1e-9)
            raw_var_ratio = raw_variability / bl_var
        else:
            raw_var_ratio = 1.0  # geen baseline → neutraal
    elif thorax_raw is not None:
        raw_variability = float(np.std(thorax_raw[onset_idx:end_idx]))
        raw_var_ratio = 1.0
    elif abdomen_raw is not None:
        raw_variability = float(np.std(abdomen_raw[onset_idx:end_idx]))
        raw_var_ratio = 1.0

    # ── Stap 3: Paradoxale thoracoabdominale beweging ──
    paradox_corr = None
    if thorax_raw is not None and abdomen_raw is not None:
        t_seg = thorax_raw[onset_idx:end_idx]
        a_seg = abdomen_raw[onset_idx:end_idx]
        if len(t_seg) > 10 and np.std(t_seg) > 1e-9 and np.std(a_seg) > 1e-9:
            try:
                corr, _ = pearsonr(t_seg, a_seg)
                paradox_corr = float(corr)
            except Exception:
                paradox_corr = None

    # ── Stap 4: Eerste/tweede helft effort (gemengd detectie) ──
    half = seg_len // 2
    first_half_effort  = []
    second_half_effort = []
    for seg in effort_segs.values():
        fb = float(np.mean(seg[:half]))
        sb = float(np.mean(seg[half:]))
        first_half_effort.append(fb / effort_baseline if effort_baseline > 1e-9 else 0)
        second_half_effort.append(sb / effort_baseline if effort_baseline > 1e-9 else 0)
    first_ratio  = float(np.mean(first_half_effort))
    second_ratio = float(np.mean(second_half_effort))

    # ── Kwart-per-kwart effort ──
    quarter = max(1, seg_len // 4)
    quarter_efforts = []
    for q in range(4):
        qs = q * quarter; qe = min(qs + quarter, seg_len)
        q_vals = [float(np.mean(s[qs:qe])) for s in effort_segs.values()]
        q_ratio = np.mean(q_vals) / effort_baseline if effort_baseline > 1e-9 else 0
        quarter_efforts.append(q_ratio)

    detail = {
        "effort_ratio":          safe_r(effort_ratio, 3),
        "raw_var_ratio":         safe_r(raw_var_ratio, 3),
        "first_half_effort":     safe_r(first_ratio,  3),
        "second_half_effort":    safe_r(second_ratio, 3),
        "quarter_efforts":       [safe_r(q, 3) for q in quarter_efforts],
        "paradox_correlation":   safe_r(paradox_corr, 3),
    }

    # ═══════════════════════════════════════════════════════════
    # BESLISSINGSLOGICA (v14 — correcte AASM-volgorde)
    #
    # BELANGRIJK: Paradoxale ademhaling = OBSTRUCTIEF, ongeacht
    # hoe laag de individuele effort-enveloppen zijn.
    #
    # Bij obstructieve apneu met gesloten luchtweg beweegt
    # thorax naar binnen terwijl abdomen naar buiten gaat (en
    # omgekeerd). De Hilbert envelope van elk individueel kanaal
    # kan LAAG zijn, maar de oppositie-beweging bewijst inspanning.
    # ═══════════════════════════════════════════════════════════

    is_paradox = (paradox_corr is not None and paradox_corr < -0.15)
    has_raw_movement = raw_var_ratio > 0.25  # ruwe signalen tonen beweging

    # ── REGEL 1: Paradoxale beweging = OBSTRUCTIEF ──
    # (Moet EERST geëvalueerd worden, vóór effort-amplitude check)
    if is_paradox and has_raw_movement:
        confidence = min(0.95, 0.70 + abs(paradox_corr) * 0.3)
        detail["decision_reason"] = f"paradoxical_movement_corr={safe_r(paradox_corr,3)}"
        return "obstructive", safe_r(confidence, 2), detail

    # ── REGEL 2: Ruwe signaalbeweging maar lage envelope = OBSTRUCTIEF ──
    # (Effort IS aanwezig, maar Hilbert envelope onderschat het door paradox)
    if raw_var_ratio > 0.40 and effort_ratio < EFFORT_PRESENT_RATIO:
        if paradox_corr is None or paradox_corr < 0.3:
            confidence = min(0.85, 0.50 + raw_var_ratio * 0.3)
            detail["decision_reason"] = f"raw_movement_present_var={safe_r(raw_var_ratio,3)}"
            return "obstructive", safe_r(confidence, 2), detail

    # ── REGEL 3: Gemengd apnea ──
    mixed_pattern = (
        first_ratio  < EFFORT_ABSENT_RATIO and
        second_ratio > EFFORT_PRESENT_RATIO
    )
    if mixed_pattern:
        confidence = min(0.95, 0.6 + (second_ratio - first_ratio))
        detail["decision_reason"] = "first_half_absent_second_half_present"
        return "mixed", safe_r(confidence, 2), detail

    # ── REGEL 4: Duidelijk effort aanwezig = OBSTRUCTIEF ──
    effort_present = effort_ratio > EFFORT_PRESENT_RATIO
    if effort_present:
        confidence = min(0.95, 0.5 + (effort_ratio - EFFORT_PRESENT_RATIO))
        detail["decision_reason"] = f"effort_present_ratio={safe_r(effort_ratio,3)}"
        return "obstructive", safe_r(confidence, 2), detail

    # ── REGEL 5: CENTRAAL — alleen als GEEN beweging in ruwe signalen ──
    # Centraal = echt vlakke thorax EN abdomen (geen variabiliteit)
    truly_flat = raw_var_ratio < 0.15 and effort_ratio < EFFORT_ABSENT_RATIO
    quarters_absent = sum(1 for q in quarter_efforts if q < EFFORT_ABSENT_RATIO)
    no_paradox = (paradox_corr is None or paradox_corr > 0.0)

    if truly_flat and quarters_absent >= 3 and no_paradox:
        confidence = min(0.90, 0.5 + (EFFORT_ABSENT_RATIO - effort_ratio) * 3)
        detail["decision_reason"] = f"truly_flat_var={safe_r(raw_var_ratio,3)}_effort={safe_r(effort_ratio,3)}"
        return "central", safe_r(confidence, 2), detail

    # ── REGEL 6: Grensgebied — default OBSTRUCTIEF ──
    # AASM-consensus: bij twijfel = obstructief (meest conservatief,
    # leidt tot CPAP-behandeling die voor beide types effectief is)
    detail["decision_reason"] = f"borderline_default_obstructive_var={safe_r(raw_var_ratio,3)}_effort={safe_r(effort_ratio,3)}"
    return "obstructive", 0.40, detail


# ═══════════════════════════════════════════════════════════════
# APNEA / HYPOPNEA DETECTIE
# ═══════════════════════════════════════════════════════════════

def detect_respiratory_events(
    flow_data:    np.ndarray,
    thorax_data:  np.ndarray | None,
    abdomen_data: np.ndarray | None,
    spo2_data:    np.ndarray | None,
    sf_flow:      float,
    sf_spo2:      float,
    hypno:        list,
    only_during_sleep: bool = True,
    artifact_epochs: list = None,
    hypop_flow:   np.ndarray | None = None,
    sf_hypop:     float | None = None,
    pos_data:     np.ndarray | None = None,
    sf_pos:       float | None = None,
) -> dict:
    """
    Detecteer en classificeer apnea's en hypopnea's conform AASM 2.6.

    AASM 2.6 sensor-aanbeveling:
      - flow_data  : oronasale thermistor (voor apnea, cessatie-detectie)
      - hypop_flow : nasale druk transducer (voor hypopnea, gevoeliger)
      Als hypop_flow niet opgegeven: flow_data voor beide (backward compatible).

    Apnea-types:
      obstructive  — flow <10%, inspanningsindicatoren aanwezig
      central      — flow <10%, geen inspiratoire inspanning
      mixed        — centraal begin (geen effort) → obstructief einde (effort)
      hypopnea     — flow 10–70%, ≥10s, + desaturatie ≥3% OF arousal
    """
    result = {"success": False, "events": [], "summary": {}, "error": None}
    try:
        # ── Preprocesseer apnea-flow (thermistor — GEEN √-transformatie) ──
        flow_env = preprocess_flow(flow_data, sf_flow, is_nasal_pressure=False)
        baseline = compute_dynamic_baseline(flow_env, sf_flow)
        pos_changes = []  # geïnitialiseerd voor beide kanalen

        # v0.8.0: MMSD voor robuuste apnea-validatie (drift-onafhankelijk)
        try:
            flow_filt_mmsd = bandpass_flow(flow_data, sf_flow)
            mmsd = compute_mmsd(flow_filt_mmsd, sf_flow, window_s=1.0)
            # Normaliseer: MMSD-basislijn = mediaan van stabiele ademhaling
            mmsd_stable = mmsd[mmsd > np.percentile(mmsd, 10)]
            mmsd_baseline = float(np.median(mmsd_stable)) if len(mmsd_stable) > 100 else 1.0
            mmsd_norm = mmsd / max(mmsd_baseline, 1e-9)
            # Apnea = MMSD < 15% van basislijn (vrijwel geen ademhaling)
            MMSD_APNEA_THRESH = 0.15
            result["mmsd_available"] = True
            logger.info("MMSD basislijn: %.4f, threshold apnea: %.4f",
                       mmsd_baseline, mmsd_baseline * MMSD_APNEA_THRESH)
        except Exception as e:
            mmsd_norm = None
            result["mmsd_available"] = False
            logger.debug("MMSD berekening mislukt: %s", e)

        # ── v14: Stadium-specifieke basislijn (REM vs NREM) ──
        try:
            stage_bl = compute_stage_baseline(flow_env, sf_flow, hypno, artifact_epochs)
            # Gebruik gemiddelde van dynamisch + stadium-specifiek (robuuster)
            baseline = (baseline + stage_bl) / 2
            result["stage_baseline_used"] = True
        except Exception as e:
            logger.debug("Stage-baseline fallback: %s", e)
            result["stage_baseline_used"] = False

        # ── v14: Positie-reset basislijn ──
        if pos_data is not None and sf_pos is not None:
            try:
                pos_changes = detect_position_changes(pos_data, sf_pos)
                if pos_changes:
                    # Converteer positie-samples naar flow sample-rate
                    pos_changes_flow = []
                    for pc in pos_changes:
                        pc_flow = dict(pc)
                        pc_flow["sample"] = int(pc["time_s"] * sf_flow)
                        pos_changes_flow.append(pc_flow)
                    baseline = reset_baseline_at_position_changes(
                        baseline, flow_env, sf_flow, pos_changes_flow)
                    result["n_position_changes"] = len(pos_changes)
                    logger.info("Positie-reset: %d veranderingen gedetecteerd", len(pos_changes))
            except Exception as e:
                logger.debug("Positie-reset fallback: %s", e)

        flow_norm = np.clip(flow_env / baseline, 0, 2)

        # ── Preprocesseer hypopnea-flow (nasale druk of zelfde kanaal) ──
        if hypop_flow is not None and sf_hypop is not None:
            # v0.8.0: √-linearisatie voor nasale druk (AASM 2.6 Rule 3)
            # Nasale druk ∝ flow² (Bernoulli) → zonder √ wordt 50% flowdaling
            # getoond als 75% amplitudereductie → hypopnea-overschatting
            hypop_env = preprocess_flow(hypop_flow, sf_hypop, is_nasal_pressure=True)
            hypop_baseline = compute_dynamic_baseline(hypop_env, sf_hypop)
            # Stadium-specifieke basislijn voor hypopnea kanaal
            try:
                hypop_stage_bl = compute_stage_baseline(hypop_env, sf_hypop, hypno, artifact_epochs)
                hypop_baseline = (hypop_baseline + hypop_stage_bl) / 2
            except Exception:
                pass
            # Positie-reset voor hypopnea kanaal
            if pos_data is not None and sf_pos is not None:
                try:
                    if not pos_changes:
                        pos_changes = detect_position_changes(pos_data, sf_pos)
                    if pos_changes:
                        pc_hypop = [dict(pc, sample=int(pc["time_s"] * sf_hypop)) for pc in pos_changes]
                        hypop_baseline = reset_baseline_at_position_changes(
                            hypop_baseline, hypop_env, sf_hypop, pc_hypop)
                except Exception:
                    pass
            hypop_norm = np.clip(hypop_env / hypop_baseline, 0, 2)
            sf_hy = sf_hypop
            result["dual_sensor"] = True
        else:
            hypop_env = flow_env
            hypop_baseline = baseline
            hypop_norm = flow_norm
            sf_hy = sf_flow
            result["dual_sensor"] = False

        # ── Breath-by-breath analyse (v14 — AASM-conform) ──
        # Gebruik het nasale druk kanaal (gevoeligst) voor breath detectie
        hypop_raw = hypop_flow if hypop_flow is not None else flow_data
        hypop_sf  = sf_hypop if sf_hypop is not None else sf_flow
        try:
            flow_filt = bandpass_flow(hypop_raw, hypop_sf)
            breaths = detect_breaths(flow_filt, hypop_sf)
            if len(breaths) > 10:
                breath_ratios = compute_breath_amplitudes(breaths, hypop_sf)

                # Breath-by-breath flattening per ademhaling
                for b_i, br in enumerate(breaths):
                    if br.get("insp_segment") is not None and len(br["insp_segment"]) > 3:
                        breaths[b_i]["flattening"] = compute_flattening_index(br["insp_segment"])
                    else:
                        breaths[b_i]["flattening"] = None

                bb_apneas, bb_hypopneas = detect_breath_events(
                    breaths, breath_ratios, hypop_sf, hypno)

                result["breath_analysis"] = {
                    "n_breaths": len(breaths),
                    "n_bb_apneas": len(bb_apneas),
                    "n_bb_hypopneas": len(bb_hypopneas),
                    "avg_flattening": safe_r(float(np.mean([
                        b["flattening"] for b in breaths
                        if b.get("flattening") is not None])), 3) if any(
                        b.get("flattening") is not None for b in breaths) else None,
                }
                logger.info("Breath-by-breath: %d ademhalingen, %d apneus, %d hypopneus",
                           len(breaths), len(bb_apneas), len(bb_hypopneas))
            else:
                bb_apneas, bb_hypopneas = [], []
                breath_ratios = np.array([])
                result["breath_analysis"] = {"n_breaths": len(breaths), "fallback": True}
                logger.warning("Te weinig ademhalingen gedetecteerd (%d), fallback naar envelope", len(breaths))
        except Exception as e:
            bb_apneas, bb_hypopneas = [], []
            breath_ratios = np.array([])
            result["breath_analysis"] = {"error": str(e), "fallback": True}
            logger.warning("Breath-by-breath analyse mislukt: %s, fallback naar envelope", e)

        # Effort-envelopes vooraf berekenen (één keer, niet per event)
        thorax_env  = preprocess_effort(thorax_data,  sf_flow) if thorax_data  is not None else None
        abdomen_env = preprocess_effort(abdomen_data, sf_flow) if abdomen_data is not None else None

        # Effort-basislijn (stabiel segment buiten apnea's)
        effort_baseline = _compute_effort_baseline(
            thorax_env, abdomen_env, flow_norm, sf_flow)

        # ── Sleep mask ──
        # (sleep_mask_ap en sleep_mask_hy worden per event-type aangemaakt)

        # ── Globale SpO2-basislijn (voor cluster-apneu robuustheid) ──
        global_spo2_bl = None
        if spo2_data is not None:
            spo2_sleep_mask = build_sleep_mask(hypno, sf_spo2, len(spo2_data), artifact_epochs)
            spo2_clean = spo2_data[(spo2_data >= 50) & (spo2_data <= 100) & spo2_sleep_mask]
            if len(spo2_clean) > 100:
                global_spo2_bl = float(np.percentile(spo2_clean, 95))

        # ── Event maskers ──
        # AASM: apneu op thermistor (flow_norm), hypopneu op nasale druk (hypop_norm)
        apnea_raw    = flow_norm < APNEA_THRESHOLD
        hypopnea_raw = (hypop_norm < HYPOPNEA_THRESHOLD) & ~(hypop_norm < APNEA_THRESHOLD)
        # Sleep mask op basis van primaire flow samplefrequentie
        sleep_mask_ap = build_sleep_mask(hypno, sf_flow, len(flow_norm), artifact_epochs)
        sleep_mask_hy = build_sleep_mask(hypno, sf_hy, len(hypop_norm), artifact_epochs)

        events = []
        rejected_hypopneas = []  # Rule 1B kandidaten (geen desat, wacht op arousal)

        # ── Apnea events (thermistor) ──
        labeled_ap, n_ap = label(apnea_raw & sleep_mask_ap)
        for i in range(1, n_ap + 1):
            indices = np.where(labeled_ap == i)[0]
            dur_s   = len(indices) / sf_flow
            if dur_s < APNEA_MIN_DUR_S:
                continue

            # v0.8.0: MMSD-validatie — bevestig dat ademhaling echt afwezig is
            # (voorkomt vals-positieven door langzame baseline-drift)
            if mmsd_norm is not None:
                event_mmsd = float(np.mean(mmsd_norm[indices[0]:indices[-1]+1]))
                if event_mmsd > 0.40:
                    # MMSD >40% van basislijn = er is nog ademhalingsactiviteit
                    # → geen echte apnea, waarschijnlijk baseline-drift artefact
                    continue

            onset_s = indices[0]  / sf_flow
            end_s   = indices[-1] / sf_flow
            ep_idx  = int(onset_s // EPOCH_LEN_S)
            stage   = hypno[ep_idx] if ep_idx < len(hypno) else "W"

            # ── Pre-event basislijn (2 min vóór event) ──
            pre_start = max(0, indices[0] - int(120 * sf_flow))
            pre_end   = max(0, indices[0] - int(5 * sf_flow))  # 5s buffer
            if pre_end > pre_start:
                pre_seg = flow_env[pre_start:pre_end]
                pre_baseline = float(np.percentile(pre_seg[pre_seg > 0], 75)) if np.any(pre_seg > 0) else float(baseline[indices[0]])
            else:
                pre_baseline = float(baseline[indices[0]])

            # Flow reductie t.o.v. pre-event basislijn
            event_flow_mean = float(np.mean(flow_env[indices[0]:indices[-1]+1]))
            flow_reduction_pct = safe_r((1 - event_flow_mean / pre_baseline) * 100) if pre_baseline > 0 else None

            # ── AASM-classificatie ──
            ev_type, confidence, detail = classify_apnea_type(
                onset_idx    = indices[0],
                end_idx      = indices[-1] + 1,
                thorax_env   = thorax_env,
                abdomen_env  = abdomen_env,
                thorax_raw   = thorax_data,
                abdomen_raw  = abdomen_data,
                effort_baseline = effort_baseline,
                sf           = sf_flow,
            )

            # ── SpO2 desaturatie ──
            desat, min_spo2 = _get_desaturation(
                spo2_data, onset_s, dur_s, sf_spo2, global_spo2_bl)

            # ── Nadir luchtstroom ──
            flow_nadir = safe_r(float(np.min(flow_norm[indices[0]:indices[-1]+1])), 3)

            events.append({
                "type":             ev_type,
                "onset_s":          safe_r(onset_s),
                "duration_s":       safe_r(dur_s),
                "stage":            stage,
                "desaturation_pct": desat,
                "min_spo2":         min_spo2,
                "flow_nadir":       flow_nadir,
                "flow_reduction_pct": flow_reduction_pct,
                "pre_baseline":     safe_r(pre_baseline, 2),
                "confidence":       confidence,
                "classify_detail":  detail,
                "epoch":            ep_idx,
            })

        # ── Hypopnea events (nasale druk) ──
        # v0.8.0 FIX: sluit tijdregio's uit die al als apnea gedetecteerd zijn
        # Voorkomt dat de flanken van een apnea als extra hypopnea geteld worden
        apnea_exclusion_mask = np.zeros(len(hypop_norm), dtype=bool)
        for ev in events:  # events bevat nu alleen apneas
            ev_onset_hy = int(ev["onset_s"] * sf_hy)
            ev_end_hy   = int((ev["onset_s"] + ev["duration_s"]) * sf_hy)
            # 5s marge aan beide kanten (ramp-zone van envelope)
            margin = int(5 * sf_hy)
            m_start = max(0, ev_onset_hy - margin)
            m_end   = min(len(apnea_exclusion_mask), ev_end_hy + margin)
            apnea_exclusion_mask[m_start:m_end] = True

        labeled_hy, n_hy = label(hypopnea_raw & sleep_mask_hy & ~apnea_exclusion_mask)
        for i in range(1, n_hy + 1):
            indices = np.where(labeled_hy == i)[0]
            dur_s   = len(indices) / sf_hy
            if dur_s < HYPOPNEA_MIN_DUR_S:
                continue

            onset_s = indices[0] / sf_hy
            ep_idx  = int(onset_s // EPOCH_LEN_S)
            stage   = hypno[ep_idx] if ep_idx < len(hypno) else "W"

            # ── Pre-event basislijn (nasale druk kanaal) ──
            pre_start = max(0, indices[0] - int(120 * sf_hy))
            pre_end   = max(0, indices[0] - int(5 * sf_hy))
            if pre_end > pre_start:
                pre_seg = hypop_env[pre_start:pre_end]
                pre_baseline = float(np.percentile(pre_seg[pre_seg > 0], 75)) if np.any(pre_seg > 0) else float(hypop_baseline[indices[0]])
            else:
                pre_baseline = float(hypop_baseline[indices[0]])

            event_flow_mean = float(np.mean(hypop_env[indices[0]:indices[-1]+1]))
            flow_reduction_pct = safe_r((1 - event_flow_mean / pre_baseline) * 100) if pre_baseline > 0 else None

            desat, min_spo2 = _get_desaturation(
                spo2_data, onset_s, dur_s, sf_spo2, global_spo2_bl)

            # AASM 2.6 Rule 1A: desaturatie ≥3%
            # Rule 1B: arousal (wordt later geëvalueerd in 2e pass)
            rule1a = desat is not None and desat >= DESATURATION_DROP_PCT
            if not rule1a:
                # Bewaar als kandidaat voor Rule 1B (arousal check na stap 7)
                rejected_hypopneas.append({
                    "onset_s":     safe_r(onset_s),
                    "duration_s":  safe_r(dur_s),
                    "stage":       stage,
                    "desat":       desat,
                    "min_spo2":    min_spo2,
                    "indices":     (indices[0], indices[-1] + 1),
                    "epoch":       ep_idx,
                })
                continue

            # Hypopnea sub-type: obstructief of centraal
            # Indices zijn in sf_hy ruimte; effort is in sf_flow ruimte
            if sf_hy != sf_flow:
                hy_onset_idx = int(onset_s * sf_flow)
                hy_end_idx   = int((onset_s + dur_s) * sf_flow)
            else:
                hy_onset_idx = indices[0]
                hy_end_idx   = indices[-1] + 1
            hy_subtype, hy_conf, hy_detail = classify_apnea_type(
                onset_idx   = hy_onset_idx,
                end_idx     = hy_end_idx,
                thorax_env  = thorax_env,
                abdomen_env = abdomen_env,
                thorax_raw  = thorax_data,
                abdomen_raw = abdomen_data,
                effort_baseline = effort_baseline,
                sf          = sf_flow,
            )
            hy_label = f"hypopnea_{hy_subtype}" if hy_subtype != "obstructive" \
                       else "hypopnea"

            flow_reduction = safe_r(
                1.0 - float(np.mean(hypop_norm[indices[0]:indices[-1]+1])), 3)

            events.append({
                "type":             hy_label,
                "onset_s":          safe_r(onset_s),
                "duration_s":       safe_r(dur_s),
                "stage":            stage,
                "desaturation_pct": desat,
                "min_spo2":         min_spo2,
                "flow_reduction":   flow_reduction,
                "flow_reduction_pct": flow_reduction_pct,
                "pre_baseline":     safe_r(pre_baseline, 2),
                "confidence":       hy_conf,
                "classify_detail":  hy_detail,
                "epoch":            ep_idx,
            })

        events.sort(key=lambda x: x["onset_s"])

        # ── STATISTIEKEN ──
        result["events"]  = events
        result["rejected_hypopneas"] = rejected_hypopneas
        result["summary"] = _compute_respiratory_summary(events, hypno, artifact_epochs)
        result["success"] = True

    except Exception as e:
        result["error"]     = str(e)
        result["traceback"] = traceback.format_exc()
    return result


def _compute_effort_baseline(
    thorax_env:  np.ndarray | None,
    abdomen_env: np.ndarray | None,
    flow_norm:   np.ndarray,
    sf: float,
) -> float:
    """
    Bereken de effort-basislijn op stabiele ademhalingsperiodes
    (where flow is between 60–130% of baseline = normaal ademen).
    """
    stable_mask = (flow_norm > 0.60) & (flow_norm < 1.30)
    baselines = []
    if thorax_env is not None:
        stable = thorax_env[stable_mask]
        if len(stable) > int(sf * 30):
            baselines.append(float(np.percentile(stable, 75)))
    if abdomen_env is not None:
        stable = abdomen_env[stable_mask]
        if len(stable) > int(sf * 30):
            baselines.append(float(np.percentile(stable, 75)))
    return float(np.mean(baselines)) if baselines else 1.0


def _get_desaturation(
    spo2_data: np.ndarray | None,
    onset_s: float,
    dur_s: float,
    sf_spo2: float,
    global_spo2_baseline: float = None,
) -> tuple[float | None, float | None]:
    """
    Bereken SpO2-desaturatie tijdens en na een respiratoir event.

    AASM: desaturatie = daling ≥3% t.o.v. pre-event basislijn.

    v0.8.0 verbeteringen:
      - SpO2-nadir moet vallen BINNEN event-onset tot 30s na event-einde
        (circulatoire vertraging 10-30s). Nadir vóór event = toeval.
      - Desaturation window verkort van 45s naar 30s na event-einde
        (conform gevalideerd algoritme Uddin et al., Physiol Meas 2021)
    """
    if spo2_data is None:
        return None, None
    try:
        # SpO2-venster: van event-onset tot 30s na event-einde
        # (was: 45s — te lang, vangt desaturaties van andere events op)
        POST_EVENT_WINDOW_S = 30
        spo2_start = max(0, int(onset_s * sf_spo2))
        spo2_end   = min(len(spo2_data),
                         int((onset_s + dur_s + POST_EVENT_WINDOW_S) * sf_spo2))
        spo2_seg   = spo2_data[spo2_start:spo2_end]
        spo2_seg   = spo2_seg[(spo2_seg >= 50) & (spo2_seg <= 100)]
        if len(spo2_seg) < 3:
            return None, None

        # Basislijn = 90e percentiel van 120s vóór event
        pre_start = max(0, int((onset_s - 120) * sf_spo2))
        pre_end   = spo2_start
        pre_seg   = spo2_data[pre_start:pre_end]
        pre_seg   = pre_seg[(pre_seg >= 50) & (pre_seg <= 100)]

        if len(pre_seg) > 3:
            spo2_bl = float(np.percentile(pre_seg, 90))
        else:
            spo2_bl = float(np.percentile(spo2_seg, 90))

        # Bij ernstige OSAS: pre-event basislijn kan al in desaturatie zitten.
        # Gebruik de HOGERE van lokale basislijn en globale basislijn.
        if global_spo2_baseline is not None and global_spo2_baseline > spo2_bl:
            spo2_bl = global_spo2_baseline

        min_spo2 = float(np.min(spo2_seg))
        desat    = spo2_bl - min_spo2

        # v0.8.0: Check dat nadir NA event-onset valt (niet ervoor)
        # De nadir-index in het segment moet > 0 zijn (= niet helemaal aan het begin)
        nadir_idx = int(np.argmin(spo2_seg))
        samples_from_onset = nadir_idx  # al relatief aan event-onset
        min_delay_samples = int(3 * sf_spo2)  # nadir minstens 3s na onset
        if samples_from_onset < min_delay_samples and desat < 5:
            # Zeer vroege nadir met kleine desaturatie = waarschijnlijk toeval
            return None, safe_r(min_spo2)

        return safe_r(desat), safe_r(min_spo2)
    except Exception:
        return None, None


def _compute_respiratory_summary(events: list, hypno: list,
                                  artifact_epochs: list = None) -> dict:
    """Bereken alle AHI-statistieken uit de event-lijst.

    TST excludeert zowel Wake als artefact-epochs (consistent met event-detectie).
    """
    artifact_set = set(artifact_epochs or [])
    total_sleep_s = sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                        if s != "W" and i not in artifact_set)
    total_sleep_h = max(total_sleep_s / 3600, 0.001)
    rem_h  = max(sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                     if is_rem(s) and i not in artifact_set) / 3600, 0.001)
    nrem_h = max(sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                     if is_nrem(s) and i not in artifact_set) / 3600, 0.001)

    apneas     = [e for e in events if e["type"] in
                  ("obstructive","central","mixed")]
    hypopneas  = [e for e in events if "hypopnea" in e["type"]]
    obstr      = [e for e in events if e["type"] == "obstructive"]
    central    = [e for e in events if e["type"] == "central"]
    mixed      = [e for e in events if e["type"] == "mixed"]

    def idx(n, h): return safe_r(n / h) if h > 0 else 0

    def split_rem_nrem(ev_list):
        """Splits events in REM- en NREM-subgroepen voor positionele analyse."""
        rem  = [e for e in ev_list if is_rem(e["stage"])]
        nrem = [e for e in ev_list if is_nrem(e["stage"])]
        return rem, nrem

    obstr_rem,   obstr_nrem  = split_rem_nrem(obstr)
    central_rem, central_nrem= split_rem_nrem(central)
    mixed_rem,   mixed_nrem  = split_rem_nrem(mixed)
    hyp_rem,     hyp_nrem    = split_rem_nrem(hypopneas)

    ahi = idx(len(apneas) + len(hypopneas), total_sleep_h)

    # Gemiddelde betrouwbaarheid van classificaties
    confidences = [e.get("confidence", 0.5) for e in apneas if e.get("confidence")]
    avg_conf    = safe_r(float(np.mean(confidences))) if confidences else None

    return {
        # Totale counts
        "n_obstructive":          len(obstr),
        "n_central":              len(central),
        "n_mixed":                len(mixed),
        "n_apnea_total":          len(apneas),
        "n_hypopnea":             len(hypopneas),
        "n_ah_total":             len(apneas) + len(hypopneas),

        # Indices (per uur slaap)
        "ahi_total":              ahi,
        "oahi":                   idx(len(obstr) + len(hypopneas), total_sleep_h),
        "ahi_rem":                idx(len([e for e in events if is_rem(e["stage"])]),  rem_h),
        "ahi_nrem":               idx(len([e for e in events if is_nrem(e["stage"])]), nrem_h),
        "obstructive_index":      idx(len(obstr),   total_sleep_h),
        "central_index":          idx(len(central), total_sleep_h),
        "mixed_index":            idx(len(mixed),   total_sleep_h),
        "hypopnea_index":         idx(len(hypopneas), total_sleep_h),

        # REM vs Non-REM uitsplitsing
        "obstructive_rem":        len(obstr_rem),
        "obstructive_nrem":       len(obstr_nrem),
        "central_rem":            len(central_rem),
        "central_nrem":           len(central_nrem),
        "mixed_rem":              len(mixed_rem),
        "mixed_nrem":             len(mixed_nrem),
        "hypopnea_rem":           len(hyp_rem),
        "hypopnea_nrem":          len(hyp_nrem),

        # Duurstatistieken
        "max_apnea_dur_s":        safe_r(max((e["duration_s"] for e in apneas),    default=0)),
        "avg_apnea_dur_s":        safe_r(float(np.mean([e["duration_s"] for e in apneas])))    if apneas    else None,
        "max_hypopnea_dur_s":     safe_r(max((e["duration_s"] for e in hypopneas), default=0)),
        "avg_hypopnea_dur_s":     safe_r(float(np.mean([e["duration_s"] for e in hypopneas]))) if hypopneas else None,

        # SpO2 statistieken uit events
        "avg_desaturation":       safe_r(float(np.mean([
            e["desaturation_pct"] for e in events
            if e.get("desaturation_pct") is not None]))) if any(
            e.get("desaturation_pct") for e in events) else None,

        # Classificatiekwaliteit
        "avg_classification_confidence": avg_conf,
        "n_low_confidence":       sum(1 for e in apneas
                                      if (e.get("confidence") or 1) < 0.5),

        # OSAS classificatie
        "severity":               _classify_ahi(ahi),
        "oahi_severity":          _classify_ahi(idx(len(obstr) + len(hypopneas), total_sleep_h)),

        # TST en artefact-info (transparantie)
        "tst_hours":              safe_r(total_sleep_h),
        "tst_minutes":            safe_r(total_sleep_s / 60),
        "n_artifact_epochs_excluded": len(artifact_set),

        # Klinische waarschuwingen
        "warnings":               _generate_warnings(
            len(central), len(obstr), len(mixed), ahi,
            avg_conf, total_sleep_h),
    }


def _classify_ahi(ahi: float) -> str:
    """Classificeer AHI-ernst: normaal (<5), licht (5-15), matig (15-30), ernstig (>30)."""
    if ahi is None:
        return "unknown"
    if ahi < 5:   return "normal"
    if ahi < 15:  return "mild"
    if ahi < 30:  return "moderate"
    return "severe"


def _generate_warnings(n_central, n_obstr, n_mixed, ahi, avg_conf, sleep_h) -> list:
    """Genereer klinische waarschuwingen op basis van event-patroon."""
    warnings = []

    # Centraal overwicht
    total_ap = n_central + n_obstr + n_mixed
    if total_ap > 0:
        central_pct = n_central / total_ap * 100
        if central_pct > 50:
            warnings.append({
                "level": "warning",
                "code":  "CENTRAL_DOMINANT",
                "msg":   f"Centrale apnea's overheersen ({central_pct:.0f}% van totaal). "
                         "Overweeg centrale slaapapneu (CSA) of Cheyne-Stokes respiratie. "
                         "Nader cardiologisch/neurologisch onderzoek aanbevolen.",
            })
        elif central_pct > 25:
            warnings.append({
                "level": "info",
                "code":  "CENTRAL_SIGNIFICANT",
                "msg":   f"Significant aantal centrale apnea's ({central_pct:.0f}%). "
                         "Mogelijke CSA-component. nCPAP kan centralen verergeren — "
                         "ASV (adaptive servo-ventilation) overwegen.",
            })

    # Gemengde apnea's
    if n_mixed > 5:
        warnings.append({
            "level": "info",
            "code":  "MIXED_APNEAS",
            "msg":   f"{n_mixed} gemengde apnea's. Klassiek patroon bij ernstig OSAS "
                     "met collaps bovenste luchtweg na centrale periode. "
                     "Verwacht goed CPAP-respons.",
        })

    # Lage classificatiebetrouwbaarheid
    if avg_conf is not None and avg_conf < 0.5:
        warnings.append({
            "level": "warning",
            "code":  "LOW_CONFIDENCE",
            "msg":   "Lage classificatiebetrouwbaarheid. Mogelijke oorzaken: "
                     "slechte signaalqualiteit effort-kanalen, "
                     "bewegingsartefacten, of ontbrekende thorax/abdomen-kanalen. "
                     "Handmatige verificatie aanbevolen.",
        })

    # Voldoende slaaptijd voor betrouwbare AHI
    if sleep_h < 2:
        warnings.append({
            "level": "warning",
            "code":  "SHORT_SLEEP",
            "msg":   f"Korte slaaptijd ({sleep_h:.1f}u). AHI-schatting minder betrouwbaar.",
        })

    return warnings


# ═══════════════════════════════════════════════════════════════
# SpO2 ANALYSE
# ═══════════════════════════════════════════════════════════════

def analyze_spo2(spo2_data: np.ndarray, sf: float, hypno: list) -> dict:
    """Analyseer SpO2-kanaal: detecteer desaturaties, bereken ODI, nadir en tijdspercentages."""
    result = {"success": False, "summary": {}, "desaturations": [], "error": None}
    try:
        spo2_clean = spo2_data.copy().astype(float)
        spo2_clean[(spo2_clean < 50) | (spo2_clean > 100)] = np.nan

        sleep_mask = build_sleep_mask(hypno, sf, len(spo2_clean))
        spo2_sleep = spo2_clean[sleep_mask]
        spo2_sleep = spo2_sleep[~np.isnan(spo2_sleep)]

        if len(spo2_sleep) == 0:
            result["error"] = "Geen bruikbare SpO2-data tijdens slaap"
            return result

        total_sleep_s = float(np.sum(sleep_mask)) / sf
        total_sleep_h = max(total_sleep_s / 3600, 0.001)
        baseline_spo2 = float(np.percentile(spo2_sleep, 90))
        min_spo2      = float(np.nanmin(spo2_sleep))
        avg_spo2      = float(np.nanmean(spo2_sleep))

        def pct_below(thresh):
            """Bereken het percentage van de tijd dat SpO2 onder een drempel valt."""
            n   = np.sum(spo2_sleep < thresh)
            t_s = float(n) / sf
            pct = t_s / total_sleep_s * 100 if total_sleep_s > 0 else 0
            return safe_r(pct), _fmt_time(t_s)

        pct90, t90 = pct_below(90)
        pct80, t80 = pct_below(80)
        pct70, t70 = pct_below(70)

        desaturations = _detect_desaturations(spo2_clean, sf, sleep_mask, 3.0)

        rem_mask  = np.zeros(len(spo2_clean), dtype=bool)
        nrem_mask = np.zeros(len(spo2_clean), dtype=bool)
        hypno_num = hypno_to_numeric(hypno)
        spe = int(sf * EPOCH_LEN_S)
        for ep_i, stage in enumerate(hypno_num):
            s = ep_i * spe
            e = min(s + spe, len(rem_mask))
            if is_rem(stage):
                rem_mask[s:e]  = True
            elif is_nrem(stage):
                nrem_mask[s:e] = True

        def spo2_stats_for_mask(mask):
            """Bereken SpO2-statistieken (gemiddelde, nadir, ODI) voor een slaapmasker."""
            seg = spo2_clean[mask]
            seg = seg[~np.isnan(seg)]
            if len(seg) == 0:
                return None, None, None
            dur = float(np.sum(mask)) / sf
            n90 = float(np.sum(seg < 90))
            t90_s = n90 / sf
            pct   = t90_s / dur * 100 if dur > 0 else 0
            return safe_r(pct), _fmt_time(t90_s), safe_r(float(np.min(seg)))

        rem_pct90,  rem_t90,  rem_min  = spo2_stats_for_mask(rem_mask)
        nrem_pct90, nrem_t90, nrem_min = spo2_stats_for_mask(nrem_mask)

        result["summary"] = {
            "baseline_spo2":     safe_r(baseline_spo2),
            "min_spo2":          safe_r(min_spo2),
            "avg_spo2":          safe_r(avg_spo2),
            "n_desaturations":   len(desaturations),
            "desat_index":       safe_r(len(desaturations) / total_sleep_h),
            "pct_below_90":      pct90,
            "time_below_90":     t90,
            "pct_below_80":      pct80,
            "time_below_80":     t80,
            "pct_below_70":      pct70,
            "time_below_70":     t70,
            "total_sleep_s":     safe_r(total_sleep_s),
            "rem_pct_below_90":  rem_pct90,
            "rem_time_below_90": rem_t90,
            "rem_min_spo2":      rem_min,
            "nrem_pct_below_90": nrem_pct90,
            "nrem_time_below_90":nrem_t90,
            "nrem_min_spo2":     nrem_min,
        }
        result["desaturations"] = desaturations[:200]
        result["success"] = True

    except Exception as e:
        result["error"]     = str(e)
        result["traceback"] = traceback.format_exc()
    return result


def _detect_desaturations(spo2: np.ndarray, sf: float,
                           sleep_mask: np.ndarray,
                           drop_pct: float = 3.0) -> list:
    """
    Detecteer SpO2-desaturaties (≥drop_pct% daling).

    v7.1 FIX: geoptimaliseerd — berekent baseline via vectorized rolling max
    ipv per-sample np.nanpercentile() loop.
    """
    win = max(1, int(sf * 3))
    spo2_smooth = np.convolve(
        np.nan_to_num(spo2, nan=95.0),
        np.ones(win) / win, mode="same")

    # Vectorized rolling baseline: 95e percentiel over 60s venster
    # Benader met rolling max (veel sneller, conservatief genoeg)
    baseline_win = int(sf * 60)
    rolling_peak = maximum_filter1d(spo2_smooth, size=baseline_win, origin=-baseline_win // 2)

    # Desaturatie-masker: daling ≥ drop_pct onder rolling peak
    desat_mask = (spo2_smooth <= rolling_peak - drop_pct) & sleep_mask

    # Identificeer aaneengesloten desaturatie-episodes
    events = []
    labeled, n_events = label(desat_mask)
    for i in range(1, n_events + 1):
        indices = np.where(labeled == i)[0]
        if len(indices) < int(sf * 3):  # minimaal 3 seconden
            continue
        onset_idx = indices[0]
        end_idx   = indices[-1]
        nadir     = float(np.min(spo2_smooth[onset_idx:end_idx + 1]))
        peak      = float(rolling_peak[onset_idx])
        drop      = peak - nadir
        dur_s     = len(indices) / sf
        if drop >= drop_pct:
            events.append({
                "onset_s":    safe_r(onset_idx / sf),
                "duration_s": safe_r(dur_s),
                "nadir_spo2": safe_r(nadir),
                "drop_pct":   safe_r(drop),
            })
    return events


# ═══════════════════════════════════════════════════════════════
# POSITIE-ANALYSE
# ═══════════════════════════════════════════════════════════════

def analyze_position(pos_data: np.ndarray, sf: float, hypno: list,
                      resp_events: list) -> dict:
    """Analyseer lichaamshouding tijdens slaap: positieverdeling, positioneel OSAS-percentage."""
    result = {"success": False, "summary": {}, "error": None}
    try:
        spe      = int(sf * EPOCH_LEN_S)
        n_epochs = len(hypno)
        pos_per_epoch = []
        for ep in range(n_epochs):
            s   = ep * spe
            e   = min(s + spe, len(pos_data))
            seg = pos_data[s:e]
            if len(seg) == 0:
                pos_per_epoch.append(-1)
                continue
            vals, cnts = np.unique(seg.astype(int), return_counts=True)
            pos_per_epoch.append(int(vals[np.argmax(cnts)]))

        pos_names  = {0: "Prone", 1: "Left", 2: "Supine", 3: "Right", 4: "Upright"}
        sleep_time = {}
        ahi_pos    = {}
        for pos_code, pos_name in pos_names.items():
            mask    = [i for i, (p, s) in enumerate(zip(pos_per_epoch, hypno))
                       if p == pos_code and s != "W"]
            dur_min = len(mask) * 0.5
            sleep_time[pos_name] = safe_r(dur_min)
            n_events = sum(1 for ev in resp_events
                           if pos_per_epoch[ev.get("epoch", 0)] == pos_code)
            dur_h = dur_min / 60
            ahi_pos[pos_name] = safe_r(n_events / dur_h) if dur_h > 0 else 0

        total_sleep_min = sum(v for v in sleep_time.values() if v)
        pct = {k: safe_r(v / total_sleep_min * 100) if total_sleep_min > 0 else 0
               for k, v in sleep_time.items()}

        result["summary"] = {
            "sleep_time_min": sleep_time,
            "sleep_pct":      pct,
            "ahi_per_pos":    ahi_pos,
        }
        result["pos_per_epoch"] = pos_per_epoch
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════
# HARTRITME
# ═══════════════════════════════════════════════════════════════

def analyze_heart_rate(hr_data: np.ndarray, sf: float, hypno: list) -> dict:
    """Analyseer hartritme uit ECG/HR-kanaal: gemiddelde, variabiliteit, bradycardie/tachycardie."""
    result = {"success": False, "summary": {}, "error": None}
    try:
        hr_clean = hr_data.copy().astype(float)
        hr_clean[(hr_clean < 20) | (hr_clean > 250)] = np.nan
        sleep_mask = build_sleep_mask(hypno, sf, len(hr_clean))
        hr_sleep   = hr_clean[sleep_mask]
        hr_sleep   = hr_sleep[~np.isnan(hr_sleep)]
        if len(hr_sleep) == 0:
            result["error"] = "Geen HR-data tijdens slaap"
            return result
        result["summary"] = {
            "avg_hr":         safe_r(float(np.mean(hr_sleep))),
            "min_hr":         safe_r(float(np.min(hr_sleep))),
            "max_hr":         safe_r(float(np.max(hr_sleep))),
            "std_hr":         safe_r(float(np.std(hr_sleep))),
            "n_tachycardia":  int(np.sum(hr_sleep > 100)),
            "n_bradycardia":  int(np.sum(hr_sleep < 50)),
        }
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════
# SNURKEN
# ═══════════════════════════════════════════════════════════════

def analyze_snore(snore_data: np.ndarray, sf: float, hypno: list) -> dict:
    """Analyseer snurk-kanaal: detecteer snurkepisodes en bereken snurk-index per uur."""
    result = {"success": False, "summary": {}, "error": None}
    try:
        win = int(sf)
        n_windows = len(snore_data) // win
        rms = np.array([
            np.sqrt(np.mean(snore_data[i*win:(i+1)*win]**2))
            for i in range(n_windows)])
        threshold   = float(np.percentile(rms, 60))
        snore_mask  = rms > threshold
        sleep_mask_1s = np.zeros(n_windows, dtype=bool)
        hypno_num   = hypno_to_numeric(hypno)
        for ep_i, stage in enumerate(hypno_num):
            s = ep_i * EPOCH_LEN_S
            e = min(s + EPOCH_LEN_S, n_windows)
            if stage > 0:
                sleep_mask_1s[s:e] = True
        snore_sleep   = snore_mask & sleep_mask_1s
        total_sleep_s = float(np.sum(sleep_mask_1s))
        snore_s       = float(np.sum(snore_sleep))
        snore_min     = snore_s / 60
        total_sleep_h = total_sleep_s / 3600
        result["summary"] = {
            "snore_min":      safe_r(snore_min),
            "snore_pct_tst":  safe_r(snore_s / total_sleep_s * 100) if total_sleep_s > 0 else 0,
            "snore_index":    safe_r(snore_min / 60 / total_sleep_h * 60) if total_sleep_h > 0 else 0,
        }
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════
# PLM
# ═══════════════════════════════════════════════════════════════

def analyze_plm(leg_l: np.ndarray | None, leg_r: np.ndarray | None,
                sf: float, hypno: list,
                resp_events: list = None,
                artifact_epochs: list = None) -> dict:
    """
    Periodieke beenbewegingen (PLM) detectie conform AASM 2.6.

    AASM criteria:
      - Leg Movement (LM): EMG-toename >=8 uV boven resting, 0.5-10s duur
      - Bilaterale LMs binnen 0.5s = 1 LM
      - PLM-serie: >=4 opeenvolgende LMs met 5-90s inter-movement interval
      - PLMS: PLM tijdens slaap (excl. wake)
      - Resp-geassocieerde LMs: LM binnen 0.5s voor tot 0.5s na resp event einde
        -> worden NIET meegeteld als PLMS (AASM 2.6 sectie V)
      - PLMI: PLMs per uur slaap (normaal <15/u, significant >=15/u)
    """
    LM_MIN_DUR_S      = 0.5
    LM_MAX_DUR_S      = 10.0
    LM_AMPLITUDE_UV   = 8.0    # >=8 uV boven resting EMG
    PLM_MIN_INTERVAL_S = 5.0
    PLM_MAX_INTERVAL_S = 90.0
    PLM_MIN_SERIES     = 4
    BILATERAL_WINDOW_S = 0.5
    RESP_EXCLUSION_S   = 0.5   # LM binnen 0.5s van resp event = excluis

    result = {"success": False, "summary": {}, "events": [], "error": None}
    try:
        if leg_l is None and leg_r is None:
            result["error"] = "Geen been-EMG kanalen beschikbaar"
            return result

        def _detect_lm_channel(data, sf):
            """Detecteer LMs op 1 EMG kanaal conform AASM."""
            # v0.8.0 FIX: raw.get_data() levert Volt, AASM drempel is 8 µV
            # Converteer naar µV als het signaal duidelijk in Volt is
            data_uv = data.copy()
            if np.max(np.abs(data_uv)) < 0.1:
                # Signaal is in Volt → converteer naar µV
                data_uv = data_uv * 1e6
                logger.debug("PLM EMG: geconverteerd van V naar µV (max=%.1f µV)",
                            np.max(np.abs(data_uv)))

            # Bandpass 10-100 Hz (AASM EMG filter)
            nyq = sf / 2
            lo = min(10.0 / nyq, 0.99)
            hi = min(100.0 / nyq, 0.99)
            if lo >= hi:
                lo, hi = 0.1, 0.99
            b, a = signal.butter(4, [lo, hi], btype="band")
            filt = signal.filtfilt(b, a, data_uv)

            # RMS in 0.1s vensters
            win = max(1, int(sf * 0.1))
            n_w = len(filt) // win
            rms = np.array([
                np.sqrt(np.mean(filt[i*win:(i+1)*win]**2))
                for i in range(n_w)])

            # Resting EMG: 10e percentiel (baseline in rustige segmenten)
            resting = float(np.percentile(rms, 10))

            # Drempel: resting + 8 uV (AASM criterium)
            threshold = resting + LM_AMPLITUDE_UV
            lm_mask = rms > threshold

            labeled, n_bursts = label(lm_mask)
            lms = []
            for i in range(1, n_bursts + 1):
                idx = np.where(labeled == i)[0]
                dur_s = len(idx) * 0.1
                onset_s = idx[0] * 0.1
                amplitude = float(np.max(rms[idx]))
                if LM_MIN_DUR_S <= dur_s <= LM_MAX_DUR_S:
                    lms.append({
                        "onset_s": onset_s,
                        "duration_s": round(dur_s, 2),
                        "amplitude_uv": round(amplitude, 1),
                    })
            return lms

        # Detecteer LMs per been
        lms_l = _detect_lm_channel(leg_l, sf) if leg_l is not None else []
        lms_r = _detect_lm_channel(leg_r, sf) if leg_r is not None else []

        # Bilaterale samenvoeging: LMs binnen 0.5s op L+R = 1 LM
        all_lms = []
        used_r = set()
        for lm in lms_l:
            merged = False
            for j, rlm in enumerate(lms_r):
                if j in used_r:
                    continue
                if abs(lm["onset_s"] - rlm["onset_s"]) <= BILATERAL_WINDOW_S:
                    # Neem de vroegste onset en langste duur
                    all_lms.append({
                        "onset_s":      min(lm["onset_s"], rlm["onset_s"]),
                        "duration_s":   max(lm["duration_s"], rlm["duration_s"]),
                        "amplitude_uv": max(lm["amplitude_uv"], rlm["amplitude_uv"]),
                        "bilateral":    True,
                    })
                    used_r.add(j)
                    merged = True
                    break
            if not merged:
                lm["bilateral"] = False
                all_lms.append(lm)
        for j, rlm in enumerate(lms_r):
            if j not in used_r:
                rlm["bilateral"] = False
                all_lms.append(rlm)

        # Sorteer op onset
        all_lms.sort(key=lambda x: x["onset_s"])

        # Voeg slaapstadium toe en filter wake
        sleep_lms = []
        for lm in all_lms:
            ep_idx = int(lm["onset_s"] // EPOCH_LEN_S)
            stage = hypno[ep_idx] if ep_idx < len(hypno) else "W"
            lm["stage"] = stage
            lm["epoch"] = ep_idx
            if stage != "W":  # PLMS alleen tijdens slaap
                sleep_lms.append(lm)

        # Resp-geassocieerde LM exclusie (AASM 2.6)
        resp_ends = []
        if resp_events:
            for e in resp_events:
                try:
                    resp_ends.append(float(e["onset_s"]) + float(e["duration_s"]))
                except (KeyError, TypeError):
                    pass

        plm_eligible = []
        n_resp_associated = 0
        for lm in sleep_lms:
            onset = lm["onset_s"]
            end = onset + lm["duration_s"]
            is_resp = False
            for re in resp_ends:
                # LM binnen 0.5s voor tot 0.5s na resp event einde
                if (re - RESP_EXCLUSION_S) <= onset <= (re + RESP_EXCLUSION_S):
                    is_resp = True
                    break
            lm["resp_associated"] = is_resp
            if is_resp:
                n_resp_associated += 1
            else:
                plm_eligible.append(lm)

        # PLM-series detectie: >=4 opeenvolgende LMs met 5-90s interval
        plm_series = []
        plm_count = 0
        if len(plm_eligible) >= PLM_MIN_SERIES:
            seq = [plm_eligible[0]]
            for j in range(1, len(plm_eligible)):
                interval = plm_eligible[j]["onset_s"] - plm_eligible[j-1]["onset_s"]
                if PLM_MIN_INTERVAL_S <= interval <= PLM_MAX_INTERVAL_S:
                    seq.append(plm_eligible[j])
                else:
                    if len(seq) >= PLM_MIN_SERIES:
                        plm_count += len(seq)
                        plm_series.append({
                            "start_s": seq[0]["onset_s"],
                            "end_s":   seq[-1]["onset_s"] + seq[-1]["duration_s"],
                            "n_lms":   len(seq),
                        })
                    seq = [plm_eligible[j]]
            if len(seq) >= PLM_MIN_SERIES:
                plm_count += len(seq)
                plm_series.append({
                    "start_s": seq[0]["onset_s"],
                    "end_s":   seq[-1]["onset_s"] + seq[-1]["duration_s"],
                    "n_lms":   len(seq),
                })

        # Mark PLM LMs
        for lm in plm_eligible:
            lm["is_plm"] = False
        for series in plm_series:
            for lm in plm_eligible:
                if series["start_s"] <= lm["onset_s"] <= series["end_s"]:
                    lm["is_plm"] = True

        artifact_set = set(artifact_epochs or [])
        total_sleep_s = sum(EPOCH_LEN_S for i, s in enumerate(hypno)
                            if s != "W" and i not in artifact_set)
        total_sleep_h = max(total_sleep_s / 3600, 0.001)
        total_rec_h   = len(hypno) * EPOCH_LEN_S / 3600

        plmi = safe_r(plm_count / total_sleep_h)
        lmi  = safe_r(len(sleep_lms) / total_sleep_h)

        # Klinische classificatie
        if plmi is None or plmi == 0:
            plm_severity = "normal"
        elif plmi < 5:
            plm_severity = "normal"
        elif plmi < 15:
            plm_severity = "mild"
        elif plmi < 25:
            plm_severity = "moderate"
        else:
            plm_severity = "severe"

        result["events"] = plm_eligible[:200]  # max 200 events in output
        result["summary"] = {
            "n_lm_total":         len(all_lms),
            "n_lm_sleep":         len(sleep_lms),
            "n_lm_wake":          len(all_lms) - len(sleep_lms),
            "n_resp_associated":  n_resp_associated,
            "n_plm_eligible":     len(plm_eligible),
            "n_plm":              plm_count,
            "n_plm_series":       len(plm_series),
            "lm_index":           lmi,
            "plm_index":          plmi,
            "plm_severity":       plm_severity,
            "total_sleep_h":      safe_r(total_sleep_h),
        }
        result["series"] = plm_series
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
    return result




# ═══════════════════════════════════════════════════════════════
# HYPOPNEA RULE 1B — AROUSAL-GEBASEERDE HERACTIVATIE
# ═══════════════════════════════════════════════════════════════

RULE1B_AROUSAL_WINDOW_S = 15.0  # arousal binnen 15s na einde hypopnea

def reinstate_rule1b_hypopneas(
    rejected: list,
    arousal_events: list,
    resp_events: list,
    hypno: list,
) -> tuple:
    """
    AASM 2.6 Hypopnea Rule 1B: flow ≥30% reductie ≥10s + arousal
    (zonder vereiste ≥3% desaturatie).

    Evalueert eerder afgewezen hypopnea-kandidaten tegen arousal events.
    Geeft (reinstated_events, updated_summary) terug.

    Parameters
    ----------
    rejected       : lijst van afgewezen hypopnea-kandidaten
    arousal_events : lijst van gedetecteerde arousals
    resp_events    : bestaande (Rule 1A) events
    hypno          : slaapstadia lijst
    """
    if not rejected or not arousal_events:
        return [], resp_events

    # Bouw arousal onset-set voor snelle lookup
    arousal_times = [(a.get("onset_s", 0), a.get("duration_s", 3))
                     for a in arousal_events]

    reinstated = []
    for cand in rejected:
        onset = float(cand["onset_s"])
        dur   = float(cand["duration_s"])
        end   = onset + dur

        # Check of een arousal binnen 15s na einde event valt
        has_arousal = False
        for a_onset, a_dur in arousal_times:
            # Arousal start moet binnen [event_start, event_end + 15s]
            if onset <= a_onset <= end + RULE1B_AROUSAL_WINDOW_S:
                has_arousal = True
                break

        if has_arousal:
            reinstated.append({
                "type":             "hypopnea",
                "onset_s":          cand["onset_s"],
                "duration_s":       cand["duration_s"],
                "stage":            cand["stage"],
                "desaturation_pct": cand.get("desat"),
                "min_spo2":         cand.get("min_spo2"),
                "flow_reduction":   None,
                "confidence":       0.7,
                "classify_detail":  {"rule": "1B_arousal"},
                "epoch":            cand["epoch"],
                "rule1b":           True,
            })

    if reinstated:
        # Voeg toe aan events en hersorteer
        all_events = resp_events + reinstated
        all_events.sort(key=lambda x: float(x["onset_s"]))
        return reinstated, all_events

    return [], resp_events

# ═══════════════════════════════════════════════════════════════
# MASTER FUNCTIE
# ═══════════════════════════════════════════════════════════════

def run_pneumo_analysis(
    raw: mne.io.BaseRaw,
    hypno: list,
    channel_map: dict = None,
    artifact_epochs: list = None,
) -> dict:
    """
    Voert alle pneumologische analyses uit op één EDF-opname.

    Parameters
    ----------
    raw         : geladen MNE raw object
    hypno       : slaapfase-lijst ['W','N1','N2','N3','R', ...]
    channel_map : optionele handmatige kanaalkeuze (overrulet auto-detectie)
    """
    auto_map = detect_channels(raw.ch_names)
    ch = {**auto_map, **(channel_map or {})}

    output = {
        "meta": {
            "channels_used": ch,
            "all_channels":  raw.ch_names,
            "sfreq":         raw.info["sfreq"],
            "duration_min":  round(raw.times[-1] / 60, 1),
        },
        "channel_availability": {k: (v in raw.ch_names) for k, v in ch.items()},
    }

    def get(ch_type):
        """Haal een kanaal op uit de MNE raw data op basis van de kanaalmap."""
        name = ch.get(ch_type)
        if name and name in raw.ch_names:
            return raw.get_data(picks=[name])[0], raw.info["sfreq"]
        return None, None

    flow_data,   sf_flow   = get("flow")
    # AASM 2.6: apneu op thermistor, hypopneu op nasale druk
    flow_pressure_data, sf_fp   = get("flow_pressure")
    flow_therm_data,    sf_ft   = get("flow_thermistor")

    # Intelligente kanaal-toewijzing:
    # - Als BEIDE beschikbaar: apneu op thermistor, hypopneu op druk
    # - Als alleen "flow" generiek: gebruik voor beide (backward compatible)
    if flow_pressure_data is not None or flow_therm_data is not None:
        # Specifieke kanalen gevonden
        apnea_flow  = flow_therm_data if flow_therm_data is not None else (flow_pressure_data if flow_pressure_data is not None else flow_data)
        hypop_flow  = flow_pressure_data if flow_pressure_data is not None else (flow_therm_data if flow_therm_data is not None else flow_data)
        sf_apnea    = sf_ft if flow_therm_data is not None else (sf_fp if flow_pressure_data is not None else sf_flow)
        sf_hypop    = sf_fp if flow_pressure_data is not None else (sf_ft if flow_therm_data is not None else sf_flow)
        # Gebruik druk als primaire flow voor detectie (gevoeligst)
        if flow_data is None:
            flow_data = flow_pressure_data if flow_pressure_data is not None else flow_therm_data
            sf_flow   = sf_fp if flow_pressure_data is not None else sf_ft
        output["meta"]["flow_channels"] = {
            "apnea_sensor":   ch.get("flow_thermistor") or ch.get("flow_pressure") or ch.get("flow"),
            "hypopnea_sensor": ch.get("flow_pressure") or ch.get("flow_thermistor") or ch.get("flow"),
            "dual_sensor":    flow_pressure_data is not None and flow_therm_data is not None,
        }
        logger.info("Flow kanalen: apneu=%s, hypopneu=%s, dual=%s",
                    output["meta"]["flow_channels"]["apnea_sensor"],
                    output["meta"]["flow_channels"]["hypopnea_sensor"],
                    output["meta"]["flow_channels"]["dual_sensor"])
    else:
        apnea_flow = flow_data
        hypop_flow = flow_data
        sf_apnea = sf_flow
        sf_hypop = sf_flow
        output["meta"]["flow_channels"] = {
            "apnea_sensor":   ch.get("flow", "—"),
            "hypopnea_sensor": ch.get("flow", "—"),
            "dual_sensor":    False,
        }

    thorax_data, _         = get("thorax")
    abdomen_data, _        = get("abdomen")
    spo2_data,   sf_spo2   = get("spo2")
    pulse_data,  sf_pulse  = get("pulse")
    pos_data,    sf_pos    = get("position")
    snore_data,  sf_snore  = get("snore")
    leg_l_data,  sf_leg    = get("leg_l")
    leg_r_data,  _         = get("leg_r")

    # EEG voor arousal-detectie (primair kanaal)
    eeg_ch_name = ch.get("eeg")
    if not eeg_ch_name:
        # Fallback: eerste EEG-achtig kanaal
        for candidate in raw.ch_names:
            cu = candidate.upper()
            if any(p in cu for p in ["EEG","C3","C4","F3","F4","CZ"]):
                eeg_ch_name = candidate
                break
    eeg_data, sf_eeg = (
        (raw.get_data(picks=[eeg_ch_name])[0], raw.info["sfreq"])
        if eeg_ch_name and eeg_ch_name in raw.ch_names
        else (None, None)
    )

    # EMG voor REM arousal criterium (chin-EMG)
    emg_ch_name = ch.get("emg")
    if not emg_ch_name:
        for candidate in raw.ch_names:
            cu = candidate.upper()
            if any(p in cu for p in ["EMG", "CHIN", "MENT"]):
                emg_ch_name = candidate
                break
    emg_data = (
        raw.get_data(picks=[emg_ch_name])[0]
        if emg_ch_name and emg_ch_name in raw.ch_names
        else None
    )

    logger.info("[pneumo 1/7] Apnea/hypopnea classificatie (AASM 2.6)...")
    if flow_data is not None:
        resp = detect_respiratory_events(
            flow_data    = apnea_flow,       # thermistor voor apneu (of fallback)
            hypop_flow   = hypop_flow,        # nasale druk voor hypopneu (of fallback)
            sf_hypop     = sf_hypop,
            thorax_data  = thorax_data,
            abdomen_data = abdomen_data,
            spo2_data    = spo2_data,
            sf_flow      = sf_apnea,
            sf_spo2      = sf_spo2 or sf_flow,
            hypno        = hypno,
            artifact_epochs = artifact_epochs,
            pos_data     = pos_data,
            sf_pos       = sf_pos,
        )
    else:
        resp = {"success": False,
                "error": "Geen luchtstroom-kanaal gevonden",
                "events": [], "summary": {}}
    output["respiratory"] = resp

    logger.info("[pneumo 2/7] SpO2 analyse...")
    if spo2_data is not None:
        output["spo2"] = analyze_spo2(spo2_data, sf_spo2, hypno)
    else:
        output["spo2"] = {"success": False, "error": "Geen SpO2-kanaal", "summary": {}}

    logger.info("[pneumo 3/7] Positie-analyse...")
    if pos_data is not None:
        output["position"] = analyze_position(
            pos_data, sf_pos, hypno, resp.get("events", []))
    else:
        output["position"] = {"success": False, "error": "Geen positie-kanaal", "summary": {}}

    logger.info("[pneumo 4/7] Hartritme...")
    hr_data, sf_hr = (pulse_data, sf_pulse) if pulse_data is not None else get("ecg")
    if hr_data is not None:
        output["heart_rate"] = analyze_heart_rate(hr_data, sf_hr, hypno)
    else:
        output["heart_rate"] = {"success": False, "error": "Geen HR/ECG-kanaal", "summary": {}}

    logger.info("[pneumo 5/7] Snurkanalyse...")
    if snore_data is not None:
        output["snore"] = analyze_snore(snore_data, sf_snore, hypno)
    else:
        output["snore"] = {"success": False, "error": "Geen snurk-kanaal", "summary": {}}

    logger.info("[pneumo 6/7] PLM detectie...")
    if leg_l_data is not None or leg_r_data is not None:
        output["plm"] = analyze_plm(
            leg_l_data, leg_r_data,
            sf_leg or raw.info["sfreq"], hypno,
            resp_events=resp.get("events", []),
            artifact_epochs=artifact_epochs)
    else:
        output["plm"] = {"success": False, "error": "Geen been-EMG kanalen", "summary": {}}

    # ── Stap 7: Arousal-respiratoir verband (NIEUW v7.2) ──────────
    logger.info("[pneumo 7/7] Arousal detectie & respiratoire koppeling...")
    if eeg_data is not None and _AROUSAL_AVAILABLE:
        # Hergebruik de genormaliseerde flow uit de resp-analyse indien beschikbaar
        flow_env_norm = None
        if flow_data is not None:
            try:
                flow_env      = preprocess_flow(flow_data, sf_flow)
                flow_baseline = compute_dynamic_baseline(flow_env, sf_flow)
                flow_env_norm = np.clip(flow_env / flow_baseline, 0, 2)
            except Exception:
                flow_env_norm = None

        output["arousal"] = run_arousal_respiratory_analysis(
            eeg_data    = eeg_data,
            sf_eeg      = sf_eeg,
            flow_data   = flow_data,
            flow_norm   = flow_env_norm,
            sf_flow     = sf_flow,
            resp_events = resp.get("events", []),
            hypno       = hypno,
            emg_data    = emg_data,
            artifact_epochs = artifact_epochs,
        )
    else:
        reason = "Geen EEG-kanaal" if eeg_data is None else "arousal_analysis module niet geladen"
        output["arousal"] = {
            "success": False,
            "error":   reason,
            "summary": {
                "arousal_index": None,
                "n_respiratory_arousals": None,
                "n_spontaneous_arousals": None,
                "pct_respiratory_arousals": None,
                "n_reras": 0,
                "rera_index": 0,
                "rdi": None,
                "clinical_interpretation": [],
            },
        }

    # ── Stap 8: Rule 1B heractivatie (na arousal detectie) ──────
    rejected_hyps = resp.get("rejected_hypopneas", [])
    arousal_evts  = output.get("arousal", {}).get("events", [])
    if rejected_hyps and arousal_evts:
        logger.info("[pneumo 8] Rule 1B: %d kandidaten vs %d arousals...",
                    len(rejected_hyps), len(arousal_evts))
        reinstated, updated_events = reinstate_rule1b_hypopneas(
            rejected     = rejected_hyps,
            arousal_events = arousal_evts,
            resp_events  = resp.get("events", []),
            hypno        = hypno,
        )
        if reinstated:
            logger.info("[pneumo 8] Rule 1B: %d hypopnea's heractiveerd", len(reinstated))
            output["respiratory"]["events"]  = updated_events
            output["respiratory"]["summary"] = _compute_respiratory_summary(
                updated_events, hypno, artifact_epochs)
            output["respiratory"]["rule1b_reinstated"] = len(reinstated)
    else:
        output["respiratory"]["rule1b_reinstated"] = 0

    # ── Stap 9: Cheyne-Stokes detectie (v14) ──
    logger.info("[pneumo 9] Cheyne-Stokes respiratie detectie...")
    if flow_data is not None:
        try:
            # v0.8.0 FIX: gebruik sf_flow (altijd gedefinieerd als flow_data niet None is)
            # sf_apnea is een lokale variabele die mogelijk niet bestaat in alle code-paden
            sf_csr = sf_flow
            flow_env_csr = preprocess_flow(flow_data, sf_csr)
            output["cheyne_stokes"] = detect_cheyne_stokes(flow_env_csr, sf_csr, hypno)
            if output["cheyne_stokes"].get("csr_detected"):
                logger.info("[pneumo 9] CSR gedetecteerd: periodiciteit %.0fs, %d minuten",
                           output["cheyne_stokes"]["periodicity_s"],
                           output["cheyne_stokes"]["csr_minutes"])
        except Exception as e:
            logger.warning("[pneumo 9] CSR detectie mislukt: %s", e)
            output["cheyne_stokes"] = {"success": False, "csr_detected": False, "error": str(e)}
    else:
        output["cheyne_stokes"] = {"success": False, "csr_detected": False}

    logger.info("✅ Pneumo-analyse voltooid.")
    return output
