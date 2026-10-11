"""
signal_quality_channels.py — Per-channel signal quality assessment for PSG.

Detects:
  1. Flat-line segments (electrode disconnect / amplifier saturation)
  2. Clipping (signal at ADC rail)
  3. High-impedance noise (50/60 Hz dominant)
  4. Montage plausibility (cross-correlation sanity checks)

Distinct from ``signal_quality.py`` (which assesses RIP-effort channel quality);
the two are surfaced separately as ``output["channel_quality"]`` and
``output["signal_quality"]``.
"""
from __future__ import annotations  # PEP 604 `X | None` annotations on Python 3.9

import numpy as np
from scipy import signal as sp_signal
import logging

logger = logging.getLogger("psgscoring.signal_quality_channels")

EPOCH_LEN_S = 30


#: Rollen met een DISCREET bereik: een SpO2 die minutenlang 96 % aanwijst of
#: een positiekanaal dat tussen een handvol waarden springt is per definitie
#: "vlak" en "op zijn maximum" zonder dat er iets mis is. Clipping heeft hier
#: geen betekenis; vlakheid alleen als het kanaal buiten zijn fysieke bereik
#: staat (SpO2 < 50 % of > 100 % = sensor los).
DISCRETE_ROLES = ("spo2", "position")
SPO2_PLAUSIBLE = (50.0, 100.0)

#: Vlak = de standaarddeviatie in een 2 s-venster ligt onder deze FRACTIE
#: van de robuuste schaal van het hele kanaal (1,4826 x MAD). Een verhouding
#: binnen hetzelfde signaal, dus onafhankelijk van de eenheid waarin het EDF
#: is opgeschreven. De oude drempel was een absolute std in ruwe eenheden
#: (0,5 µV voor EEG, 1e-4 voor ademhaling): een RIP-kanaal dat zijn eenheid
#: als µV declareert komt via MNE 1e-5x zo klein binnen en was "100 % vlak".
#:
#: 0,3 %, niet 1 %: een apneu met 2 % restflow (de golden-case `apnea_clean`)
#: staat op 1,6 % van de schaal en telde bij 1 % mee als vlakke lijn -- dan
#: wordt een ernstige OSA-nacht "poor" omdat ze ziek is, niet omdat de sensor
#: stuk is. Een dode lijn (digitale nul) en een losse sensor met 1 LSB
#: ADC-ruis op een 12-bits bereik (0,02 % van de piek) blijven er ruim onder.
FLAT_REL_STD = 0.003

#: Clipping = een piek in het amplitudehistogram op de rail. De bovenste band
#: van CLIP_RAIL_FRACTION x het bereik moet CLIP_SPIKE_RATIO keer zoveel
#: monsters dragen als de banden er direct onder, en minstens
#: CLIP_MIN_FRACTION van het kanaal zijn. Een gekwantiseerde of opgesamplede
#: sinus houdt zijn hoogste niveau even lang vast als het niveau eronder en
#: haalt die verhouding niet; een echte ADC-rail stapelt alles wat erboven
#: had gezeten op één waarde.
CLIP_RAIL_FRACTION = 0.005
CLIP_SPIKE_RATIO = 5.0
CLIP_MIN_FRACTION = 0.005


def assess_signal_quality(
    raw,
    channel_map: dict,
    hypno: list | None = None,
) -> dict:
    """Run signal quality assessment on all mapped channels.

    Parameters
    ----------
    raw          : MNE Raw object
    channel_map  : dict mapping role -> channel name
    hypno        : accepted for the positional call in pipeline.py and
                   intentionally unused: a disconnect during wake is as much
                   a montage fact as one during sleep, so the statistics are
                   whole-recording on purpose.

    Returns
    -------
    dict with keys:
      channels : {role: {channel, flat_pct, clip_pct, noise_pct, quality_grade}}
                 Keyed by ROLE: two roles can share one physical channel
                 (`flow` and `flow_thermistor` on a single thermistor) and
                 keyed by name they overwrote each other. The channel name
                 is in ``channel``.
      montage_warnings : list of str
      overall_grade : "good" | "acceptable" | "poor"
    """
    sf = raw.info["sfreq"]
    results = {"channels": {}, "montage_warnings": [], "overall_grade": "good"}

    for role, ch_name in channel_map.items():
        if ch_name is None or ch_name not in raw.ch_names:
            continue
        try:
            data = raw.get_data(picks=[ch_name])[0]
            ch_result = _assess_channel(data, sf, ch_name, role)
            results["channels"][role] = {"channel": ch_name, **ch_result}
        except Exception as e:
            logger.warning("Quality check failed for %s (%s): %s", ch_name, role, e)
            results["channels"][role] = {
                "channel": ch_name, "quality_grade": "unknown", "error": str(e)
            }

    # Montage plausibility checks
    results["montage_warnings"] = _check_montage(raw, channel_map)

    # Overall grade
    grades = [v.get("quality_grade", "good")
              for v in results["channels"].values()]
    if grades.count("poor") >= 2:
        results["overall_grade"] = "poor"
    elif "poor" in grades or grades.count("acceptable") >= 3:
        results["overall_grade"] = "acceptable"

    return results


def _assess_channel(data: np.ndarray, sf: float,
                    ch_name: str, role: str) -> dict:
    """Assess quality of a single channel."""
    n_samples = len(data)

    if role in DISCRETE_ROLES:
        # 1./2. Een discreet kanaal is per definitie "vlak" en "op zijn
        # maximum" zonder dat er iets mis is. Wat hier telt is of het buiten
        # zijn fysieke bereik staat (sensor los); clipping betekent niets.
        buiten = _out_of_range_mask(data, role)
        n_flat = int(buiten.sum())
        clip_pct = 0.0
        disconnects = _mask_runs_s(buiten, sf)
    else:
        # 1. Flat-line detection: std in een 2 s-venster onder een fractie
        #    van de robuuste schaal van het kanaal zelf (eenheidsvrij).
        flat_threshold = _flat_threshold(data)
        n_flat = _count_flat_samples(data, sf, flat_threshold)
        # 2. Clipping detection (signal at min/max rail)
        clip_pct = round(100 * _count_clipped(data) / max(n_samples, 1), 1)
        # 4. Disconnects (sudden flat after active signal)
        disconnects = _detect_disconnects(data, sf, flat_threshold)
    flat_pct = round(100 * n_flat / max(n_samples, 1), 1)

    # 3. High-impedance noise (50/60 Hz power ratio)
    noise_pct = round(_estimate_line_noise_pct(data, sf), 1)

    # Grade
    if flat_pct > 20 or clip_pct > 10:
        grade = "poor"
    elif flat_pct > 5 or clip_pct > 3 or noise_pct > 40:
        grade = "acceptable"
    else:
        grade = "good"

    result = {
        "flat_pct": flat_pct,
        "clip_pct": clip_pct,
        "noise_pct": noise_pct,
        "n_disconnects": len(disconnects),
        "disconnect_intervals_s": disconnects[:10],
        "quality_grade": grade,
    }

    if grade != "good":
        logger.info("[quality] %s (%s): grade=%s flat=%.1f%% clip=%.1f%% "
                    "noise=%.1f%% disconnects=%d",
                    ch_name, role, grade, flat_pct, clip_pct,
                    noise_pct, len(disconnects))

    return result


def _flat_threshold(data: np.ndarray) -> float:
    """Absolute std-drempel voor "vlak", afgeleid uit het kanaal zelf.

    FLAT_REL_STD maal de robuuste schaal (1,4826 x MAD). Is de MAD nul --
    een kanaal dat grotendeels op één waarde staat -- dan de gewone std; is
    ook die nul, dan is het hele kanaal constant en is elk venster vlak.

    Verving `_flat_threshold_for_role`, dat per rol een absolute std in ruwe
    eenheden gaf (0,5 µV EEG, 1e-4 ademhaling, 1e-5 overig). EDF-eenheden
    zijn per kanaal vrij, dus die drempel mat de eenhedendeclaratie: een
    RIP-kanaal in µV kwam via MNE 1e-5x zo klein binnen en was "100 % vlak".
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.inf
    schaal = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    if schaal <= 0:
        schaal = float(np.std(x))
    if schaal <= 0:
        return np.inf
    return FLAT_REL_STD * schaal


def _out_of_range_mask(data: np.ndarray, role: str) -> np.ndarray:
    """Monsters buiten het fysieke bereik van een discreet kanaal."""
    x = np.asarray(data, dtype=float)
    buiten = ~np.isfinite(x)
    if role == "spo2":
        lo, hi = SPO2_PLAUSIBLE
        buiten |= (x < lo) | (x > hi)
    return buiten


def _mask_runs_s(mask: np.ndarray, sf: float, min_s: float = 10.0) -> list:
    """Aaneengesloten True-stukken van minstens ``min_s`` als [onset, duur]."""
    if not np.any(mask):
        return []
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [len(mask)]))
    out = []
    for s, e in zip(starts, ends):
        dur = (e - s) / sf
        if dur >= min_s:
            out.append([round(s / sf, 1), round(dur, 1)])
    return out


def _count_flat_samples(data: np.ndarray, sf: float,
                        threshold: float) -> int:
    """Count samples in flat-line segments (sliding 2s window)."""
    win = max(int(2.0 * sf), 10)
    if len(data) < win:
        return 0

    # Rolling std via cumsum trick (fast)
    cumsum = np.cumsum(data)
    cumsum2 = np.cumsum(data ** 2)

    n = len(data)
    mean_w = (cumsum[win:] - cumsum[:-win]) / win
    var_w = (cumsum2[win:] - cumsum2[:-win]) / win - mean_w ** 2
    var_w = np.maximum(var_w, 0)
    std_w = np.sqrt(var_w)

    flat_mask = std_w < threshold
    # Each True at index i means data[i : i+win] is flat. Count the union
    # of those (overlapping) spans — the number of distinct flat samples —
    # via a difference array so overlapping windows are not double-counted.
    # (The old formula reduced to sum(flat_mask) * win, overcounting by ~win
    #  and letting flat_pct exceed 100%.)
    if not np.any(flat_mask):
        return 0
    starts = np.flatnonzero(flat_mask)
    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, np.minimum(starts + win, n), -1)
    covered = np.cumsum(delta[:-1]) > 0
    return int(covered.sum())


def _count_runs(mask: np.ndarray, min_run: int = 10) -> int:
    """Monsters in aaneengesloten True-stukken van minstens ``min_run``."""
    if not np.any(mask):
        return 0
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [len(mask)]))
    runs = ends[:len(starts)] - starts[:len(ends)]
    return int(np.sum(runs[runs >= min_run]))


def _count_clipped(data: np.ndarray) -> int:
    """Monsters op een ADC-rail.

    De oude regel -- "op het maximum én minstens tien keer achter elkaar" --
    sloeg aan op alles wat zijn hoogste waarde even vasthoudt: een SpO2 die
    minutenlang 96 % aanwijst, een opgesampled kanaal, een gekwantiseerde
    sinus die bij 512 Hz honderden monsters per ademtop op zijn bovenste
    niveau blijft. Een constant kanaal telde bovendien als 100 % geclipt,
    terwijl dat vlak is, niet op de rail.

    Nu: een rail is een PIEK in het amplitudehistogram. Het bereik wordt in
    banden van CLIP_RAIL_FRACTION x het bereik verdeeld (of één
    kwantisatiestap, als die groter is), uitgelijnd op het minimum en op
    niveau-index afgerond zodat een kwantisatieniveau nooit op een bandrand
    valt. De bovenste band moet CLIP_SPIKE_RATIO keer zoveel monsters dragen
    als de twee banden er direct onder (per band), en minstens
    CLIP_MIN_FRACTION van het kanaal. Een sinus -- ook gekwantiseerd -- haalt
    hooguit ~2x (arcsinusverdeling); een rail waarop 5 % van de amplitude is
    weggevallen haalt ~11x, 10 % ~21x. Daarna tellen, zoals voorheen, alleen
    stukken van minstens tien monsters. Verhoudingen binnen hetzelfde kanaal,
    dus eenheidsvrij.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n < 20:
        return 0
    d_min, d_max = float(np.min(x)), float(np.max(x))
    span = d_max - d_min
    if not np.isfinite(span) or span <= 0:
        return 0                               # constant: vlak, niet geclipt

    w = CLIP_RAIL_FRACTION * span
    # Kwantisatiestap: kleinste positieve afstand tussen voorkomende waarden.
    # Is die groter dan de band, dan vallen er banden leeg en zegt de
    # vergelijking met de banden eronder niets meer.
    niveaus = np.unique(x)
    if niveaus.size > 1:
        stap = float(np.min(np.diff(niveaus)))
        if stap > w:
            w = stap

    # Band-index per monster. `rint` in plaats van `floor`: bij gekwantiseerde
    # data is (x - d_min) / w een geheel getal plus afrondingsruis, en die
    # ruis mag een niveau niet in de band ernaast duwen.
    k = np.rint((x - d_min) / w).astype(np.int64)
    k_max = int(k.max())

    def _rail(rail_k: int, onder_k: tuple) -> int:
        rail = k == rail_k
        p_rail = float(rail.mean())
        p_onder = float(np.isin(k, onder_k).mean()) / 2.0   # per band
        if p_rail < CLIP_MIN_FRACTION:
            return 0
        if p_rail < CLIP_SPIKE_RATIO * max(p_onder, 1.0 / n):
            return 0
        return _count_runs(rail)

    return (_rail(k_max, (k_max - 1, k_max - 2))
            + _rail(0, (1, 2)))


def _estimate_line_noise_pct(data: np.ndarray, sf: float) -> float:
    """Estimate percentage of power in 49-51 Hz + 59-61 Hz bands."""
    if sf < 120 or len(data) < int(4 * sf):
        return 0.0

    # Use first 60s for efficiency
    seg = data[:min(len(data), int(60 * sf))]
    freqs, psd = sp_signal.welch(seg, fs=sf, nperseg=min(int(2*sf), len(seg)))

    total = np.sum(psd[(freqs >= 1) & (freqs <= sf/2 - 1)])
    if total < 1e-20:
        return 0.0

    line_50 = np.sum(psd[(freqs >= 49) & (freqs <= 51)])
    line_60 = np.sum(psd[(freqs >= 59) & (freqs <= 61)])

    return float((line_50 + line_60) / total * 100)


def _detect_disconnects(data: np.ndarray, sf: float,
                        flat_threshold: float) -> list:
    """Detect transitions from active signal to flat-line (disconnects).

    Returns list of [onset_s, duration_s] pairs.
    """
    win = max(int(2.0 * sf), 10)
    step = win  # non-overlapping for speed
    disconnects = []
    prev_active = True
    flat_start = None

    for i in range(0, len(data) - win, step):
        seg_std = float(np.std(data[i:i+win]))
        is_flat = seg_std < flat_threshold

        if is_flat and prev_active:
            flat_start = i / sf
        elif not is_flat and not prev_active and flat_start is not None:
            flat_dur = i / sf - flat_start
            if flat_dur >= 10.0:  # only report >=10s disconnects
                disconnects.append([round(flat_start, 1), round(flat_dur, 1)])
            flat_start = None

        prev_active = not is_flat

    # Handle trailing flat
    if flat_start is not None:
        flat_dur = len(data) / sf - flat_start
        if flat_dur >= 10.0:
            disconnects.append([round(flat_start, 1), round(flat_dur, 1)])

    return disconnects


def _check_montage(raw, channel_map: dict) -> list:
    """Basic montage plausibility checks via cross-correlation.

    Checks:
    - EEG and EOG should NOT be identical (copy error)
    - Left and right EOG should be anti-correlated (if both present)
    - ECG should not correlate highly with EEG (swapped leads)
    - Flow and effort should not be identical
    """
    warnings = []
    sf = raw.info["sfreq"]
    n_check = min(int(60 * sf), raw.n_times)  # first 60s

    def _get(role):
        name = channel_map.get(role)
        if name and name in raw.ch_names:
            return raw.get_data(picks=[name], start=0, stop=n_check)[0]
        return None

    def _corr(a, b):
        if a is None or b is None or len(a) < 100:
            return None
        a_z = a - np.mean(a)
        b_z = b - np.mean(b)
        denom = np.std(a_z) * np.std(b_z) * len(a_z)
        if denom < 1e-15:
            return None
        return float(np.sum(a_z * b_z) / denom)

    eeg = _get("eeg")
    eog = _get("eog")
    emg = _get("emg")
    flow = _get("flow")
    if flow is None:
        flow = _get("flow_pressure")
    thorax = _get("thorax")
    abdomen = _get("abdomen")

    # EEG ↔ EOG should not be identical
    r = _corr(eeg, eog)
    if r is not None and abs(r) > 0.95:
        warnings.append(
            f"EEG and EOG are nearly identical (r={r:.2f}) — "
            "possible montage error or shared reference")

    # Flow ↔ effort should not be identical
    r = _corr(flow, thorax)
    if r is not None and abs(r) > 0.95:
        warnings.append(
            f"Flow and thorax are nearly identical (r={r:.2f}) — "
            "possible channel duplication")

    # Thorax ↔ abdomen: anti-correlation suggests paradoxical breathing
    # but perfect correlation >0.99 suggests duplication
    r = _corr(thorax, abdomen)
    if r is not None and abs(r) > 0.98:
        warnings.append(
            f"Thorax and abdomen are nearly identical (r={r:.2f}) — "
            "possible channel duplication (would prevent OA/CA classification)")

    return warnings
