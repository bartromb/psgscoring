"""
psgscoring.ancillary
====================
Ancillary signal analyses: body position, heart rate, snoring,
and Cheyne-Stokes respiration detection.

Dependencies: numpy, scipy, psgscoring.constants, psgscoring.utils
"""

from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal

from .constants import EPOCH_LEN_S, POSITION_MIN_MINUTES
from .utils import (
    build_sleep_mask, fmt_time, hypno_to_numeric,
    is_nrem, is_rem, safe_r,
)


# ---------------------------------------------------------------------------
# Body position
# ---------------------------------------------------------------------------

def analyze_position(
    pos_data: np.ndarray,
    sf: float,
    hypno: list,
    resp_events: list,
) -> dict:
    """
    Compute sleep time and AHI per body position.

    Returns
    -------
    dict with keys: success, summary (sleep_time_min, sleep_pct,
    ahi_per_pos), pos_per_epoch, error.
    """
    result: dict = {"success": False, "summary": {}, "error": None}
    try:
        spe           = int(sf * EPOCH_LEN_S)
        n_epochs      = len(hypno)
        # v0.8.12: auto-map raw ADC/voltage to 0-4 codes
        pos_mapped, pos_method = _map_position_signal(pos_data)
        pos_per_epoch = [_modal_position(pos_mapped, ep, spe) for ep in range(n_epochs)]

        pos_names = {0: "Prone", 1: "Left", 2: "Supine", 3: "Right", 4: "Upright"}
        sleep_time: dict[str, float | None] = {}
        ahi_pos:    dict[str, float | None] = {}
        n_events:   dict[str, int] = {}

        for code, name in pos_names.items():
            sleep_epochs = [
                i for i, (p, s) in enumerate(zip(pos_per_epoch, hypno))
                if p == code and s != "W"
            ]
            dur_min      = len(sleep_epochs) * (EPOCH_LEN_S / 60)
            sleep_time[name] = safe_r(dur_min)
            n_ev = sum(
                1 for ev in resp_events
                if 0 <= ev.get("epoch", 0) < len(pos_per_epoch)
                and pos_per_epoch[ev.get("epoch", 0)] == code
            )
            n_events[name] = n_ev
            dur_h         = dur_min / 60
            # Twee onwaarheden kwamen uit de oude regel. Onder de ondergrens
            # was het een opgeblazen index (1 event in 0,5 min = 120/u); bij
            # nul minuten was het `0`, wat leest als "geen events in die
            # houding" terwijl de patiënt er niet gelegen heeft. Allebei None:
            # de index bestaat niet, en dat is een uitspraak.
            ahi_pos[name] = (safe_r(n_ev / dur_h)
                             if dur_min >= POSITION_MIN_MINUTES else None)

        total_sleep_min = sum(v for v in sleep_time.values() if v)
        pct = {
            k: safe_r(v / total_sleep_min * 100) if total_sleep_min > 0 else 0
            for k, v in sleep_time.items()
        }

        result["summary"]       = {
            "sleep_time_min": sleep_time,
            "sleep_pct":      pct,
            "ahi_per_pos":    ahi_pos,
            # De TELLING erbij. `_compute_phenotypes` reconstrueerde het aantal
            # events uit index x uren -- een omweg langs een afronding die
            # bovendien breekt zodra de index None kan zijn.
            "n_events_per_pos": n_events,
            "min_minutes_for_index": POSITION_MIN_MINUTES,
            # "coded" = de codering van de recorder; "levels"/"percentile" =
            # een gok op de rangorde van de ruwe waarden. Zie
            # _map_position_signal.
            "position_mapping_method": pos_method,
        }
        result["pos_per_epoch"] = pos_per_epoch
        result["success"]       = True
    except Exception as e:
        result["error"] = str(e)
    return result


# Een houdingssignaal is bijna altijd DISCREET: de recorder schrijft een code
# of een vast spanningsniveau per houding. Zoveel van het signaal moet in een
# handvol niveaus zitten voordat we het als discreet behandelen.
_POSITION_LEVEL_COVERAGE = 0.90   # zoveel van de nacht moet in de niveaus zitten
_POSITION_LEVEL_MIN_FRAC = 0.005  # kleiner dan dit is ruis, geen houding
_POSITION_MAX_LEVELS     = 8
_POSITION_LEVEL_TOL      = 0.02   # clustertolerantie als fractie van het bereik
_POSITION_LEVEL_SPREAD   = 3.0    # een plateau mag zoveel maal de tolerantie breed zijn


def _position_levels(valid: np.ndarray) -> list[float] | None:
    """De discrete niveaus in een houdingssignaal, of None als het continu is.

    Een houdingssignaal is bijna altijd discreet: de recorder schrijft per
    houding een code of een vast spanningsniveau. Groeperen gebeurt op WAARDE,
    met een tolerantie die met het bereik meeschaalt, zodat de uitkomst niet
    verandert als de recorder in mV of in ADC-eenheden exporteert.

    Twee dingen die deze functie NIET mag doen, want allebei zijn ze de bug die
    hij vervangt:

    * stoppen zodra de dekking gehaald is. Ligt de patiënt 340 van de 420
      minuten in één houding, dan haalt dat ene plateau in zijn eentje al 80 %
      en zouden de overige houdingen wegvallen -- precies de duurafhankelijkheid
      die we kwijt willen. Elk niveau dat groot genoeg is telt mee.
    * een continu signaal als discreet behandelen. Een dichte verdeling heeft
      geen gaten groter dan de tolerantie en levert dus ÉÉN cluster over het
      hele bereik. Vandaar de eis dat een niveau ook echt smal is.
    """
    span = float(np.max(valid) - np.min(valid))
    if span <= 0:
        return [float(valid[0])]
    tol  = span * _POSITION_LEVEL_TOL
    orde = np.sort(valid)
    stukken = np.split(orde, np.flatnonzero(np.diff(orde) > tol) + 1)

    min_n = max(1, int(len(orde) * _POSITION_LEVEL_MIN_FRAC))
    niveaus, gedekt = [], 0
    for stuk in stukken:
        if len(stuk) < min_n:
            continue
        if float(stuk[-1] - stuk[0]) > tol * _POSITION_LEVEL_SPREAD:
            return None          # geen plateau maar een continuüm
        niveaus.append(float(np.median(stuk)))
        gedekt += len(stuk)
    if len(niveaus) < 2 or len(niveaus) > _POSITION_MAX_LEVELS:
        return None
    if gedekt / len(orde) < _POSITION_LEVEL_COVERAGE:
        return None
    return sorted(niveaus)


def _map_position_signal(pos_data: np.ndarray) -> tuple[np.ndarray, str]:
    """Map raw position signal to 0-4 codes (Prone/Left/Supine/Right/Upright).

    Geeft ``(codes, methode)`` terug. De methode hoort in de provenance: zonder
    dat is achteraf niet te zien of een positielabel uit de codering van de
    recorder komt of uit een gok op de rangorde.

    WAT HIER MIS WAS
    ----------------
    De ruwe tak quantiseerde op PERCENTIELEN van de nachtverdeling
    (``np.percentile(valid, [0,20,40,60,80,100])``). De bingrenzen schoven dus
    mee met hoe lang de patiënt in elke houding lag. Vier vaste
    plateauwaarden kregen drie verschillende labelsets, afhankelijk van de
    duurverdeling, en twee onderscheiden houdingen vielen samen op één code --
    waarna hun events en minuten op één hoop kwamen. De positie-AHI, het
    POSA-fenotype en de therapieaanbeveling hangen daaraan.

    WAT DIT NIET OPLOST
    -------------------
    Welke houding een niveau IS. Zowel de oude als de nieuwe tak nemen aan dat
    oplopende ruwe waarden overeenkomen met de volgorde van ``POSITION_MAP``.
    Dat is een aanname over de recorder, geen meting. Alleen een
    fabrikantspecifieke codetabel kan dat beslissen; tot die er is meldt de
    methode eerlijk dat het een rangorde-gok is (``"levels"``/``"percentile"``)
    en niet de codering van het apparaat (``"coded"``).
    """
    rounded = np.round(pos_data).astype(int)
    unique_vals = np.unique(rounded)

    # Already coded 0-4 → use as-is
    if len(unique_vals) <= 6 and np.all((unique_vals >= 0) & (unique_vals <= 5)):
        return np.clip(rounded, 0, 4), "coded"

    valid = pos_data[~np.isnan(pos_data)]
    if len(valid) == 0:
        return np.zeros(len(pos_data), dtype=int), "empty"

    niveaus = _position_levels(valid)
    if niveaus:
        # Rangorde op WAARDE, niet op duur. Elk niveau krijgt het volgende
        # codenummer; een nacht met drie houdingen krijgt er drie, geen vijf.
        # Grenzen halverwege twee niveaus, zodat ruis rond een plateau bij dat
        # plateau blijft.
        edges = [(a + b) / 2 for a, b in zip(niveaus, niveaus[1:])]
        mapped = np.digitize(pos_data, edges)
        return np.clip(mapped, 0, 4), "levels"

    # Continu signaal (bv. een hoek uit een accelerometer): er is geen
    # discreet niveau om op te rangschikken. Laatste redmiddel, en het blijft
    # duurafhankelijk -- vandaar dat de methode meegegeven wordt.
    edges = np.percentile(valid, [0, 20, 40, 60, 80, 100])
    mapped = np.digitize(pos_data, edges[1:-1])  # 0-4
    return np.clip(mapped, 0, 4), "percentile"


def _modal_position(pos_data: np.ndarray, ep: int, spe: int) -> int:
    """Bepaal de meest voorkomende slaappositie (modus) voor een epoch."""
    s   = ep * spe
    e   = min(s + spe, len(pos_data))
    seg = pos_data[s:e]
    if len(seg) == 0:
        return -1
    vals, cnts = np.unique(seg.astype(int), return_counts=True)
    return int(vals[np.argmax(cnts)])


# ---------------------------------------------------------------------------
# Heart rate
# ---------------------------------------------------------------------------

HR_IMPLAUSIBLE_LOW  = 20.0
HR_IMPLAUSIBLE_HIGH = 250.0
HR_DROPOUT_MAX_FRAC = 0.01
"""Boven dit aandeel onmogelijke waarden tijdens slaap is het extremum niet meer
te vertrouwen: het is dan de filtergrens, niet de patiënt."""


def analyze_heart_rate(
    hr_data: np.ndarray,
    sf: float,
    hypno: list,
    source: str = "pulse",
) -> dict:
    """
    Basic heart-rate statistics during sleep.

    Physiologically implausible values (< 20 or > 250 bpm) are removed.

    ``min_hr``/``max_hr`` zijn absolute extremen en dus zo betrouwbaar als het
    slechtste sample. Op drie opnames stond er een minimum van 20,2 · 32,6 ·
    20,0 bpm — twee daarvan liggen exact op de ondergrens van dit filter, en dat
    is de signatuur van een oximeter die even loslaat, niet van bradycardie.
    Daarom komen er robuuste percentielen naast te staan plus een expliciet
    oordeel: liever "niet betrouwbaar" tonen dan een fout getal.

    ``source`` benoemt waar het signaal vandaan komt. Bij ``"ecg"`` gaat er een
    ruwe ECG-golfvorm door deze functie — geen bpm, en geen RR-detectie
    onderweg — en zijn de getallen per definitie geen hartfrequentie.
    """
    result: dict = {"success": False, "summary": {}, "error": None}
    try:
        raw_hr = hr_data.copy().astype(float)
        hr = raw_hr.copy()
        hr[(hr < HR_IMPLAUSIBLE_LOW) | (hr > HR_IMPLAUSIBLE_HIGH)] = np.nan

        sleep_mask = build_sleep_mask(hypno, sf, len(hr))
        n_sleep    = int(np.sum(sleep_mask))
        hr_sleep   = hr[sleep_mask]
        hr_sleep   = hr_sleep[~np.isnan(hr_sleep)]

        if len(hr_sleep) == 0:
            result["error"] = "No HR data during sleep"
            return result

        n_implausible = max(0, n_sleep - len(hr_sleep))
        frac_bad = (n_implausible / n_sleep) if n_sleep else 0.0

        if source != "pulse":
            reliable, reason = False, (
                f"afgeleid van kanaalrol '{source}' zonder R-piekdetectie — "
                "geen hartfrequentie")
        elif frac_bad > HR_DROPOUT_MAX_FRAC:
            reliable, reason = False, (
                f"{frac_bad * 100:.1f}% van de slaap onmogelijke waarden "
                "(sensoruitval); extremen liggen op de filtergrens")
        else:
            reliable, reason = True, ""

        result["summary"] = {
            "avg_hr":        safe_r(float(np.mean(hr_sleep))),
            "min_hr":        safe_r(float(np.min(hr_sleep))),
            "max_hr":        safe_r(float(np.max(hr_sleep))),
            "std_hr":        safe_r(float(np.std(hr_sleep))),
            "n_tachycardia": int(np.sum(hr_sleep > 100)),
            "n_bradycardia": int(np.sum(hr_sleep < 50)),
            # Robuuste tegenhangers van min/max: één uitvalsample verschuift
            # deze niet.
            "hr_p1":         safe_r(float(np.percentile(hr_sleep, 1))),
            "hr_p99":        safe_r(float(np.percentile(hr_sleep, 99))),
            "hr_source":     source,
            "hr_implausible_pct": safe_r(frac_bad * 100.0),
            "hr_reliable":   reliable,
            "hr_unreliable_reason": reason,
        }
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
    return result


# ---------------------------------------------------------------------------
# Snoring
# ---------------------------------------------------------------------------

def analyze_snore(
    snore_data: np.ndarray,
    sf: float,
    hypno: list,
) -> dict:
    """
    Estimate snoring duration and index from a microphone / snore channel.

    Uses 1-second RMS windows; the 60th percentile of the RMS distribution
    is used as the snoring threshold.
    """
    result: dict = {"success": False, "summary": {}, "error": None}
    try:
        win        = int(sf)
        n_windows  = len(snore_data) // win
        rms        = np.array([
            np.sqrt(np.mean(snore_data[i * win : (i + 1) * win] ** 2))
            for i in range(n_windows)
        ])
        threshold  = float(np.percentile(rms, 60))
        snore_mask = rms > threshold

        # Build 1-s sleep mask
        hypno_num      = hypno_to_numeric(hypno)
        sleep_mask_1s  = np.zeros(n_windows, dtype=bool)
        for ep_i, stage in enumerate(hypno_num):
            s = ep_i * EPOCH_LEN_S
            e = min(s + EPOCH_LEN_S, n_windows)
            if stage > 0:
                sleep_mask_1s[s:e] = True

        snore_sleep    = snore_mask & sleep_mask_1s
        total_sleep_s  = float(np.sum(sleep_mask_1s))
        snore_s        = float(np.sum(snore_sleep))
        total_sleep_h  = total_sleep_s / 3600

        result["summary"] = {
            "snore_min":     safe_r(snore_s / 60),
            "snore_pct_tst": (
                safe_r(snore_s / total_sleep_s * 100) if total_sleep_s > 0 else 0
            ),
            "snore_index":   (
                safe_r((snore_s / 60) / total_sleep_h) if total_sleep_h > 0 else 0
            ),
        }
        # v0.8.12: expose RMS timeseries for PDF overview plot
        result["rms_1s"] = rms.tolist()
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
    return result


# ---------------------------------------------------------------------------
# Cheyne-Stokes Respiration
# ---------------------------------------------------------------------------

def detect_cheyne_stokes(
    flow_env: np.ndarray,
    sf: float,
    hypno: list,
    min_cycle_s: float = 40.0,
    max_cycle_s: float = 120.0,
) -> dict:
    """
    Detect Cheyne-Stokes Respiration (CSR) via autocorrelation of the
    very-low-frequency flow envelope (0.005–0.05 Hz = 20–200 s periods).

    A normalised autocorrelation peak > 0.3 in the 40–120 s lag range
    indicates a significant crescendo-decrescendo pattern.

    NOTE (v0.4.4 review): the literature on autocorrelation-based CSR
    detection uses tighter thresholds (Trinder et al., Sleep 1991: >0.4;
    He et al., EHJ 2023: >0.5). At 0.3 this detector is more sensitive
    but may over-flag non-CSR periodic breathing, leading to
    over-aggressive central reclassification downstream in
    postprocess.reclassify_csr_events. Default kept at 0.3 for backward
    compatibility with paper v31 numerics; tighten to 0.4-0.5 for a more
    conservative CSR call by passing a higher threshold to the caller
    (currently the parameter is hard-coded; v0.5 will expose it).

    Returns
    -------
    dict: success, csr_detected, periodicity_s, csr_minutes,
    csr_pct_sleep, error.
    """
    out = {
        "success":      False,
        "csr_detected": False,
        "periodicity_s": None,
        "csr_minutes":  0,
    }
    try:
        n = len(flow_env)
        if n < int(sf * min_cycle_s * 3):
            out["success"] = True
            return out

        nyq = sf / 2
        lo  = max(0.005 / nyq, 0.0001)
        hi  = min(0.05  / nyq, 0.49)
        if lo >= hi:
            out["success"] = True
            return out

        b, a      = sp_signal.butter(2, [lo, hi], btype="band")
        slow_env  = np.abs(sp_signal.filtfilt(b, a, flow_env))

        min_lag = int(min_cycle_s * sf)
        max_lag = min(int(max_cycle_s * sf), n // 2)
        if max_lag <= min_lag:
            out["success"] = True
            return out

        centered = slow_env - np.mean(slow_env)
        var      = np.var(centered)
        if var < 1e-12:
            out["success"] = True
            return out

        lags  = np.arange(min_lag, max_lag, max(1, int(sf * 2)))
        acorr = np.array([
            np.mean(centered[: n - lag] * centered[lag:]) / var
            for lag in lags
        ])

        if len(acorr) > 2:
            peak_idx = int(np.argmax(acorr))
            peak_val = float(acorr[peak_idx])

            if peak_val > 0.3:  # paper v31 default; see docstring NOTE above
                period_s = lags[peak_idx] / sf
                out["periodicity_s"] = safe_r(period_s)
                out["csr_detected"]  = True

                # Count 2-minute CSR segments
                seg_len    = int(120 * sf)
                csr_min    = 0
                for seg_start in range(0, n - seg_len, seg_len):
                    if np.var(centered[seg_start : seg_start + seg_len]) > 0.5 * var:
                        csr_min += 2
                out["csr_minutes"] = csr_min

                sleep_min = sum(
                    EPOCH_LEN_S for s in hypno if s != "W"
                ) / 60
                out["csr_pct_sleep"] = safe_r(
                    csr_min / max(sleep_min, 1) * 100
                )

        out["success"] = True
    except Exception as e:
        out["error"] = str(e)
    return out
