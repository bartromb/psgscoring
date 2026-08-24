"""
psgscoring.plm
==============
Periodic Limb Movement (PLM) detection per AASM criteria.

AASM criteria summary
---------------------
- Leg Movement (LM): EMG >= 8 µV above resting, duration 0.5–10 s
- Bilateral LMs within 0.5 s -> merged to one LM
- PLM series: >= 4 consecutive LMs, inter-movement interval 5–90 s
- Wake LMs excluded; respiratory-associated LMs excluded
- PLMI: PLMs per hour of sleep (significant >= 15/h)

Dependencies: numpy, scipy, psgscoring.constants, psgscoring.utils
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import label

from .constants import EPOCH_LEN_S
from .indices import per_hour
from .utils import safe_r

# AASM thresholds
LM_MIN_DUR_S       = 0.5
LM_MAX_DUR_S       = 10.0
LM_AMPLITUDE_UV    = 8.0     # µV above resting EMG
LM_OFFSET_UV       = 2.0
"""Einde-drempel van AASM regel 4.A, in µV boven rust.

De regel legt onset en einde op VERSCHILLENDE drempels: onset bij een stijging
van 8 µV, einde bij het begin van een periode van >= LM_MIN_DUR_S waarin het
EMG niet boven 2 µV boven rust komt. Deze module gebruikte 8 µV voor allebei,
waardoor de gemeten duur de tijd BOVEN 8 µV was -- stelselmatig korter, zodat
een beweging die net boven de drempel uitkomt onder het minimum van 0,5 s zakt.

Alleen actief met `offset_aasm=True`; zie de profielvlag `plm_offset_aasm`.
"""

# Hoeveel events er hoogstens in `result["events"]` gaan. Een payloadgrens,
# geen scoringsregel: `summary["n_events_truncated"]` zegt hoeveel er wegviel.
EVENT_LIST_CAP     = 200

_log_plm = logging.getLogger(__name__)
PLM_MIN_INTERVAL_S = 5.0
PLM_MAX_INTERVAL_S = 90.0
PLM_MIN_SERIES     = 4
BILATERAL_WIN_S    = 0.5
RESP_EXCLUSION_S   = 0.5     # LM within 0.5 s of resp event end -> excluded


def analyze_plm(
    leg_l: np.ndarray | None,
    leg_r: np.ndarray | None,
    sf: float,
    hypno: list,
    resp_events: list | None = None,
    artifact_epochs: list | None = None,
    leg_unit: str = "auto",
    time_base_fix: bool = True,
    event_list_cap: int | None = EVENT_LIST_CAP,
    offset_aasm: bool = False,
) -> dict:
    """
    Detect PLMs on left and/or right tibialis anterior EMG channels.

    Parameters
    ----------
    time_base_fix   : convert RMS-window index to seconds with the window's
                      real length (`win / sf`) instead of the 0.1 s it was
                      derived from. **Default True since 21-08-2026**; pass
                      False only to reproduce pre-repair output. See the
                      comment in `_detect_lm_channel` and
                      docs/plm_tijdbasis_bevinding.md.

    leg_l / leg_r   : raw EMG arrays
    sf              : sample rate
    hypno           : string hypnogram
    resp_events     : respiratory events (used for resp-associated exclusion)
    artifact_epochs : epochs to exclude from TST denominator
    leg_unit        : EMG physical unit. One of:
                       'V'    : volts (will be scaled ×1e6 to µV)
                       'mV'   : millivolts (will be scaled ×1e3 to µV)
                       'uV'   : already in µV (no scaling)
                       'auto' : amplitude-based heuristic (default)
                     The 8 µV detection threshold (AASM) is sensitive
                     to scaling errors; pass an explicit unit when the EDF
                     physical_unit is available rather than relying on
                     'auto'.

    Returns
    -------
    dict with keys: success, summary, events, series, error
    """
    result: dict = {"success": False, "summary": {}, "events": [], "error": None}
    try:
        if leg_l is None and leg_r is None:
            result["error"] = "No leg-EMG channels available"
            return result

        lms_l = (_detect_lm_channel(leg_l, sf, unit=leg_unit,
                                    time_base_fix=time_base_fix,
                                  offset_aasm=offset_aasm)
                 if leg_l is not None else [])
        lms_r = (_detect_lm_channel(leg_r, sf, unit=leg_unit,
                                    time_base_fix=time_base_fix,
                                  offset_aasm=offset_aasm)
                 if leg_r is not None else [])
        all_lms = _merge_bilateral(lms_l, lms_r)
        all_lms.sort(key=lambda x: x["onset_s"])

        # Tag with sleep stage; filter wake
        sleep_lms: list[dict] = []
        for lm in all_lms:
            ep_idx  = int(lm["onset_s"] // EPOCH_LEN_S)
            stage   = hypno[ep_idx] if ep_idx < len(hypno) else "W"
            lm["stage"] = stage
            lm["epoch"] = ep_idx
            if stage != "W":
                sleep_lms.append(lm)

        # Respiratory-associated exclusion
        resp_ends = [
            float(e["onset_s"]) + float(e["duration_s"])
            for e in (resp_events or [])
            if "onset_s" in e and "duration_s" in e
        ]
        plm_eligible, n_resp = _exclude_resp_associated(sleep_lms, resp_ends)

        # PLM series detection
        plm_series, plm_count = _detect_series(plm_eligible)

        # Mark PLM membership
        for lm in plm_eligible:
            lm["is_plm"] = False
        for series in plm_series:
            for lm in plm_eligible:
                if series["start_s"] <= lm["onset_s"] <= series["end_s"]:
                    lm["is_plm"] = True

        artifact_set  = set(artifact_epochs or [])
        total_sleep_s = sum(
            EPOCH_LEN_S for i, s in enumerate(hypno)
            if s != "W" and i not in artifact_set
        )
        total_sleep_h = total_sleep_s / 3600   # zie psgscoring/indices.py

        plmi = per_hour(plm_count, total_sleep_h)
        lmi  = per_hour(len(sleep_lms), total_sleep_h)

        # De lijst wordt afgekapt op `event_list_cap`. Dat is een payloadgrens,
        # geen scoringsregel, en ze was tot 21-08-2026 onzichtbaar: op PSG-IPA
        # SN1 zijn dat 200 van 660 bewegingen.
        #
        # `event_list_cap=None` kapt niet af. De pipeline gebruikt dat: die
        # rekent eerst `plm_arousal_index` over de VOLLEDIGE lijst en kapt pas
        # daarna af (zie pipeline.py, na `_compute_arousal_etiology`). Tot
        # 22-08-2026 gebeurde dat andersom, en dan telde een afgeleide index
        # alleen de eerste 200 bewegingen van de nacht -- een klinisch getal
        # dat van een payloadgrens afhing.
        #
        # De index zelf (`plm_index`, `n_plm`) is nooit geraakt: die wordt
        # hierboven uit `plm_eligible` en `plm_series` berekend, vóór het
        # afkappen.
        if event_list_cap is None:
            n_truncated = 0
            result["events"] = plm_eligible
        else:
            n_truncated = max(0, len(plm_eligible) - event_list_cap)
            if n_truncated:
                _log_plm.warning(
                    "PLM: %d van %d geschikte bewegingen niet in events[] "
                    "(grens %d); n_events_truncated staat in de samenvatting",
                    n_truncated, len(plm_eligible), event_list_cap,
                )
            result["events"] = plm_eligible[:event_list_cap]
        result["series"]  = plm_series
        result["summary"] = {
            "n_events_truncated": n_truncated,
            "n_lm_total":        len(all_lms),
            "n_lm_sleep":        len(sleep_lms),
            "n_lm_wake":         len(all_lms) - len(sleep_lms),
            "n_resp_associated": n_resp,
            "n_plm_eligible":    len(plm_eligible),
            "n_plm":             plm_count,
            "n_plm_series":      len(plm_series),
            "lm_index":          lmi,
            "plm_index":         plmi,
            "plm_severity":      _classify_plmi(plmi),
            "total_sleep_h":     safe_r(total_sleep_h),
        }
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_lm_channel(
    data: np.ndarray,
    sf: float,
    unit: str = "auto",
    time_base_fix: bool = True,
    offset_aasm: bool = False,
) -> list[dict]:
    """Detect LM events on a single EMG channel (AASM).

    The AASM amplitude criterion (≥8 µV above resting) is unit-sensitive,
    so the input must be scaled correctly. ``unit`` controls scaling:

      'V'    → multiply by 1e6  (volts → µV)
      'mV'   → multiply by 1e3  (millivolts → µV)
      'uV'   → leave as-is
      'auto' → amplitude heuristic with logging of likely scaling errors

    The 'auto' heuristic (v0.4.4 hardened):

      max|x| < 0.01    → assume V    (×1e6 → µV)
      max|x| < 10      → assume mV   (×1e3 → µV) and warn
      otherwise        → assume µV
    """
    import logging
    _log = logging.getLogger(__name__)

    data_uv = np.asarray(data, dtype=float).copy()
    if unit == "V":
        data_uv = data_uv * 1e6
    elif unit == "mV":
        data_uv = data_uv * 1e3
    elif unit == "uV":
        pass
    elif unit == "auto":
        max_abs = float(np.max(np.abs(data_uv))) if data_uv.size else 0.0
        if max_abs < 1e-12:
            # All-zero signal (stuck/disconnected channel); leave as-is.
            pass
        elif max_abs < 0.01:
            # Almost certainly volts (raw V-scaled EMG max typically 1–5 mV
            # = 0.001–0.005 V). Scale ×1e6 to µV.
            data_uv = data_uv * 1e6
        elif max_abs < 10:
            # Plausibly millivolts (e.g. EDF stored with mV physical_unit),
            # max ≈ 1–5 mV. Scale ×1e3 and emit a warning so callers can
            # pass `unit='mV'` explicitly to silence it.
            _log.warning(
                "PLM EMG amplitude max=%.3f looks like mV-scaled data; "
                "scaling ×1000 to µV. Pass leg_unit='mV' to silence this warning.",
                max_abs,
            )
            data_uv = data_uv * 1e3
        # else: max_abs >= 10 → assume already in µV, no scaling
    else:
        raise ValueError(
            f"Unknown leg_unit={unit!r}; expected 'V', 'mV', 'uV', or 'auto'"
        )

    nyq = sf / 2
    lo  = min(10.0 / nyq, 0.99)
    hi  = min(100.0 / nyq, 0.99)
    if lo >= hi:
        lo, hi = 0.1, 0.99
    b, a = sp_signal.butter(4, [lo, hi], btype="band")
    filt = sp_signal.filtfilt(b, a, data_uv)

    win = max(1, int(sf * 0.1))
    n_w = len(filt) // win
    rms = np.array([
        np.sqrt(np.mean(filt[i * win : (i + 1) * win] ** 2))
        for i in range(n_w)
    ])

    # `win` is een geheel aantal STALEN, dus een venster duurt `win / sf` --
    # niet de 0,1 s waar het uit is afgeleid. Bij 256 Hz is int(25,6) = 25
    # stalen = 0,09766 s. De omzetting van vensterindex naar tijd hieronder
    # rekende met 0,1, waardoor elke gerapporteerde tijd 2,3 % voorliep:
    # LINEAIR OPLOPEND tot +620 s aan het eind van een nacht van 7,4 u, en
    # +1013 s op een MESA-opname van 12 u. Het treft 256 Hz (2,3 %) en 128 Hz
    # (6,3 %); bij 100, 200 en 500 Hz is sf*0,1 toevallig geheel en is er geen
    # fout -- daarom viel het nooit op, en daarom klopte het AANTAL al.
    #
    # Gemeten op PSG-IPA tegen twaalf scoorders, per been, IoU 0,20:
    # mediane event-F1 0,038 -> 0,692 (mens onderling 0,820).
    # Zie docs/plm_tijdbasis_bevinding.md.
    # Default True sinds 21-08-2026; `time_base_fix=False` reproduceert het
    # gedrag van vóór die datum en is wat `mesa_shhs` en `chicago_1999`
    # via hun profiel doorgeven.
    step_s = (win / sf) if time_base_fix else 0.1

    resting   = float(np.percentile(rms, 10))
    threshold = resting + LM_AMPLITUDE_UV

    if offset_aasm:
        # AASM regel 4.A: onset bij +8 µV, EINDE bij het begin van een periode
        # van >= LM_MIN_DUR_S waarin het EMG niet boven +2 µV komt. De duur is
        # dus de tijd tot het signaal tot rust komt, niet de tijd boven 8 µV.
        return _lm_events_aasm_offset(rms, step_s, resting)

    labeled, n_bursts = label(rms > threshold)
    lms: list[dict] = []
    for i in range(1, n_bursts + 1):
        idx   = np.where(labeled == i)[0]
        dur_s = len(idx) * step_s
        if LM_MIN_DUR_S <= dur_s <= LM_MAX_DUR_S:
            lms.append({
                "onset_s":     idx[0] * step_s,
                "duration_s":  round(dur_s, 2),
                "amplitude_uv": round(float(np.max(rms[idx])), 1),
            })
    return lms


def _lm_events_aasm_offset(rms, step_s: float, resting: float) -> list[dict]:
    """Bewegingen volgens de twee-drempelregel van AASM 4.A."""
    hoog = rms > resting + LM_AMPLITUDE_UV
    laag = rms <= resting + LM_OFFSET_UV
    n_min = max(1, int(round(LM_MIN_DUR_S / step_s)))
    out: list[dict] = []
    i, N = 0, len(rms)
    while i < N:
        if not hoog[i]:
            i += 1
            continue
        start = i
        j = i
        while j < N:
            if laag[j]:
                k = j
                while k < N and laag[k]:
                    k += 1
                if (k - j) >= n_min:
                    break          # rustperiode lang genoeg: hier eindigt hij
                j = k
            else:
                j += 1
        dur_s = (j - start) * step_s
        if LM_MIN_DUR_S <= dur_s <= LM_MAX_DUR_S:
            out.append({
                "onset_s":      start * step_s,
                "duration_s":   round(dur_s, 2),
                "amplitude_uv": round(float(np.max(rms[start:max(j, start + 1)])), 1),
            })
        i = max(j, start + 1)
    return out


def _merge_bilateral(
    lms_l: list[dict],
    lms_r: list[dict],
) -> list[dict]:
    """Merge bilateral LMs (within 0.5 s) into a single LM."""
    used_r: set[int] = set()
    merged: list[dict] = []

    for lm in lms_l:
        found = False
        for j, rlm in enumerate(lms_r):
            if j in used_r:
                continue
            if abs(lm["onset_s"] - rlm["onset_s"]) <= BILATERAL_WIN_S:
                merged.append({
                    "onset_s":      min(lm["onset_s"],     rlm["onset_s"]),
                    "duration_s":   max(lm["duration_s"],  rlm["duration_s"]),
                    "amplitude_uv": max(lm["amplitude_uv"], rlm["amplitude_uv"]),
                    "bilateral":    True,
                })
                used_r.add(j)
                found = True
                break
        if not found:
            merged.append({**lm, "bilateral": False})

    for j, rlm in enumerate(lms_r):
        if j not in used_r:
            merged.append({**rlm, "bilateral": False})

    return merged


def _exclude_resp_associated(
    sleep_lms: list[dict],
    resp_ends: list[float],
) -> tuple[list[dict], int]:
    """Remove LMs within 0.5 s of a respiratory-event end."""
    eligible: list[dict] = []
    n_resp = 0
    for lm in sleep_lms:
        onset   = lm["onset_s"]
        is_resp = any(
            (re - RESP_EXCLUSION_S) <= onset <= (re + RESP_EXCLUSION_S)
            for re in resp_ends
        )
        lm["resp_associated"] = is_resp
        if is_resp:
            n_resp += 1
        else:
            eligible.append(lm)
    return eligible, n_resp


def _detect_series(
    plm_eligible: list[dict],
) -> tuple[list[dict], int]:
    """Identify PLM series (>= 4 LMs with 5–90 s intervals)."""
    series: list[dict] = []
    count = 0
    if len(plm_eligible) < PLM_MIN_SERIES:
        return series, count

    seq = [plm_eligible[0]]
    for j in range(1, len(plm_eligible)):
        interval = plm_eligible[j]["onset_s"] - plm_eligible[j - 1]["onset_s"]
        if PLM_MIN_INTERVAL_S <= interval <= PLM_MAX_INTERVAL_S:
            seq.append(plm_eligible[j])
        else:
            if len(seq) >= PLM_MIN_SERIES:
                count += len(seq)
                series.append(_series_dict(seq))
            seq = [plm_eligible[j]]
    if len(seq) >= PLM_MIN_SERIES:
        count += len(seq)
        series.append(_series_dict(seq))

    return series, count


def _series_dict(seq: list[dict]) -> dict:
    """Bouw een PLM-serie dictionary met start, einde en aantal bewegingen."""
    return {
        "start_s": seq[0]["onset_s"],
        "end_s":   seq[-1]["onset_s"] + seq[-1]["duration_s"],
        "n_lms":   len(seq),
    }


def _classify_plmi(plmi: float | None) -> str:
    """Classificeer PLM-index ernst: normaal (<5), licht (5-25), matig (25-50), ernstig (>50).

    ``None`` is "unknown", niet "normal". Een index die niet berekend kon
    worden — geen slaaptijd, geen beenkanaal — is geen schone uitslag maar een
    ontbrekende uitslag, en "normaal" leest als het eerste. Dat is dezelfde
    verwarring die elders "AHI 0,0" naast 81 events opleverde.
    ``_classify_arousal_index`` deed dit al goed; deze niet.
    """
    if plmi is None:
        return "unknown"
    if plmi == 0:
        return "normal"     # nul gemeten bewegingen IS een schone uitslag
    if plmi < 5:
        return "normal"
    if plmi < 15:
        return "mild"
    if plmi < 25:
        return "moderate"
    return "severe"
