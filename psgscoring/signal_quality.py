"""
signal_quality.py — RIP sensor quality validation for psgscoring v0.2.96+

Motivation
----------
Respiratory event classification (obstructive/central/mixed) depends on
bilateral effort signals (thorax + abdomen RIP). When one channel fails
(sensor disconnect, calibration drift, movement artifact), paradoxical
phase detection becomes impossible and the classifier defaults to
obstructive. This module detects such failures BEFORE classification and
provides:

1. Per-channel quality assessment
2. Pair comparison (thorax vs abdomen)
3. Single-channel fallback classification
4. Clinical warning text for reports

Empirically validated
---------------------
Clinical case "Loos" (AZORG, April 2026):
- Thorax MAD: 0.0017, breath-band energy: 3e-04 → classified 'failed'
- Abdomen MAD: 0.046, breath-band energy: 2.0  → classified 'ok'
- Energy ratio: 6862× → pair flagged as unreliable
- Psgscoring v0.2.951 reported: OSAS, AHI 56.6, CAI 3.8
- With abdomen-only fallback: CSAS, CAI 45.1 (217 events reclassified)

References
----------
- AASM Manual 2.6, Chapter 2 (Respiratory signals)
- Kushida et al. 2005 — PSG sensor reliability
- Redline et al. 2004 — Inter-scorer variability due to signal quality
"""

from __future__ import annotations
import logging
from typing import Literal, TypedDict

import numpy as np
from scipy.signal import hilbert, welch

logger = logging.getLogger("psgscoring.signal_quality")


# ════════════════════════════════════════════════════════════════════
#  Type definitions
# ════════════════════════════════════════════════════════════════════

ChannelStatus = Literal["ok", "weak", "failed"]
EventClassification = Literal["obstructive", "central", "uncertain"]


class ChannelQuality(TypedDict):
    mad: float
    breath_energy: float
    peak_freq: float | None
    status: ChannelStatus
    reason: str


class PairQuality(TypedDict):
    thorax: ChannelQuality
    abdomen: ChannelQuality
    energy_ratio: float
    warnings: list[str]
    classification_reliable: bool
    recommended_mode: Literal["bilateral", "single-channel", "unreliable"]
    working_channel: Literal["thorax", "abdomen", "none"] | None


# ════════════════════════════════════════════════════════════════════
#  Thresholds (empirically calibrated)
# ════════════════════════════════════════════════════════════════════

# Per-channel thresholds
MAD_FAILED_BELOW      = 0.005   # Below this: sensor is dead
MAD_WEAK_BELOW        = 0.020   # Below this: sensor is weak
ENERGY_FAILED_BELOW   = 0.001   # Breath-band energy (Welch PSD sum)
ENERGY_WEAK_BELOW     = 0.050

# Pair thresholds
ENERGY_RATIO_WARN     = 10.0    # 10× asymmetry is suspicious
ENERGY_RATIO_FAIL     = 100.0   # 100× means one channel is ~dead

# Breathing band for energy computation
BREATH_FREQ_LOW       = 0.10    # Hz (6 breaths/min minimum)
BREATH_FREQ_HIGH      = 0.50    # Hz (30 breaths/min maximum)

# Single-channel fallback thresholds
FALLBACK_CENTRAL_RATIO      = 0.20   # Event envelope <20% baseline → central
FALLBACK_OBSTRUCTIVE_RATIO  = 0.50   # Event envelope >50% baseline → obstructive
                                     # NOTE (v0.4.4 review): single-channel
                                     # fallback cannot detect paradox, so
                                     # this threshold tolerates events where
                                     # cardiac pulsation alone (20-50% of
                                     # baseline) might pass as "obstructive".
                                     # A stricter threshold (0.70) is more
                                     # conservative but excludes valid
                                     # obstructive events with smaller
                                     # residual envelope. Default kept at
                                     # 0.50 for backward compatibility;
                                     # callers concerned about cardiac
                                     # pulsation should pass an explicit
                                     # higher value or add an explicit
                                     # cardiac-band-power check upstream.
FALLBACK_BASELINE_WINDOW_S  = 120.0  # Baseline = preceding 2 minutes
FALLBACK_BASELINE_PERCENTILE = 75    # Robust to event clusters


# ════════════════════════════════════════════════════════════════════
#  Core functions
# ════════════════════════════════════════════════════════════════════

def assess_rip_channel(
    signal: np.ndarray,
    sf: float,
    label: str = "",
) -> ChannelQuality:
    """
    Assess quality of a single RIP channel.

    Parameters
    ----------
    signal : np.ndarray
        Raw RIP signal (time domain)
    sf : float
        Sample rate in Hz
    label : str
        Optional label for logging

    Returns
    -------
    dict with keys: mad, breath_energy, peak_freq, status, reason
    """
    if signal is None or len(signal) == 0:
        return {
            "mad": 0.0,
            "breath_energy": 0.0,
            "peak_freq": None,
            "status": "failed",
            "reason": "Empty signal",
        }

    # v0.2.963 SQUEEZE2D MARKER: MNE raw.get_data() returns shape (1, N).
    # welch() on 2D produces 2D psd, breaking 1D boolean masking later.
    signal = np.asarray(signal, dtype=float).squeeze()
    if signal.ndim != 1:
        return {
            "mad": 0.0,
            "breath_energy": 0.0,
            "peak_freq": None,
            "status": "failed",
            "reason": f"Expected 1D signal, got {signal.ndim}D shape {signal.shape}",
        }

    mad = float(np.median(np.abs(signal - np.median(signal))))

    # Welch PSD for breath-band energy
    nperseg = int(min(60 * sf, max(len(signal) // 4, 64)))
    try:
        f, psd = welch(signal, sf, nperseg=nperseg)
    except Exception as e:
        logger.debug(f"[{label}] Welch failed: {e}")
        return {
            "mad": mad,
            "breath_energy": 0.0,
            "peak_freq": None,
            "status": "failed",
            "reason": f"PSD computation failed: {e}",
        }

    breath_mask = (f >= BREATH_FREQ_LOW) & (f <= BREATH_FREQ_HIGH)
    breath_energy = float(np.sum(psd[breath_mask]))

    if np.any(breath_mask):
        peak_freq = float(f[breath_mask][np.argmax(psd[breath_mask])])
    else:
        peak_freq = None

    # Classification
    if mad < MAD_FAILED_BELOW or breath_energy < ENERGY_FAILED_BELOW:
        status = "failed"
        reason = (f"MAD={mad:.4f} (<{MAD_FAILED_BELOW}), "
                  f"energy={breath_energy:.2e} (<{ENERGY_FAILED_BELOW:.0e})")
    elif mad < MAD_WEAK_BELOW or breath_energy < ENERGY_WEAK_BELOW:
        status = "weak"
        reason = (f"MAD={mad:.4f}, energy={breath_energy:.2e} — "
                  f"below normal but above failure threshold")
    else:
        status = "ok"
        reason = (f"MAD={mad:.4f}, energy={breath_energy:.2e} — within normal range")

    if label:
        logger.debug(f"[{label}] quality: {status} — {reason}")

    return {
        "mad": mad,
        "breath_energy": breath_energy,
        "peak_freq": peak_freq,
        "status": status,
        "reason": reason,
    }


def compare_rip_pair(
    thorax: np.ndarray,
    abdomen: np.ndarray,
    sf: float,
) -> PairQuality:
    """
    Compare thorax + abdomen RIP pair to detect channel failure,
    inversion, or extreme asymmetry.

    Parameters
    ----------
    thorax, abdomen : np.ndarray
        Raw RIP signals (time domain, same length)
    sf : float
        Sample rate in Hz

    Returns
    -------
    dict with:
        thorax, abdomen: ChannelQuality
        energy_ratio: float (max/min of breath-band energies)
        warnings: list of clinical warning strings
        classification_reliable: bool
        recommended_mode: 'bilateral' | 'single-channel' | 'unreliable'
        working_channel: 'thorax' | 'abdomen' | 'none' (when single-channel)
    """
    thor_q = assess_rip_channel(thorax, sf, "thorax")
    abd_q = assess_rip_channel(abdomen, sf, "abdomen")

    # Energy ratio (max / min)
    thor_e = max(thor_q["breath_energy"], 1e-12)
    abd_e = max(abd_q["breath_energy"], 1e-12)
    ratio = max(thor_e, abd_e) / min(thor_e, abd_e)

    warnings_list: list[str] = []
    thor_ok = thor_q["status"] == "ok"
    abd_ok = abd_q["status"] == "ok"
    thor_failed = thor_q["status"] == "failed"
    abd_failed = abd_q["status"] == "failed"

    # Determine mode
    if thor_failed and abd_failed:
        mode = "unreliable"
        working_ch = "none"
        warnings_list.append(
            "Both RIP channels failed. No effort-based classification possible. "
            "Results should be treated as uninterpretable for central/obstructive typing."
        )
    elif thor_failed and not abd_failed:
        mode = "single-channel"
        working_ch = "abdomen"
        warnings_list.append(
            "Thorax RIP failed — abdomen-only classification. "
            "Paradoxical phase detection unavailable."
        )
    elif abd_failed and not thor_failed:
        mode = "single-channel"
        working_ch = "thorax"
        warnings_list.append(
            "Abdomen RIP failed — thorax-only classification. "
            "Paradoxical phase detection unavailable."
        )
    elif ratio > ENERGY_RATIO_FAIL:
        mode = "single-channel"
        weak = "thorax" if thor_e < abd_e else "abdomen"
        working_ch = "abdomen" if weak == "thorax" else "thorax"
        warnings_list.append(
            f"RIP energy ratio {ratio:.0f}× — {weak} likely disconnected. "
            f"Using {working_ch}-only classification."
        )
    elif ratio > ENERGY_RATIO_WARN:
        mode = "bilateral"
        working_ch = None
        warnings_list.append(
            f"RIP energy ratio {ratio:.1f}× — modest asymmetry. "
            f"Classification proceeds bilaterally but review recommended."
        )
    elif not thor_ok or not abd_ok:
        mode = "bilateral"
        working_ch = None
        warnings_list.append(
            f"One or both RIP channels weak (thorax={thor_q['status']}, "
            f"abdomen={abd_q['status']}). Effort classification may be less reliable."
        )
    else:
        mode = "bilateral"
        working_ch = None

    classification_reliable = (mode == "bilateral" and len(warnings_list) == 0)

    return {
        "thorax": thor_q,
        "abdomen": abd_q,
        "energy_ratio": float(ratio),
        "warnings": warnings_list,
        "classification_reliable": classification_reliable,
        "recommended_mode": mode,
        "working_channel": working_ch,
    }


def single_channel_fallback_classify(
    apnea_start_s: float,
    apnea_end_s: float,
    effort_signal: np.ndarray,
    sf: float,
    baseline_window_s: float = FALLBACK_BASELINE_WINDOW_S,
) -> EventClassification:
    """
    Classify event using only ONE effort signal (when bilateral fails).

    Parameters
    ----------
    apnea_start_s, apnea_end_s : event boundaries (seconds)
    effort_signal : array — the working (thorax OR abdomen) signal
    sf : sample rate
    baseline_window_s : pre-event baseline window (default 120s)

    Returns
    -------
    'central' | 'obstructive' | 'uncertain'

    Logic
    -----
    - Event envelope (Hilbert median) compared to baseline P75 envelope
    - <20% of baseline → central
    - >50% of baseline → obstructive
    - Between → uncertain (flagged for manual review)
    """
    i0 = int(apnea_start_s * sf)
    i1 = int(apnea_end_s * sf)
    bl_i0 = max(0, i0 - int(baseline_window_s * sf))

    if i1 - i0 < int(2 * sf):
        return "uncertain"
    if i0 - bl_i0 < int(10 * sf):
        return "uncertain"

    bl_seg = effort_signal[bl_i0:i0]
    ev_seg = effort_signal[i0:i1]

    try:
        bl_env = np.abs(hilbert(bl_seg))
        ev_env = np.abs(hilbert(ev_seg))
    except Exception:
        return "uncertain"

    bl_amp = float(np.percentile(bl_env, FALLBACK_BASELINE_PERCENTILE))
    ev_amp = float(np.median(ev_env))

    if bl_amp < 1e-9:
        return "uncertain"

    ratio = ev_amp / bl_amp

    if ratio < FALLBACK_CENTRAL_RATIO:
        return "central"
    elif ratio > FALLBACK_OBSTRUCTIVE_RATIO:
        return "obstructive"
    else:
        return "uncertain"


def quality_warning_text(quality: PairQuality, lang: str = "en") -> str | None:
    """
    Generate clinical warning text for PDF report / dashboard.
    Multilingual: en, nl, fr, de.

    Returns None if classification is reliable (no warning needed).
    """
    if quality["classification_reliable"]:
        return None

    I18N = {
        "en": {
            "header": "⚠ RESPIRATORY EFFORT SIGNAL QUALITY WARNING",
            "impact": "IMPACT ON SCORING:",
            "impact_items": [
                "Central/mixed apnea classification may be INCORRECT",
                "Obstructive classifications may be FALSE POSITIVES",
                "Manual review of effort signals strongly recommended",
            ],
            "recommendation": "RECOMMENDATION:",
            "rec_text": (
                "Verify sensor placement and calibration. If signal cannot be "
                "salvaged, consider re-study or expert scorer review."
            ),
        },
        "nl": {
            "header": "⚠ WAARSCHUWING — KWALITEIT RESPIRATOIRE EFFORT-SIGNALEN",
            "impact": "IMPACT OP SCORING:",
            "impact_items": [
                "Classificatie centraal/gemengde apneu mogelijk ONJUIST",
                "Obstructieve classificaties mogelijk VALS-POSITIEF",
                "Manuele review van effort-signalen sterk aanbevolen",
            ],
            "recommendation": "AANBEVELING:",
            "rec_text": (
                "Verifieer sensorplaatsing en kalibratie. Als signaal niet "
                "kan worden hersteld: overweeg herhaalstudie of expert-scoring."
            ),
        },
        "fr": {
            "header": "⚠ AVERTISSEMENT — QUALITÉ DES SIGNAUX D'EFFORT RESPIRATOIRE",
            "impact": "IMPACT SUR LE SCORING:",
            "impact_items": [
                "Classification apnée centrale/mixte possiblement INCORRECTE",
                "Classifications obstructives possiblement FAUX POSITIFS",
                "Révision manuelle des signaux d'effort fortement recommandée",
            ],
            "recommendation": "RECOMMANDATION:",
            "rec_text": (
                "Vérifier placement et calibration des capteurs. "
                "Envisager une nouvelle étude si le signal ne peut être récupéré."
            ),
        },
        "de": {
            "header": "⚠ WARNUNG — QUALITÄT DER ATEMANSTRENGUNGSSIGNALE",
            "impact": "AUSWIRKUNG AUF SCORING:",
            "impact_items": [
                "Klassifikation zentraler/gemischter Apnoen möglicherweise FALSCH",
                "Obstruktive Klassifikationen möglicherweise FALSCH-POSITIV",
                "Manuelle Überprüfung der Effort-Signale dringend empfohlen",
            ],
            "recommendation": "EMPFEHLUNG:",
            "rec_text": (
                "Sensorpositionierung und Kalibrierung überprüfen. "
                "Bei irreparablem Signal: Wiederholung oder Experten-Scoring erwägen."
            ),
        },
    }

    t = I18N.get(lang, I18N["en"])

    parts = [t["header"], ""]
    for w in quality["warnings"]:
        parts.append(f"  • {w}")
    parts.extend([
        "",
        t["impact"],
        *(f"  - {item}" for item in t["impact_items"]),
        "",
        t["recommendation"],
        f"  {t['rec_text']}",
    ])
    return "\n".join(parts)


def quality_badge_summary(quality: PairQuality) -> dict:
    """
    Compact badge info for dashboard UI.

    Returns
    -------
    dict with:
        level: 'ok' | 'warning' | 'danger'  — for color coding
        label: short label (single word)
        tooltip: detailed explanation
    """
    mode = quality["recommended_mode"]
    if mode == "bilateral" and quality["classification_reliable"]:
        return {
            "level": "ok",
            "label": "OK",
            "tooltip": "Both RIP channels functioning normally.",
        }
    elif mode == "bilateral":
        return {
            "level": "warning",
            "label": "Weak",
            "tooltip": (quality["warnings"][0]
                        if quality["warnings"]
                        else "One or both effort channels weak."),
        }
    elif mode == "single-channel":
        working = quality["working_channel"] or "unknown"
        return {
            "level": "warning",
            "label": f"{working.capitalize()}-only",
            "tooltip": (quality["warnings"][0]
                        if quality["warnings"]
                        else f"Single-channel fallback using {working}."),
        }
    else:  # unreliable
        return {
            "level": "danger",
            "label": "Failed",
            "tooltip": (quality["warnings"][0]
                        if quality["warnings"]
                        else "Both RIP channels failed — classification unreliable."),
        }


# ════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════

__all__ = [
    "assess_rip_channel",
    "compare_rip_pair",
    "single_channel_fallback_classify",
    "quality_warning_text",
    "quality_badge_summary",
    "ChannelQuality",
    "PairQuality",
    "ChannelStatus",
    "EventClassification",
]


# ---------------------------------------------------------------------------
# Flow-sensorpaar: mag de thermistor de apneu-sensor worden?
# ---------------------------------------------------------------------------
# De AASM schrijft de oronasale thermistor voor apneus voor, en psgscoring
# wijst die rol toe zodra het kanaal herkend wordt. Dat ging mis op montages
# waar de thermistor aanwezig is maar geen bruikbaar ademsignaal draagt: op
# een echte opname zakte het aantal apneus van 93 naar 0 en verschoof de
# uitkomst van matig CSAS (bevestigd door menselijke scoring) naar mild SAS.
#
# De beslissende eigenschap is niet of de thermistor ademband-energie heeft —
# ruis kan die ook hebben — maar of hij het met de neusdruk EENS is over
# wanneer de ademhaling wegvalt. Dat is wat de envelope-correlatie meet.
#
# LET OP bij de drempel hieronder. Gemeten op 8 opnames scheidden de klassen
# elkaar NIET: bekend-slecht liep tot +0,225, bekend-goed begon op +0,226.
# 0,40 is daarom bewust conservatief gekozen, niet afgeleid: hij ligt boven
# elk gemeten slecht paar en houdt ook enkele bruikbare paren tegen. Dat is
# de veilige kant, want een gemiste centrale apneu weegt zwaarder dan een
# milde overdetectie op de neusdruk. Met meer opnames hoort dit opnieuw
# bepaald te worden.
THERMISTOR_AGREEMENT_MIN = 0.40
THERMISTOR_ENV_SMOOTH_S  = 10.0

# ── Alternatieve poort: één kanaal, geen vergelijking ──────────────────────
# De maat hierboven vergelijkt twee sensoren en beantwoordt daarmee een andere
# vraag dan hij lijkt te stellen. Synthetisch aangetoond op zes signalen die
# ALLEMAAL op 0,25 Hz ademen: de uitkomst loopt van -0,985 tot +1,000, puur
# naar gelang hun trage amplitudemodulatie. Of beide sensoren dezelfde
# ademhaling zien telt niet mee; een thermische en een drukopnemer moduleren
# hun amplitude nu eenmaal verschillend.
#
# Deze maat kijkt naar één kanaal en stelt de vraag die er werkelijk toe doet:
# draagt DIT signaal ademhaling? Het aandeel van het vermogen dat in de
# ademband valt. Een losgeraakte of dode sensor haalt dat niet, ongeacht wat
# de andere sensor doet.
THERMISTOR_BAND_POWER_MIN = 0.70
"""Afgeleid, niet gekozen. Gemeten op 9 onderscheiden montages (Somnomedics,
`Pressure Flow` + `Flow Th.`), mediaan over tien vensters per nacht:

    0,982  0,981  0,977  0,970   |   0,441  0,396  0,318  0,036  0,000

Een gat van 0,53 met niets erin. 0,70 ligt op het midden daarvan — de waarde
met de grootste marge naar beide klassen — en laat zich lezen als "minstens
70% van het vermogen van dit kanaal is ademhaling".

Ter vergelijking: de neusdruk haalde op dezelfde opnames 0,566 tot 0,914.
Deze drempel geldt dus UITSLUITEND voor de thermistor; op de neusdruk
toegepast zou hij bruikbare kanalen afwijzen.

n = 9. Klein. Hoort met meer opnames opnieuw bepaald te worden, net als
THERMISTOR_AGREEMENT_MIN — maar anders dan die drempel scheidt deze de
gemeten klassen wél."""

THERMISTOR_BAND_POWER_WINDOWS = 10
"""Vensters verspreid over de nacht; de mediaan telt. Eén venster uit het
midden is niet representatief — op echte opnames scheelde dat een factor twee."""

THERMISTOR_BAND_POWER_WIN_S = 600.0

BAND_POWER_TOTAL_HZ = (0.02, 4.0)
"""Noemer: van trager dan ademhaling tot boven de hartslag. Niet tot Nyquist,
anders bepaalt hoogfrequente ruis de breuk in plaats van het signaal."""


def respiratory_band_power(x: np.ndarray, sf: float,
                           n_windows: int = THERMISTOR_BAND_POWER_WINDOWS,
                           win_s: float = THERMISTOR_BAND_POWER_WIN_S) -> dict:
    """Welk deel van het vermogen van dit kanaal valt in de ademband?

    Retourneert ``{"fraction", "peak_hz", "n_windows"}``; ``fraction`` is None
    wanneer er niets te meten valt. Eén kanaal, geen tweede sensor nodig.
    """
    from scipy.signal import welch

    out = {"fraction": None, "peak_hz": None, "n_windows": 0}
    x = np.asarray(x, dtype=float).squeeze()
    if x.ndim != 1 or sf <= 0 or x.size < int(sf * win_s):
        return out

    q = max(1, int(sf // 8))              # ~8 Hz volstaat en is veel sneller
    d, sfd = x[::q], sf / max(1, int(sf // 8))
    n = int(win_s * sfd)
    if d.size < n:
        return out

    starts = np.unique(np.linspace(0, d.size - n, n_windows).astype(int))
    fracs, peaks = [], []
    for s in starts:
        seg = d[s:s + n]
        # Relatieve toets, niet ``== 0``: filtfilt/welch op een constante geeft
        # numerieke ruis van orde 1e-15, en een exacte nulvergelijking laat die
        # door. Zo kreeg een volledig plat kanaal een berekende correlatie over
        # ruis toebedeeld in plaats van te worden afgewezen.
        if np.std(seg) <= 1e-9 * max(1.0, float(np.max(np.abs(seg)))):
            fracs.append(0.0)
            continue
        f, p = welch(seg - seg.mean(), fs=sfd, nperseg=min(2048, n))
        band = (f >= BREATH_FREQ_LOW) & (f <= BREATH_FREQ_HIGH)
        total = (f >= BAND_POWER_TOTAL_HZ[0]) & (f <= BAND_POWER_TOTAL_HZ[1])
        tot = float(p[total].sum())
        if tot <= 0 or not band.any():
            fracs.append(0.0)
            continue
        fracs.append(float(p[band].sum() / tot))
        peaks.append(float(f[band][int(np.argmax(p[band]))]))

    if not fracs:
        return out
    out["fraction"] = round(float(np.median(fracs)), 3)
    out["peak_hz"] = round(float(np.median(peaks)), 3) if peaks else None
    out["n_windows"] = len(fracs)
    return out


def _breath_envelope(x: np.ndarray, sf: float,
                     smooth_s: float = THERMISTOR_ENV_SMOOTH_S) -> np.ndarray:
    """Gladgestreken ademband-amplitude: 'hoeveel wordt er nu geademd'."""
    from scipy.signal import butter, filtfilt
    x = np.asarray(x, dtype=float).squeeze()
    if x.ndim != 1 or x.size < int(sf * 60):
        return np.empty(0)
    x = x - np.median(x)
    nyq = sf / 2.0
    hi = min(BREATH_FREQ_HIGH / nyq, 0.99)
    lo = BREATH_FREQ_LOW / nyq
    if not (0 < lo < hi < 1):
        return np.empty(0)
    b, a = butter(2, [lo, hi], btype="band")
    y = np.abs(filtfilt(b, a, x))
    n = max(1, int(smooth_s * sf))
    return np.convolve(y, np.ones(n) / n, mode="same")


def assess_flow_sensor_agreement(
    pressure: np.ndarray, sf_pressure: float,
    thermistor: np.ndarray, sf_thermistor: float,
    min_agreement: float | None = None,
) -> dict:
    """Is de thermistor bruikbaar als apneu-sensor naast deze neusdruk?

    Retourneert ``{"agreement", "usable", "reason"}``. ``usable=False``
    betekent: val terug op de neusdruk voor apneus, en zeg dat erbij.
    """
    # De drempel wordt HIER opgezocht, niet als default-argument: een default
    # wordt bij functiedefinitie geevalueerd, waardoor de constante achteraf
    # overschrijven geen effect heeft. Dat maakte een meting stil ongeldig —
    # de poort bleef aan terwijl hij uit leek te staan, en beide armen van de
    # vergelijking draaiden op hetzelfde kanaal.
    if min_agreement is None:
        min_agreement = THERMISTOR_AGREEMENT_MIN
    out = {"agreement": None, "usable": False, "reason": ""}
    if pressure is None or thermistor is None:
        out["reason"] = "een van beide kanalen ontbreekt"
        return out
    if abs(sf_pressure - sf_thermistor) > 1e-6:
        out["reason"] = (f"verschillende samplefrequenties "
                         f"({sf_pressure:g} vs {sf_thermistor:g} Hz)")
        return out

    ep = _breath_envelope(pressure, sf_pressure)
    et = _breath_envelope(thermistor, sf_thermistor)
    if ep.size == 0 or et.size == 0:
        out["reason"] = "signaal te kort voor een envelope"
        return out

    n = min(ep.size, et.size)
    step = max(1, int(sf_pressure))          # 1 Hz volstaat en is stabieler
    a, b = ep[:n:step], et[:n:step]
    if a.size < 30 or float(np.std(a)) == 0 or float(np.std(b)) == 0:
        out["reason"] = "envelope zonder variatie"
        return out

    corr = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(corr):
        out["reason"] = "correlatie niet berekenbaar"
        return out

    out["agreement"] = round(corr, 3)
    if corr >= min_agreement:
        out["usable"] = True
        out["reason"] = (f"envelope-overeenstemming {corr:.2f} "
                         f">= {min_agreement:.2f}")
    else:
        out["reason"] = (f"envelope-overeenstemming {corr:.2f} "
                         f"< {min_agreement:.2f}: de thermistor volgt de "
                         f"ademhaling niet zoals de neusdruk")
    return out


def assess_thermistor_band_power(thermistor: np.ndarray, sf_thermistor: float,
                                 min_fraction: float | None = None) -> dict:
    """Draagt DIT kanaal ademhaling? Eén sensor, geen vergelijking.

    Dezelfde vraag als ``assess_flow_sensor_agreement`` — is de thermistor
    bruikbaar als apneu-sensor — maar zonder de neusdruk als maatstaf te
    nemen. Dat verschil is niet cosmetisch. De overeenstemmingsmaat keurde op
    negen montages acht thermistors af, waaronder drie die 98% van hun
    vermogen in de ademband hebben en hun ademfrequentie tot op 0,002 Hz met
    de neusdruk delen. Twee fysisch verschillende opnemers hoeven hun
    amplitude niet gelijk te moduleren; dat ze dat niet doen zegt niets over
    de vraag of de thermistor werkt.

    Retourneert dezelfde sleutels als ``assess_flow_sensor_agreement``, zodat
    het rapport en de meta-blokken niet hoeven te weten welke poort er draaide.
    ``agreement`` blijft None: dit is geen overeenstemmingsmaat, en er een
    getal in schrijven dat op een andere schaal leeft zou het rapport laten
    liegen. ``band_power`` draagt de eigenlijke waarde.
    """
    if min_fraction is None:
        min_fraction = THERMISTOR_BAND_POWER_MIN
    out = {"agreement": None, "band_power": None, "peak_hz": None,
           "usable": False, "reason": "", "gate": "respiratory_band"}
    if thermistor is None:
        out["reason"] = "thermistorkanaal ontbreekt"
        return out
    if not sf_thermistor or sf_thermistor <= 0:
        out["reason"] = "geen bruikbare samplefrequentie"
        return out

    m = respiratory_band_power(thermistor, sf_thermistor)
    if m["fraction"] is None:
        out["reason"] = "signaal te kort voor een spectrum"
        return out

    out["band_power"] = m["fraction"]
    out["peak_hz"] = m["peak_hz"]
    if m["fraction"] >= min_fraction:
        out["usable"] = True
        out["reason"] = (f"ademband-vermogen {m['fraction']:.2f} "
                         f">= {min_fraction:.2f}")
        if m["peak_hz"]:
            out["reason"] += f" (piek {m['peak_hz'] * 60:.0f}/min)"
    else:
        out["reason"] = (f"ademband-vermogen {m['fraction']:.2f} "
                         f"< {min_fraction:.2f}: dit kanaal draagt "
                         f"overwegend geen ademhaling")
    return out
