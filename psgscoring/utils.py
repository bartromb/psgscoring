"""
psgscoring.utils
================
Pure-Python helpers used across all other submodules.

Dependencies: numpy, psgscoring.constants
No imports from other psgscoring modules.
"""

from __future__ import annotations
import numpy as np
from .constants import CHANNEL_PATTERNS, EPOCH_LEN_S


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def safe_r(val, dec: int = 1):
    """Round *val* to *dec* decimal places; return None for None/NaN."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), dec)
    except Exception:
        return None


def fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS string."""
    if seconds is None:
        return "--:--:--"
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Hypnogram helpers
# ---------------------------------------------------------------------------

_HYPNO_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


def hypno_to_numeric(hypno: list) -> np.ndarray:
    """Convert string hypnogram ['W','N1',...] to numeric array (W=0..R=4)."""
    return np.array([_HYPNO_MAP.get(s, -1) for s in hypno])


def is_nrem(stage) -> bool:
    """Controleer of een slaapstadium NREM is (N1, N2 of N3)."""
    return stage in (1, 2, 3, "N1", "N2", "N3")


def is_rem(stage) -> bool:
    """Controleer of een slaapstadium REM is."""
    return stage in (4, "R")


def is_sleep(stage) -> bool:
    """Controleer of een slaapstadium slaap is (niet Wake)."""
    return stage not in (0, -1, "W")


# ---------------------------------------------------------------------------
# Sleep / artifact mask
# ---------------------------------------------------------------------------

def build_sleep_mask(
    hypno: list,
    sf: float,
    total_samples: int,
    artifact_epochs: list | None = None,
) -> np.ndarray:
    """
    Build a sample-level boolean mask: True = valid sleep.

    Excludes Wake (stage 0) and artifact epochs supplied by YASA's artifact
    detector.

    Parameters
    ----------
    hypno           : string hypnogram list
    sf              : sample rate of the target signal
    total_samples   : length of the target signal
    artifact_epochs : list of epoch indices containing artefacts
    """
    hypno_num   = hypno_to_numeric(hypno)
    artifact_set = set(artifact_epochs or [])
    spe  = int(sf * EPOCH_LEN_S)
    mask = np.zeros(total_samples, dtype=bool)
    for ep_i, stage in enumerate(hypno_num):
        s = ep_i * spe
        e = min(s + spe, total_samples)
        if stage > 0 and ep_i not in artifact_set:
            mask[s:e] = True
    return mask


# ---------------------------------------------------------------------------
# Channel detection
# ---------------------------------------------------------------------------

# Rol -> rollen waarvan hij het kanaal niet mag overnemen. Zie de toelichting
# in detect_channels(); beide gevallen komen van een patroon van twee tekens.
# Bewust géén algemene regel "een kanaal hoort bij één rol": "flow" en
# "flow_thermistor" wijzen op een montage met één flowkanaal terecht naar
# hetzelfde kanaal.
_ROLE_MAY_NOT_TAKE: dict[str, tuple[str, ...]] = {
    "pulse": ("flow_pressure", "flow_thermistor", "flow"),
    # "emg" draagt een kaal "emg", substring van "EMG Tib L"/"EMG La": op een
    # montage zonder kin-EMG kreeg de rol anders het beenkanaal, en
    # emg_var_ratio zou dan op beenbewegingen draaien in plaats van op de
    # kin. Dat is geen ontbrekend EMG maar een fout EMG -- niet herkenbaar
    # als fout in de output.
    "emg":   ("leg_l", "leg_r"),
    "eeg":   ("flow_pressure", "flow_thermistor", "flow", "thorax", "abdomen",
              "spo2", "pulse", "ecg", "position", "snore", "leg_l", "leg_r",
              "emg"),
}


def _channel_patterns() -> dict[str, list[str]]:
    """CHANNEL_PATTERNS, optioneel uitgebreid met dataset-specifieke namen.

    Aanleiding: de MESA/NSRR-documentatie vermeldt dat het Sleep Reading
    Center in de eerste maanden vaststelde dat de thermistor onbetrouwbaar
    was en overstapte op een ThermiSense-unit. De oude configuratie draagt
    een kanaal ``Therm``, de nieuwe een kanaal ``Aux_AC``. Geen enkel
    patroon in ``flow_thermistor`` matcht ``Aux_AC``, dus op die opnames
    blijft de rol leeg en reduceert elk duaal profiel stilzwijgend tot zijn
    één-sensor-ouder. Een gemeten thermistor-passage van 45% op MESA kan
    daardoor evengoed een kanaalnaam als een sensoreigenschap zijn.

    ``Aux_AC`` is echter een generieke hulpkanaalnaam: bij een andere
    fabrikant kan er van alles op staan. De uitbreiding staat daarom UIT
    tenzij expliciet gevraagd, volgens dezelfde conventie als
    ``PSGSCORING_BASELINE_MODE``::

        PSGSCORING_NSRR_AUX_AC=1

    Ook aan blijft de bestaande thermistorpoort beslissen of het kanaal als
    apneu-sensor toelaatbaar is; deze vlag maakt het kanaal alleen
    zichtbaar. Default uit betekent: elk bestaand profiel is byte-identiek.
    """
    import os
    if os.environ.get("PSGSCORING_NSRR_AUX_AC", "").strip().lower() not in (
            "1", "true", "yes", "on"):
        return CHANNEL_PATTERNS
    patterns = {k: list(v) for k, v in CHANNEL_PATTERNS.items()}
    # Achteraan: elke specifieke thermistornaam wint hiervan.
    patterns["flow_thermistor"].append("aux_ac")
    return patterns


def detect_channels(ch_names: list[str]) -> dict[str, str]:
    """
    Pattern-match EDF channel names to functional roles.

    Returns a dict mapping role -> original channel name for the first match
    found per role.  Matching is case-insensitive substring search.
    """
    ch_lower = {ch.lower(): ch for ch in ch_names}
    found: dict[str, str] = {}
    patterns_by_role = _channel_patterns()
    for ch_type, patterns in patterns_by_role.items():
        # Twee rollen dragen een patroon dat te kort is om alleen te staan, en
        # allebei zijn ze een kanaal gaan opeisen dat al een andere betekenis
        # had. De rollen die ze mogen inpikken staan vóór hen in
        # CHANNEL_PATTERNS en zijn hier dus al ingevuld — die volgorde is
        # semantiek, geen cosmetica.
        #
        # "pulse" draagt "pr", substring van "Pres" en "Pressure Flow": op een
        # montage zonder eigen hartslagkanaal kreeg de rol de neusdruk, waarna
        # analyze_heart_rate een flowsignaal als bpm behandelde en er een
        # "minimale hartfrequentie" uitrolde die op de filterondergrens lag.
        #
        # "eeg" draagt "o2", substring van "SpO2" en "SaO2": op een montage
        # zonder EEG-kanaal kreeg de rol de saturatie. _pick_eeg() leest die rol
        # rechtstreeks, dus de arousal-detectie zou op de SpO2-curve draaien.
        # Geblokkeerd valt _pick_eeg terug op zijn eigen, striktere lijst
        # (EEG/C3/C4/F3/F4/CZ, zonder O1/O2) of op niets — allebei beter dan
        # een saturatiecurve die als EEG wordt gelezen.
        blocked = {
            found[r] for r in _ROLE_MAY_NOT_TAKE.get(ch_type, ()) if found.get(r)
        }
        for pat in patterns:
            match = next(
                (orig for lc, orig in ch_lower.items()
                 if pat in lc and orig not in blocked),
                None,
            )
            if match:
                found[ch_type] = match
                break

    # Eén kanaal kan niet allebei de benen zijn. Matching is substring-based en
    # per rol first-match-wins, dus een patroon dat op beide rollen past (of een
    # kanaalnaam die beide patronen bevat) wijst hetzelfde kanaal twee keer toe
    # en verdubbelt de PLM-telling. Liever één been dan twee keer hetzelfde.
    if found.get("leg_l") and found.get("leg_l") == found.get("leg_r"):
        taken = found["leg_l"]
        alt = next(
            (orig for lc, orig in ch_lower.items()
             if orig != taken and any(p in lc for p in CHANNEL_PATTERNS["leg_r"])),
            None,
        )
        if alt:
            found["leg_r"] = alt
        else:
            found.pop("leg_r")

    return found


def channel_map_from_user(
    user_map: dict | None,
    ch_names: list[str],
) -> dict[str, str]:
    """
    Merge auto-detected channel map with optional manual overrides.

    Manual overrides take precedence; invalid channel names are ignored.
    """
    auto   = detect_channels(ch_names)
    merged = {**auto}
    for k, v in (user_map or {}).items():
        if v and v in ch_names:
            merged[k] = v
    return merged
