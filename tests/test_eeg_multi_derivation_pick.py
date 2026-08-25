"""De afleidingsset voor multi-arousal mag geen saturatiecurve bevatten.

Op een klinische opname (AHI 62, psgscoring 0.27.3) leverde de provenance:

    n_derivations: 2
    derivations:   ['C3', 'SpO2']
    n_per_derivation: {'C3': 142, 'SpO2': 0}

`_pick_eeg_multi` zoekt occipitaal met onder meer een KAAL "O2", en "SPO2"
bevat "O2". Dat is dezelfde val die `_ROLE_MAY_NOT_TAKE["eeg"]` in
`detect_channels` al afvangt -- maar deze functie heeft haar eigen zoektocht
en die guard ontbrak.

Twee gevolgen, en het tweede is het ergste:

1. De arousaldetectie draait een volledige analyse op een saturatiecurve. Hier
   gaf dat nul events; dat is geluk, geen ontwerp. Een SpO2-curve met dalingen
   kan "arousals" opleveren die niemand kan herkennen als onzin.
2. `C4` stond gewoon in dezelfde raw en werd NOOIT overwogen: de picker zoekt
   alleen occipitaal en frontaal, nooit een tweede CENTRALE afleiding. De
   meest voorkomende klinische montage (C3 + C4) krijgt dus geen union, en de
   provenance meldde `n_derivations: 2` alsof multi gewerkt had.
"""
import numpy as np
import pytest

import mne

from psgscoring.pipeline import _pick_eeg_multi

SF = 100.0


def _raw(namen):
    rng = np.random.default_rng(3)
    n = int(SF * 120)
    data = []
    for naam in namen:
        if naam.upper().startswith(("SPO2", "SAO2")):
            data.append(np.full(n, 96.0))          # saturatie: geen EEG
        else:
            data.append(rng.normal(0, 20e-6, n))
    info = mne.create_info(list(namen), SF, ch_types="misc", verbose="ERROR")
    return mne.io.RawArray(np.vstack(data), info, verbose="ERROR")


def _namen(raw, ch=None):
    return [n for n, _d, _s in _pick_eeg_multi(raw, ch or {})]


# ══════════════════════════════════════════════════════════════
# De saturatiecurve
# ══════════════════════════════════════════════════════════════

def test_the_saturation_curve_is_never_an_eeg_derivation():
    raw = _raw(["Snore", "Pressure Flow", "SpO2", "Pulse", "C3", "C4"])
    got = _namen(raw)
    assert "SpO2" not in got, (
        f"de arousaldetectie draait op een saturatiecurve: {got}")


@pytest.mark.parametrize("sat", ["SpO2", "SaO2", "SPO2", "Sat O2"])
def test_every_spelling_of_the_saturation_is_excluded(sat):
    raw = _raw(["C3", sat, "Pulse"])
    assert sat not in _namen(raw)


def test_a_real_occipital_channel_is_still_used():
    raw = _raw(["C3", "O2-M1", "SpO2"])
    got = _namen(raw)
    assert "O2-M1" in got, got
    assert "SpO2" not in got


# ══════════════════════════════════════════════════════════════
# De tweede centrale afleiding
# ══════════════════════════════════════════════════════════════

def test_the_second_central_derivation_is_used():
    """C3 + C4 is de gangbaarste klinische montage. Zonder C4 in de zoektocht
    krijgt die nacht geen union en is multi-modus een lege huls."""
    raw = _raw(["Snore", "SpO2", "C3", "C4", "Pulse"])
    got = _namen(raw)
    assert got[0] == "C3"
    assert "C4" in got, f"tweede centrale afleiding ontbreekt: {got}"


def test_the_central_pick_stays_first():
    """Element 0 moet de single-channel pick blijven, anders is single-modus
    geen strikte deelverzameling meer van multi."""
    raw = _raw(["C4", "O2-M1", "F4-M1", "C3"])
    assert _namen(raw)[0] == "C4"


def test_a_configured_eeg_channel_still_wins():
    raw = _raw(["C3", "C4", "O2-M1"])
    assert _namen(raw, {"eeg": "C4"})[0] == "C4"


def test_no_channel_appears_twice():
    raw = _raw(["C3", "C4", "O2-M1", "F4-M1"])
    got = _namen(raw)
    assert len(got) == len(set(got)), got


def test_a_single_eeg_montage_yields_one_derivation():
    """Met één EEG hoort de lijst lengte 1 te hebben -- de pijplijn valt dan
    terug op single, en dat is de gedocumenteerde invariant."""
    raw = _raw(["Snore", "SpO2", "C3", "Pulse"])
    assert _namen(raw) == ["C3"]


def test_the_clinical_montage_that_exposed_this():
    """Precies de pneumo-raw van de opname die de bug blootlegde."""
    raw = _raw(["Snore", "Pressure Flow", "Flow Th.", "RIP Thora", "RIP Abdom",
                "SpO2", "PLMl", "PLMr", "EMG1", "Pos.", "Pulse", "ECG II",
                "C3", "C4"])
    got = _namen(raw)
    assert got == ["C3", "C4"], got
