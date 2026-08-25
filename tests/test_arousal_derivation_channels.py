"""Welke EEG-kanalen de arousalstap als afleidingen wil, uit namen alleen.

WAAROM DIT PUBLIEK MOET
-----------------------
YASAFlaskified bouwt de pneumo-raw uit `detect_channels`, dat ÉÉN kanaal per rol
teruggeeft. Op een klinische opname stonden daar C3 en C4 in -- twee kanalen uit
DEZELFDE regio -- terwijl het EDF ook O1/O2 en F3/F4 bevatte. De arousalstap
kiest zijn afleidingen uit wat er ís, dus die zag nooit een frontale of
occipitale afleiding.

Gemeten op PSG-IPA (n=5, 12 scoorders), arousal-F1 per combinatie van regio's:

    F+C+O  0,514      F  0,442
    F+C    0,501      C  0,439
    F+O    0,485      O  0,316
    C+O    0,460

Van één naar twee regio's is +0,06; de derde doet er nog +0,013 bij. En geen
enkele regio wint overal: op SN4 is occipitaal de beste (0,59) waar hij
gemiddeld de zwakste is. AASM V.A Note 1 schrijft frontaal, centraal EN
occipitaal voor, en dit is waarom.

De kennis hoort HIER, niet in YASAFlaskified: laat de aanroeper raden welke
kanalen de picker straks kiest, en het loopt mis zodra de picker verandert.
Dat is precies wat er met de SpO2-afleiding gebeurde.
"""
import pytest

from psgscoring.pipeline import arousal_derivation_channels as adc


def test_the_three_regions_are_all_requested():
    got = adc(["Snore", "Pres", "SpO2", "C3", "C4", "O1", "O2", "F3", "F4"])
    boven = {g.upper() for g in got}
    assert any("C" in n and n[0] == "C" for n in boven), got
    assert any(n.startswith("O") for n in boven), got
    assert any(n.startswith("F") for n in boven), got


def test_the_clinical_montage_gains_frontal_and_occipital():
    """Precies de EDF die dit blootlegde: de pneumo-raw droeg C3 en C4."""
    edf = ["Snore", "Pressure Flow", "Flow Th.", "RIP Thora", "RIP Abdom",
           "Sum RIP", "SpO2", "PLMl", "PLMr", "EMG1", "Pos.", "Pleth",
           "C4:A1", "Pulse", "ECG II", "C3", "C4", "O1", "O2", "A1", "A2",
           "F3", "F4", "EOG1", "EOG2"]
    got = adc(edf)
    assert len(got) >= 3, got
    boven = " ".join(got).upper()
    assert "O1" in boven or "O2" in boven, f"geen occipitale afleiding: {got}"
    assert "F3" in boven or "F4" in boven, f"geen frontale afleiding: {got}"


def test_the_saturation_curve_is_never_requested():
    got = adc(["C3", "C4", "SpO2", "SaO2", "Pleth"])
    assert not any(g.upper().startswith(("SPO2", "SAO2", "PLETH")) for g in got), got


def test_a_montage_with_one_eeg_asks_for_one():
    assert adc(["Snore", "Pres", "SpO2", "C4"]) == ["C4"]


def test_no_eeg_asks_for_nothing():
    assert adc(["Snore", "Pres", "SpO2", "Pulse"]) == []


def test_a_configured_channel_leads():
    """Element 0 moet de single-channel pick blijven, anders is single-modus
    geen strikte deelverzameling meer van multi."""
    got = adc(["C3", "C4", "O2", "F4"], {"eeg": "C4"})
    assert got[0] == "C4"


def test_nothing_appears_twice():
    got = adc(["C3", "C4", "O1", "O2", "F3", "F4", "Cz"])
    assert len(got) == len(set(got)), got


def test_it_agrees_with_the_picker_that_uses_it():
    """De naamkiezer en _pick_eeg_multi mogen niet uiteenlopen -- dan vraagt
    YASAFlaskified andere kanalen op dan de arousalstap gebruikt."""
    mne = pytest.importorskip("mne")
    import numpy as np

    from psgscoring.pipeline import _pick_eeg_multi
    namen = ["Snore", "SpO2", "C3", "C4", "O2", "F4", "Pulse"]
    info = mne.create_info(namen, 100.0, ch_types="misc", verbose="ERROR")
    rng = np.random.default_rng(2)
    raw = mne.io.RawArray(rng.normal(0, 1e-5, (len(namen), 6000)), info,
                          verbose="ERROR")
    uit_picker = [n for n, _d, _s in _pick_eeg_multi(raw, {})]
    assert adc(namen) == uit_picker, (adc(namen), uit_picker)
