"""``_pick_emg`` / ``_pick_eog``: een afwezige geconfigureerde naam blokkeerde
de patroonzoektocht.

De code las eerst ``ch.get("emg")``. Stond daar een naam die niet in het EDF
zit — bijvoorbeeld een default "EMG1" uit een oudere jobconfig, of een naam uit
een montage die de recorder intussen anders exporteert — dan werd de
substringfallback OVERGESLAGEN en gaf de functie ``None`` terug, terwijl er
verderop in dezelfde ``raw`` een kanaal "Chin1-Chin2" stond.

Dat is niet cosmetisch: zonder EMG degenereert de LGBM-arousalclassifier (zie
tests/test_arousal_no_emg_guard.py) en zonder EMG is een REM-arousal per AASM
niet conform scoorbaar.

Tegelijk mag de fallback NOOIT een been-EMG als kin-EMG aanwijzen: "EMG Tib L"
bevat "EMG" en werd door de oude patronen zonder meer geaccepteerd. Een
tibialis-signaal als kin-EMG lezen zet ``emg_var_ratio`` op beenbewegingen —
erger dan geen EMG, want het is niet als ontbrekend herkenbaar.
"""
import mne
import numpy as np
import pytest

from psgscoring.pipeline import _pick_emg, _pick_eog

SF = 100.0


def _raw(names):
    info = mne.create_info(list(names), SF, ch_types="misc", verbose="ERROR")
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 1e-5, (len(names), int(SF * 60)))
    return mne.io.RawArray(data, info, verbose="ERROR")


# ══════════════════════════════════════════════════════════════
# fall-through
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("chin", ["Chin1-Chin2", "Kin EMG", "EMG Chin",
                                  "Menton"])
def test_an_absent_configured_name_falls_through_to_the_patterns(chin, caplog):
    raw = _raw(["C3:A2", chin, "SpO2"])
    with caplog.at_level("WARNING", logger="psgscoring.pipeline"):
        data = _pick_emg(raw, {"emg": "EMG1"})
    assert data is not None, (
        f"{chin!r} staat in de raw, maar de geconfigureerde 'EMG1' blokkeerde "
        f"de patroonzoektocht"
    )
    np.testing.assert_allclose(data, raw.get_data(picks=[chin])[0])
    assert any("EMG1" in r.getMessage() for r in caplog.records), (
        "stil terugvallen verbergt dat de jobconfig een kanaal noemt dat niet "
        "bestaat"
    )


def test_a_present_configured_name_still_wins():
    raw = _raw(["C3:A2", "EMG1", "Chin1-Chin2"])
    data = _pick_emg(raw, {"emg": "EMG1"})
    np.testing.assert_allclose(data, raw.get_data(picks=["EMG1"])[0])


def test_the_eog_picker_falls_through_too():
    raw = _raw(["C3:A2", "E1-M2", "SpO2"])
    data = _pick_eog(raw, {"eog": "EOG-L"})
    assert data is not None
    np.testing.assert_allclose(data, raw.get_data(picks=["E1-M2"])[0])


# ══════════════════════════════════════════════════════════════
# been-EMG is geen kin-EMG
# ══════════════════════════════════════════════════════════════

def test_a_leg_emg_is_never_picked_as_chin_emg():
    raw = _raw(["C3:A2", "EMG Tib L", "EMG Tib R", "SpO2"])
    assert _pick_emg(raw, {"emg": "EMG1"}) is None, (
        "een tibialiskanaal als kin-EMG zet emg_var_ratio op beenbewegingen"
    )
    assert _pick_emg(raw, {}) is None


@pytest.mark.parametrize("leg", ["PLMl", "Leg L", "Tibialis Ant L",
                                 "EMG LAT", "Bein li"])
def test_the_leg_labels_are_all_excluded(leg):
    assert _pick_emg(_raw(["C3:A2", leg]), {}) is None, leg


def test_the_chin_wins_when_both_are_present():
    raw = _raw(["C3:A2", "EMG Tib L", "Chin EMG", "EMG Tib R"])
    data = _pick_emg(raw, {})
    np.testing.assert_allclose(data, raw.get_data(picks=["Chin EMG"])[0])
