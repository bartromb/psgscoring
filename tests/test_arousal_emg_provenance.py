"""End-to-end: welk kin-EMG heeft de arousalstap gebruikt, en zegt de output het?

De regressie die dit afdwingt was maandenlang onzichtbaar omdat NIETS in de
output vertelde dat de LGBM-arousalclassifier zonder EMG draaide. Het rapport
toonde een arousal-index, de logs een waarschuwing die niemand las, en de twee
gevallen die het aan het licht brachten waren gewoon een getal dat klinisch
niet kon: AI 3,5/u bij AHI 42.

Twee velden dekken dat gat af:
  meta["arousal_emg_channel"]            -- welk kanaal, of None
  arousal.summary["lgbm_skipped_reason"] -- waarom de classifier niet draaide
"""
import numpy as np
import pytest


def _raw(kanalen: list[str]):
    mne = pytest.importorskip("mne")
    sf, minutes = 64.0, 30
    n = int(sf * 60 * minutes)
    t = np.arange(n) / sf
    rng = np.random.default_rng(4)
    eeg = rng.normal(0, 20e-6, n)
    for start in range(60, 60 * minutes // 2, 90):
        a, b = int(start * sf), int((start + 5) * sf)
        eeg[a:b] += 70e-6 * np.sin(2 * np.pi * 10.0 * t[a:b])

    data, namen, typen = [], [], []
    for naam in kanalen:
        namen.append(naam)
        if naam.startswith("EEG"):
            data.append(eeg); typen.append("eeg")
        elif naam in ("SaO2",):
            data.append(np.full(n, 97.0)); typen.append("misc")
        elif naam.startswith("Resp"):
            data.append(np.sin(2 * np.pi * 0.25 * t)); typen.append("misc")
        else:
            data.append(rng.normal(0, 5e-6, n)); typen.append("misc")
    info = mne.create_info(namen, sf, typen)
    raw = mne.io.RawArray(np.vstack(data), info, verbose=False)
    return raw, ["N2"] * int(np.ceil(raw.times[-1] / 30.0))


def _run(kanalen, **kw):
    import psgscoring
    raw, hypno = _raw(kanalen)
    return psgscoring.run_pneumo_analysis(
        raw, hypno=hypno, scoring_profile="aasm_v3_breath", **kw)


BASIS = ["Resp nasal", "SaO2", "EEG C4-M1"]


def test_the_output_names_the_chin_emg_it_used():
    out = _run(BASIS + ["Chin1-Chin2"])
    assert out["meta"]["arousal_emg_channel"] == "Chin1-Chin2"


def test_a_montage_without_chin_emg_says_so():
    out = _run(BASIS)
    assert out["meta"]["arousal_emg_channel"] is None


def test_a_leg_channel_is_not_reported_as_the_chin_emg():
    """Precies de AZORG-montage die de regressie zichtbaar maakte: been-EMG
    aanwezig, kin-EMG niet."""
    out = _run(BASIS + ["PLMl", "PLMr"])
    assert out["meta"]["arousal_emg_channel"] is None


def test_without_chin_emg_the_summary_says_why_the_classifier_did_not_run():
    pytest.importorskip("lightgbm")
    out = _run(BASIS)
    s = out["arousal"]["summary"]
    assert s.get("lgbm_available") is False
    assert str(s.get("lgbm_skipped_reason", "")).startswith("no_emg_channel"), (
        f"{s.get('lgbm_skipped_reason')!r}")


def test_a_configured_but_absent_emg_name_does_not_lose_the_channel():
    """Een jobconfig met een default "EMG1" uit een andere montage mag geen
    EMG-loze run opleveren als er een kin-EMG in het bestand zit."""
    out = _run(BASIS + ["Chin1-Chin2"],
               channel_map={"eeg": "EEG C4-M1", "emg": "EMG1"})
    assert out["meta"]["arousal_emg_channel"] == "Chin1-Chin2"
