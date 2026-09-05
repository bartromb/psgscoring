"""Gegradeerde Rule 1B: een arousalKANS als bevestiging, nooit als telling.

DE AANLEIDING (verliesrekening 2026-09-04)
==========================================
Op de 15 zwaarst onderdetecterende hoge-AHI-nachten was 18,7 % van de
terughaalbare verliezen een `no_desaturation`-afwijzing: Regel 1A eist desat
óf arousal, en de arousal die er fysiologisch wel was haalde ons werkpunt
(0,70/0,80) niet. ABED (Nat Commun 2026) voedt arousalkansen als invoer; dit
is onze regelgebaseerde vorm van hetzelfde idee.

DE SCHEIDING DIE HEILIG IS
==========================
De kandidaatkans mag een hypopneu BEVESTIGEN (reinstatement), maar komt NOOIT
in de arousal-index of de eventlijst: de index houdt het geijkte werkpunt.
Een bevestiging vraagt minder zekerheid dan een telling — dat is precies wat
"gegradeerd" hier betekent, en de provenance maakt elk gegradeerd gekoppeld
event herkenbaar (`coupled_arousal_proba`).
"""
import numpy as np

from psgscoring.constants import _profile_to_legacy_dict as _L
from psgscoring.profiles import PROFILES
from psgscoring.respiratory import reinstate_rule1a_arousal_hypopneas


def _afgewezen(onset):
    return {"type": "hypopnea", "onset_s": onset, "duration_s": 15.0,
            "stage": "N2", "epoch": int(onset // 30),
            "reject_reason": "no_desaturation"}


def test_profielvelden_bestaan_en_default_is_uit():
    d = _L(PROFILES["aasm_v3_rec"])
    assert d["RULE1B_GRADED"] is False, "default uit tot de meting er ligt"
    assert d["RULE1B_MIN_PROBA"] == 0.50


def test_kandidaat_boven_de_koppelkans_bevestigt():
    """Geen enkel vol arousal-event, wel een kandidaat p=0,62 vlak na het
    event-einde: met graded aan wordt de hypopneu hersteld."""
    rein, alle = reinstate_rule1a_arousal_hypopneas(
        rejected=[_afgewezen(100.0)], arousal_events=[],
        resp_events=[], hypno=["N2"] * 20,
        graded_candidates=[{"onset_s": 116.0, "duration_s": 4.0,
                            "proba": 0.62}],
        graded_min_proba=0.50)
    assert len(rein) == 1
    assert rein[0]["coupled_arousal_proba"] == 0.62, (
        "de provenance moet het gegradeerde koppelen herkenbaar maken")


def test_kandidaat_onder_de_koppelkans_bevestigt_NIET():
    rein, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=[_afgewezen(100.0)], arousal_events=[],
        resp_events=[], hypno=["N2"] * 20,
        graded_candidates=[{"onset_s": 116.0, "duration_s": 4.0,
                            "proba": 0.31}],
        graded_min_proba=0.50)
    assert rein == []


def test_zonder_graded_verandert_er_NIETS():
    """Bestaand gedrag: geen kandidatenlijst -> geen herstel zonder vol event."""
    rein, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=[_afgewezen(100.0)], arousal_events=[],
        resp_events=[], hypno=["N2"] * 20)
    assert rein == []


def test_een_vol_arousal_event_wint_van_de_kandidaat():
    """Als een echt event koppelt, hoort de provenance GEEN kandidaatkans te
    dragen — anders leest een vol gekoppeld event als een zwak bevestigd."""
    rein, _ = reinstate_rule1a_arousal_hypopneas(
        rejected=[_afgewezen(100.0)],
        arousal_events=[{"onset_s": 116.0, "duration_s": 4.0}],
        resp_events=[], hypno=["N2"] * 20,
        graded_candidates=[{"onset_s": 116.0, "duration_s": 4.0,
                            "proba": 0.55}],
        graded_min_proba=0.50)
    assert len(rein) == 1
    assert "coupled_arousal_proba" not in rein[0]


def test_kandidaten_dragen_hun_kans_naar_buiten():
    """Leveringsoppervlak in de arousalstap: de kandidatenlijst met kansen
    moet bestaan zodra de classifier draait, en de gehouden events zijn er
    een deelverzameling van (op onset)."""
    import mne

    import psgscoring

    sf, n_s = 100.0, 600.0
    t = np.arange(int(sf * n_s)) / sf
    rng = np.random.default_rng(5)
    eeg = rng.normal(0, 2e-5, len(t))
    for s0 in (90.0, 180.0, 270.0, 360.0, 450.0, 540.0):
        m = (t >= s0) & (t < s0 + 7)
        eeg[m] += 8e-5 * np.sin(2 * np.pi * 11 * t[m])
    info = mne.create_info(["EEG1", "EEG2", "EEG3", "EMG", "Pres"], sf,
                           ch_types="misc", verbose=False)
    raw = mne.io.RawArray(np.vstack([
        eeg, rng.normal(0, 2e-5, len(t)), rng.normal(0, 2e-5, len(t)),
        rng.normal(0, 1e-5, len(t)),
        np.sin(2 * np.pi * 0.25 * t)]), info, verbose=False)
    o = psgscoring.run_pneumo_analysis(raw, hypno=["N2"] * int(n_s // 30),
                                       scoring_profile="aasm_v3_rec")
    ar = o.get("arousal") or {}
    # kandidaten nesten onder ["arousals"] (wrapperstructuur)
    kand = (ar.get("arousals") or {}).get("lgbm_candidates")
    if kand is None:
        import pytest
        pytest.skip("LGBM niet gedraaid in deze omgeving")
    assert len(kand) > 0, "het fixture hoort kandidaten te leveren"
    assert all("proba" in c for c in kand)
    onsets = {round(c["onset_s"], 1) for c in kand}
    for e in ar.get("events") or []:
        assert round(e["onset_s"], 1) in onsets, (
            "gehouden events horen een deelverzameling van de kandidaten te zijn")
