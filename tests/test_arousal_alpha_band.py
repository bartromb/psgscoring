"""De alfaband is versmald tot 8-11 Hz; de manual versmalt hem niet.

DE REGEL
--------
AASM v3, V.A.1: *"an abrupt shift of EEG frequency including **alpha**, theta
and/or frequencies greater than 16 Hz (but not spindles)"*.

De manual noemt alfa zonder ondergrens of bovengrens; conventioneel is dat
8-13 Hz. Wij gebruiken 8-11.

WAAROM DIE VERSMALLING ER KWAM
------------------------------
Om spindeloverlap te vermijden (`ALPHA_NARROW_BAND = (8, 11)  # Alpha ZONDER
spindle-overlap`). Maar er staat AL een spindelbeveiliging in de keten, en die
werkt per event: Check C verwerpt een kandidaat wanneer de sigmaband verhoogd
is EN alfa+beta onder 50 % daarvan blijft.

De versmalling is dus een tweede, grovere beveiliging bovenop een fijnere. Ze
kost gevoeligheid in 11-13 Hz -- precies het gebied waar alfa en spindels
elkaar raken, en waar een arousal met gemengde inhoud valt.

WAAROM DIT ERTOE DOET
---------------------
Van 2760 menselijke arousals stellen wij er 62,7 % nooit als kandidaat voor.
Dat is een recall-plafond in de bandvermogenstap, en de bandgrenzen zijn daar
de meest directe knop.
"""
import numpy as np
import pytest

from psgscoring.arousal import ALPHA_NARROW_BAND, detect_arousals

SF = 64.0


def _eeg(freq, minuten=20, seed=3, amp=70e-6):
    n = int(SF * 60 * minuten)
    t = np.arange(n) / SF
    rng = np.random.default_rng(seed)
    eeg = rng.normal(0, 20e-6, n)
    onsets = list(range(60, 60 * minuten - 60, 120))
    for s0 in onsets:
        a, b = int(s0 * SF), int((s0 + 5) * SF)
        eeg[a:b] += amp * np.sin(2 * np.pi * freq * t[a:b])
    return eeg, onsets


def test_de_huidige_band_stopt_bij_elf():
    assert ALPHA_NARROW_BAND == (8, 11)


def test_een_arousal_op_twaalf_hz_wordt_nu_gemist():
    """12 Hz is alfa volgens de manual en sigma volgens onze band. Zonder de
    bredere band valt hij tussen wal en schip."""
    eeg, _o = _eeg(12.0)
    smal = detect_arousals(eeg, SF, ["N2"] * 40)
    breed = detect_arousals(eeg, SF, ["N2"] * 40, alpha_band_wide=True)
    n_smal = len(smal.get("events") or [])
    n_breed = len(breed.get("events") or [])
    assert n_breed > n_smal, (
        f"de bredere band vindt niet meer op 12 Hz: {n_smal} tegen {n_breed}")


def test_alfa_op_tien_hz_verandert_niet():
    """Binnen de oude band mag er niets veranderen."""
    eeg, _o = _eeg(10.0)
    n_smal = len(detect_arousals(eeg, SF, ["N2"] * 40).get("events") or [])
    n_breed = len(detect_arousals(eeg, SF, ["N2"] * 40,
                                  alpha_band_wide=True).get("events") or [])
    assert n_breed >= n_smal


def test_een_echte_spindel_wordt_nog_steeds_geweerd():
    """De reden dat de band versmald werd, mag niet terugkomen.

    Een spindel is 11-16 Hz ZONDER alfa- of beta-inhoud eromheen. Check C
    verwerpt die per event; als dat werkt, is de versmalling overbodig.
    """
    n = int(SF * 60 * 20)
    t = np.arange(n) / SF
    rng = np.random.default_rng(7)
    eeg = rng.normal(0, 20e-6, n)
    # zuivere 13,5 Hz-spindels van 1 s -- geen alfa, geen beta
    for s0 in range(60, 60 * 20 - 60, 60):
        a, b = int(s0 * SF), int((s0 + 1.0) * SF)
        eeg[a:b] += 80e-6 * np.sin(2 * np.pi * 13.5 * t[a:b])
    r = detect_arousals(eeg, SF, ["N2"] * 40, alpha_band_wide=True)
    n_ev = len(r.get("events") or [])
    assert n_ev <= 3, (
        f"{n_ev} arousals op zuivere spindels; de per-event sigmatoets houdt "
        f"ze niet tegen en de bredere band is dan niet veilig")


def test_de_default_verandert_niets():
    eeg, _o = _eeg(12.0)
    a = len(detect_arousals(eeg, SF, ["N2"] * 40).get("events") or [])
    b = len(detect_arousals(eeg, SF, ["N2"] * 40,
                            alpha_band_wide=False).get("events") or [])
    assert a == b


def test_de_vlag_overleeft_de_hele_keten():
    """Dezelfde drie lagen als bij `score_wake_arousals`, waar alleen de
    onderste bedraden nul arousals opleverde in acht tests."""
    import inspect

    from psgscoring.arousal import (
        detect_arousals,
        detect_arousals_multi,
        run_arousal_respiratory_analysis,
    )
    for fn in (detect_arousals, detect_arousals_multi,
               run_arousal_respiratory_analysis):
        assert "alpha_band_wide" in inspect.signature(fn).parameters, fn.__name__
    src = inspect.getsource(run_arousal_respiratory_analysis)
    assert src.count("alpha_band_wide=alpha_band_wide") >= 2
    assert "alpha_band_wide=alpha_band_wide" in inspect.getsource(
        detect_arousals_multi)
