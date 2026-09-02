"""Arousals in WAKE-epochs tellen mee, zegt de manual.

DE REGEL
--------
AASM v3, V.A Note 3 (RECOMMENDED):

    "Arousals meeting all scoring criteria but occurring during an AWAKE epoch
     in the recorded time between 'lights out' and 'lights on' should be scored
     and used for computation of the arousal index."

WAT ER STOND
------------
`detect_arousals` bouwt een slaapmasker per sample en zoekt alleen binnen
slaap-epochs naar kandidaten. Een arousal die volledig in een wake-epoch valt,
bestaat voor ons niet.

WAAROM DIT ERTOE DOET
---------------------
Gemeten op 45 MESA-opnames, gelijk verdeeld over drie scoorders: wij zitten
~6/u ONDER de menselijke arousalindex (bias -7,1 / -6,1 / +0,2). En van 2760
menselijke arousals wordt 62,7 % nooit als kandidaat voorgesteld -- niet door
de classifier verworpen, maar nooit voorgedragen.

Wake-epochs zijn een voor de hand liggende bron: bij een gefragmenteerde nacht
valt een arousal vaak in een epoch die als W gescoord wordt, juist omdat die
arousal er is. De AASM-regel bestaat precies daarvoor.

DE NOEMER VERANDERT NIET
------------------------
De arousalindex deelt door TOTALE SLAAPTIJD. Extra arousals in wake-epochs
verhogen de teller, niet de noemer -- dat is wat de manual voorschrijft, en het
is ook de enige lezing die niet circulair is.
"""
import numpy as np
import pytest

from psgscoring.arousal import detect_arousals

SF = 64.0


def _eeg_met_arousals(minuten=20, sf=SF, seed=3):
    """Rustige achtergrond met alfa/beta-bursts op vaste tijden."""
    n = int(sf * 60 * minuten)
    t = np.arange(n) / sf
    rng = np.random.default_rng(seed)
    eeg = rng.normal(0, 20e-6, n)
    onsets = list(range(60, 60 * minuten - 60, 120))
    for start in onsets:
        a, b = int(start * sf), int((start + 5) * sf)
        # 10 Hz = alfa. NIET 12 Hz: dat valt in de sigmaband die als spindel
        # wordt UITGESLOTEN, en de eerste versie van deze fixture vond daardoor
        # nul arousals in elke arm -- een test die niets mat.
        eeg[a:b] += 70e-6 * np.sin(2 * np.pi * 10.0 * t[a:b])
    return eeg, onsets


def _hypno(minuten, wake_epochs=()):
    h = ["N2"] * int(minuten * 2)
    for i in wake_epochs:
        if 0 <= i < len(h):
            h[i] = "W"
    return h


def test_een_arousal_in_een_wake_epoch_wordt_nu_gevonden():
    """De kern: dezelfde burst, één keer in N2 en één keer in W."""
    eeg, onsets = _eeg_met_arousals()
    # de epoch waarin de derde burst valt
    doel = int(onsets[2] // 30)
    h_slaap = _hypno(20)
    h_wake = _hypno(20, wake_epochs=(doel,))

    r_slaap = detect_arousals(eeg, SF, h_slaap, score_wake_arousals=True)
    r_wake = detect_arousals(eeg, SF, h_wake, score_wake_arousals=True)
    assert r_slaap.get("success") and r_wake.get("success")

    def _rond(res, t0):
        return [e for e in (res.get("events") or [])
                if abs(float(e["onset_s"]) - t0) < 20.0]

    assert _rond(r_slaap, onsets[2]), "de fixture levert geen arousal in slaap"
    assert _rond(r_wake, onsets[2]), (
        "dezelfde burst in een wake-epoch wordt niet gescoord, terwijl "
        "V.A Note 3 dat voorschrijft")


def test_de_default_laat_wake_epochs_nog_weg():
    """Werkregel 1: meten gaat vóór aanzetten."""
    eeg, onsets = _eeg_met_arousals()
    doel = int(onsets[2] // 30)
    r = detect_arousals(eeg, SF, _hypno(20, wake_epochs=(doel,)))
    rond = [e for e in (r.get("events") or [])
            if abs(float(e["onset_s"]) - onsets[2]) < 20.0]
    assert not rond, "de default gedraagt zich al als de manual"


def test_de_noemer_blijft_de_slaaptijd():
    """Extra arousals in wake verhogen de TELLER, niet de noemer. Anders zou
    een gefragmenteerde nacht zichzelf wegdelen."""
    eeg, _o = _eeg_met_arousals()
    wake = tuple(range(10, 20))          # vijf minuten wake middenin
    r_uit = detect_arousals(eeg, SF, _hypno(20, wake_epochs=wake))
    r_aan = detect_arousals(eeg, SF, _hypno(20, wake_epochs=wake),
                            score_wake_arousals=True)
    for r in (r_uit, r_aan):
        assert r.get("success")
    s_uit = (r_uit.get("summary") or {})
    s_aan = (r_aan.get("summary") or {})
    # dezelfde slaaptijd in beide armen
    if s_uit.get("total_sleep_h") and s_aan.get("total_sleep_h"):
        assert s_uit["total_sleep_h"] == pytest.approx(s_aan["total_sleep_h"])


def test_meer_arousals_of_evenveel_maar_nooit_minder():
    """De regel voegt toe; hij mag niets wegnemen."""
    eeg, _o = _eeg_met_arousals()
    wake = tuple(range(12, 22))
    n_uit = len(detect_arousals(eeg, SF, _hypno(20, wake_epochs=wake)
                                ).get("events") or [])
    n_aan = len(detect_arousals(eeg, SF, _hypno(20, wake_epochs=wake),
                                score_wake_arousals=True).get("events") or [])
    assert n_aan >= n_uit, (n_uit, n_aan)


def test_volledig_wakkere_opname_levert_geen_index_maar_wel_events():
    """Zonder slaap is er geen noemer. De events bestaan wel, de index niet --
    dezelfde regel als bij de AHI (geen ondergrens op de noemer)."""
    eeg, _o = _eeg_met_arousals()
    r = detect_arousals(eeg, SF, ["W"] * 40, score_wake_arousals=True)
    if not r.get("success"):
        pytest.skip("detector weigert een volledig wakkere opname")
    idx = (r.get("summary") or {}).get("arousal_index")
    assert idx is None or idx == 0


def test_de_vlag_overleeft_de_hele_keten():
    """De vlag moet door DRIE lagen: de pijplijn roept
    `run_arousal_respiratory_analysis` aan, die roept `detect_arousals` of
    `detect_arousals_multi` aan, en die laatste roept `detect_arousals` per
    afleiding aan.

    De eerste versie gaf de vlag alleen aan `detect_arousals` mee. De pijplijn
    riep de wrapper aan, die het onbekende argument als TypeError opving in een
    stille `except` -- gevolg: nul arousals in acht tests, en een golden die op
    twee gevallen omviel. De fout zag eruit als een detectieprobleem.
    """
    import inspect

    from psgscoring.arousal import (
        detect_arousals,
        detect_arousals_multi,
        run_arousal_respiratory_analysis,
    )

    for fn in (detect_arousals, detect_arousals_multi,
               run_arousal_respiratory_analysis):
        assert "score_wake_arousals" in inspect.signature(fn).parameters, (
            f"{fn.__name__} kent de vlag niet; de keten breekt daar stil af")

    # en de wrapper moet hem ook DOORGEVEN, niet alleen accepteren
    src = inspect.getsource(run_arousal_respiratory_analysis)
    assert src.count("score_wake_arousals=score_wake_arousals") >= 2, (
        "de wrapper geeft de vlag niet aan beide detectiepaden door")
    src_multi = inspect.getsource(detect_arousals_multi)
    assert "score_wake_arousals=score_wake_arousals" in src_multi, (
        "de multi-afleidingsdetector geeft de vlag niet per afleiding door")
