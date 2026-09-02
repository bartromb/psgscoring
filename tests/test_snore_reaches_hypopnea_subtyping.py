"""Het snurkkanaal moet criterium 1 van de hypopnee-subtypering voeden.

DE REGEL
--------
AASM v3, VIII.D.2: een hypopnee is obstructief bij ten minste één van
(a) **snurken tijdens het event**, (b) toegenomen inspiratoire afvlakking,
(c) paradox tijdens maar niet vóór het event. Centraal alleen als geen van
de drie aanwezig is (VIII.D.3).

WAT ER MIS WAS
--------------
`classify_hypopnea_type` kreeg `snore_present=None` hardgecodeerd: "niet
gemeten". Daardoor rustte het oordeel altijd op twee van de drie criteria, en
met abstentie werd 70 % van de hypopneus `uncertain` in plaats van gescoord --
tegen een menselijk ijkpunt van 5,9 % centraal.

De aanname was dat snurken niet beschikbaar is. Dat klopt voor MESA en PSG-IPA,
maar NIET voor de klinische opnames: SOMNOmedics-montages dragen een
snurkkanaal. De kanaalmap herkende het alleen niet, omdat DOMINO Duitse labels
schrijft ("Schnarchen", "Mikro").

WAT DE MANUAL OVER DE SENSOR ZEGT
---------------------------------
VIII.A.8 RECOMMENDED: *"For monitoring snoring, use an acoustic sensor (e.g.,
microphone), piezoelectric sensor, or nasal pressure transducer."* Een echt
snurkkanaal is dus de aanbevolen weg -- en beschikbaar.
"""
import numpy as np
import pytest

from psgscoring.classify import classify_hypopnea_type


def _ademteugen(n, flattening, start=0.0, periode=4.0):
    return [{"onset_s": start + i * periode, "duration_s": periode * 0.4,
             "flattening": flattening} for i in range(n)]


def _effort(n_s, sf=32.0, amp=1.0):
    t = np.arange(int(n_s * sf)) / sf
    x = amp * np.sin(2 * np.pi * 0.25 * t)
    return x, x.copy()


def test_snurken_tijdens_het_event_maakt_het_obstructief():
    th, ab = _effort(200.0)
    sub, _c, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=_ademteugen(35, 0.10),
        thorax_env=th, abdomen_env=ab, sf=32.0, snore_present=True)
    assert sub == "obstructive"
    assert "snoring" in det["criteria_met"]


def test_met_een_snurkkanaal_is_het_oordeel_VOLLEDIG():
    """Dit is het verschil dat het kanaal maakt: `complete` wordt True, en dan
    betekent 'centraal' werkelijk 'geen van de drie' in plaats van 'geen van de
    twee die we konden toetsen'."""
    th, ab = _effort(200.0)
    _s, _c, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=_ademteugen(35, 0.10),
        thorax_env=th, abdomen_env=ab, sf=32.0, snore_present=False)
    assert det["complete"] is True, det
    assert "snoring" not in det["criteria_unavailable"]


def test_zonder_kanaal_blijft_het_oordeel_onvolledig():
    th, ab = _effort(200.0)
    _s, _c, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=_ademteugen(35, 0.10),
        thorax_env=th, abdomen_env=ab, sf=32.0, snore_present=None)
    assert det["complete"] is False
    assert "snoring" in det["criteria_unavailable"]


def test_geen_snurken_en_geen_ander_kenmerk_geeft_CENTRAAL_niet_uncertain():
    """Met een snurkkanaal mag de regel wél een centrale hypopnee aanwijzen.
    Zonder kanaal abstineert ze -- dat verschil is de hele winst."""
    th, ab = _effort(200.0)
    sub_met, _c, _d = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=_ademteugen(35, 0.10),
        thorax_env=th, abdomen_env=ab, sf=32.0, snore_present=False)
    sub_zonder, _c2, _d2 = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=_ademteugen(35, 0.10),
        thorax_env=th, abdomen_env=ab, sf=32.0, snore_present=None)
    assert sub_met == "central"
    assert sub_zonder == "uncertain"


# ── De keten ──────────────────────────────────────────────────────────────

def test_de_duitse_labels_worden_herkend():
    """SOMNOmedics/DOMINO schrijft Duits. Zonder deze patronen bleef het kanaal
    ongebruikt op precies de opnames waar het aanwezig is."""
    from psgscoring.constants import CHANNEL_PATTERNS
    for naam in ("Schnarchen", "Mikro", "Snore", "Snoring", "Geräusch"):
        k = naam.lower()
        assert any(p in k for p in CHANNEL_PATTERNS["snore"]), naam


def test_de_detectieketen_geeft_het_snurksignaal_door():
    """Vier keer op één dag bleek een signaal zijn consument niet te halen.

    `_detect_hypopneas` moet het snurksignaal kennen; anders blijft
    `snore_present=None` hardgecodeerd en is het kanaal er wel maar doet het
    niets.
    """
    import inspect

    from psgscoring.respiratory import _detect_hypopneas

    params = inspect.signature(_detect_hypopneas).parameters
    assert "snore_data" in params, (
        "de hypopnee-detector kent het snurksignaal niet")
    src = inspect.getsource(_detect_hypopneas)
    assert "snore_present=None" not in src, (
        "snore_present staat nog hardgecodeerd op None")
