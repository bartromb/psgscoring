"""Een kanaal dat de snurkband niet kan dragen, mag geen antwoord geven.

DE AANLEIDING, GEMETEN
======================
Het MESA-kanaal dat `Snore` heet is bemonsterd op **32 Hz** — uit de EDF-kop,
niet uit MNE, die alle kanalen naar 256 Hz optilt. Nyquist ligt dus op 16 Hz.

Snurken is akoestische/vibratie-energie met een grondtoon rond 30 tot 250 Hz.
Een kanaal met een Nyquist van 16 Hz kan dat verschijnsel niet representeren —
niet slecht, maar principieel niet.

Alles wat op 37 MESA-opnames aan dat kanaal is gemeten (AUC 0,486 vóór het
event, 0,566 tijdens, 0,543 bij herstel, tegen menselijke apneu-subtypes) ging
dus over een laagfrequente envelope van onbekende herkomst, niet over snurken.
Criterium 1 van AASM v3 §6.1 is daarmee niet WEERLEGD op MESA — het is er
nooit getoetst.

Dit is dezelfde fout als bij de RIP-poort, die de EDF-eenheid mat in plaats van
de ademhaling, en bij de vlakke-detectie, die het bestand mat in plaats van de
patiënt. De maat moet weigeren wanneer het signaal de vraag niet kan dragen.
"""
import numpy as np

from psgscoring.respiratory import SNORE_MIN_SF_HZ, _snore_during


def _burst(sf, n_s=600.0, f_snore=45.0):
    """Een nacht met een snurksalvo van 20 s vanaf t=300."""
    n = int(n_s * sf)
    t = np.arange(n) / sf
    x = np.random.default_rng(0).normal(0, 0.01, n)
    m = (t >= 300) & (t < 320)
    x[m] += np.sin(2 * np.pi * f_snore * t[m])
    return x


def test_een_kanaal_onder_de_snurkband_geeft_GEEN_oordeel():
    """32 Hz is wat MESA levert. Nyquist 16 Hz — de band is onbereikbaar."""
    sf = 32.0
    assert _snore_during(_burst(sf), sf, 300.0, 20.0) is None, (
        "een kanaal dat de snurkband niet kan dragen hoort 'niet gemeten' te "
        "zeggen, niet 'niet gesnurkt'")


def test_de_grens_ligt_op_twee_maal_de_ONDERrand_van_de_band():
    """Niet een verzonnen getal: Nyquist moet boven 30 Hz uitkomen."""
    assert SNORE_MIN_SF_HZ == 60.0
    assert _snore_during(_burst(60.0), 60.0, 300.0, 20.0) is None
    assert _snore_during(_burst(200.0), 200.0, 300.0, 20.0) is not None


def test_een_echt_snurkkanaal_wordt_NIET_geweigerd():
    """Zonder deze toets zou een te strenge grens alles onbereikbaar maken."""
    sf = 200.0
    uit = _snore_during(_burst(sf), sf, 300.0, 20.0)
    assert uit is True, ("een salvo van 45 Hz op 200 Hz hoort herkend te "
                         "worden; anders meet de grens de verkeerde kant op")


def test_de_weigering_is_ONAFHANKELIJK_van_de_inhoud():
    """Ook een kanaal vol snurken wordt geweigerd als de band onbereikbaar is.

    Anders zou de grens soms wel en soms niet gelden, en dat is geen grens.
    """
    sf = 32.0
    luid = _burst(sf) * 100.0
    assert _snore_during(luid, sf, 300.0, 20.0) is None


def test_de_vlag_bereikt_de_subtypering():
    """Leveringsoppervlak: het oordeel moet als 'niet gemeten' aankomen."""
    from psgscoring.classify import classify_hypopnea_type

    sub, _conf, det = classify_hypopnea_type(
        onset_s=300.0, duration_s=20.0, breaths=None,
        thorax_env=None, abdomen_env=None, sf=32.0,
        snore_present=_snore_during(_burst(32.0), 32.0, 300.0, 20.0))
    assert "snoring" in det["criteria_unavailable"]
    assert det["complete"] is False
    assert sub == "uncertain", (
        "zonder toetsbaar snurkcriterium mag er geen centrale hypopnee "
        "uit de restbak komen")
