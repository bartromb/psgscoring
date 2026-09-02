"""Snurken tijdens het event moet tegen de ADEMHALING ERVOOR gemeten worden.

DE METING DIE DIT AANLEIDING GAF
--------------------------------
Op 24 MESA-opnames met een absolute drempel (60e percentiel van de RMS over de
hele nacht):

    menselijke hypopneus   2512 events, 30,1 % 'snurkt'
    normale ademhaling     1440 vensters, 39,7 % 'snurkt'

Snurken kwam VAKER voor buiten de events dan erin. Het criterium vuurde
willekeurig, en in de keten leverde dat 45,3 % centrale hypopneus tegen een
menselijk ijkpunt van 5,9 %.

WAAROM DE ABSOLUTE DREMPEL NIET KAN WERKEN
------------------------------------------
Snurken is een eigenschap van de NACHT, niet van het event. Een snurkende
patiënt snurkt vrijwel doorlopend; een drempel op het nachtpercentiel markeert
dan overal "snurkt", en het onderscheid verdwijnt.

De manual zegt *"snoring during the event"* in een opsomming met "**increased**
inspiratory flattening compared to baseline breathing" en paradox "during the
event but **not** during pre-event breathing". Alle drie de criteria zijn
CONTRASTEN met de pre-event-ademhaling. Criterium 1 hoort dat ook te zijn.

DE VORM
-------
Snurkintensiteit tijdens het event tegen die van de twee minuten ervóór --
hetzelfde venster als de AASM-basislijn (VIII.B Note 1) en dezelfde vorm als
criterium 2.
"""
import numpy as np
import pytest

from psgscoring.respiratory import _snore_during

SF = 32.0


def _signaal(n_s, snurk_overal=False, snurk_venster=None, sf=SF, seed=1):
    """Ruis, met optioneel snurken overal en/of in één venster."""
    n = int(n_s * sf)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.0, n)
    if snurk_overal:
        x += rng.normal(0, 4.0, n)
    if snurk_venster:
        a, b = int(snurk_venster[0] * sf), int(snurk_venster[1] * sf)
        x[a:b] += rng.normal(0, 8.0, b - a)
    return x


def test_snurken_dat_bij_het_event_begint_telt():
    """Stille basislijn, luid tijdens het event."""
    x = _signaal(600.0, snurk_venster=(300.0, 320.0))
    assert _snore_during(x, SF, 300.0, 20.0) is True


def test_een_patient_die_de_hele_nacht_snurkt_geeft_GEEN_kenmerk():
    """Dit is de kern. Met de absolute drempel was dit True voor elk event, en
    dan is criterium 1 betekenisloos."""
    x = _signaal(600.0, snurk_overal=True)
    assert _snore_during(x, SF, 300.0, 20.0) is False


def test_stilte_tijdens_het_event_bij_een_snurkende_nacht_telt_niet():
    """Minder snurken dan ervoor is geen obstructiekenmerk."""
    x = _signaal(600.0, snurk_overal=True)
    a, b = int(300.0 * SF), int(320.0 * SF)
    x[a:b] *= 0.1
    assert _snore_during(x, SF, 300.0, 20.0) is False


def test_zonder_kanaal_blijft_het_None():
    assert _snore_during(None, SF, 300.0, 20.0) is None
    assert _snore_during(np.zeros(10), None, 300.0, 20.0) is None


def test_een_event_aan_het_begin_zonder_basislijn_geeft_None():
    """Zonder pre-event-ademhaling is er niets om tegen af te zetten, en dan is
    'niet gemeten' het eerlijke antwoord -- niet 'niet gesnurkt'."""
    x = _signaal(600.0, snurk_venster=(0.0, 20.0))
    assert _snore_during(x, SF, 0.0, 20.0) is None


def test_de_maat_is_schaalvrij():
    """Een montage met dubbele versterking mag niet ander antwoord geven; de
    RIP-poort maakte precies die fout drie keer."""
    x = _signaal(600.0, snurk_venster=(300.0, 320.0))
    a = _snore_during(x, SF, 300.0, 20.0)
    b = _snore_during(x * 1000.0, SF, 300.0, 20.0)
    assert a == b


def test_de_lokale_maat_verschilt_AANTOONBAAR_van_de_absolute():
    """Zonder dit onderscheid bewijzen de tests hierboven niets.

    De opzet: een nacht die in de TWEEDE helft luider wordt (houding, slaapfase).
    Een absolute nachtdrempel markeert die hele helft als 'snurkt'; de lokale
    maat ziet dat er niets VERANDERT rond het event.
    """
    n_s, sf = 1200.0, SF
    n = int(n_s * sf)
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1.0, n)
    helft = n // 2
    x[helft:] += rng.normal(0, 6.0, n - helft)      # tweede helft luider

    # absolute maat: 60e percentiel over alles
    win = int(sf)
    n_win = n // win
    rms = np.sqrt(np.mean(x[:n_win * win].reshape(n_win, win) ** 2, axis=1))
    drempel = float(np.percentile(rms, 60.0))
    ev0, ev1 = 900, 920                              # in de luide helft
    absoluut = bool(np.mean(rms[ev0:ev1] > drempel) > 0.5)

    lokaal = _snore_during(x, sf, float(ev0), 20.0)

    assert absoluut is True, "de fixture reproduceert de absolute fout niet"
    assert lokaal is False, (
        "de lokale maat markeert een constant luide periode óók als snurken; "
        "dan is er geen verschil met de absolute drempel")


def test_de_drempel_is_een_KEUZE_en_staat_als_zodanig_in_de_signatuur():
    """De manual kwantificeert 'snoring during the event' niet. Een verborgen
    constante zou dat verhullen."""
    import inspect

    sig = inspect.signature(_snore_during)
    assert "ratio" in sig.parameters
    assert sig.parameters["ratio"].default == 1.30
    assert "baseline_s" in sig.parameters
    assert sig.parameters["baseline_s"].default == 120.0, (
        "het basislijnvenster hoort 2 minuten te zijn, zoals AASM VIII.B Note 1")
