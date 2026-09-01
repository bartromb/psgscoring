"""Hypopneus krijgen hun eigen subtyperingsregel, want de manual heeft er een.

WAT ER MIS WAS
--------------
`classify_apnea_type` implementeert AASM sectie 3B -- de APNEUregel -- en werd
ongewijzigd op hypopneus toegepast. Haar centrale regel luidt "truly flat: no
raw movement, low envelope" -> centraal. Bij een apneu is dat zinnig. Bij een
hypopneu is de inspanning **per definitie nooit vlak**: de flow daalt 30 tot
90 %, niet naar nul. Die regel kan daar dus vrijwel alleen vuren wanneer het
EFFORTKANAAL zwak is -- een losgekoppelde band, lage versterking, een RIP-poort
die de paradoxdetectie uitschakelt.

`hypopnea_central` is daarmee in de praktijk vaker een uitspraak over de
meetopstelling dan over de patiënt.

WAT DE MANUAL WEL ZEGT (§6.1, optioneel)
----------------------------------------
Omgekeerd ontworpen. **Obstructief** bij ten minste één van:

  1. snurken tijdens het event,
  2. toegenomen inspiratoire afvlakking van de neusdrukcurve ten opzichte van
     de basislijnademteugen,
  3. thoracoabdominale paradox die TIJDENS het event optreedt maar NIET in de
     ademhaling ervóór.

**Centraal** alleen als geen van de drie aanwezig is.

Drie positieve obstructiekenmerken met centraal als restcategorie. Geen
effort-vlakheid. En criterium 3 met zijn pre-event-vergelijking is precies de
bescherming tegen de meetopstellingsartefacten hierboven: een chronisch
paradoxale of losgekoppelde band vuurt niet, want die is vóór het event al
paradoxaal.

HET IJKPUNT
-----------
Menselijke scoorders in PSG-IPA noemen **5,9 %** van hun hypopneus centraal
(95 van 1601).
"""
import collections

import numpy as np
import pytest

from psgscoring.classify import classify_hypopnea_type

SF = 32.0


def _ademteugen(n, flattening, start=0.0, periode=4.0):
    return [{"onset_s": start + i * periode, "duration_s": periode * 0.4,
             "flattening": flattening} for i in range(n)]


def _effort(n_s, sf=SF, paradox_vanaf=None, amp=1.0):
    """Thorax en abdomen; vanaf `paradox_vanaf` lopen ze in tegenfase."""
    t = np.arange(int(n_s * sf)) / sf
    thorax = amp * np.sin(2 * np.pi * 0.25 * t)
    abdomen = amp * np.sin(2 * np.pi * 0.25 * t)
    if paradox_vanaf is not None:
        k = int(paradox_vanaf * sf)
        abdomen[k:] = -abdomen[k:]
    return thorax, abdomen


# ── Criterium 2: toegenomen afvlakking ────────────────────────────────────

def test_toegenomen_afvlakking_maakt_het_obstructief():
    """Basislijn driehoekig (0,10), tijdens het event een plateau (0,45)."""
    ademteugen = _ademteugen(30, 0.10, start=0.0) + \
                 _ademteugen(5, 0.45, start=120.0)
    th, ab = _effort(200.0)
    sub, _conf, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=ademteugen,
        thorax_env=th, abdomen_env=ab, sf=SF)
    assert sub == "obstructive", det
    assert "flattening" in det["criteria_met"], det


def test_gelijke_afvlakking_pleit_niet_voor_obstructie():
    """Zonder TOENAME zegt criterium 2 niets -- de manual vraagt een
    vergelijking met de basislijn, geen absolute waarde."""
    ademteugen = _ademteugen(30, 0.35, start=0.0) + \
                 _ademteugen(5, 0.35, start=120.0)
    th, ab = _effort(200.0)
    _sub, _c, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=ademteugen,
        thorax_env=th, abdomen_env=ab, sf=SF)
    assert "flattening" not in det["criteria_met"], det


# ── Criterium 3: paradox tijdens maar niet ervóór ─────────────────────────

def test_paradox_die_pas_bij_het_event_begint_is_obstructief():
    th, ab = _effort(200.0, paradox_vanaf=120.0)
    sub, _c, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=_ademteugen(35, 0.10),
        thorax_env=th, abdomen_env=ab, sf=SF)
    assert sub == "obstructive", det
    assert "paradox" in det["criteria_met"], det


def test_chronische_paradox_pleit_NIET_voor_obstructie():
    """Dit is de kern van criterium 3, en de reparatie van de oude fout.

    Een band die de hele nacht al paradoxaal staat -- verwisselde polariteit,
    losgekoppeld, verkeerd geplakt -- is een meetopstelling, geen fysiologie.
    De pre-event-vergelijking vangt dat; een absolute paradoxtoets niet.
    """
    th, ab = _effort(200.0, paradox_vanaf=0.0)
    _sub, _c, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=_ademteugen(35, 0.10),
        thorax_env=th, abdomen_env=ab, sf=SF)
    assert "paradox" not in det["criteria_met"], (
        "chronische paradox telt als obstructiekenmerk; dan vuurt elke "
        "losgekoppelde band")


# ── Criterium 1: snurken ──────────────────────────────────────────────────

def test_snurken_maakt_het_obstructief():
    th, ab = _effort(200.0)
    sub, _c, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=_ademteugen(35, 0.10),
        thorax_env=th, abdomen_env=ab, sf=SF, snore_present=True)
    assert sub == "obstructive"
    assert "snoring" in det["criteria_met"]


def test_zonder_snurkkanaal_wordt_dat_gemeld_en_niet_verzwegen():
    """Met twee van de drie criteria is 'centraal' zwakker dan de manual
    bedoelt: het betekent 'geen van de twee die we kónden toetsen'.

    Het bandfilter (0,05-3 Hz) knipt snurktrillingen er bovendien juist uit,
    dus zonder apart snurkkanaal is criterium 1 op dit pad onbereikbaar.
    """
    th, ab = _effort(200.0)
    _sub, _c, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=_ademteugen(35, 0.10),
        thorax_env=th, abdomen_env=ab, sf=SF, snore_present=None)
    assert "snoring" in det["criteria_unavailable"], det
    assert det["complete"] is False, (
        "een oordeel op twee van de drie criteria mag zich niet als volledig "
        "voordoen")


# ── Centraal als RESTcategorie ────────────────────────────────────────────

def test_centraal_alleen_als_geen_enkel_kenmerk_aanwezig_is():
    th, ab = _effort(200.0)
    sub, _c, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=_ademteugen(35, 0.10),
        thorax_env=th, abdomen_env=ab, sf=SF, snore_present=False)
    assert sub == "central", det
    assert det["criteria_met"] == []
    assert det["complete"] is True


def test_effortvlakheid_speelt_geen_enkele_rol():
    """De apneuregel gebruikt effort-vlakheid; deze regel niet.

    Een zwak effortsignaal -- de situatie die `hypopnea_central` nu
    produceert -- mag op zichzelf niets beslissen.
    """
    th, ab = _effort(200.0, amp=0.001)          # vrijwel vlakke banden
    sub, _c, _det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0,
        breaths=_ademteugen(30, 0.10) + _ademteugen(5, 0.45, start=120.0),
        thorax_env=th, abdomen_env=ab, sf=SF)
    assert sub == "obstructive", (
        "de afvlakking pleit voor obstructie; een vlakke effortband mag dat "
        "niet omkeren")


def test_zonder_bruikbare_gegevens_geen_bewering():
    """Geen ademteugen, geen banden, geen snurkkanaal: dan is er niets om op
    te oordelen, en `uncertain` is het eerlijke antwoord."""
    sub, conf, det = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=[],
        thorax_env=None, abdomen_env=None, sf=SF)
    assert sub == "uncertain", det
    assert conf <= 0.5
    assert len(det["criteria_unavailable"]) == 3


@pytest.mark.parametrize("aantal_kenmerken,verwacht", [
    (0, "central"), (1, "obstructive"), (2, "obstructive"), (3, "obstructive"),
])
def test_een_kenmerk_is_genoeg(aantal_kenmerken, verwacht):
    """`ten minste één van` -- geen stemming, geen weging."""
    ademteugen = _ademteugen(30, 0.10) + _ademteugen(
        5, 0.45 if aantal_kenmerken >= 2 else 0.10, start=120.0)
    th, ab = _effort(200.0,
                     paradox_vanaf=120.0 if aantal_kenmerken >= 3 else None)
    sub, _c, _d = classify_hypopnea_type(
        onset_s=120.0, duration_s=20.0, breaths=ademteugen,
        thorax_env=th, abdomen_env=ab, sf=SF,
        snore_present=(aantal_kenmerken >= 1))
    assert sub == verwacht


# ── Bereikt de vlag de detectieketen? ─────────────────────────────────────

def test_de_vlag_verandert_de_labels_van_een_echte_run():
    """Vier keer op één dag bleek een vlag zijn consument niet te halen, en
    elke keer zag de meting er geslaagd uit terwijl ze niets mat.

    Deze test ving twee echte fouten voordat hij groen werd:
    een `NameError` in de detectieketen (de parameter zat in de verkeerde
    functie, en de detectie faalde stil met `success=False`), en drie fixtures
    die nul events opleverden -- afgewezen door het stabiliteitsfilter omdat
    perfect regelmatige synthetische ademhaling een CV onder 0,45 heeft.

    Wat deze test NIET zegt: of de nieuwe regel klinisch beter is. De richting
    op een synthetische opname zegt niets; daarvoor is de cohortmeting tegen
    het ijkpunt van 5,9 %.
    """
    import os

    import psgscoring

    mne = pytest.importorskip("mne")
    sf, minuten = 32.0, 30
    n = int(sf * 60 * minuten)
    t = np.arange(n) / sf
    rng = np.random.default_rng(11)
    # Variabele ademamplitude: het stabiliteitsfilter wijst regelmatige
    # ademhaling af als normale variatie, en dan meet de test niets.
    amp = 1.0 + 0.7 * np.sin(2 * np.pi * 0.011 * t) + 0.25 * rng.normal(0, 1, n)
    flow = amp * np.sin(2 * np.pi * 0.25 * t)
    spo2 = np.full(n, 97.0)
    for start in range(60, 60 * minuten - 60, 120):
        a, b = int(start * sf), int((start + 25) * sf)
        # Plateau in plaats van sinus: gereduceerd én afgevlakt.
        flow[a:b] = 0.30 * np.sign(np.sin(2 * np.pi * 0.25 * t[a:b]))
        d0 = b + int(4 * sf)          # circulatievertraging
        spo2[d0:d0 + int(20 * sf)] -= 7.0
    zwak = 0.02 * np.sin(2 * np.pi * 0.25 * t)   # nauwelijks effort
    info = mne.create_info(["Resp nasal", "SaO2", "Thorax", "Abdomen"],
                           sf, ["misc"] * 4)
    raw = mne.io.RawArray(np.vstack([flow, spo2, zwak, zwak]), info,
                          verbose=False)
    hypno = ["N2"] * int(np.ceil(raw.times[-1] / 30.0))

    labels = {}
    for arm in ("0", "1"):
        os.environ["PSGSCORING_HYPOPNEA_SUBTYPE_AASM"] = arm
        try:
            out = psgscoring.run_pneumo_analysis(
                raw, hypno=hypno, scoring_profile="aasm_v3_rec")
        finally:
            os.environ.pop("PSGSCORING_HYPOPNEA_SUBTYPE_AASM", None)
        assert out["respiratory"].get("success") is True, (
            f"arm {arm}: {out['respiratory'].get('error')}")
        ev = [e for e in out["respiratory"]["events"] if "hypopnea" in e["type"]]
        assert ev, f"arm {arm} levert geen hypopneus — de test meet niets"
        labels[arm] = collections.Counter(e["type"] for e in ev)

    assert labels["0"] != labels["1"], (
        f"de vlag verandert niets aan de labels: {labels} — hij bereikt de "
        f"detectieketen niet")


def test_de_criteria_staan_per_event_in_de_uitvoer():
    """Een subtype zonder zijn onderbouwing is niet te reviewen.

    `criteria_unavailable` moet snurken noemen: op dit pad is het niet
    toetsbaar, en 'centraal' betekent dan 'geen van de TWEE die we konden
    toetsen'. Dat verschil hoort per event zichtbaar te zijn.
    """
    import os

    import psgscoring

    mne = pytest.importorskip("mne")
    sf, minuten = 32.0, 30
    n = int(sf * 60 * minuten)
    t = np.arange(n) / sf
    rng = np.random.default_rng(11)
    amp = 1.0 + 0.7 * np.sin(2 * np.pi * 0.011 * t) + 0.25 * rng.normal(0, 1, n)
    flow = amp * np.sin(2 * np.pi * 0.25 * t)
    spo2 = np.full(n, 97.0)
    for start in range(60, 60 * minuten - 60, 120):
        a, b = int(start * sf), int((start + 25) * sf)
        flow[a:b] = 0.30 * np.sign(np.sin(2 * np.pi * 0.25 * t[a:b]))
        d0 = b + int(4 * sf)
        spo2[d0:d0 + int(20 * sf)] -= 7.0
    zwak = 0.02 * np.sin(2 * np.pi * 0.25 * t)
    info = mne.create_info(["Resp nasal", "SaO2", "Thorax", "Abdomen"],
                           sf, ["misc"] * 4)
    raw = mne.io.RawArray(np.vstack([flow, spo2, zwak, zwak]), info,
                          verbose=False)
    hypno = ["N2"] * int(np.ceil(raw.times[-1] / 30.0))

    os.environ["PSGSCORING_HYPOPNEA_SUBTYPE_AASM"] = "1"
    try:
        out = psgscoring.run_pneumo_analysis(
            raw, hypno=hypno, scoring_profile="aasm_v3_rec")
    finally:
        os.environ.pop("PSGSCORING_HYPOPNEA_SUBTYPE_AASM", None)

    hyp = [e for e in out["respiratory"]["events"] if "hypopnea" in e["type"]]
    assert hyp
    for e in hyp:
        d = e.get("classify_detail") or {}
        assert d.get("rule", "").startswith("AASM v3"), d
        assert "snoring" in d.get("criteria_unavailable", []), d
        assert d.get("complete") is False, (
            "zonder snurkkanaal mag het oordeel zich niet als volledig "
            "voordoen")
    # En het criterium moet ook echt kunnen vuren, niet alleen bestaan.
    obstructief = [e for e in hyp if e["type"] == "hypopnea"]
    assert obstructief, "geen enkel event haalde een obstructiekenmerk"
    d = obstructief[0].get("classify_detail") or {}
    assert d["criteria_met"], d
    assert d["flattening_event"] > d["flattening_baseline"], d


def test_de_default_blijft_de_oude_regel():
    """Werkregel 1: meten gaat vóór aanzetten."""
    from psgscoring.constants import _profile_to_legacy_dict
    from psgscoring.pipeline import _hypopnea_subtype_aasm
    from psgscoring.profiles import get_profile

    for naam in ("aasm_v3_rec", "aasm_v3_breath", "mesa_shhs"):
        d = _profile_to_legacy_dict(get_profile(naam))
        assert _hypopnea_subtype_aasm(d) is False, naam
