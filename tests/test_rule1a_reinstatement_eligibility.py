"""
tests/test_rule1a_reinstatement_eligibility.py
==============================================

De arousal-tak van AASM Rule 1A mag alleen kandidaten herbeoordelen die het
BEVESTIGINGSCRITERIUM misten.

De AASM stelt de flowreductie (>=30 %) en de duur (>=10 s) in beide takken
verplicht; alleen de bevestiging is een disjunctie: >=3 % desaturatie OF een
arousal. `reinstate_rule1a_arousal_hypopneas` liep echter over de VOLLEDIGE
lijst afgewezen kandidaten zonder ooit naar `reject_reason` te kijken. Die
lijst bevat ook:

  * `local_reduction_..pct<..pct`   -- flowreductie te klein
  * `pre_event_reduction_..pct<..`  -- flowreductie te klein (pre-event modus)
  * `stable_breathing_cv_..<..`     -- kwaliteitsveto op een event dat zijn
                                       desaturatie al HAD

Een arousal kon dus een event promoveren dat een verplicht criterium nooit
haalde. Gemeten op de kale functie: 4 van 4 kandidaten kwamen terug, ook die
drie.

Waarom dit nu telt terwijl geen enkel profiel de tak aan heeft: `fase 4` gaat
hem juist aanzetten om de kalibratie te bepalen. Zonder deze reparatie meet
dat experiment een vervuilde tak, en dan is de uitkomst niet te gebruiken --
de winst en de vervuiling zijn achteraf niet uit elkaar te halen.

De tweede helft van deze module is de afleverkant: een allow-list is niets
waard als de producent de markering nooit zet. `test_de_detector_zet_de_reden`
draait de echte detector en controleert dat een kandidaat die alleen zijn
desaturatie miste de reden ook werkelijk draagt.
"""
import numpy as np
import pytest

from psgscoring.respiratory import (
    REINSTATABLE_REJECTIONS,
    reinstate_rule1a_arousal_hypopneas,
)

ONSET, DUUR = 100.0, 20.0
EINDE = ONSET + DUUR


def _kandidaat(reject_reason=None):
    k = {"onset_s": ONSET, "duration_s": DUUR, "stage": "N2",
         "epoch": int(ONSET // 30), "desat": None, "min_spo2": None}
    if reject_reason is not None:
        k["reject_reason"] = reject_reason
    return k


def _herstel(kandidaten, stats=None):
    """Arousal 5 s na het eventeinde -- ruim binnen elk koppelvenster."""
    return reinstate_rule1a_arousal_hypopneas(
        rejected=list(kandidaten),
        arousal_events=[{"onset_s": EINDE + 5.0, "duration_s": 3.0}],
        resp_events=[], hypno=["N2"] * 40, breaths=[], stats=stats,
    )[0]


# ══════════════════════════════════════════════════════════════════════
#  1. Wat NIET terug mag komen
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("reden", [
    "local_reduction_8.3pct<20pct",
    "local_reduction_28.6pct<20pct",
    "pre_event_reduction_12.0pct<30.0pct",
    "stable_breathing_cv_0.11<0.45",
])
def test_een_gemist_verplicht_criterium_blijft_afgewezen(reden):
    """Amplitude, duur en het kwaliteitsveto zijn geen bevestigingscriterium."""
    assert _herstel([_kandidaat(reden)]) == [], (
        f"kandidaat afgewezen om '{reden}' kwam terug als hypopnee; de "
        "arousal-tak vervangt daarmee een criterium dat de AASM in BEIDE "
        "takken verplicht stelt")


def test_een_onbekende_reden_wordt_geweerd_niet_toegelaten():
    """Fail-safe richting: wie een nieuwe grond toevoegt en niets beslist,
    krijgt géén stille herintrede."""
    assert _herstel([_kandidaat("een_nieuwe_grond_van_volgend_jaar")]) == []


# ══════════════════════════════════════════════════════════════════════
#  2. Wat WEL terug moet komen -- de tak mag niet doorslaan
# ══════════════════════════════════════════════════════════════════════

def test_de_ontbrekende_desaturatie_kwalificeert_nog_steeds():
    herstel = _herstel([_kandidaat("no_desaturation")])
    assert len(herstel) == 1
    assert herstel[0]["rule1a_arousal"] is True


def test_de_historische_vorm_zonder_reden_blijft_werken():
    """Externe aanroepers leveren kandidaten zonder `reject_reason` aan."""
    assert len(_herstel([_kandidaat(None)])) == 1


def test_de_allowlist_bevat_precies_die_twee_vormen():
    assert REINSTATABLE_REJECTIONS == frozenset({"", "no_desaturation"})


# ══════════════════════════════════════════════════════════════════════
#  3. De tellers maken de uitsluiting zichtbaar
# ══════════════════════════════════════════════════════════════════════

def test_de_statistiek_toont_wie_er_buiten_viel_en_waarom():
    stats = {}
    kandidaten = [_kandidaat("no_desaturation"),
                  _kandidaat("local_reduction_8.3pct<20pct"),
                  _kandidaat("stable_breathing_cv_0.11<0.45")]
    herstel = _herstel(kandidaten, stats=stats)

    assert len(herstel) == 1
    assert stats["n_candidates_tested"] == 1, (
        "'getest' hoort te slaan op wat de tak werkelijk beoordeeld heeft")
    assert stats["n_ineligible"] == 2
    assert stats["ineligible_by_reason"] == {
        "local_reduction_8.3pct<20pct": 1,
        "stable_breathing_cv_0.11<0.45": 1,
    }


# ══════════════════════════════════════════════════════════════════════
#  4. De afleverkant: zet de DETECTOR de reden werkelijk?
# ══════════════════════════════════════════════════════════════════════

def _opname_met_hypopneus_zonder_desaturatie():
    """Vier flowreducties van 60 %, SpO2 vlak -- dus geen enkele bevestiging.

    Het stabiliteitsfilter en de lokale vloer staan uit: die zouden dezelfde
    kandidaten om een ANDERE reden afwijzen, en dan meet deze test niet wat
    ze bedoelt.
    """
    sf, dur, BR = 32.0, 900, 0.25
    t = np.arange(int(sf * dur)) / sf
    rng = np.random.default_rng(7)
    amp = np.ones(t.size)
    fac = rng.lognormal(0.0, 0.25, int(dur * BR) + 2)
    for i in range(int(dur * BR) + 2):
        amp[int(i / BR * sf):int((i + 1) / BR * sf)] = fac[i]
    flow = amp * np.sin(2 * np.pi * BR * t) + rng.normal(0, 0.005, t.size)
    for start in range(120, dur - 120, 180):
        flow[int(start * sf):int((start + 18) * sf)] *= 0.40
    return flow, np.full(dur, 97.0), ["N2"] * (dur // 30), sf


def test_de_detector_zet_de_reden():
    """Zonder deze markering is de allow-list een filter op een leeg veld."""
    from psgscoring.respiratory import detect_respiratory_events

    flow, spo2, hypno, sf = _opname_met_hypopneus_zonder_desaturatie()
    r = detect_respiratory_events(
        flow_data=flow, thorax_data=None, abdomen_data=None, spo2_data=spo2,
        sf_flow=sf, sf_spo2=1.0, hypno=hypno,
        scoring_profile={"STABILITY_FILTER_CV": 0.0,
                         "LOCAL_BL_MIN_REDUCTION_PCT": 10.0,
                         "LOCAL_BL_STRICT_RED": 10.0,
                         "HYPOPNEA_THRESHOLD": 0.70,
                         "DESATURATION_DROP_PCT": 3.0})
    afgewezen = r.get("rejected_hypopneas", [])
    assert afgewezen, "fixture levert geen afgewezen kandidaten -- meet niets"

    redenen = {str(x.get("reject_reason") or "") for x in afgewezen}
    assert redenen == {"no_desaturation"}, (
        f"detector benoemt de desaturatie-afwijzing niet: {redenen}")
    assert redenen <= REINSTATABLE_REJECTIONS, (
        "de producent en de allow-list zijn uit de pas gelopen")
