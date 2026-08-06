"""`aasm_v3_breath_dual` en `aasm_v3_prob_dual` — het lege kruisvlak.

De profielen varieerden langs twee assen die elkaar niet raakten: welke
hypopneedetector draait, en op hoeveel flowsensoren apneus worden gezocht.
Deze twee profielen zetten beide aan. Wat hier getoetst wordt is precies dat
"elkaar niet raken", want dat is een aanname en geen wet:

  * op een montage met één flowkanaal moeten ze IDENTIEK zijn aan hun ouder —
    de nieuwe as kan daar niets doen, en als hij er tóch iets doet is de
    combinatie niet wat hij beweert te zijn;
  * op een montage met twee sensoren mag de tweede sensor apneus TOEVOEGEN en
    er nooit een verwijderen;
  * de ademteugsegmentatie hoort op de neusdruk te gebeuren, ook wanneer het
    apneukanaal de thermistor is. Dat is de val waar dit ontwerp op stond of
    viel: segmenteren op een traag, afgerond signaal meet iets anders dan
    `aasm_v3_breath` meet.

Niet getoetst, en dat is geen omissie maar de stand van zaken: of de tweede
sensor de AHI DICHTER bij een menselijke scoorder brengt. Daar is een cohort
met twee werkende flowsensoren én menselijke scoring voor nodig.
"""

import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES, get_profile

mne = pytest.importorskip("mne")

SF = 32.0
DUR_S = 600
N = int(DUR_S * SF)
T = np.arange(N) / SF


# ─────────────────────────────────────────────────────────────────
#  De definities
# ─────────────────────────────────────────────────────────────────

PAIRS = [("aasm_v3_breath", "aasm_v3_breath_dual"),
         ("aasm_v3_prob", "aasm_v3_prob_dual")]


@pytest.mark.parametrize("parent,child", PAIRS)
def test_the_child_differs_from_its_parent_on_exactly_three_fields(parent, child):
    """Alles wat niet de sensor-as is, moet van de ouder komen.

    Vooral het operatiepunt: `hypopnea_strictness` 0,50 en — bij prob —
    arousal-gewicht 0,70 zijn gemeten waarden. Een kind dat daar stilletjes
    van afwijkt meet iets anders dan zijn naam belooft.
    """
    a = get_profile(parent).post_processing
    b = get_profile(child).post_processing
    moved = {f for f in vars(a) if getattr(a, f) != getattr(b, f)}
    assert moved == {"dual_sensor_apnea", "flow_reference",
                     "thermistor_gate"}, moved
    assert b.hypopnea_strictness == a.hypopnea_strictness
    assert b.hypopnea_arousal_weight == a.hypopnea_arousal_weight
    assert b.hypopnea_arousal_latency_grading == a.hypopnea_arousal_latency_grading
    assert get_profile(child).hypopnea == get_profile(parent).hypopnea


@pytest.mark.parametrize("parent,child", PAIRS)
def test_the_nested_rules_are_copies_and_not_the_parents_own_object(parent, child):
    """`replace()` geeft standaard het geneste object van de ouder door. Twee
    profielen die één mutabel object delen is een val die pas veel later
    afgaat, en dan in een ander profiel dan waar iemand aan het werk was."""
    for f in ("hypopnea", "apnea", "spo2", "post_processing"):
        assert getattr(get_profile(child), f) is not getattr(get_profile(parent), f), f


@pytest.mark.parametrize("_,child", PAIRS)
def test_the_reference_channel_is_the_nasal_pressure(_, child):
    """De val uit het ontwerpdocument, expliciet vastgelegd."""
    assert get_profile(child).post_processing.flow_reference == "hypopnea"
    assert SCORING_PROFILES[child]["FLOW_REFERENCE"] == "hypopnea"


@pytest.mark.parametrize("_,child", PAIRS)
def test_the_new_axes_reach_the_dict_the_pipeline_actually_reads(_, child):
    """Twee keer eerder is een veld gepatcht dat niemand leest — de
    dataclass-naam is niet de legacy-sleutel. Daarom via SCORING_PROFILES."""
    d = SCORING_PROFILES[child]
    assert d["DUAL_SENSOR_APNEA"] is True
    assert d["HYPOPNEA_DETECTOR"] == "breath_graded"


@pytest.mark.parametrize("_,child", PAIRS)
def test_they_are_exploratory_and_say_so(_, child):
    p = get_profile(child)
    assert p.family == "exploratory"
    assert "experimental" in p.description.lower() or "EXPERIMENTAL" in p.description


def test_no_existing_profile_moved_onto_the_new_axis():
    """Alles is additief. Wie hiervoor enkelsensor was, blijft dat."""
    expected_dual = {"aasm_v3_dual", "aasm_v3_fusion",
                     "aasm_v3_breath_dual", "aasm_v3_prob_dual"}
    actual = {n for n, p in PROFILES.items() if p.post_processing.dual_sensor_apnea}
    assert actual == expected_dual, actual


# ─────────────────────────────────────────────────────────────────
#  Synthetische montages
# ─────────────────────────────────────────────────────────────────

def _breathing(amp=1.0, rate=0.25, seed=0):
    rng = np.random.default_rng(seed)
    mod = 1.0 + 0.4 * np.sin(2 * np.pi * 0.01 * T)
    return amp * mod * np.sin(2 * np.pi * rate * T) + 0.05 * amp * rng.normal(size=N)


def _flatten(sig, t0, t1, factor=0.02):
    out = sig.copy()
    out[int(t0 * SF):int(t1 * SF)] *= factor
    return out


#: Apneus die BEIDE sensoren zien.
SHARED = [(60, 78), (180, 198), (300, 318)]
#: Eén apneu die alleen de neusdruk ziet — de thermistor ademt daar door.
#: Dit is wat de tweede pas hoort toe te voegen.
PRESSURE_ONLY = (450, 468)


def _spo2_with_dips(events):
    spo2 = np.full(N, 96.0)
    for t0, t1 in events:
        a, b = int((t1 + 2) * SF), int((t1 + 18) * SF)
        spo2[a:min(b, N)] = 90.0
    return spo2


def _dual_montage():
    """Twee flowsensoren die het grotendeels eens zijn.

    De thermistor moet de kwaliteitstoets halen, anders is hij niet het
    apneukanaal en toetst de additiviteitstest iets anders dan hij beweert.
    """
    pres = _breathing(seed=1)
    therm = _breathing(seed=2)
    for t0, t1 in SHARED:
        pres = _flatten(pres, t0, t1)
        therm = _flatten(therm, t0, t1)
    pres = _flatten(pres, *PRESSURE_ONLY)      # alleen de druk valt hier weg

    data = np.vstack([pres, therm,
                      _breathing(amp=0.6, seed=3), _breathing(amp=0.6, seed=4),
                      _spo2_with_dips(SHARED + [PRESSURE_ONLY])])
    info = mne.create_info(["Pressure Flow", "Flow Th.", "THORAX", "ABDOMEN", "SPO2"],
                           SF, ch_types=["misc"] * 5)
    cmap = {"flow_pressure": "Pressure Flow", "flow_thermistor": "Flow Th.",
            "thorax": "THORAX", "abdomen": "ABDOMEN", "spo2": "SPO2"}
    return mne.io.RawArray(data, info, verbose="ERROR"), cmap


def _single_flow_montage():
    """Eén flowkanaal — de montage van PSG-IPA, en van de meeste opnames waar
    `aasm_v3_breath` op gemeten is."""
    pres = _breathing(seed=1)
    for t0, t1 in SHARED:
        pres = _flatten(pres, t0, t1)
    pres = _flatten(pres, *PRESSURE_ONLY)

    data = np.vstack([pres, _breathing(amp=0.6, seed=3), _breathing(amp=0.6, seed=4),
                      _spo2_with_dips(SHARED + [PRESSURE_ONLY])])
    info = mne.create_info(["Pressure Flow", "THORAX", "ABDOMEN", "SPO2"],
                           SF, ch_types=["misc"] * 4)
    cmap = {"flow_pressure": "Pressure Flow", "thorax": "THORAX",
            "abdomen": "ABDOMEN", "spo2": "SPO2"}
    return mne.io.RawArray(data, info, verbose="ERROR"), cmap


HYPNO = ["W"] + ["N2"] * (DUR_S // 30 - 1)


def _run(raw, cmap, profile):
    import psgscoring
    return psgscoring.run_pneumo_analysis(raw, hypno=HYPNO, channel_map=cmap,
                                          scoring_profile=profile)


def _events(out):
    return [(round(float(e["onset_s"]), 3), round(float(e.get("duration_s") or 0), 3),
             str(e.get("type", "")))
            for e in out["respiratory"]["events"]]


def _apneas(out):
    return [e for e in _events(out) if "hypopnea" not in e[2].lower()]


@pytest.fixture(scope="module")
def single():
    raw, cmap = _single_flow_montage()
    return {n: _run(raw, cmap, n)
            for n in ("aasm_v3_breath", "aasm_v3_breath_dual",
                      "aasm_v3_prob", "aasm_v3_prob_dual")}


@pytest.fixture(scope="module")
def dual():
    raw, cmap = _dual_montage()
    return {n: _run(raw, cmap, n)
            for n in ("aasm_v3_breath", "aasm_v3_breath_dual")}


# ─────────────────────────────────────────────────────────────────
#  De belangrijkste test: waar de as niets kan, doet hij niets
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("parent,child", PAIRS)
def test_on_a_single_flow_montage_the_child_equals_its_parent(single, parent, child):
    """Precies zoals `rec`, `dual` en `pressure` op PSG-IPA samenvallen.

    Dit is de test die bewijst dat de nieuwe profielen op alle bestaande
    validatiedata hetzelfde meten als de profielen waarop die validatie is
    gedaan — en dus dat de PSG-IPA-cijfers voor `breath` en `prob` er
    ongewijzigd voor gelden.
    """
    assert single[parent]["respiratory"]["events"], "geen events — dan toetst dit niets"
    assert _events(single[child]) == _events(single[parent])


def test_the_single_flow_run_actually_found_something(single):
    """Zonder deze bewaking zou de gelijkheidstest ook slagen op twee lege
    lijsten, en dat is precies hoe zo'n test stil waardeloos wordt."""
    ap = _apneas(single["aasm_v3_breath"])
    assert len(ap) >= 3, f"maar {len(ap)} apneus in een montage met 4"


def test_the_dual_fallback_is_recorded_and_not_silently_performed(single):
    """Een profiel dat om twee sensoren vraagt op een montage met één moet dat
    melden, niet stil een ander algoritme draaien dan de gebruiker koos."""
    fb = single["aasm_v3_breath_dual"]["meta"].get("dual_sensor_fallback") or {}
    assert fb.get("requested") is True
    assert fb.get("performed") is False
    assert fb.get("reason") == "single_flow_channel"


# ─────────────────────────────────────────────────────────────────
#  Waar de as wél iets kan: toevoegen, nooit weghalen
# ─────────────────────────────────────────────────────────────────

def test_the_thermistor_is_the_apnea_sensor_so_the_test_measures_what_it_claims(dual):
    fc = dual["aasm_v3_breath_dual"]["meta"]["flow_channels"]
    assert fc["apnea_sensor"] == "Flow Th."
    assert fc["thermistor_check"]["usable"] is True, fc["thermistor_check"]


def test_the_second_sensor_only_adds_apneas(dual):
    """De kern van het additieve ontwerp: bij twijfel blijft een apneu staan."""
    assert len(_apneas(dual["aasm_v3_breath_dual"])) >= len(_apneas(dual["aasm_v3_breath"]))


def test_and_here_it_really_does_add_one(dual):
    """De montage heeft één apneu die alleen de neusdruk ziet. Vindt het duale
    profiel die niet, dan is de tweede pas een dure no-op."""
    extra = len(_apneas(dual["aasm_v3_breath_dual"])) - len(_apneas(dual["aasm_v3_breath"]))
    assert extra >= 1, "de druk-only apneu is niet toegevoegd"
    cd = dual["aasm_v3_breath_dual"]["respiratory"].get("dual_sensor_apnea") or {}
    assert cd.get("n_pressure_only", 0) >= 1, cd


def test_the_breath_segmentation_runs_on_the_nasal_pressure(dual):
    """De val uit het ontwerpdocument, gemeten in plaats van beredeneerd.

    Beide kanalen ademen op 0,25 Hz, dus een teller alleen zegt niets. Wat het
    wél zegt: de ademteugtelling van het duale profiel is gelijk aan die van
    zijn enkelsensor-ouder, terwijl hun apneukanaal identiek is en alleen de
    referentie verschuift. Zou de segmentatie de referentie volgen, dan lag
    hier een verschil.
    """
    a = dual["aasm_v3_breath"]["respiratory"]["breath_analysis"]
    b = dual["aasm_v3_breath_dual"]["respiratory"]["breath_analysis"]
    assert a.get("n_breaths", 0) > 10, a
    assert b["n_breaths"] == a["n_breaths"]


def test_the_hypopneas_still_come_from_the_graded_detector(dual):
    """Het duale profiel mag de hypopneedetector niet stil terugzetten."""
    for name in ("aasm_v3_breath", "aasm_v3_breath_dual"):
        diag = dual[name]["respiratory"].get("breath_detector") or {}
        assert diag, f"{name}: geen gegradeerde diagnostiek — draaide de envelope-detector?"
        assert "n_candidates" in diag, diag
