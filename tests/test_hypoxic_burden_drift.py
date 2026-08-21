"""Hypoxic burden mag niet meebewegen met een dalende basislijn.

Aanleiding: de melding dat de burden "onderschat bij aanhoudende hypoxemie".
Gemeten klopt die richting NIET — een vlakke basislijn van 85 % geeft exact
dezelfde burden als 96 % bij dezelfde dipdiepte, en dat is correct: de maat van
Azarbarzin is per definitie oppervlak onder de EIGEN basislijn.

Wat er wel misgaat is drift. `baseline = max(lokaal, globaal)` legt het globale
95e percentiel van de héle nacht als ondergrens onder elk event, dus events laat
in de nacht worden gemeten tegen een basislijn van vroeg in de nacht. Op een
trend van 94 % naar 82 % verdubbelt de burden bijna zonder dat er iets aan de
events verandert — precies het beeld van COPD of obesitas-hypoventilatie.

De code waarschuwde hier zelf voor in een v0.4.4-review. `local_baseline_only`
is die poort, expliciet in plaats van als drempel op 88 %.
"""

import numpy as np
import pytest

from psgscoring.spo2 import compute_hypoxic_burden

SF, DUR_S = 4.0, 8 * 3600
N = int(DUR_S * SF)
T = np.arange(N) / SF
HYPNO = ["N2"] * (DUR_S // 30)


def _nacht(baseline_fn, diepte=5.0, n=60, ev_dur=25.0):
    """SpO2 met n identieke desaturaties op een gegeven basislijnverloop."""
    s = baseline_fn(T).astype(float)
    events = []
    for i in range(n):
        t0 = 300 + i * 120
        a, b = int(t0 * SF), int((t0 + ev_dur) * SF)
        herstel = int(20 * SF)
        s[a:b] -= diepte
        s[b:b + herstel] -= np.linspace(diepte, 0, herstel)
        events.append({"onset_s": float(t0), "duration_s": ev_dur,
                       "desaturation_pct": diepte})
    return s, events


VLAK_HOOG = lambda t: np.full(len(t), 96.0)
VLAK_LAAG = lambda t: np.full(len(t), 85.0)
DRIFT     = lambda t: 94.0 - 12.0 * t / t[-1]


def _hb(fn, **kw):
    s, ev = _nacht(fn)
    return compute_hypoxic_burden(s, SF, ev, HYPNO, **kw)["hypoxic_burden"]


# ─────────────────────────────────────────────────────────────
#  Wat NIET het probleem is
# ─────────────────────────────────────────────────────────────

def test_a_flat_low_baseline_gives_the_same_burden_as_a_normal_one():
    """De gemelde "onderschatting bij lage basislijn" bestaat niet. Dezelfde
    dipdiepte op 85% geeft dezelfde burden als op 96% — correct, want de maat
    is oppervlak onder de eigen basislijn."""
    assert _hb(VLAK_LAAG) == pytest.approx(_hb(VLAK_HOOG), rel=0.02)


def test_that_is_a_property_of_the_measure_not_a_bug():
    """Wel het vastleggen waard: de burden is blind voor het ABSOLUTE
    saturatieniveau. Een daling van 85 naar 80 telt even zwaar als van 96 naar
    91, terwijl de dissociatiecurve onder 90% veel steiler loopt. Dat is de
    definitie van Azarbarzin, geen implementatiefout — maar wie de burden
    klinisch leest, moet het weten."""
    assert _hb(VLAK_LAAG) == pytest.approx(_hb(VLAK_HOOG), rel=0.02)


# ─────────────────────────────────────────────────────────────
#  Wat het wél is
# ─────────────────────────────────────────────────────────────

def test_a_drifting_baseline_inflates_the_burden():
    """Zonder de vlag: dezelfde events, bijna dubbele burden."""
    assert _hb(DRIFT) > 1.6 * _hb(VLAK_HOOG)


def test_the_flag_removes_the_drift_sensitivity():
    """Met de vlag komt de drift-nacht op dezelfde burden uit als de vlakke."""
    assert _hb(DRIFT, local_baseline_only=True) == pytest.approx(
        _hb(VLAK_HOOG), rel=0.05)


def test_the_flag_changes_nothing_on_a_stable_recording():
    """De prijs mag niet zijn dat stabiele nachten verschuiven."""
    for fn in (VLAK_HOOG, VLAK_LAAG):
        assert _hb(fn, local_baseline_only=True) == pytest.approx(_hb(fn), rel=0.01)


# ─────────────────────────────────────────────────────────────
#  De vlag zelf
# ─────────────────────────────────────────────────────────────

def test_no_profile_turns_it_on_by_default():
    """Dit verschuift een gepubliceerde grootheid op elke opname met drift."""
    from psgscoring.profiles import PROFILES
    aan = [n for n, p in PROFILES.items()
           if p.post_processing.hypoxic_burden_local_baseline]
    assert aan == [], f"profielen met de vlag aan: {aan}"


def test_the_flag_reaches_the_dict_the_pipeline_reads():
    """De dataclass-naam is niet de legacy-sleutel — twee keer eerder is een
    veld gepatcht dat niemand las."""
    from psgscoring.constants import SCORING_PROFILES
    assert SCORING_PROFILES["aasm_v3_rec"]["HYPOXIC_BURDEN_LOCAL_BASELINE"] is False


# ══════════════════════════════════════════════════════════════
# De gepubliceerde definitie — `baseline_method="azarbarzin"`
# ══════════════════════════════════════════════════════════════
#
# Azarbarzin et al. (Eur Heart J 2019): "the maximum SpO2 during the 100
# seconds before the end of the event is considered as the pre-event baseline
# oxygen saturation", en het zoekvenster is "the interval between the
# pre-event and post-event maximum oxygen saturation values", afgeleid uit het
# ensemble-gemiddelde.
#
# Onze twee bestaande paden wijken daar allebei van af, in tegengestelde
# richting. Zie docs/hypoxic_burden_bevinding.md.

def test_azarbarzin_baseline_comes_from_before_the_event_end():
    """De basislijn is het MAXIMUM van de 100 s vóór het eventeinde.

    Constructie: de saturatie DAALT geleidelijk in de aanloop naar het event,
    van 97 % tot 92 %. De gepubliceerde basislijn is het maximum over de hele
    100 s ervoor en pakt dus de 97 %; een basislijn die alleen naar de rand van
    het zoekvenster kijkt, pakt de 92 % die daar toevallig staat.

    Zonder dalende aanloop vallen beide samen -- na de venstercorrectie begint
    het venster immers vóór het eventeinde, waar de saturatie nog hoog is. Een
    fixture met vlakke aanloop meet dit verschil dus niet.
    """
    import numpy as np

    from psgscoring.spo2 import compute_hypoxic_burden

    sf = 1.0
    n = int(4 * 3600 * sf)
    spo2 = np.full(n, 97.0)
    events = []
    for k in range(20):
        t0 = 300 + k * 150
        end = t0 + 30
        # geleidelijke daling 97 -> 92 over de 100 s vóór het eventeinde
        pre = np.linspace(97.0, 92.0, int(100 * sf))
        spo2[int((end - 100) * sf):int(end * sf)] = pre
        spo2[int(end * sf):int((end + 40) * sf)] = 88.0
        spo2[int((end + 40) * sf):int((end + 90) * sf)] = 93.0
        events.append({"onset_s": float(t0), "duration_s": 30.0})
    hypno = ["N2"] * int(n / sf / 30)

    spec = compute_hypoxic_burden(spo2, sf, events, hypno,
                                  baseline_method="azarbarzin")
    ens = compute_hypoxic_burden(spo2, sf, events, hypno,
                                 baseline_method="ensemble")
    assert spec["baseline_method"] == "azarbarzin"
    assert spec["hypoxic_burden"] is not None
    assert ens["hypoxic_burden"] is not None
    assert spec["hypoxic_burden"] > ens["hypoxic_burden"], (
        "de gepubliceerde basislijn (max vóór het eventeinde) hoort hoger te "
        f"liggen dan die uit het venster: {spec['hypoxic_burden']} vs "
        f"{ens['hypoxic_burden']}"
    )


def test_search_window_left_edge_precedes_event_end():
    """De linkerflank van het zoekvenster ligt vóór het eventeinde.

    Bij dicht opeenvolgende events bevat de ensemble-span van +/-60 s meerdere
    cycli; de nadir landt dan in een latere cyclus en het maximum ervóór kan
    NA het eventeinde liggen. Gemeten op mesa-sleep-1374: [+29,4, +60,0] s.
    Het venster omsluit dan niet meer het event waar het bij hoort.
    """
    import numpy as np

    from psgscoring.spo2 import _ensemble_search_window

    sf = 1.0
    n = int(2 * 3600 * sf)
    spo2 = np.full(n, 96.0)
    events = []
    for k in range(60):          # elke 45 s -- dicht opeen
        t0 = 200 + k * 45
        end = t0 + 20
        spo2[int(end * sf):int((end + 15) * sf)] = 89.0
        events.append({"onset_s": float(t0), "duration_s": 20.0})
    left, right, _, _ = _ensemble_search_window(spo2, sf, events)
    assert left is not None
    assert left <= 0.0, f"linkerflank op {left:+.1f}s, dus NA het eventeinde"
    assert right > left


def test_the_summary_says_which_definition_produced_the_number():
    """De burden kan op vier manieren berekend worden.

    Op dezelfde opname lopen die uiteen met een factor 0,29 tot 2,34
    (docs/hypoxic_burden_bevinding.md). Wie het getal leest -- het PDF-rapport,
    een export, een vergelijking tussen centra -- moet kunnen zien welke
    definitie eronder zit, want de gepubliceerde afkapwaarden gelden alleen
    voor de gepubliceerde definitie.

    Dit is dezelfde afleverkant-eis als bij `plm_time_base` en
    `lgbm_available`: een keuze die het getal bepaalt, hoort naast het getal
    te staan.
    """
    import numpy as np
    import pytest

    mne = pytest.importorskip("mne")
    import psgscoring

    # De eerste versie van deze fixture (flow x0,3 gedurende 20 s) leverde NUL
    # respiratoire events op, waardoor de test stilletjes skipte en dus niets
    # mat. x0,1 gedurende 25 s met een desaturatie van 5 % geeft 11 events en
    # een burden van 7,14 -- pas dan is er iets om te labelen.
    sf, minutes = 32.0, 14
    n = int(sf * 60 * minutes)
    t = np.arange(n) / sf
    rng = np.random.default_rng(3)
    flow = np.sin(2 * np.pi * 0.25 * t)
    spo2 = np.full(n, 97.0)
    for start in range(90, 60 * minutes - 90, 60):
        flow[int(start * sf):int((start + 25) * sf)] *= 0.1
        e = start + 25
        spo2[int(e * sf):int((e + 20) * sf)] = 92.0
    info = mne.create_info(["Resp nasal", "SaO2", "EEG C4-M1", "EMG chin"],
                           sf, ["misc", "misc", "eeg", "emg"])
    raw = mne.io.RawArray(
        np.vstack([flow, spo2, rng.normal(0, 20e-6, n), rng.normal(0, 5e-6, n)]),
        info, verbose=False)
    hypno = ["N2"] * int(np.ceil(raw.times[-1] / 30.0))
    out = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile="aasm_v3_rec")
    ss = (out.get("spo2") or {}).get("summary") or {}
    assert ss.get("hypoxic_burden"), (
        "fixture levert geen burden -- dan meet deze test niets; zie de "
        "opmerking bij de fixture")
    assert ss.get("hypoxic_burden_method"), (
        "het getal staat er zonder te zeggen welke definitie het opleverde")
    assert ss["hypoxic_burden_method"] in (
        "percentile", "ensemble", "azarbarzin",
        "percentile (ensemble fallback)")
