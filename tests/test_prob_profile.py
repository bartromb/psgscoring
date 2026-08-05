"""`aasm_v3_prob`: de arousal-as was de enige die niet gegradeerd was.

De ademteug-detector beoordeelt elk AASM-criterium gegradeerd — behalve de
arousal. ``p_arousal`` sprong naar ``arousal_weight`` (0,90) zodra er een
arousal in het koppelvenster lag, en omdat de bevestiging een noisy-OR is van
desaturatie en arousal kon ``p_confirm`` daarna nooit meer onder 0,90 komen,
hoe klein de desaturatie ook was. Een drempel in een verder continu model.

Aanleiding is een meting op PSG-IPA SN1: `aasm_v3_breath` markeert daar tien
events die géén van de twaalf scoorders markeerde, en vijf daarvan hebben een
desaturatie onder 1,2 % terwijl het laagste consensus-event op 2,16 % ligt.

Deze tests bewaken twee dingen: dat het nieuwe profiel doet wat het zegt, en
dat het puur additief is — geen bestaand profiel mag bewegen.
"""
from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES, PostProcessingRules


def test_the_defaults_are_the_old_behaviour():
    """De veilige faalstand: wie de velden niet zet, krijgt v0.13.0-gedrag."""
    pp = PostProcessingRules()
    assert pp.hypopnea_arousal_weight == 0.90
    assert pp.hypopnea_arousal_latency_grading is False


def test_no_existing_profile_moves():
    """Additief betekent additief."""
    moved = [n for n, p in PROFILES.items()
             if n != "aasm_v3_prob"
             and (p.post_processing.hypopnea_arousal_weight != 0.90
                  or p.post_processing.hypopnea_arousal_latency_grading)]
    assert moved == [], f"profielen met een gewijzigde arousal-as: {moved}"


def test_the_new_profile_differs_only_on_the_arousal_axis():
    """Alles behalve die twee assen moet gelijk zijn aan aasm_v3_breath —
    anders meet een vergelijking tussen de twee iets anders dan de arousal."""
    import dataclasses
    a = PROFILES["aasm_v3_breath"].post_processing
    b = PROFILES["aasm_v3_prob"].post_processing
    diff = [f.name for f in dataclasses.fields(a)
            if getattr(a, f.name) != getattr(b, f.name)]
    assert sorted(diff) == ["hypopnea_arousal_latency_grading",
                            "hypopnea_arousal_weight"], diff
    # en de hypopnee-regels zelf zijn identiek: het is dezelfde AASM-regel
    assert PROFILES["aasm_v3_breath"].hypopnea == PROFILES["aasm_v3_prob"].hypopnea


def test_the_new_profile_uses_the_graded_detector():
    p = PROFILES["aasm_v3_prob"].post_processing
    assert p.hypopnea_detector == "breath_graded"
    assert p.hypopnea_arousal_weight < 0.90, "anders is er niets veranderd"
    assert p.hypopnea_arousal_latency_grading is True


def test_an_arousal_alone_demands_more_than_it_did():
    """De kern van de wijziging, als rekensom.

    p_confirm = 1 - (1 - p_desat)(1 - p_arousal). Zonder desaturatie is dat
    gelijk aan p_arousal, en p = p_flow * p_dur * p_confirm. Bij gewicht 0,90
    haalt een arousal-only event de drempel van 0,50 al bij p_flow * p_dur >
    0,56 — vrijwel elke kandidaat. Bij 0,70 moet dat product boven 0,71.
    """
    p = PROFILES["aasm_v3_prob"].post_processing
    needed_new = p.hypopnea_strictness / p.hypopnea_arousal_weight
    needed_old = p.hypopnea_strictness / 0.90
    assert needed_new > 0.70, "een arousal alleen mag geen vrijkaart meer zijn"
    assert needed_new > needed_old


def test_the_axis_is_reachable_through_the_legacy_dict():
    """De pijplijn leest de legacy-dict, niet het dataclass-veld. Twee keer
    eerder is een meting stukgelopen op een sleutel die niemand las."""
    d = SCORING_PROFILES["aasm_v3_prob"]
    assert d["HYPOPNEA_AROUSAL_WEIGHT"] == 0.70
    assert d["HYPOPNEA_AROUSAL_LATENCY"] is True
    assert SCORING_PROFILES["aasm_v3_breath"]["HYPOPNEA_AROUSAL_WEIGHT"] == 0.90
    assert SCORING_PROFILES["aasm_v3_breath"]["HYPOPNEA_AROUSAL_LATENCY"] is False


def test_the_profile_is_exploratory_not_clinical():
    """Het werkpunt is niet opnieuw afgeleid; het hoort niet in de klinische
    familie tot de validatie dat draagt."""
    assert PROFILES["aasm_v3_prob"].family == "exploratory"


# ─────────────────────────────────────────────────────────────
#  aasm_v3_fusion — sensorovereenstemming als gewicht (kimi.doc §3.1)
# ─────────────────────────────────────────────────────────────

def test_fusion_differs_from_dual_only_on_the_weighting_axis():
    """Anders meet een vergelijking tussen de twee iets anders dan de fusie."""
    import dataclasses
    a = PROFILES["aasm_v3_dual"].post_processing
    b = PROFILES["aasm_v3_fusion"].post_processing
    diff = [f.name for f in dataclasses.fields(a)
            if getattr(a, f.name) != getattr(b, f.name)]
    assert diff == ["thermistor_agreement_weighting"], diff
    assert PROFILES["aasm_v3_dual"].hypopnea == PROFILES["aasm_v3_fusion"].hypopnea


def test_fusion_keeps_the_thermistor_additive():
    """Falsifiëren met een sensor die je niet vertrouwt is hoe een echte
    opname 83 centrale apneus verloor. De fusie mag toevoegen, nooit afwijzen."""
    p = PROFILES["aasm_v3_fusion"].post_processing
    assert p.dual_sensor_apnea is True
    assert p.dual_sensor_corroboration is False


def test_no_existing_profile_gets_the_weighting():
    on = [n for n, p in PROFILES.items()
          if n != "aasm_v3_fusion" and p.post_processing.thermistor_agreement_weighting]
    assert on == [], f"profielen met ongevraagde weging: {on}"


def test_the_weighting_axis_is_reachable_through_the_legacy_dict():
    assert SCORING_PROFILES["aasm_v3_fusion"]["THERMISTOR_AGREEMENT_WEIGHTING"] is True
    assert SCORING_PROFILES["aasm_v3_dual"]["THERMISTOR_AGREEMENT_WEIGHTING"] is False


def test_both_new_profiles_are_exploratory():
    """Geen van beide is gevalideerd; ze horen niet in de klinische familie.
    `aasm_v3_fusion` is bovendien op PSG-IPA principieel niet te meten — die
    montage heeft één flowkanaal."""
    for n in ("aasm_v3_prob", "aasm_v3_fusion"):
        assert PROFILES[n].family == "exploratory", n


def test_the_desat_width_axis_defaults_to_the_old_behaviour():
    assert PostProcessingRules().hypopnea_desat_width == 0.80
    moved = [n for n, p in PROFILES.items()
             if p.post_processing.hypopnea_desat_width != 0.80]
    assert moved == [], f"profielen met een gewijzigde desat-breedte: {moved}"
