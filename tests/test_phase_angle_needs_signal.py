"""Een fasehoek op een vlak effortsignaal is ruis, en beslist nu wél.

DE METING DIE DIT AANLEIDING GAF
--------------------------------
PSG-IPA, vijf opnames, apneus gekoppeld aan de scoorder binnen 10 s:

    mens \\ wij        centraal   obstructief
      centraal              15            60
      gemengd                2            55
      obstructief            1           153

Van de 75 menselijk-centrale apneus noemen wij er 60 obstructief. De andere
richting klopt bijna perfect (153 van 154). Dit is dus geen classificatiefout
maar een eenzijdige bias.

Per event uitgesplitst was de grootste enkele oorzaak:

    phase_angle                33x   <- deze test
    borderline_default_var     27x
    truly_flat_var             10x   (terecht centraal)
    low_effort_default_central  5x   (terecht centraal)

WAT ER MISGAAT
--------------
Regel 0 vuurt vóór alle andere: fasehoek >= 45 graden -> obstructief, met
confidence tot 0,97. Er staat geen amplitudedrempel onder.

Bij een centrale apneu bewegen thorax en abdomen per definitie nauwelijks. De
Hilbert-fase van twee bijna-vlakke signalen is die van RUIS, en ruis is niet in
fase. `_compute_phase_angle` is daar ook expliciet op ontworpen -- de docstring
zegt "ook wanneer de amplitude-envelop laag is". Dat is zinnig als je een
obstructief event met lage amplitude wilt vangen, en het is precies verkeerd
wanneer de lage amplitude JUIST het centrale kenmerk is.

DE REPARATIE
------------
Vertrouw de fasehoek niet wanneer er geen effortsignaal onder ligt. Onder
`EFFORT_ABSENT_RATIO` is afwezige inspanning de bevinding; de fase eroverheen
is dan geen tegenbewijs.

Achter `phase_angle_needs_effort`, default uit (werkregel 1).
"""
import numpy as np
import pytest

from psgscoring.classify import classify_apnea_type
from psgscoring.constants import EFFORT_ABSENT_RATIO

SF = 32.0


def _banden(n_s, amplitude, sf=SF, tegenfase=True, seed=0):
    """Thorax en abdomen. Bij `tegenfase` lopen ze 180 graden uit elkaar --
    op grote amplitude is dat paradox, op ruisniveau is het toeval."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_s * sf)) / sf
    th = amplitude * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.01, len(t))
    ab = amplitude * np.sin(2 * np.pi * 0.25 * t + (np.pi if tegenfase else 0.0))
    ab = ab + rng.normal(0, 0.01, len(t))
    return th, ab


def _classificeer(th, ab, baseline, **kw):
    o, e = int(60 * SF), int(80 * SF)
    return classify_apnea_type(
        onset_idx=o, end_idx=e, thorax_env=np.abs(th), abdomen_env=np.abs(ab),
        thorax_raw=th, abdomen_raw=ab, effort_baseline=baseline, sf=SF, **kw)


# ── Het gedrag dat moet blijven ───────────────────────────────────────────

def test_paradox_met_echte_amplitude_blijft_obstructief():
    """Een obstructief event heeft inspanning; daar hoort de fasehoek te
    beslissen en dat mag deze reparatie niet wegnemen."""
    th, ab = _banden(140.0, amplitude=1.0)
    sub, _c, det = _classificeer(th, ab, baseline=1.0,
                                 phase_angle_needs_effort=True)
    assert sub == "obstructive", det
    assert "phase_angle" in str(det.get("decision_reason")), det


# ── Het gedrag dat moet veranderen ────────────────────────────────────────

def test_fase_op_een_vlak_signaal_beslist_niet_meer():
    """De 33 fouten uit de meting: amplitude ver onder de effortdrempel, en
    toch een fasehoek die obstructief roept."""
    th, ab = _banden(140.0, amplitude=0.02)      # 2 % van de basislijn
    sub_uit, _c1, det_uit = _classificeer(th, ab, baseline=1.0,
                                          phase_angle_needs_effort=False)
    sub_aan, _c2, det_aan = _classificeer(th, ab, baseline=1.0,
                                          phase_angle_needs_effort=True)
    assert "phase_angle" in str(det_uit.get("decision_reason")), (
        "de fixture reproduceert de fout niet: regel 0 vuurt niet eens")
    assert sub_uit == "obstructive"
    assert "phase_angle" not in str(det_aan.get("decision_reason")), (
        f"regel 0 vuurt nog steeds op een vlak signaal: {det_aan}")
    assert "paradox_corr" not in str(det_aan.get("decision_reason")), (
        f"de fout verplaatste naar regel 1 (paradoxcorrelatie): {det_aan}")
    assert sub_aan == "central", det_aan


def test_de_reden_vermeldt_waarom_de_fase_genegeerd_werd():
    """Een genegeerde regel moet zichtbaar zijn, anders is de uitkomst niet te
    reviewen."""
    th, ab = _banden(140.0, amplitude=0.02)
    _s, _c, det = _classificeer(th, ab, baseline=1.0,
                                phase_angle_needs_effort=True)
    assert det.get("phase_angle_ignored") is True, det
    assert det.get("phase_angle_deg") is not None, (
        "de hoek zelf hoort bewaard te blijven, ook als hij niet beslist")


# De effortverhouding is de GEMIDDELDE ENVELOPPE gedeeld door de basislijn, niet
# de amplitude zelf: een sinus van amplitude a heeft een gemiddelde absolute
# waarde van 2a/pi ~ 0,64a. Vandaar dat 0,25 amplitude uitkomt op ratio 0,159 en
# dus ONDER de drempel valt. De fixture noemt daarom de gemeten ratio.
@pytest.mark.parametrize("amplitude,verwacht_fase_telt", [
    (1.00, True),    # ratio 0,64 -- volle inspanning
    (0.50, True),    # ratio 0,32 -- ruim boven de drempel
    (0.35, True),    # ratio 0,22 -- net boven EFFORT_ABSENT_RATIO
    (0.25, False),   # ratio 0,16 -- eronder: fase is ruis
    (0.02, False),
])
def test_de_grens_ligt_op_de_effortdrempel(amplitude, verwacht_fase_telt):
    th, ab = _banden(140.0, amplitude=amplitude)
    _s, _c, det = _classificeer(th, ab, baseline=1.0,
                                phase_angle_needs_effort=True)
    genegeerd = bool(det.get("phase_angle_ignored"))
    assert genegeerd != verwacht_fase_telt, (
        f"amplitude {amplitude}: effort_ratio={det.get('effort_ratio')}, "
        f"drempel={EFFORT_ABSENT_RATIO}, genegeerd={genegeerd}")


def test_de_default_verandert_niets():
    """Werkregel 1: meten gaat vóór aanzetten."""
    th, ab = _banden(140.0, amplitude=0.02)
    sub, _c, det = _classificeer(th, ab, baseline=1.0)
    assert sub == "obstructive"
    assert not det.get("phase_angle_ignored")
