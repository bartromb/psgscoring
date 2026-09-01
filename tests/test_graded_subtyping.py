"""Vormmaten wegen naar rato van het signaal eronder, niet aan of uit.

WAT DE HARDE POORT LEERDE
-------------------------
`phase_angle_needs_effort` zet fasehoek, paradox en ruwe beweging UIT zodra de
effortverhouding onder 0,20 komt. Gemeten op PSG-IPA (286 gekoppelde apneus):

    arm                recall centraal  recall obstructief  gebalanceerd  kappa
    uit (huidig)              20,0 %            99,4 %          59,7 %   0,139
    aan (harde poort)         98,7 %            44,8 %          71,7 %   0,250

De diagnose klopte -- de vormmaten dreven de misclassificatie -- maar de poort
ruilt de ene bias voor de andere. 85 van de 154 menselijk-obstructieve apneus
werden centraal.

WAAROM EEN DREMPEL HIER NIET KAN WERKEN
---------------------------------------
`EFFORT_ABSENT_RATIO` is geen natuurlijke grens. Een obstructieve apneu met een
slecht zittende band heeft een lage effortverhouding en toch echte paradox; een
centrale apneu met hartpulsatie heeft een iets hogere verhouding en toch alleen
ruis. Een harde schakelaar op een grootheid die geleidelijk verloopt, kan
alleen kiezen tussen twee verkeerde uitersten.

DE GEGRADEERDE VORM
-------------------
Hetzelfde wat dit pakket met AASM Rule 1A deed: geen keten van ja/nee-sneden
maar een gewicht dat meeloopt. Het BEWIJSGEWICHT van elke vormmaat schaalt met
hoeveel signaal eronder ligt:

    w = clip(effort_ratio / EFFORT_PRESENT_RATIO, 0, 1)

Bij volle inspanning (>= 0,40) telt de fasehoek volledig; bij afwezige
inspanning (0) telt hij niet; daartussen geleidelijk. De drempels van de regels
zelf blijven staan -- alleen hun ZEGGINGSKRACHT schaalt.

Het werkpunt van die schaal is een profielveld, zodat de meting hem kan ijken
tegen bovenstaande matrix.
"""
import numpy as np
import pytest

from psgscoring.classify import shape_evidence_weight
from psgscoring.constants import EFFORT_ABSENT_RATIO, EFFORT_PRESENT_RATIO


# ── Het gewicht zelf ──────────────────────────────────────────────────────

def test_volle_inspanning_geeft_vol_gewicht():
    assert shape_evidence_weight(EFFORT_PRESENT_RATIO) == pytest.approx(1.0)
    assert shape_evidence_weight(0.80) == pytest.approx(1.0)


def test_afwezige_inspanning_geeft_geen_gewicht():
    assert shape_evidence_weight(0.0) == pytest.approx(0.0)


def test_het_gewicht_loopt_geleidelijk_en_monotoon():
    """Geen sprong: dat is het hele verschil met de harde poort."""
    xs = np.linspace(0.0, 0.6, 25)
    ws = [shape_evidence_weight(x) for x in xs]
    assert all(b >= a - 1e-9 for a, b in zip(ws, ws[1:])), ws
    # en er zit geen enkele stap groter dan wat de rasterafstand rechtvaardigt
    sprongen = np.diff(ws)
    assert sprongen.max() < 0.12, f"grootste sprong {sprongen.max():.3f}"


def test_op_de_oude_drempel_is_het_gewicht_niet_nul_maar_gedeeltelijk():
    """De harde poort maakte er 0 van, en dat kostte 85 obstructieve apneus."""
    w = shape_evidence_weight(EFFORT_ABSENT_RATIO)
    assert 0.2 < w < 0.8, (
        f"op de oude poortgrens is het gewicht {w:.2f}; niet 0 en niet 1")


@pytest.mark.parametrize("schaal", [0.5, 1.0, 2.0])
def test_de_schaal_is_instelbaar(schaal):
    """Het werkpunt moet te ijken zijn tegen de PSG-IPA-matrix."""
    w = shape_evidence_weight(0.20, scale=schaal)
    assert 0.0 <= w <= 1.0
    if schaal < 1.0:
        assert w > shape_evidence_weight(0.20, scale=1.0)


# ── Wat het in de classificatie doet ──────────────────────────────────────

SF = 32.0


def _banden(amp, seed=0, n_s=140.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_s * SF)) / SF
    th = amp * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.01, len(t))
    ab = amp * np.sin(2 * np.pi * 0.25 * t + np.pi) + rng.normal(0, 0.01, len(t))
    return th, ab


def _kl(amp, **kw):
    from psgscoring.classify import classify_apnea_type
    th, ab = _banden(amp)
    return classify_apnea_type(
        onset_idx=int(60 * SF), end_idx=int(80 * SF),
        thorax_env=np.abs(th), abdomen_env=np.abs(ab),
        thorax_raw=th, abdomen_raw=ab, effort_baseline=1.0, sf=SF, **kw)


def test_echte_paradox_blijft_obstructief():
    sub, _c, det = _kl(1.0, shape_evidence_grading=True)
    assert sub == "obstructive", det


def test_ruis_op_een_vlak_signaal_wordt_centraal():
    sub, _c, det = _kl(0.02, shape_evidence_grading=True)
    assert sub == "central", det


def test_het_gewicht_staat_in_de_uitvoer():
    """Zonder dit is niet te reviewen waarom een event zo geclassificeerd is."""
    _s, _c, det = _kl(0.15, shape_evidence_grading=True)
    assert "shape_weight" in det, det
    assert 0.0 <= det["shape_weight"] <= 1.0


def test_de_default_verandert_niets():
    sub_uit, _c, det = _kl(0.02)
    assert sub_uit == "obstructive"
    assert "shape_weight" not in det
