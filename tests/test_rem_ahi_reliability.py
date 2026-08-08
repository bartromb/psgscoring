"""Een REM-AHI over een halfuur REM is iets anders dan een REM-AHI.

Bij 22,5 minuten REM is één event al 2,7 per uur. Dat getal is wiskundig
correct en klinisch onbruikbaar: het leest als een meting terwijl het de
afronding van een handvol events is.

Dit is NIET hetzelfde als de noemer-bugs van vandaag. Daar bestond de index
niet en werd hij toch getoond; hier bestaat hij wel, maar is hij niet te
vertrouwen. Het verschil bepaalt de reactie: niet weglaten maar kwalificeren,
zodat de lezer het getal ziet én weet waar het op rust.

De grens van 30 minuten is niet nieuw verzonnen — `_compute_phenotypes`
hanteert hem al voordat het REM-predominant fenotype gesteld mag worden. Eén
drempel voor dezelfde vraag.
"""

import pytest

from psgscoring.respiratory import MIN_STAGE_MIN_FOR_INDEX, _compute_summary


def _ev(n, stage):
    return [{"type": "hypopnea", "onset_s": 100.0 + i * 60, "duration_s": 20.0,
             "stage": stage, "confidence": 0.7} for i in range(n)]


def _hypno(rem_min, nrem_min=240):
    """Hypnogram met een gegeven hoeveelheid REM en NREM, in minuten."""
    return ["R"] * int(rem_min * 2) + ["N2"] * int(nrem_min * 2)


# ─────────────────────────────────────────────────────────────
#  De kwalificatie
# ─────────────────────────────────────────────────────────────

def test_little_rem_marks_the_rem_ahi_as_unreliable():
    s = _compute_summary(_ev(1, "R"), _hypno(rem_min=22.5))
    assert s["ahi_rem_reliable"] is False
    assert s["ahi_rem_caveat"] and "22" in s["ahi_rem_caveat"]


def test_the_number_itself_is_still_reported():
    """Kwalificeren is geen weglaten. Wie het getal wil narekenen moet het
    zien, met de slaaptijd erbij."""
    s = _compute_summary(_ev(1, "R"), _hypno(rem_min=22.5))
    assert s["ahi_rem"] is not None
    assert s["ahi_rem"] == pytest.approx(1 / (22.5 / 60), rel=0.05)
    assert s["rem_min"] == pytest.approx(22.5, abs=0.5)


def test_enough_rem_carries_no_caveat():
    s = _compute_summary(_ev(10, "R"), _hypno(rem_min=90))
    assert s["ahi_rem_reliable"] is True
    assert s["ahi_rem_caveat"] is None


@pytest.mark.parametrize("rem_min,betrouwbaar", [
    (0.0, False),
    (MIN_STAGE_MIN_FOR_INDEX - 1, False),
    (MIN_STAGE_MIN_FOR_INDEX, True),
    (MIN_STAGE_MIN_FOR_INDEX + 1, True),
])
def test_the_threshold_sits_where_it_says(rem_min, betrouwbaar):
    s = _compute_summary(_ev(2, "R"), _hypno(rem_min=rem_min))
    assert s["ahi_rem_reliable"] is betrouwbaar


def test_the_threshold_is_the_same_one_the_phenotype_uses():
    """Twee drempels voor dezelfde vraag lopen vroeg of laat uiteen."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "psgscoring" / "pipeline.py").read_text()
    assert "rem_min >= 30" in src, (
        "de fenotype-poort is verzet; MIN_STAGE_MIN_FOR_INDEX hoort mee")
    assert MIN_STAGE_MIN_FOR_INDEX == 30.0


# ─────────────────────────────────────────────────────────────
#  Geen REM is nog iets anders
# ─────────────────────────────────────────────────────────────

def test_no_rem_at_all_gives_no_index_and_no_caveat_about_reliability():
    """Zonder REM bestaat de index niet — dat is de noemer-regel, niet de
    betrouwbaarheidsregel. Beide signalen horen te kloppen."""
    s = _compute_summary(_ev(10, "N2"), _hypno(rem_min=0))
    assert s["ahi_rem"] is None
    assert s["ahi_rem_reliable"] is False
    assert s["rem_min"] == 0


def test_the_nrem_side_is_reported_too():
    s = _compute_summary(_ev(10, "N2"), _hypno(rem_min=90, nrem_min=240))
    assert s["nrem_min"] == pytest.approx(240, abs=1)
    assert s["ahi_nrem"] is not None


# ─────────────────────────────────────────────────────────────
#  Env-overrides horen zichtbaar te zijn
# ─────────────────────────────────────────────────────────────

def test_the_active_overrides_are_reported_in_the_metadata():
    """Dezelfde profielnaam mag op twee machines niet stil iets anders
    betekenen."""
    import importlib
    import os

    import psgscoring.pipeline as P
    os.environ["PSGSCORING_BREATH_AROUSAL_LATENCY"] = "1"
    try:
        importlib.reload(P)
        assert P._breath_env_overrides() == {"arousal_latency_grading": True}
    finally:
        os.environ.pop("PSGSCORING_BREATH_AROUSAL_LATENCY", None)
        importlib.reload(P)


def test_no_overrides_is_an_empty_dict_not_none():
    """Het rapport toont de regel alleen wanneer er iets aan staat; een lege
    dict is dan makkelijker dan None."""
    import psgscoring.pipeline as P
    assert P._breath_env_overrides() == {}


def test_the_pipeline_puts_them_in_the_metadata():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "psgscoring" / "pipeline.py").read_text()
    assert '"env_overrides":   _breath_env_overrides()' in src
