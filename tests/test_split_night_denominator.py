"""De segmentindices van een split-night delen door DEZELFDE noemer als de nacht.

Sinds 0.38.0 delen de klinische profielen de AHI-familie door de volle
slaaptijd (`index_denominator="sleep"`). `split_night.segment_summaries` riep
`_compute_summary` aan zonder noemer en viel dus terug op de default
`"artifact_free_sleep"`: de diagnostische AHI van een split-night deelde door
een kleinere slaaptijd dan de nacht-AHI waar hij naast staat. Gevonden bij de
codelezing van 10-11-2026.

De segmentfuncties nemen nu `denominator` aan (default het oude gedrag, voor
wie ze rechtstreeks aanroept) en de pijplijn geeft de noemer van het profiel
door.
"""
import pytest

from psgscoring import split_night as sn

E = 30.0
BREUK_S = 3600.0
HYPNO = ["N2"] * 240                       # twee uur slaap
ARTEFACT = list(range(60))              # het eerste halfuur


def _events():
    """Tien hypopneus in het tweede halfuur van het diagnostische deel."""
    return [{"type": "hypopnea", "onset_s": 1900.0 + i * 120.0,
             "duration_s": 20.0, "stage": "N2", "confidence": 0.9}
            for i in range(10)]


def test_de_default_blijft_de_artefactvrije_slaap():
    seg = sn.segment_indices(_events(), HYPNO, BREUK_S, E, ARTEFACT)
    assert seg["diagnostic"]["sleep_h"] == pytest.approx(0.5)
    assert seg["diagnostic"]["ahi"] == pytest.approx(20.0)


def test_met_de_noemer_sleep_telt_het_hele_segment():
    seg = sn.segment_indices(_events(), HYPNO, BREUK_S, E, ARTEFACT,
                             denominator="sleep")
    assert seg["diagnostic"]["sleep_h"] == pytest.approx(1.0)
    assert seg["diagnostic"]["ahi"] == pytest.approx(10.0)


def test_de_losse_noemer_volgt_dezelfde_keuze():
    h = sn.segment_denominator_h(HYPNO, BREUK_S, E, ARTEFACT,
                                 denominator="sleep")
    assert h["diagnostic"] == pytest.approx(1.0)
    assert h["therapeutic"] == pytest.approx(1.0)


@pytest.mark.parametrize("functie, extra, sleutel", [
    ("segment_arousals", ([{"onset_s": 2000.0, "type": "respiratory"}],),
     "arousal_index"),
    ("segment_plm", ([{"onset_s": 2000.0, "is_plm": True}],), "plm_index"),
])
@pytest.mark.parametrize("noemer, verwacht", [
    ("sleep", 1.0),                    # één gebeurtenis in één uur slaap
    ("artifact_free_sleep", 2.0),      # dezelfde, op het artefactvrije halfuur
])
def test_arousal_en_plm_per_deel_delen_door_dezelfde_uren(
        functie, extra, sleutel, noemer, verwacht):
    uit = getattr(sn, functie)(*extra, HYPNO, BREUK_S, E, ARTEFACT,
                               denominator=noemer)
    assert uit["diagnostic"][sleutel] == pytest.approx(verwacht)


def test_de_rdi_per_deel_gebruikt_dezelfde_noemer():
    uit = sn.segment_rdi(_events(), [2000.0], HYPNO, BREUK_S, E, ARTEFACT,
                         denominator="sleep")
    assert uit["diagnostic"]["rdi"] == pytest.approx(11.0)


@pytest.mark.slow
def test_de_pijplijn_geeft_de_noemer_van_het_profiel_door():
    """Op een klinisch profiel zijn de twee helften samen de nacht."""
    mne = pytest.importorskip("mne")
    import psgscoring
    from tests.test_split_night_indices_reach_the_output import (
        BREUK_S as BREUK,
    )
    from tests.test_split_night_indices_reach_the_output import (
        _split_night_raw,
    )
    raw, hypno = _split_night_raw(mne)
    uit = psgscoring.run_pneumo_analysis(
        raw, hypno=hypno, scoring_profile="aasm_v3_rec",
        artifact_epochs=list(range(10, 50)),
        split_night="manual", split_night_breakpoint_s=BREUK)
    nacht = uit["respiratory"]["summary"]["index_denominator_h"]
    split = uit["split_night"]
    seg = split["segments"]
    samen = seg["diagnostic"]["sleep_h"] + seg["therapeutic"]["sleep_h"]
    assert samen == pytest.approx(nacht, abs=0.02)
    # Elke familie per deel deelt door dezelfde uren als `segments`: de
    # samenvatting rechtstreeks, de andere via aantal / index.
    d = seg["diagnostic"]["sleep_h"]
    assert split["summaries"]["diagnostic"]["index_denominator_h"] == pytest.approx(d, abs=0.01)
    for familie, teller, index in (("arousal", "n_arousals", "arousal_index"),
                                   ("plm", "n_plm", "plm_index")):
        blok = split[familie]["diagnostic"]
        if blok.get(teller) and blok.get(index):
            assert blok[teller] / blok[index] == pytest.approx(d, rel=0.03), familie
    rdi = split["rdi"]["diagnostic"]
    assert rdi["rdi"] == pytest.approx(
        seg["diagnostic"]["ahi"] + (rdi.get("rera_index") or 0.0), abs=0.11)
