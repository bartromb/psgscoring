"""Bilaterale beenbewegingen: de manual zegt 5 s, de code zei 0,5 s.

DE REGEL
--------
AASM v3, hoofdstuk beenbewegingen: bewegingen aan twee verschillende benen die
minder dan **5 seconden** uit elkaar beginnen, tellen als ÉÉN beweging. Het
venster is onset-tot-onset gedefinieerd, niet op overlap.

WAT ER STOND
------------
`BILATERAL_WIN_S = 0.5`, een factor tien te krap. Bewegingen tussen 0,5 en 5 s
uit elkaar werden daardoor als TWEE bewegingen geteld. Dat inflateert de
LM-telling, en via de reekstelling ook de PLMI -- en het maakt bovendien
PLM-reeksen mogelijk die uit dubbeltellingen van hetzelfde bilaterale event
bestaan, want het PLM-interval begint bij 5 s.

Dat laatste is de scherpe rand: een bilaterale beweging met 2 s tussen de benen
levert twee onsets die 2 s uit elkaar liggen. Die vallen NIET in het
PLM-interval (5-90 s) en breken dus geen reeks -- maar ze verhogen wel `n_lm`
en daarmee de LM-index.

DE REPARATIE STAAT ACHTER EEN VLAG
----------------------------------
`plm_bilateral_window_s`, default 0,5 (het huidige gedrag). De manualwaarde 5,0
is een keuze die eerst gemeten wordt op beide cohorten. Zie werkregel 1 in
docs/AASM_v3_conformiteit.md.
"""
import pytest

from psgscoring.plm import BILATERAL_WIN_S, _merge_bilateral


def _lm(onset, dur=1.0, amp=20.0):
    return {"onset_s": onset, "duration_s": dur, "amplitude_uv": amp}


# ── De regel zelf ─────────────────────────────────────────────────────────

def test_de_manualwaarde_is_vijf_seconden():
    """Het venster is een REGEL uit de manual, geen afstelbare parameter."""
    from psgscoring.plm import BILATERAL_WIN_AASM_S
    assert BILATERAL_WIN_AASM_S == 5.0


@pytest.mark.parametrize("delta,samen", [
    (0.2, True),    # vrijwel gelijktijdig
    (0.4, True),
    (2.0, True),    # HIER zat de fout: 2 s is onder 5 s, dus één beweging
    (4.9, True),
    (5.1, False),   # boven het venster: twee bewegingen
    (30.0, False),
])
def test_het_venster_van_de_manual_voegt_samen_tot_vijf_seconden(delta, samen):
    uit = _merge_bilateral([_lm(100.0)], [_lm(100.0 + delta)],
                           window_s=5.0)
    assert len(uit) == (1 if samen else 2), (
        f"{delta} s uit elkaar: verwacht "
        f"{'één' if samen else 'twee'} beweging(en), kreeg {len(uit)}")


def test_het_oude_venster_blijft_bereikbaar():
    """Voor reproductie van eerdere metingen."""
    uit = _merge_bilateral([_lm(100.0)], [_lm(102.0)], window_s=0.5)
    assert len(uit) == 2


def test_de_default_is_nog_het_oude_gedrag():
    """Werkregel 1: meten gaat vóór aanzetten."""
    assert BILATERAL_WIN_S == 0.5
    assert len(_merge_bilateral([_lm(100.0)], [_lm(102.0)])) == 2


# ── Waar het klinisch op neerkomt ─────────────────────────────────────────

def test_een_nacht_met_bilaterale_bewegingen_telt_er_de_helft_minder():
    """Twintig bilaterale bewegingen, elk met 2 s tussen de benen.

    Onder de manualregel zijn dat twintig bewegingen. Onder het oude venster
    veertig -- en de LM-index verdubbelt daarmee.
    """
    links = [_lm(60.0 + 30 * i) for i in range(20)]
    rechts = [_lm(62.0 + 30 * i) for i in range(20)]
    assert len(_merge_bilateral(links, rechts, window_s=5.0)) == 20
    assert len(_merge_bilateral(links, rechts, window_s=0.5)) == 40


def test_de_samengevoegde_beweging_houdt_de_vroegste_onset():
    """De onset bepaalt het PLM-interval; die mag niet verschuiven."""
    uit = _merge_bilateral([_lm(100.0)], [_lm(103.0)], window_s=5.0)
    assert len(uit) == 1
    assert uit[0]["onset_s"] == 100.0
    assert uit[0]["bilateral"] is True


def test_elke_rechterbeweging_wordt_hoogstens_een_keer_gebruikt():
    """Twee linkerbewegingen vlak bij één rechter: de rechter mag niet twee
    keer meedoen, anders verdwijnt er een beweging uit de telling."""
    uit = _merge_bilateral([_lm(100.0), _lm(101.0)], [_lm(100.5)],
                           window_s=5.0)
    assert len(uit) == 2, [e["onset_s"] for e in uit]


# ── Bereikt de vlag zijn consument? ───────────────────────────────────────

def test_de_profielvlag_bereikt_de_samenvoeging():
    """Vier keer op één dag bleek een instelling zijn consument niet te halen,
    en elke keer zag de meting er geslaagd uit terwijl ze niets mat.

    De eerste versie van deze fixture spoot een gelijkspanningsstap in en vond
    NUL bewegingen bij beide vensters -- groen door leegte. Een EMG-burst is
    oscillatoir; de RMS van een constante is nul na middelingsverwijdering.
    """
    import numpy as np

    from psgscoring.plm import analyze_plm

    sf = 200.0
    n = int(600 * sf)
    rng = np.random.default_rng(3)
    basis = rng.normal(0, 1.0, n)
    links, rechts = basis.copy(), basis.copy()
    # Twintig bilaterale bewegingen met 2 s tussen de benen: onder de
    # manualregel één beweging, onder het huidige venster twee.
    for i in range(20):
        t0 = 60.0 + 25 * i
        a, b = int(t0 * sf), int((t0 + 1.0) * sf)
        links[a:b] += 40 * rng.normal(0, 1, b - a)
        c, d = int((t0 + 2.0) * sf), int((t0 + 3.0) * sf)
        rechts[c:d] += 40 * rng.normal(0, 1, d - c)
    hypno = ["N2"] * 20

    krap = (analyze_plm(links, rechts, sf, hypno,
                        bilateral_window_s=0.5).get("summary") or {})
    ruim = (analyze_plm(links, rechts, sf, hypno,
                        bilateral_window_s=5.0).get("summary") or {})
    assert krap.get("n_lm_total") == 40, krap
    assert ruim.get("n_lm_total") == 20, ruim


def test_de_env_override_werkt():
    """Zonder override is de manualarm niet in één run te scheiden."""
    import os

    from psgscoring.pipeline import _plm_bilateral_window_s

    assert _plm_bilateral_window_s({"PLM_BILATERAL_WINDOW_S": 0.5}) == 0.5
    os.environ["PSGSCORING_PLM_BILATERAL_WINDOW_S"] = "5.0"
    try:
        assert _plm_bilateral_window_s({"PLM_BILATERAL_WINDOW_S": 0.5}) == 5.0
    finally:
        del os.environ["PSGSCORING_PLM_BILATERAL_WINDOW_S"]
