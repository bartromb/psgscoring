"""
tests/test_arousal_min_interval.py — de AASM-regel van 10 s tussen arousals.

De AASM eist dat een arousal wordt voorafgegaan door ten minste 10 s stabiele
slaap. `detect_arousals` toetst daarvan alleen de HYPNOGRAMKANT (check A: is
>=60 % van de voorgaande 10 s als slaap gescoord). Een epoch waarin net een
arousal zat heet nog steeds N2, dus die check ziet een voorgaande arousal niet.
Gemeten op een synthetisch EEG met twee alpha/beta-bursts van 4 s: bij 4 s
tussenruimte scoorde de detector er TWEE, waar er één hoort te staan.

In multi-derivatie telt het dubbel: `_union_arousals` fuseert alleen bij
temporele OVERLAP, dus twee afleidingen die 2 s na elkaar vuren leveren twee
events op.

De ingreep kan alleen events WEGNEMEN — het spiegelbeeld van
`bridge_event_gaps`. De richting staat daarmee vóór elke meting vast, en dat
maakt hem een zuivere PRECISIE-ingreep: precies waar het gemeten arousalgat zit
(F1 0,546 tegen een menselijk plafond van 0,679).
"""
import numpy as np
import pytest

from psgscoring.arousal import (
    AROUSAL_MIN_INTERVAL_S,
    detect_arousals,
    enforce_min_arousal_interval,
)

SF, DUUR = 128.0, 600


def _ev(onset, dur=4.0, **extra):
    e = {"onset_s": onset, "end_s": onset + dur, "duration_s": dur,
         "stage": "N2", "dominant_band": "alpha"}
    e.update(extra)
    return e


def _spans(events):
    return [(round(e["onset_s"], 2), round(e["end_s"], 2)) for e in events]


# ══════════════════════════════════════════════════════════════════════
#  1. Uit is uit
# ══════════════════════════════════════════════════════════════════════

def test_elk_profiel_heeft_de_regel_uit():
    from psgscoring.constants import SCORING_PROFILES
    from psgscoring.profiles import list_profiles
    for naam in list_profiles():
        assert SCORING_PROFILES[naam]["AROUSAL_MIN_INTERVAL_S"] == 0.0, (
            f"{naam} zet de tussenafstandsregel aan")


def test_uitgeschakeld_geeft_dezelfde_lijst_terug():
    ev = [_ev(100.0), _ev(106.0)]
    assert enforce_min_arousal_interval(ev, 0.0) is ev
    assert enforce_min_arousal_interval(ev, -1.0) is ev


def test_de_aasm_waarde_is_tien_seconden():
    assert AROUSAL_MIN_INTERVAL_S == 10.0


# ══════════════════════════════════════════════════════════════════════
#  2. Wat hij doet
# ══════════════════════════════════════════════════════════════════════

def test_twee_arousals_binnen_tien_seconden_worden_er_een():
    """4 s slaap ertussen -> één arousal van onset eerste tot einde laatste."""
    st = {}
    uit = enforce_min_arousal_interval([_ev(100.0), _ev(108.0)], 10.0, stats=st)
    assert _spans(uit) == [(100.0, 112.0)]
    assert st["n_merged"] == 1 and st["n_before"] == 2 and st["n_after"] == 1
    assert uit[0]["merged_from"] == 2


def test_ruim_gescheiden_arousals_blijven_apart():
    """12 s slaap ertussen -> twee arousals."""
    uit = enforce_min_arousal_interval([_ev(100.0), _ev(116.0)], 10.0)
    assert _spans(uit) == [(100.0, 104.0), (116.0, 120.0)]


def test_precies_tien_seconden_is_genoeg():
    """De eis is >=10 s; exact 10 mag apart blijven."""
    assert len(enforce_min_arousal_interval([_ev(100.0), _ev(114.0)], 10.0)) == 2
    assert len(enforce_min_arousal_interval([_ev(100.0), _ev(113.9)], 10.0)) == 1


def test_een_keten_wordt_een_enkel_event():
    ev = [_ev(100.0), _ev(107.0), _ev(114.0)]
    uit = enforce_min_arousal_interval(ev, 10.0)
    assert _spans(uit) == [(100.0, 118.0)]
    assert uit[0]["merged_from"] == 3


def test_de_langste_bijdrager_levert_band_en_stadium():
    """Zelfde regel als `_union_arousals`, zodat de twee niet uiteenlopen."""
    ev = [_ev(100.0, 3.0, dominant_band="theta", stage="N1"),
          _ev(105.0, 9.0, dominant_band="beta", stage="N2")]
    uit = enforce_min_arousal_interval(ev, 10.0)
    assert uit[0]["dominant_band"] == "beta"
    assert uit[0]["stage"] == "N2"


def test_afleidingen_worden_verenigd_niet_weggegooid():
    ev = [_ev(100.0, derivations=["C4-M1"]), _ev(107.0, derivations=["O2-M1"])]
    uit = enforce_min_arousal_interval(ev, 10.0)
    assert uit[0]["derivations"] == ["C4-M1", "O2-M1"]


def test_de_invoerlijst_wordt_niet_gemuteerd():
    ev = [_ev(100.0), _ev(107.0)]
    enforce_min_arousal_interval(ev, 10.0)
    assert _spans(ev) == [(100.0, 104.0), (107.0, 111.0)]


# ══════════════════════════════════════════════════════════════════════
#  3. Gegarandeerde richting -- vóór de meting bekend
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", range(8))
def test_de_regel_neemt_alleen_weg_en_voegt_nooit_toe(seed):
    rng = np.random.default_rng(seed)
    onsets = sorted(rng.uniform(0, 3000, 60))
    ev = [_ev(float(o), float(rng.uniform(3, 15))) for o in onsets]
    uit = enforce_min_arousal_interval(ev, 10.0)
    assert len(uit) <= len(ev)
    # Geen enkel oorspronkelijk event valt buiten de resulterende dekking.
    # TOL = een halve afrondingsstap: `_safe()` rondt onsets en eindes op één
    # decimaal af, net als `_union_arousals`, dus een samengevoegd event kan
    # 0,05 s binnen zijn bijdragers vallen. Zonder deze marge faalt de test op
    # de afronding in plaats van op de logica.
    TOL = 0.05
    dekking = [(e["onset_s"], e["end_s"]) for e in uit]
    for e in ev:
        assert any(a - TOL <= e["onset_s"] and e["end_s"] <= b + TOL
                   for a, b in dekking), (
            f"event {e['onset_s']:.3f}-{e['end_s']:.3f} valt buiten elke "
            f"resulterende span")


# ══════════════════════════════════════════════════════════════════════
#  4. Door de detector heen
# ══════════════════════════════════════════════════════════════════════

def _eeg_met_twee_bursts(gap_s):
    """Delta-achtergrond met twee alpha+beta-bursts van 4 s, `gap_s` ertussen."""
    t = np.arange(int(SF * DUUR)) / SF
    rng = np.random.default_rng(11)
    eeg = 40 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(0, 8, t.size)
    for start in (200.0, 200.0 + 4.0 + gap_s):
        a, b = int(start * SF), int((start + 4.0) * SF)
        tt = t[a:b]
        eeg[a:b] = (55 * np.sin(2 * np.pi * 9.5 * tt)
                    + 35 * np.sin(2 * np.pi * 20.0 * tt)
                    + rng.normal(0, 8, b - a))
    return eeg


def _tel(gap_s, min_interval_s):
    res = detect_arousals(_eeg_met_twee_bursts(gap_s), SF, ["N2"] * (DUUR // 30),
                          min_interval_s=min_interval_s)
    return [a for a in res["events"] if 195 < a["onset_s"] < 245], res


def test_zonder_de_regel_telt_de_detector_er_twee():
    ev, res = _tel(4.0, 0.0)
    assert len(ev) == 2, f"fixture toont het probleem niet meer: {len(ev)}"
    assert "n_interval_merged" not in res.get("summary", {})


def test_met_de_regel_wordt_het_er_een():
    ev, res = _tel(4.0, 10.0)
    assert len(ev) == 1, f"vier seconden ertussen hoort één arousal te zijn: {len(ev)}"
    assert res["summary"]["n_interval_merged"] >= 1
    assert res["summary"]["min_interval_s"] == 10.0


def test_ruim_gescheiden_bursts_blijven_ook_met_de_regel_twee():
    ev, _ = _tel(12.0, 10.0)
    assert len(ev) == 2, (
        "twaalf seconden slaap ertussen zijn twee arousals; de regel mag ze "
        "niet samenvoegen")


def test_de_index_volgt_de_samenvoeging():
    _, uit = _tel(4.0, 0.0)
    _, aan = _tel(4.0, 10.0)
    assert aan["summary"]["n_arousals"] < uit["summary"]["n_arousals"]
    assert aan["summary"]["arousal_index"] < uit["summary"]["arousal_index"]


# ══════════════════════════════════════════════════════════════════════
#  5. Bereikbaar zonder de pipeline
# ══════════════════════════════════════════════════════════════════════
#
# `pipeline.py` leest de env-vlag, maar de meetharnassen roepen de detector
# RECHTSTREEKS aan (sweep_arousal_threshold_psgipa.py doet dat). Zonder deze
# doorgifte meet zo'n harnas een arm die niets doet -- vandaag twee keer
# gebeurd, zie docs/rule1a_arousal_20260829.md.

def test_de_env_vlag_bereikt_de_detector_zonder_pipeline(monkeypatch):
    monkeypatch.setenv("PSGSCORING_AROUSAL_MIN_INTERVAL_S", "10")
    ev, res = _tel(4.0, 0.0)        # argument 0,0 -- de env moet winnen
    assert len(ev) == 1, "env-vlag bereikt detect_arousals niet"
    assert res["summary"]["min_interval_s"] == 10.0


def test_de_env_wint_van_het_argument(monkeypatch):
    monkeypatch.setenv("PSGSCORING_AROUSAL_MIN_INTERVAL_S", "0")
    ev, _ = _tel(4.0, 10.0)         # argument 10 -- de env zet hem uit
    assert len(ev) == 2, "env kan de regel niet uitzetten"


def test_onleesbare_env_valt_terug_op_het_argument(monkeypatch, caplog):
    monkeypatch.setenv("PSGSCORING_AROUSAL_MIN_INTERVAL_S", "tien")
    with caplog.at_level("WARNING", logger="psgscoring.arousal"):
        ev, _ = _tel(4.0, 10.0)
    assert len(ev) == 1
    assert any("geen getal" in m for m in caplog.messages)
