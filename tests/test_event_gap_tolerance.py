"""
tests/test_event_gap_tolerance.py — de duurtolerantie op de amplitudemaskers.

De detectoren labelen AANEENGESLOTEN runs onder de amplitudedrempel. Eén
ademteug die de drempel niet haalt knipt een event daarmee doormidden, en twee
helften van elk ~7 s sneuvelen allebei op de eis van >=10 s: het event verdwijnt
volledig in plaats van iets korter te worden.

De AASM staat de onderbreking uitdrukkelijk toe -- voor apneus letterlijk (>=90 %
van de EVENTDUUR moet de reductie halen, de rest is vrij), voor hypopneus via de
duurdefinitie van ademteugnadir tot eerste ademteug op basislijnniveau.

`bridge_event_gaps` staat LOS van `detect_respiratory_events`, en niet uit
netheid: verstopt in de cascade is hij alleen te toetsen door eerst een volledige
detectie door acht andere filters te praten, en een fixture die daar geen events
uit krijgt laat de test leeg slagen in plaats van falen.

De derde voorwaarde -- geen van beide zijden mag de minimale eventduur al halen --
staat niet in de AASM. Ze kwam uit een meting: zonder haar slokte de brug de
laagamplitude-ademhaling ná een event op en groeide een geldig event van 30,9 s
naar 37,3 s. Zie `test_een_zijde_die_al_geldig_is_slokt_niets_op`.
"""
import numpy as np
import pytest

from psgscoring.respiratory import (
    HYPOPNEA_MIN_DUR_S,
    _breath_gap_seconds,
    bridge_event_gaps,
    detect_respiratory_events,
)

SF = 32.0


def _masker(sf, *spans, n_s=200):
    """Bouw een masker uit (start_s, eind_s)-paren."""
    m = np.zeros(int(n_s * sf), dtype=bool)
    for a, b in spans:
        m[int(a * sf):int(b * sf)] = True
    return m


def _runs(mask, sf=SF):
    from scipy.ndimage import find_objects, label
    lab, _ = label(mask)
    return [(round(sl[0].start / sf, 2), round(sl[0].stop / sf, 2))
            for sl in find_objects(lab)]


# ══════════════════════════════════════════════════════════════════════
#  1. Uit is uit -- byte-identiek als eigenschap, niet als belofte
# ══════════════════════════════════════════════════════════════════════

def test_elk_profiel_heeft_de_tolerantie_uit():
    from psgscoring.constants import SCORING_PROFILES
    from psgscoring.profiles import list_profiles
    for naam in list_profiles():
        assert SCORING_PROFILES[naam]["EVENT_GAP_TOLERANCE_BREATHS"] == 0.0, (
            f"{naam} zet de duurtolerantie aan")


def test_uitgeschakeld_geeft_hetzelfde_object_terug():
    """Niet 'gelijk aan' maar 'hetzelfde' -- dan is er geen pad waarlangs het
    masker stilletjes toch bewerkt kan worden."""
    m = _masker(SF, (10, 17), (21, 28))
    assert bridge_event_gaps(m, SF, 0.0, HYPOPNEA_MIN_DUR_S) is m
    assert bridge_event_gaps(m, SF, -1.0, HYPOPNEA_MIN_DUR_S) is m
    assert bridge_event_gaps(m, SF, 4.0, HYPOPNEA_MIN_DUR_S,
                             min_qualifying_fraction=0.0) is m


# ══════════════════════════════════════════════════════════════════════
#  2. Wat hij WEL doet
# ══════════════════════════════════════════════════════════════════════

def test_twee_te_korte_fragmenten_worden_een_geldig_event():
    """7 s + 1 s gat + 7 s: los onbruikbaar, samen 15 s."""
    m = _masker(SF, (10, 17), (18, 25))
    st = {}
    uit = bridge_event_gaps(m, SF, 4.0, HYPOPNEA_MIN_DUR_S, stats=st)
    assert _runs(uit, SF) == [(10.0, 25.0)]
    assert st["n_bridged"] == 1
    assert st["n_runs_before"] == 2 and st["n_runs_after"] == 1


def test_de_overbrugde_samples_tellen_niet_als_reductie():
    """De fractie meet tegen het ORIGINELE masker."""
    m = _masker(SF, (10, 17), (18, 25))
    uit = bridge_event_gaps(m, SF, 4.0, HYPOPNEA_MIN_DUR_S)
    assert int(uit.sum()) == int(m.sum()) + int(1.0 * SF)


# ══════════════════════════════════════════════════════════════════════
#  3. Wat hij NIET doet -- de drie voorwaarden, elk apart
# ══════════════════════════════════════════════════════════════════════

def test_een_te_lang_gat_wordt_niet_overbrugd():
    m = _masker(SF, (10, 17), (24, 31))          # 7 s gat, tolerantie 4 s
    assert _runs(bridge_event_gaps(m, SF, 4.0, HYPOPNEA_MIN_DUR_S)) == [
        (10.0, 17.0), (24.0, 31.0)]


def test_een_te_lage_kwalificerende_fractie_wordt_geweigerd():
    """4 s + 3,5 s gat + 4 s = 11,5 s waarvan 8 s reductie -> 0,70 < 0,90."""
    m = _masker(SF, (10, 14), (17.5, 21.5))
    assert _runs(bridge_event_gaps(m, SF, 4.0, HYPOPNEA_MIN_DUR_S)) == [
        (10.0, 14.0), (17.5, 21.5)]
    # met een soepeler fractie mag het wel -- de knop doet iets
    assert _runs(bridge_event_gaps(m, SF, 4.0, HYPOPNEA_MIN_DUR_S,
                                   min_qualifying_fraction=0.65)) == [(10.0, 21.5)]


def test_een_zijde_die_al_geldig_is_slokt_niets_op():
    """De gemeten fout: een geldig event van 30,9 s slokte een run van 3,7 s
    op 2,6 s afstand op en groeide naar 37,3 s. Er viel niets te repareren."""
    m = _masker(SF, (119.22, 150.16), (152.75, 156.47), n_s=200)
    st = {}
    uit = bridge_event_gaps(m, SF, 4.0, HYPOPNEA_MIN_DUR_S, stats=st)
    assert _runs(uit, SF) == [(119.22, 150.16), (152.75, 156.47)]
    assert st["n_bridged"] == 0


def test_twee_geldige_events_blijven_twee_events():
    """14 s + 2 s herstel + 14 s: beide kanten halen de duur al."""
    m = _masker(SF, (10, 24), (26, 40), n_s=60)
    assert len(_runs(bridge_event_gaps(m, SF, 4.0, HYPOPNEA_MIN_DUR_S), SF)) == 2


# ══════════════════════════════════════════════════════════════════════
#  4. Gegarandeerde richting -- vóór de meting bekend
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", range(8))
def test_de_brug_voegt_alleen_toe_en_neemt_nooit_weg(seed):
    """Op willekeurige maskers: nooit minder reductie, nooit méér runs."""
    rng = np.random.default_rng(seed)
    m = np.zeros(int(300 * SF), dtype=bool)
    for _ in range(40):
        a = rng.integers(0, m.size - int(20 * SF))
        m[a:a + int(rng.uniform(1, 20) * SF)] = True
    uit = bridge_event_gaps(m, SF, 4.0, HYPOPNEA_MIN_DUR_S)
    assert uit.sum() >= m.sum()
    assert (uit & ~m).sum() == uit.sum() - m.sum()   # alleen toegevoegd
    assert len(_runs(uit, SF)) <= len(_runs(m, SF))


# ══════════════════════════════════════════════════════════════════════
#  5. Schaalvrij: de tolerantie staat in ademteugen, niet in seconden
# ══════════════════════════════════════════════════════════════════════

def test_de_tolerantie_volgt_de_ademfrequentie():
    """4 s is bij 30/min twee ademteugen en bij 10/min nog geen halve."""
    snel = [{"duration_s": 2.0}] * 20
    traag = [{"duration_s": 6.0}] * 20
    assert _breath_gap_seconds(snel, 1.0) == 2.0
    assert _breath_gap_seconds(traag, 1.0) == 6.0
    assert _breath_gap_seconds(traag, 2.0) == 12.0


def test_zonder_bruikbare_ademteugen_een_expliciete_terugval():
    assert _breath_gap_seconds([], 1.0) == 4.0
    assert _breath_gap_seconds([{"duration_s": 2.0}] * 5, 1.0) == 4.0
    assert _breath_gap_seconds(None, 1.0) == 4.0


def test_nul_ademteugen_tolerantie_is_nul_seconden():
    assert _breath_gap_seconds([{"duration_s": 4.0}] * 20, 0.0) == 0.0


# ══════════════════════════════════════════════════════════════════════
#  6. Door de hele detector heen
# ══════════════════════════════════════════════════════════════════════

DUUR, BR, EV_LEN, GAP_S = 900, 0.25, 18, 4.0
_BASIS = {"STABILITY_FILTER_CV": 0.0, "LOCAL_BL_MIN_REDUCTION_PCT": 10.0,
          "LOCAL_BL_STRICT_RED": 10.0, "HYPOPNEA_THRESHOLD": 0.70,
          "DESATURATION_DROP_PCT": 3.0}


def _opname_met_doorgeknipte_hypopneus():
    """Vier reducties van 18 s, elk met één VOLLEDIGE herstelademteug van 4 s
    in het midden. Losse helften van ~7 s halen de 10 s niet.

    Het stabiliteitsfilter en de lokale vloer staan uit: die zouden dezelfde
    kandidaten om een andere reden afwijzen en dan meet deze test niets.
    """
    t = np.arange(int(SF * DUUR)) / SF
    rng = np.random.default_rng(7)
    amp = np.ones(t.size)
    fac = rng.lognormal(0.0, 0.25, int(DUUR * BR) + 2)
    for i in range(int(DUUR * BR) + 2):
        amp[int(i / BR * SF):int((i + 1) / BR * SF)] = fac[i]
    flow = amp * np.sin(2 * np.pi * BR * t) + rng.normal(0, 0.005, t.size)
    spo2 = np.full(DUUR, 97.0)
    starts = list(range(120, DUUR - 120, 180))
    for start in starts:
        flow[int(start * SF):int((start + EV_LEN) * SF)] *= 0.40
        m = int((start + EV_LEN / 2 - GAP_S / 2) * SF)
        flow[m:m + int(GAP_S * SF)] /= 0.40
        spo2[start + EV_LEN + 2:start + EV_LEN + 15] = 92.0
    return flow, spo2, starts


def _scoor(extra):
    flow, spo2, starts = _opname_met_doorgeknipte_hypopneus()
    r = detect_respiratory_events(
        flow_data=flow, thorax_data=None, abdomen_data=None, spo2_data=spo2,
        sf_flow=SF, sf_spo2=1.0, hypno=["N2"] * (DUUR // 30),
        scoring_profile={**_BASIS, **extra})
    hyp = [e for e in r["events"] if "hypopnea" in str(e.get("type"))]
    return len(hyp), len(starts), r


def test_zonder_de_vlag_verdwijnt_de_helft_van_de_events():
    n, gepland, r = _scoor({})
    assert n == 2 and gepland == 4, (
        f"fixture toont het probleem niet meer: {n} van {gepland}")
    assert "event_gap_bridging" not in r, (
        "herkomstveld aanwezig terwijl de vlag uit staat")


def test_met_de_vlag_komen_ze_alle_vier_terug():
    n, gepland, r = _scoor({"EVENT_GAP_TOLERANCE_BREATHS": 1.0})
    assert n == gepland == 4
    brug = r["event_gap_bridging"]
    assert brug["tolerance_breaths"] == 1.0
    assert brug["min_qualifying_fraction"] == 0.90
    assert brug["hypopnea"]["n_bridged"] >= 2
    assert brug["tolerance_s"] == pytest.approx(4.0, abs=1.0), (
        "omrekening naar seconden volgt de mediane ademteugduur niet")


def test_de_vlag_kan_de_telling_alleen_verhogen():
    uit, _, _ = _scoor({})
    aan, _, _ = _scoor({"EVENT_GAP_TOLERANCE_BREATHS": 1.0})
    assert aan >= uit


def test_een_soepeler_fractie_slaat_door():
    """0,90 is niet willekeurig: op 0,75 scoort dezelfde fixture er zes."""
    n, gepland, _ = _scoor({"EVENT_GAP_TOLERANCE_BREATHS": 1.0,
                            "EVENT_MIN_QUALIFYING_FRACTION": 0.75})
    assert n > gepland, (
        f"0,75 hoort door te slaan; kreeg {n} bij {gepland} geplande events")


# ══════════════════════════════════════════════════════════════════════
#  7. Env-overrides -- beide armen meetbaar zonder de registry te muteren
# ══════════════════════════════════════════════════════════════════════

def test_beide_knoppen_hebben_een_env_override(monkeypatch):
    """Zonder deze twee is een arm alleen te meten door de registry te
    wijzigen, en dan meet je een andere bibliotheek dan die je uitrolt."""
    monkeypatch.setenv("PSGSCORING_EVENT_GAP_TOLERANCE_BREATHS", "1.0")
    n_aan, gepland, r = _scoor({})
    assert n_aan == gepland == 4
    assert r["scoring_thresholds"]["event_gap_tolerance_breaths"] == 1.0

    monkeypatch.setenv("PSGSCORING_EVENT_MIN_QUALIFYING_FRACTION", "0.75")
    n_los, gepland, r = _scoor({})
    assert r["scoring_thresholds"]["event_min_qualifying_fraction"] == 0.75
    assert n_los > gepland


def test_onleesbare_env_valt_terug_op_het_profiel_met_waarschuwing(monkeypatch, caplog):
    """Stil op nul terugvallen zou een meting ongeldig maken zonder dat het
    opvalt."""
    monkeypatch.setenv("PSGSCORING_EVENT_GAP_TOLERANCE_BREATHS", "een beetje")
    monkeypatch.setenv("PSGSCORING_EVENT_MIN_QUALIFYING_FRACTION", "veel")
    with caplog.at_level("WARNING", logger="psgscoring.respiratory"):
        _, _, r = _scoor({"EVENT_GAP_TOLERANCE_BREATHS": 1.0,
                          "EVENT_MIN_QUALIFYING_FRACTION": 0.90})
    st = r["scoring_thresholds"]
    assert st["event_gap_tolerance_breaths"] == 1.0
    assert st["event_min_qualifying_fraction"] == 0.90
    assert sum("geen getal" in m for m in caplog.messages) == 2
