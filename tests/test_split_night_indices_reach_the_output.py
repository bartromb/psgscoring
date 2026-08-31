"""De segmentindices moeten de UITVOER halen, niet alleen bestaan.

`segment_spo2()` is het waarschuwende voorbeeld: sinds 0.29.0 rekent de
bibliotheek ODI en T90 keurig per segment, en geen enkele consument leest het.
Een index die nergens aankomt is geen index, en die fout is in dit project
vaker gemaakt dan de rekenfout zelf -- de kin-EMG die de classifier nooit
bereikte, `analysis_warnings` zonder lezer, de topografiecheck die op een
niet-bestaande variabele draaide.

Deze test draait de hele pijplijn en kijkt in de uitvoer.
"""
import numpy as np
import pytest

BREUK_S = 3900.0          # 65 min: ruim boven MIN_SEGMENT_S van 3600 s


def _split_night_raw(mne):
    """Twee uur zware diagnostiek, daarna twee uur rustige titratie.

    Het tweede deel krijgt een kleinere flowamplitude en een hogere
    saturatiebasislijn -- de twee sporen waarop de detector werkt -- plus veel
    minder events en arousals, zoals een geslaagde titratie eruitziet.
    """
    sf, minuten = 32.0, 130
    n = int(sf * 60 * minuten)
    t = np.arange(n) / sf
    rng = np.random.default_rng(7)

    flow = np.sin(2 * np.pi * 0.25 * t)
    eeg = rng.normal(0, 20e-6, n)
    spo2 = np.full(n, 91.0)
    been = rng.normal(0, 5e-6, n)
    k = int(BREUK_S * sf)

    # Diagnostisch: apneus met desaturatie en een arousal erachteraan.
    for start in range(60, int(BREUK_S) - 60, 100):
        a, b = int(start * sf), int((start + 20) * sf)
        flow[a:b] *= 0.05
        spo2[b:b + int(15 * sf)] -= 6.0
        c, d = b, b + int(6 * sf)
        eeg[c:d] += 70e-6 * np.sin(2 * np.pi * 10.0 * t[c:d])
        # Beenbeweging kort voor het event, met dezelfde regelmaat.
        e, f = a - int(3 * sf), a - int(1 * sf)
        if e > 0:
            been[e:f] += 60e-6 * np.sin(2 * np.pi * 25.0 * t[e:f])

    # Onder therapie: kleinere amplitude, hogere basislijn, bijna niets.
    flow[k:] *= 0.2
    spo2[k:] = 96.0
    for start in range(int(BREUK_S) + 120, 60 * minuten - 120, 900):
        a, b = int(start * sf), int((start + 20) * sf)
        flow[a:b] *= 0.05
        spo2[b:b + int(15 * sf)] -= 5.0

    # Houding en snurken horen bij het beeld: rugligging en snurken in het
    # diagnostische deel, zijligging en stilte onder therapie. Zonder deze
    # kanalen zou de test de twee nieuwste families overslaan.
    positie = np.where(np.arange(n) < k, 0.0, 1.0)      # 0 = rug, 1 = links
    snurk = np.where(np.arange(n) < k,
                     rng.normal(0, 1.0, n), rng.normal(0, 0.02, n))

    info = mne.create_info(
        ["Resp nasal", "SaO2", "Thorax", "Abdomen", "EEG C4-M1",
         "Chin EMG", "Leg EMG", "Position", "Snore"],
        sf, ["misc"] * 4 + ["eeg"] + ["misc"] * 4)
    raw = mne.io.RawArray(
        np.vstack([flow, spo2,
                   np.sin(2 * np.pi * 0.25 * t) * np.where(np.arange(n) < k, 1.0, 0.2),
                   np.sin(2 * np.pi * 0.25 * t) * np.where(np.arange(n) < k, 1.0, 0.2),
                   eeg, rng.normal(0, 5e-6, n), been, positie, snurk]),
        info, verbose=False)
    return raw, ["N2"] * int(np.ceil(raw.times[-1] / 30.0))


@pytest.fixture(scope="module")
def uit():
    mne = pytest.importorskip("mne")
    import psgscoring
    raw, hypno = _split_night_raw(mne)
    return psgscoring.run_pneumo_analysis(
        raw, hypno=hypno, scoring_profile="aasm_v3_rec",
        split_night="manual", split_night_breakpoint_s=BREUK_S)


def test_de_split_wordt_gerapporteerd(uit):
    sn = uit.get("split_night") or {}
    assert sn.get("detected") is True, sn
    assert sn["breakpoint_s"] == pytest.approx(BREUK_S, abs=1.0)


@pytest.mark.parametrize("sleutel", ["segments", "summaries", "spo2",
                                     "arousal", "rdi", "plm",
                                     "position", "snore"])
def test_elke_indexfamilie_staat_in_de_uitvoer(uit, sleutel):
    """Acht families, elk met beide helften. Ontbreekt er een, dan staat er in
    het rapport een nachtgetal naast een segmentgetal."""
    blok = (uit.get("split_night") or {}).get(sleutel)
    assert blok, f"split_night['{sleutel}'] ontbreekt"
    assert set(blok) >= {"diagnostic", "therapeutic"}, blok.keys()


def test_de_arousalindex_van_het_diagnostische_deel_ligt_hoger_dan_de_nacht(uit):
    """Dit is de klinische kern: het nachtgemiddelde verdunt de diagnose."""
    sn = uit["split_night"]["arousal"]
    nacht = (uit.get("arousal") or {}).get("summary", {}).get("arousal_index")
    d = sn["diagnostic"]["arousal_index"]
    if nacht is None or d is None:
        pytest.skip("geen arousals gedetecteerd in deze synthetische opname")
    assert d > nacht, (
        f"diagnostisch {d}/u zou boven het nachtgemiddelde {nacht}/u moeten "
        "liggen; anders verdunt de titratiehelft nog steeds mee")
    assert sn["therapeutic"]["arousal_index"] < d


def test_de_noemers_lopen_niet_uiteen(uit):
    """Elke familie deelt door dezelfde slaaptijd per segment."""
    sn = uit["split_night"]
    for deel in ("diagnostic", "therapeutic"):
        h = sn["segments"][deel]["sleep_h"]
        for fam in ("arousal", "rdi", "plm"):
            assert sn[fam][deel]["sleep_h"] == pytest.approx(h, abs=1e-6), (
                f"{fam}[{deel}] rekent op een andere noemer dan de AHI")


def test_de_rdi_per_deel_is_de_ahi_plus_de_rera_index(uit):
    sn = uit["split_night"]
    for deel in ("diagnostic", "therapeutic"):
        ahi = sn["segments"][deel]["ahi"]
        r = sn["rdi"][deel]
        if ahi is None or r["rera_index"] is None:
            continue
        assert r["rdi"] == pytest.approx(ahi + r["rera_index"], abs=0.11)


def test_de_rera_onsets_worden_gepubliceerd(uit):
    """Zonder deze lijst kan geen enkele consument een RDI per helft rekenen."""
    resp = uit["respiratory"]
    assert "rera_onsets_s" in resp
    assert len(resp["rera_onsets_s"]) == (resp["summary"].get("n_rera") or 0)
    assert resp["rera_onsets_s"] == sorted(resp["rera_onsets_s"])


def test_de_segmenten_worden_geteld_voor_de_payloadgrens():
    """Volgordebewaking, en deze fout is hier al eens gemaakt.

    `_cap_plm_event_list()` kort de PLM-lijst in tot de payloadgrens. Draait die
    stap voor het split-blok, dan telt de segment-PLM-index alleen het BEGIN van
    de nacht -- en dat is stil, want een afgekapte lijst ziet er niet afgekapt
    uit. Precies dat overkwam `plm_arousal_index` voor 22-08-2026.

    De volgorde is niet uit te drukken in een gedragstest zonder een opname met
    meer dan EVENT_LIST_CAP bewegingen te bouwen; dit leest hem uit de bron.
    """
    import ast
    import inspect

    import psgscoring.pipeline as pl

    boom = ast.parse(inspect.getsource(pl))
    fn = next(n for n in ast.walk(boom)
              if isinstance(n, ast.FunctionDef) and n.name == "run_pneumo_analysis")

    def _regel(naam, attr="func"):
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                    and n.func.id == naam:
                return n.lineno
        return None

    split = _regel("segment_plm")
    cap = _regel("_cap_plm_event_list")
    assert split is not None, "het split-blok roept segment_plm niet meer aan"
    assert cap is not None, "_cap_plm_event_list is verdwenen uit de pijplijn"
    assert split < cap, (
        f"segment_plm (regel {split}) draait NA _cap_plm_event_list "
        f"(regel {cap}): de segment-PLM-index telt dan een afgekapte lijst")


def test_de_snurkdrempel_is_dezelfde_voor_beide_helften(uit):
    """Was de drempel per segment bepaald, dan trok een stille tweede helft
    zijn eigen lat omlaag en rapporteerde alsnog snurken."""
    sn = uit["split_night"]["snore"]
    d = (sn.get("diagnostic") or {}).get("snore_index")
    t_ = (sn.get("therapeutic") or {}).get("snore_index")
    if d is None or t_ is None:
        pytest.skip("geen snurkindex in deze opname")
    assert d > t_, (
        f"diagnostisch {d} zou boven {t_} moeten liggen: het snurken stopt "
        "na de titratie in deze fixture")


def test_de_houding_per_helft_wordt_gerapporteerd(uit):
    """Ligt de patiënt diagnostisch op de rug en onder therapie op de zij, dan
    verklaart de houding een deel van de daling die aan de CPAP wordt
    toegeschreven."""
    pos = uit["split_night"]["position"]
    assert (pos.get("diagnostic") or {}).get("ahi_per_pos") is not None
    assert (pos.get("therapeutic") or {}).get("ahi_per_pos") is not None
