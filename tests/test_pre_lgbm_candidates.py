"""De kandidatenlijst moet naar buiten, niet alleen haar aantal.

WAAROM
------
Wij zitten ~6/u onder de menselijke arousalindex. Of die arousals door de
classifier zijn VERWORPEN of nooit zijn VOORGESTELD, vraagt om verschillende
reparaties: een beter model tegenover een ruimere kandidaatgeneratie.

`pre_lgbm_n_arousals` gaf alleen een AANTAL. Twee pogingen om de lijst te
benaderen liepen stuk:

1. Werkpunt 0,01 als proxy gaf een KLEINERE lijst dan de normale uitvoer --
   onmogelijk voor een deelverzameling. `enforce_min_arousal_interval` draait
   NA de classifier en voegt bij een lage drempel massaal samen.
2. `AROUSAL_LGBM=0` levert geen kandidaten maar een ANDERE detector: het
   regelgebaseerde pad met eigen criteria.

Beide gaven een zichzelf tegensprekende uitkomst (694 kandidaten tegen 968 in
de uitvoer), en beide zouden zonder die controle tot de conclusie hebben geleid
dat de kandidaatgeneratie het probleem is.
"""
import numpy as np
import pytest


def _opname(mne, sf=64.0, minuten=20):
    n = int(sf * 60 * minuten)
    t = np.arange(n) / sf
    rng = np.random.default_rng(5)
    eeg = rng.normal(0, 20e-6, n)
    # arousals: bursts van alfa/beta bovenop de achtergrond
    for start in range(60, 60 * minuten - 60, 90):
        a, b = int(start * sf), int((start + 5) * sf)
        eeg[a:b] += 60e-6 * np.sin(2 * np.pi * 12.0 * t[a:b])
    flow = np.sin(2 * np.pi * 0.25 * t)
    info = mne.create_info(
        ["EEG C4-M1", "Chin EMG", "Resp nasal", "SaO2", "Thorax", "Abdomen"],
        sf, ["eeg"] + ["misc"] * 5)
    return mne.io.RawArray(
        np.vstack([eeg, rng.normal(0, 5e-6, n), flow, np.full(n, 96.0),
                   flow, flow]), info, verbose=False)


@pytest.fixture(scope="module")
def uit():
    mne = pytest.importorskip("mne")
    import psgscoring
    raw = _opname(mne)
    return psgscoring.run_pneumo_analysis(
        raw, hypno=["N2"] * int(np.ceil(raw.times[-1] / 30.0)),
        scoring_profile="aasm_v3_rec")


def test_de_kandidatenlijst_staat_in_de_uitvoer(uit):
    ar = uit.get("arousal") or {}
    nested = ar.get("arousals") or {}
    bron = nested if "pre_lgbm_events" in nested else ar
    assert "pre_lgbm_events" in bron, (
        "zonder onsets is niet te bepalen of een gemiste arousal verworpen is "
        "of nooit voorgesteld")


def test_de_lijst_telt_gelijk_aan_het_gepubliceerde_aantal(uit):
    ar = uit.get("arousal") or {}
    nested = ar.get("arousals") or {}
    bron = nested if "pre_lgbm_events" in nested else ar
    if "pre_lgbm_events" not in bron:
        pytest.skip("classifier draaide niet op deze opname")
    assert len(bron["pre_lgbm_events"]) == bron["pre_lgbm_n_arousals"]


def test_de_uitvoer_is_een_DEELVERZAMELING_van_de_kandidaten(uit):
    """De toets die twee eerdere metingen had moeten afkeuren.

    Elk gepubliceerd event moet een kandidaat hebben op dezelfde plek. Zo niet,
    dan vergelijkt de meting twee verschillende dingen -- precies wat er gebeurde
    bij werkpunt 0,01 en bij AROUSAL_LGBM=0.
    """
    ar = uit.get("arousal") or {}
    nested = ar.get("arousals") or {}
    bron = nested if "pre_lgbm_events" in nested else ar
    if "pre_lgbm_events" not in bron:
        pytest.skip("classifier draaide niet op deze opname")
    kand = np.array([e["onset_s"] for e in bron["pre_lgbm_events"]])
    ev = [e for e in (ar.get("events") or [])]
    if not len(kand) or not ev:
        pytest.skip("geen kandidaten of events")
    for e in ev:
        t = float(e.get("onset_s", 0))
        assert np.min(np.abs(kand - t)) <= 10.0, (
            f"event op {t:.1f}s heeft geen kandidaat binnen 10 s; de uitvoer "
            f"is dan geen deelverzameling van de kandidaten")
