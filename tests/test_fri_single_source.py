"""De FRI-index heeft één noemer, of het is geen index.

Eén klinisch rapport toonde **FRI 44,3/u** in de RERA-sectie en **43,2/u** in
sectie 8d, over dezelfde nacht en dezelfde teller. De oorzaak zat niet in de
teller maar in de noemer: de RERA-sectie leidde de uren af uit
`n_rera / rera_index` (de slaaptijd die psgscoring zelf gebruikt, met
artefact-epochs eruit), sectie 8d deelde door `stats["TST"]` uit de
YASA-slaapstatistiek. Twee definities van slaaptijd onder één label.

De reparatie zet de index in de bibliotheek, naast de andere indices en met
dezelfde noemer. Een rapportlaag die zelf deelt kan opnieuw uiteenlopen; een
rapportlaag die een veld leest niet.
"""
import numpy as np
import pytest


def _raw_and_hypno(mne):
    """Flowreducties zonder desaturatie, waarvan de HELFT door een arousal
    gevolgd wordt.

    Zonder die tweede helft heeft de opname geen RERA's, en dan is er geen
    tweede noemer om de FRI-noemer mee te vergelijken -- de test die de bug
    moet vangen zou dan overgeslagen worden en niets meten.
    """
    sf, minutes = 32.0, 40
    n = int(sf * 60 * minutes)
    t = np.arange(n) / sf
    rng = np.random.default_rng(5)
    flow = np.sin(2 * np.pi * 0.25 * t)
    eeg = rng.normal(0, 20e-6, n)
    for i, start in enumerate(range(90, 60 * minutes - 90, 120)):
        a, b = int(start * sf), int((start + 20) * sf)
        flow[a:b] *= 0.5
        if i % 2 == 0:                       # arousal vlak na het event
            c, d = b, b + int(6 * sf)
            eeg[c:d] += 70e-6 * np.sin(2 * np.pi * 10.0 * t[c:d])
    info = mne.create_info(
        ["Resp nasal", "SaO2", "Thorax", "Abdomen", "EEG C4-M1", "Chin EMG"],
        sf, ["misc", "misc", "misc", "misc", "eeg", "misc"])
    raw = mne.io.RawArray(
        np.vstack([flow, np.full(n, 97.0),
                   np.sin(2 * np.pi * 0.25 * t), np.sin(2 * np.pi * 0.25 * t),
                   eeg, rng.normal(0, 5e-6, n)]),
        info, verbose=False)
    return raw, ["N2"] * int(np.ceil(raw.times[-1] / 30.0))


@pytest.fixture(scope="module")
def summary():
    mne = pytest.importorskip("mne")
    import psgscoring
    raw, hypno = _raw_and_hypno(mne)
    out = psgscoring.run_pneumo_analysis(
        raw, hypno=hypno, scoring_profile="aasm_v3_breath")
    return out["respiratory"]["summary"]


def test_the_library_publishes_the_fri_index(summary):
    assert "fri_index" in summary, (
        "de rapportlaag moet zelf delen, en dat is precies hoe er twee "
        "noemers ontstonden")


def test_the_fri_index_uses_the_same_denominator_as_the_rera_index(summary):
    """De enige toets die de bug had gevangen."""
    n_fri, fri_idx = summary.get("n_fri"), summary.get("fri_index")
    n_rera, rera_idx = summary.get("n_rera"), summary.get("rera_index")
    if not (n_rera and rera_idx):
        pytest.skip("geen RERA's in deze fixture — geen tweede noemer om mee "
                    "te vergelijken")
    uren_fri = n_fri / fri_idx
    uren_rera = n_rera / rera_idx
    assert uren_fri == pytest.approx(uren_rera, rel=0.02), (
        f"FRI deelt door {uren_fri:.3f} u, RERA door {uren_rera:.3f} u")


def test_the_denominator_itself_is_published(summary):
    """Zodat een consument die tóch iets per uur wil uitrekenen niet opnieuw
    een eigen slaaptijd hoeft te verzinnen."""
    tst = summary.get("index_denominator_h")
    assert tst is not None and tst > 0
    if summary.get("n_fri"):
        assert summary["fri_index"] == pytest.approx(
            round(summary["n_fri"] / tst, 1))


def test_no_sleep_means_no_index():
    """Geen noemer, geen index — geen nul en geen opgeblazen getal.
    Zie psgscoring/indices.py."""
    mne = pytest.importorskip("mne")
    import psgscoring
    raw, hypno = _raw_and_hypno(mne)
    out = psgscoring.run_pneumo_analysis(
        raw, hypno=["W"] * len(hypno), scoring_profile="aasm_v3_breath")
    s = out["respiratory"]["summary"]
    assert s.get("fri_index") is None
    assert s.get("index_denominator_h") in (0, 0.0)
