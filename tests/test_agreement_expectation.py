"""Bij welke ziektelast is een AHI nog te vertrouwen?

WAAROM DIT BESTAAT
------------------
Drie onafhankelijke systemen vertonen dezelfde helling, gemeten op 2026-09-01:

  twaalf menselijke scoorders   F1 0,948 (273-339 events)  ->  0,553 (1-38)
  onze regelketen                93 % van het plafond      ->   52 %
  een 1D-U-Net op de golfvorm   F1 0,743 (>= 150 events)   ->  0,254 (< 20)

Geen gedeelde aannames, dezelfde uitkomst: waar weinig te vinden is, weten
mensen het onderling niet, weten onze regels het niet, en weet een netwerk het
evenmin.

Op de lichtste PSG-IPA-opname scoorde de ene expert ÉÉN event en de andere
ACHTENDERTIG, met kappa 0,000 op het subtype.

WAT DAT BETEKENT VOOR EEN RAPPORT
---------------------------------
Een AHI van 8 en een AHI van 40 dragen niet dezelfde zekerheid, en het rapport
zegt dat nu nergens. Een lezer die beide getallen als gelijkwaardig behandelt,
doet precies wat de meting verbiedt.

Dit veld zegt niet hoe zeker ONS getal is -- dat zou een claim over onszelf
zijn. Het zegt hoe goed MENSEN het bij deze ziektelast onderling eens zijn, en
dat is een eigenschap van de opname, niet van het algoritme.
"""
import pytest

from psgscoring.indices import expected_scorer_agreement


def test_een_zware_nacht_krijgt_een_hoge_verwachting():
    r = expected_scorer_agreement(n_events=300)
    assert r["f1_human"] >= 0.90
    assert r["band"] == "hoog"


def test_een_lichte_nacht_krijgt_een_lage_verwachting():
    r = expected_scorer_agreement(n_events=12)
    assert r["f1_human"] <= 0.60
    assert r["band"] == "laag"


def test_de_verwachting_loopt_monotoon_met_de_ziektelast():
    ns = [5, 15, 30, 60, 120, 250, 400]
    fs = [expected_scorer_agreement(n_events=n)["f1_human"] for n in ns]
    assert all(b >= a - 1e-9 for a, b in zip(fs, fs[1:])), fs


def test_de_meting_waarop_dit_rust_staat_erbij():
    """Een verwachting zonder bron is een mening."""
    r = expected_scorer_agreement(n_events=100)
    assert r["source"], r
    assert "PSG-IPA" in r["source"]
    assert r["n_scorer_pairs"] == 330


@pytest.mark.parametrize("n", [0, None])
def test_zonder_events_geen_bewering(n):
    r = expected_scorer_agreement(n_events=n)
    assert r["f1_human"] is None
    assert r["band"] == "onbekend"


def test_het_getal_is_geen_uitspraak_over_ons_algoritme():
    """Dit veld beschrijft de OPNAME, niet de detector. Verwarring daarover
    zou het tot een zelfbeoordeling maken."""
    r = expected_scorer_agreement(n_events=300)
    assert "human" in r["what"].lower() or "scoorder" in r["what"].lower()
    assert "algorit" not in r["what"].lower()


def test_de_grens_ligt_waar_de_meting_hem_legt():
    """De overgang hoog/laag mag geen rond getal zijn dat mooi oogt, maar moet
    volgen uit waar de gemeten curve kantelt."""
    laag = expected_scorer_agreement(n_events=20)
    hoog = expected_scorer_agreement(n_events=200)
    assert laag["band"] != hoog["band"]
    assert laag["f1_human"] < hoog["f1_human"]


def test_het_veld_bereikt_de_samenvatting():
    """Vier keer op één dag bleek een veld zijn consument niet te halen. Een
    verwachting die alleen in een functie bestaat, staat in geen enkel rapport.
    """
    import numpy as np
    import psgscoring

    mne = pytest.importorskip("mne")
    sf, minuten = 32.0, 25
    n = int(sf * 60 * minuten)
    t = np.arange(n) / sf
    rng = np.random.default_rng(11)
    amp = 1.0 + 0.7 * np.sin(2 * np.pi * 0.011 * t) + 0.25 * rng.normal(0, 1, n)
    flow = amp * np.sin(2 * np.pi * 0.25 * t)
    spo2 = np.full(n, 97.0)
    for start in range(60, 60 * minuten - 60, 120):
        a, b = int(start * sf), int((start + 25) * sf)
        flow[a:b] = 0.30 * np.sign(np.sin(2 * np.pi * 0.25 * t[a:b]))
        d0 = b + int(4 * sf)
        spo2[d0:d0 + int(20 * sf)] -= 7.0
    info = mne.create_info(["Resp nasal", "SaO2", "Thorax", "Abdomen"],
                           sf, ["misc"] * 4)
    raw = mne.io.RawArray(np.vstack([flow, spo2,
                                     np.sin(2 * np.pi * 0.25 * t),
                                     np.sin(2 * np.pi * 0.25 * t)]),
                          info, verbose=False)
    out = psgscoring.run_pneumo_analysis(
        raw, hypno=["N2"] * int(np.ceil(raw.times[-1] / 30.0)),
        scoring_profile="aasm_v3_rec")
    summ = out["respiratory"]["summary"]
    assert "scorer_agreement_expectation" in summ, (
        "de verwachting staat niet naast de AHI")
    v = summ["scorer_agreement_expectation"]
    assert v["band"] in ("hoog", "laag", "onbekend")
    assert v["source"]
