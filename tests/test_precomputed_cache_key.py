"""De gedeelde voorbewerkingscache sleutelt op ALLES wat de voorbewerking raakt.

`detect_respiratory_events` deelt envelopes, basislijnen, uitvalmaskers,
effort-basislijn en ademteugen tussen de armen van de AHI-intervalsweep via
`_precomputed`. Dat is alleen geldig zolang de sleutel elk profielveld draagt
dat de gecachte waarde beïnvloedt. Twee velden ontbraken:

  * `FLOW_GAP_SCALE_FREE` -- het uitvalmasker (`gap_resp`) stond onder een
    kale sleutel, dus een arm met schaalvrije uitvaldetectie kreeg het masker
    van de arm die het eerst draaide;
  * de basislijnvelden voor `effort_bl` -- die hangt af van `flow_norm` en
    dus van BASELINE_WINDOW_S/BASELINE_PERCENTILE, maar de sleutel droeg
    alleen de envelope-as.

Nu is er één `_pre_key` uit elk profielveld dat de voorbewerking binnengaat,
en elke cachesleutel begint ermee. Twee profielen die daarin verschillen
krijgen gescheiden cache-regels; identieke profielen delen ze nog steeds.
"""
import numpy as np

from psgscoring.constants import SCORING_PROFILES
from psgscoring.respiratory import detect_respiratory_events

SF = 32.0
DUR_S = 600.0


def _rec():
    n = int(SF * DUR_S)
    t = np.arange(n) / SF
    rng = np.random.default_rng(3)
    flow = (1.0 + 0.3 * rng.normal(0, 1, n).cumsum() / np.sqrt(n)) * np.sin(2 * np.pi * 0.25 * t)
    flow[int(100 * SF):int(115 * SF)] *= 0.02
    thorax = np.sin(2 * np.pi * 0.25 * t + 0.1)
    return flow, thorax, ["N2"] * int(DUR_S // 30)


def _score(profile, cache):
    flow, thorax, hypno = _rec()
    return detect_respiratory_events(
        flow_data=flow, thorax_data=thorax, abdomen_data=thorax.copy(),
        spo2_data=None, sf_flow=SF, sf_spo2=1.0, hypno=hypno,
        scoring_profile=profile, _precomputed=cache)


def _tel(cache, naam):
    return sum(1 for k in cache if naam in k)


def test_schaalvrije_uitvaldetectie_krijgt_een_eigen_cache_regel():
    basis = dict(SCORING_PROFILES["aasm_v3_rec"])
    anders = dict(basis, FLOW_GAP_SCALE_FREE=True)
    cache = {}
    assert _score(basis, cache)["success"]
    assert _score(anders, cache)["success"]
    assert _tel(cache, "gap_resp") == 2, sorted(cache)


def test_basislijnvelden_sleutelen_de_effort_basislijn():
    basis = dict(SCORING_PROFILES["aasm_v3_rec"])
    anders = dict(basis, BASELINE_WINDOW_S=120, BASELINE_PERCENTILE=85.0)
    cache = {}
    _score(basis, cache)
    _score(anders, cache)
    assert _tel(cache, "effort_bl") == 2, sorted(cache)


def test_identieke_voorbewerking_deelt_de_cache_nog_steeds():
    """De sweep-armen verschillen alleen in kwalificatiedrempels; die mogen
    de cache niet splitsen, anders is de besparing weg."""
    strict = SCORING_PROFILES["aasm_v3_strict"]
    rec = SCORING_PROFILES["aasm_v3_rec"]
    cache = {}
    _score(rec, cache)
    _score(strict, cache)
    assert _tel(cache, "gap_resp") == 1
    assert _tel(cache, "effort_bl") == 1
    assert _tel(cache, "flow_env") == 1


def test_elke_sleutel_draagt_dezelfde_voorbewerkingssleutel_binnen_een_profiel():
    """Achter `::`, zodat de naam vooraan blijft staan en
    `k.startswith("flow_env")` blijft werken voor wie in de cache kijkt."""
    cache = {}
    _score(SCORING_PROFILES["aasm_v3_rec"], cache)
    assert all("::" in k for k in cache), sorted(cache)
    sleutels = {k.split("::", 1)[1] for k in cache}
    assert len(sleutels) == 1, sleutels
    assert any(k.startswith("flow_env") for k in cache)
