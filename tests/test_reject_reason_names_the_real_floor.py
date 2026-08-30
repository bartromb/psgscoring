"""
De afwijzingsreden moet de drempel noemen die WERKELIJK gehanteerd is.

`_validate_local_reduction` verscherpt zijn eigen vloer wanneer de omliggende
ademhaling stabiel is: `min_reduction_pct` wordt dan opgetrokken naar
`stability_strict_reduction`. De aanroeper wist daar niets van en schreef een
hardgecodeerde 20 in de reden. Resultaat: afwijzingen als

    local_reduction_28.6pct<20pct

die er onzinnig uitzien -- 28,6 is meer dan 20 -- en het niet zijn: de vloer
stond op dat moment op 30.

Dat is niet alleen lelijk. `event_review._rejection_nearness` in
YASAFlaskified deelt die twee getallen op elkaar om te bepalen hoe dicht een
kandidaat bij scoren kwam, en kiest daarop welke events een beoordelaar te zien
krijgt. Met een verkeerde noemer is die rangschikking scheef: 28,6/20 = 1,43
(afgekapt op 1,0, dus "grensgeval") tegen de werkelijke 28,6/30 = 0,95.
"""
import numpy as np

from psgscoring.respiratory import _validate_local_reduction

SF = 32.0


def _env(pre_niveau, event_niveau, pre_s=60.0, ev_s=20.0, ruis=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pre = np.full(int(pre_s * SF), pre_niveau)
    ev = np.full(int(ev_s * SF), event_niveau)
    sig = np.concatenate([pre, ev])
    if ruis:
        sig = sig * (1.0 + rng.normal(0, ruis, sig.size))
    return sig, int(pre_s * SF), sig.size


def test_de_verscherpte_vloer_wordt_teruggegeven():
    """Stabiele ademhaling -> vloer 20 wordt 30, en dat staat in de uitvoer."""
    env, a, b = _env(1.0, 0.72)          # 28 % reductie, vlak = lage CV
    geldig, gemeten, vloer = _validate_local_reduction(
        env, a, b, SF, min_reduction_pct=20.0, stability_strict_reduction=30.0)
    assert geldig is False
    assert gemeten == 28.0
    assert vloer == 30.0, (
        f"de teruggegeven vloer is {vloer}, niet de verscherpte 30 -- dan kan "
        "de aanroeper de reden nog steeds niet correct opschrijven")


def test_zonder_verscherping_blijft_de_vloer_de_ingangswaarde():
    """Onstabiele ademhaling: de stabiliteitstak vuurt niet."""
    env, a, b = _env(1.0, 0.72, ruis=0.9, seed=3)
    _geldig, _gemeten, vloer = _validate_local_reduction(
        env, a, b, SF, min_reduction_pct=20.0, stability_strict_reduction=30.0)
    assert vloer == 20.0


def test_de_niet_gemeten_gevallen_dragen_ook_een_vloer():
    """Te weinig pre-event signaal: geen meting, maar wel een leesbare vloer."""
    env = np.full(int(30 * SF), 1.0)
    geldig, gemeten, vloer = _validate_local_reduction(
        env, 10, 100, SF, min_reduction_pct=20.0)
    assert geldig is True and np.isnan(gemeten) and vloer == 20.0


def test_de_reden_uit_de_detector_noemt_geen_hardgecodeerde_twintig():
    """Door de hele detector heen: geen enkele reden mag '<20pct' zeggen
    terwijl het profiel een andere vloer hanteert."""
    from psgscoring.respiratory import detect_respiratory_events

    dur, br = 900, 0.25
    t = np.arange(int(SF * dur)) / SF
    rng = np.random.default_rng(7)
    flow = np.sin(2 * np.pi * br * t) + rng.normal(0, 0.005, t.size)
    for start in range(120, dur - 120, 180):
        flow[int(start * SF):int((start + 18) * SF)] *= 0.65   # 35 % reductie
    # De reductie moet DOOR het 0,70-masker (dus > 30 %) en STUKLOPEN op de
    # lokale vloer. Dat kan alleen als die vloer hoger ligt dan de
    # maskerdrempel; vandaar 45. Precies daar liep de oude reden het hardst
    # mis: hij meldde "<20pct" terwijl er 45 gold.
    r = detect_respiratory_events(
        flow_data=flow, thorax_data=None, abdomen_data=None,
        spo2_data=np.full(dur, 97.0), sf_flow=SF, sf_spo2=1.0,
        hypno=["N2"] * (dur // 30),
        scoring_profile={"LOCAL_BL_MIN_REDUCTION_PCT": 45.0,
                         "LOCAL_BL_STRICT_RED": 45.0,
                         "LOCAL_BL_CV_THRESHOLD": 0.30})
    redenen = [str(x.get("reject_reason") or "")
               for x in r.get("rejected_hypopneas", [])]
    lokaal = [x for x in redenen if x.startswith("local_reduction")]
    assert lokaal, "fixture levert geen lokale-basislijn-afwijzingen -- meet niets"
    for reden in lokaal:
        gemeten, vloer = reden.replace("local_reduction_", "").replace("pct", "").split("<")
        assert float(vloer) == 45.0, (
            f"'{reden}' noemt vloer {vloer}, maar het profiel hanteert 45 -- "
            "de reden is dus nog steeds hardgecodeerd")
        assert float(gemeten) < float(vloer), (
            f"'{reden}' beweert een afwijzing terwijl de gemeten reductie de "
            "genoemde vloer haalt -- de genoemde vloer klopt dus niet")
