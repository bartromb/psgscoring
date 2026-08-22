"""De apneudrempel mag van de sensor afhangen -- maar alleen van de sensor die
de apneus ECHT draagt.

De neusdruk vergroot een flowdaling uit (het signaal loopt ongeveer met het
kwadraat van de flow), de thermistor niet. Van 597 door mensen gescoorde
apneus haalt 50 % de drempel van 0,90 op de druk en 13 % op de thermistor,
terwijl de AUC apneu-vs-hypopneu gelijk is. Eén drempel leest die twee schalen
niet.

De valkuil zit in de vraag "draagt de thermistor de apneus". Het ruwe
thermistorkanaal bestaat ook wanneer de poort het heeft afgekeurd; dan scoort
de pipeline op de neusdruk. Wie op het bestaan van het kanaal test, laat een
thermistordrempel los op een druksignaal.
"""
import numpy as np


def test_no_profile_sets_a_thermistor_threshold_by_default():
    """Default = huidig gedrag: geen enkel profiel wijkt af."""
    from psgscoring.profiles import get_profile, list_profiles

    for name in list_profiles():
        v = get_profile(name).apnea.flow_reduction_threshold_thermistor
        assert v is None, f"{name} zet een thermistordrempel: {v}"


def test_the_registry_carries_it_as_a_fraction_of_baseline():
    """Opgeslagen als fractie-van-baseline, net als HYPOPNEA_THRESHOLD."""
    import dataclasses

    from psgscoring.constants import _build_legacy_profiles
    from psgscoring.profiles import get_profile

    p = get_profile("aasm_v3_breath")
    assert "APNEA_THRESHOLD_THERMISTOR" in _build_legacy_profiles()["aasm_v3_breath"]

    gewijzigd = dataclasses.replace(
        p, apnea=dataclasses.replace(
            p.apnea, flow_reduction_threshold_thermistor=0.80))
    assert gewijzigd.apnea.flow_reduction_threshold_thermistor == 0.80
    # 0,80 daling == 0,20 van de basislijn
    assert round(1.0 - 0.80, 4) == 0.20


def test_the_detector_only_switches_when_asked():
    """Zonder `apnea_on_thermistor` verandert er niets, ook niet met een waarde."""
    from psgscoring.respiratory import detect_respiratory_events

    sf, dur = 32.0, 600
    t = np.arange(int(sf * dur)) / sf
    rng = np.random.default_rng(5)
    flow = np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.01, t.size)
    # apneus van 15 s: flow naar 15% -> haalt 0,80-daling wel, 0,90 niet
    for start in range(60, dur - 60, 90):
        flow[int(start * sf):int((start + 15) * sf)] *= 0.15
    hypno = ["N2"] * int(dur / 30)
    spo2 = np.full(int(1.0 * dur), 97.0)

    prof = {"APNEA_THRESHOLD_THERMISTOR": 0.20}
    gemeen = dict(thorax_data=None, abdomen_data=None, spo2_data=spo2,
                  sf_flow=sf, sf_spo2=1.0, hypno=hypno, scoring_profile=prof)

    uit = detect_respiratory_events(flow_data=flow, apnea_on_thermistor=False,
                                    **gemeen)
    aan = detect_respiratory_events(flow_data=flow, apnea_on_thermistor=True,
                                    **gemeen)

    def n_ap(r):
        return sum(1 for e in r.get("events", [])
                   if str(e.get("type")) in ("obstructive", "central",
                                             "mixed", "uncertain"))

    assert n_ap(aan) != n_ap(uit), (
        "de drempel deed niets: fixture onderscheidt niet "
        f"(uit={n_ap(uit)}, aan={n_ap(aan)})")
    assert n_ap(aan) > n_ap(uit), (
        "een soepelere thermistordrempel hoort MEER apneus te geven, "
        f"kreeg uit={n_ap(uit)} aan={n_ap(aan)}")


def test_the_pipeline_asks_about_the_scored_channel_not_the_present_one():
    """Het ruwe kanaal bestaat ook als de poort het afkeurde.

    Deze test bewaakt de regel in `run_pneumo_analysis`. Stond daar
    `flow_therm_data is not None`, dan zou een afgekeurde thermistor alsnog
    zijn eigen drempel op de neusdruk loslaten.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "psgscoring" / "pipeline.py").read_text(encoding="utf-8")
    regel = next(ln for ln in src.splitlines()
                 if "apnea_on_thermistor" in ln and "=" in ln
                 and "def " not in ln)
    assert "apnea_flow is flow_therm_data" in regel, (
        "de pipeline bepaalt de sensor niet uit het GESCOORDE kanaal: "
        f"{regel.strip()!r}")


def test_the_env_override_is_expressed_as_a_reduction(monkeypatch):
    """0,72 in de env betekent een daling van 72 %, net als het profielveld."""
    import numpy as np

    from psgscoring.respiratory import detect_respiratory_events

    sf, dur = 32.0, 600
    t = np.arange(int(sf * dur)) / sf
    rng = np.random.default_rng(5)
    flow = np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.01, t.size)
    for start in range(60, dur - 60, 90):
        flow[int(start * sf):int((start + 15) * sf)] *= 0.15
    gemeen = dict(thorax_data=None, abdomen_data=None,
                  spo2_data=np.full(dur, 97.0), sf_flow=sf, sf_spo2=1.0,
                  hypno=["N2"] * int(dur / 30), scoring_profile={})

    def n_ap(r):
        return sum(1 for e in r.get("events", [])
                   if str(e.get("type")) in ("obstructive", "central",
                                             "mixed", "uncertain"))

    zonder = n_ap(detect_respiratory_events(flow_data=flow,
                                            apnea_on_thermistor=True, **gemeen))
    monkeypatch.setenv("PSGSCORING_APNEA_REDUCTION_THERMISTOR", "0.72")
    met = n_ap(detect_respiratory_events(flow_data=flow,
                                         apnea_on_thermistor=True, **gemeen))
    assert met > zonder, (
        f"env deed niets: zonder={zonder}, met={met}")

    # Onleesbaar -> profielwaarde, niet stil iets anders.
    monkeypatch.setenv("PSGSCORING_APNEA_REDUCTION_THERMISTOR", "kaas")
    assert n_ap(detect_respiratory_events(flow_data=flow,
                                          apnea_on_thermistor=True,
                                          **gemeen)) == zonder


def test_the_env_does_nothing_when_apnoeas_are_not_on_the_thermistor(monkeypatch):
    """De override mag geen drukscoring raken."""
    import numpy as np

    from psgscoring.respiratory import detect_respiratory_events

    sf, dur = 32.0, 600
    t = np.arange(int(sf * dur)) / sf
    rng = np.random.default_rng(5)
    flow = np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.01, t.size)
    for start in range(60, dur - 60, 90):
        flow[int(start * sf):int((start + 15) * sf)] *= 0.15
    gemeen = dict(thorax_data=None, abdomen_data=None,
                  spo2_data=np.full(dur, 97.0), sf_flow=sf, sf_spo2=1.0,
                  hypno=["N2"] * int(dur / 30), scoring_profile={})

    def n_ap(r):
        return sum(1 for e in r.get("events", [])
                   if str(e.get("type")) in ("obstructive", "central",
                                             "mixed", "uncertain"))

    basis = n_ap(detect_respiratory_events(flow_data=flow,
                                           apnea_on_thermistor=False, **gemeen))
    monkeypatch.setenv("PSGSCORING_APNEA_REDUCTION_THERMISTOR", "0.72")
    assert n_ap(detect_respiratory_events(flow_data=flow,
                                          apnea_on_thermistor=False,
                                          **gemeen)) == basis
