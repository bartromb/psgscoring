"""Split-night: de nacht scheiden in diagnostiek en therapie.

WAAROM DIT BESTAAT
------------------
Eén AHI over diagnostiek + titratie verdunt de diagnose. Op de casus die dit
aanleiding gaf las het rapport "Mild SAS, AHI 10,1/u" bij een patiënt die de
verwijzer als ernstig kende; het diagnostische deel had ODI3 ~60/u.

WAAROM BEIDE SPOREN MOETEN MEEDOEN
----------------------------------
`test_een_flowstap_alleen_is_niet_genoeg` is de belangrijkste test hier. Op acht
gewone MESA-nachten liep de flow-amplitudeverhouding van 2,0 tot **202**: een
detector op flow alleen had acht van de acht nachten ten onrechte gesplitst.
De saturatiebasislijn draagt de specificiteit; de flowstap zegt alleen WANNEER.
"""
import numpy as np
import pytest

from psgscoring.split_night import (
    MIN_SEGMENT_S,
    detect_split_night,
    segment_indices,
)

SF = 10.0
UUR = int(3600 * SF)


def _nacht(n_uur, breuk_uur=None, amp_voor=1.0, amp_na=0.2,
           sat_voor=90.0, sat_na=96.0):
    n = int(n_uur * UUR)
    t = np.arange(n) / SF
    adem = np.sin(2 * np.pi * 0.25 * t)
    amp = np.full(n, amp_voor)
    sat = np.full(n, sat_voor)
    if breuk_uur is not None:
        k = int(breuk_uur * UUR)
        amp[k:] = amp_na
        sat[k:] = sat_na
    rng = np.random.default_rng(0)
    return adem * amp + rng.normal(scale=0.01, size=n), sat + rng.normal(scale=0.3, size=n)


def test_een_echte_split_wordt_gevonden_op_de_juiste_plek():
    flow, spo2 = _nacht(7, breuk_uur=2.5)
    r = detect_split_night(flow=flow, sf_flow=SF, spo2=spo2, sf_spo2=SF)
    assert r["detected"], r
    assert r["method"] == "flow_amplitude+spo2_baseline"
    assert abs(r["breakpoint_s"] - 2.5 * 3600) < 600, r["breakpoint_s"]


def test_een_gewone_nacht_wordt_niet_gesplitst():
    flow, spo2 = _nacht(7)
    r = detect_split_night(flow=flow, sf_flow=SF, spo2=spo2, sf_spo2=SF)
    assert not r["detected"], r


def test_een_flowstap_alleen_is_niet_genoeg():
    """Acht van acht MESA-nachten zouden anders vals-positief zijn.

    Daar liep de flowverhouding tot 202 zonder enige klinische betekenis --
    signaaluitval, verschoven canule, gewijzigde versterking.
    """
    flow, spo2 = _nacht(7, breuk_uur=2.5, sat_voor=95.0, sat_na=95.0)
    r = detect_split_night(flow=flow, sf_flow=SF, spo2=spo2, sf_spo2=SF)
    assert not r["detected"], r
    assert "saturatiestijging" in r["reason"]


def test_een_saturatiestijging_alleen_is_ook_niet_genoeg():
    flow, spo2 = _nacht(7, breuk_uur=2.5, amp_voor=1.0, amp_na=1.0)
    r = detect_split_night(flow=flow, sf_flow=SF, spo2=spo2, sf_spo2=SF)
    assert not r["detected"], r
    assert "flowamplitudestap" in r["reason"]


def test_een_te_kort_segment_telt_niet():
    """Twintig minuten therapie is een storing, geen titratie."""
    flow, spo2 = _nacht(7, breuk_uur=6.7)
    r = detect_split_night(flow=flow, sf_flow=SF, spo2=spo2, sf_spo2=SF)
    assert not r["detected"] or r["breakpoint_s"] <= 7 * 3600 - MIN_SEGMENT_S


def test_een_handmatig_breekpunt_wint_altijd():
    """Wie weet hoe laat de CPAP aanging, weet het beter dan een detector."""
    flow, spo2 = _nacht(7)          # zou niets detecteren
    r = detect_split_night(flow=flow, sf_flow=SF, spo2=spo2, sf_spo2=SF,
                           manual_breakpoint_s=9000)
    assert r["detected"] and r["method"] == "manual"
    assert r["breakpoint_s"] == 9000


def test_saturatie_die_volledig_uitvalt_laat_niets_omvallen():
    flow, spo2 = _nacht(7, breuk_uur=2.5)
    spo2[:] = 0.0                    # buiten de plausibiliteitsband
    r = detect_split_night(flow=flow, sf_flow=SF, spo2=spo2, sf_spo2=SF)
    assert not r["detected"]
    assert r["evidence"].get("spo2_before") is None


def test_zonder_signaal_geen_bewering():
    r = detect_split_night()
    assert not r["detected"] and r["reason"]


def _events(n_voor, n_na, breuk_s):
    ev = [{"onset_s": 60.0 + 120 * i, "type": "obstructive", "stage": "N2"}
          for i in range(n_voor)]
    ev += [{"onset_s": breuk_s + 60.0 + 120 * i, "type": "hypopnea", "stage": "N2"}
           for i in range(n_na)]
    return ev


def test_segmentindices_verdelen_de_events_en_de_slaaptijd():
    breuk = 2 * 3600.0
    hyp = ["N2"] * (7 * 120)                      # 7 uur slaap, epochs van 30 s
    s = segment_indices(_events(60, 6, breuk), hyp, breuk)
    assert s["diagnostic"]["n_events"] == 60
    assert s["therapeutic"]["n_events"] == 6
    assert s["diagnostic"]["sleep_h"] == pytest.approx(2.0, abs=0.05)
    assert s["diagnostic"]["ahi"] == pytest.approx(30.0, abs=0.5)
    assert s["therapeutic"]["ahi"] == pytest.approx(1.2, abs=0.3)


def test_segmentindices_tellen_uncertain_apart():
    """Bij een falende effort-band bepalen juist die het beeld."""
    breuk = 2 * 3600.0
    ev = _events(10, 2, breuk)
    ev += [{"onset_s": 100.0 + 60 * i, "type": "uncertain", "stage": "N2"}
           for i in range(20)]
    s = segment_indices(ev, ["N2"] * (7 * 120), breuk)
    assert s["diagnostic"]["n_uncertain"] == 20
    assert s["diagnostic"]["ahi"] == pytest.approx(5.0, abs=0.3)
    assert s["diagnostic"]["ahi_incl_uncertain"] == pytest.approx(15.0, abs=0.3)
