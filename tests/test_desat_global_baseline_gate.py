"""
De globale SpO₂-basislijn mag niet ALTIJD de lokale overnemen.

`get_desaturation` meet de daling tegen de lokale pre-event basislijn (90e
percentiel over 120 s) en vervangt die door de globale (95e percentiel over alle
slaap) zodra de globale hoger ligt. Dat is bedoeld voor ernstige OSAS, waar het
pre-event venster zelf al gedaald kan zijn.

Bij een CHRONISCHE desatureerder werkt het averechts. Ligt de echte basislijn
van een COPD- of OHS-patiënt op 88 %, dan tilt de overname hem naar bijvoorbeeld
92 % en wordt elke daling vier procentpunt dieper gemeten dan ze is. Dat telt te
veel, en het is nergens in de uitvoer te zien.

`global_baseline_min_local_pct` begrenst dat. De parameter BESTOND al in
`spo2.get_desaturation` maar werd door geen enkele aanroeper doorgegeven -- een
knop die niets deed. Deze module dekt de bedrading af, en vooral: dat de default
`None` het bestaande gedrag houdt.
"""
import numpy as np
import pytest

from psgscoring.spo2 import get_desaturation

SF = 1.0


def _spo2(basis, dip, dip_start=210, dip_dur=15, n=600):
    """Let op de dip_start: 10 s NA de eventonset van 200.

    `get_desaturation` verwerpt een nadir binnen 3 s na de onset tenzij de
    daling minstens `early_nadir_min_drop_pct` (5 %) is -- circulatievertraging.
    Een dip die op de onset zelf begint, wordt bij een daling van 4 % dus
    weggegooid en de test meet dan de vertragingsregel in plaats van de
    basislijnovername."""
    s = np.full(n, float(basis))
    s[dip_start:dip_start + dip_dur] = float(basis - dip)
    return s


# ══════════════════════════════════════════════════════════════════════
#  1. Default = ongewijzigd
# ══════════════════════════════════════════════════════════════════════

def test_elk_profiel_laat_de_grens_ongezet():
    from psgscoring.constants import SCORING_PROFILES
    for naam, d in SCORING_PROFILES.items():
        assert d["DESAT_GLOBAL_BL_MIN_LOCAL_PCT"] is None, (
            f"{naam} begrenst de overname; dat is een gedragswijziging")


def test_zonder_grens_neemt_de_globale_basislijn_altijd_over():
    """Bestaand gedrag: lokaal 88, globaal 92 -> gemeten tegen 92."""
    s = _spo2(88, 4)
    desat, nadir = get_desaturation(s, 200.0, 15.0, SF, global_spo2_baseline=92.0)
    assert nadir == 84.0
    assert desat == pytest.approx(8.0, abs=0.5), (
        f"verwacht ~8 (92-84), kreeg {desat} -- de overname vuurde niet")


# ══════════════════════════════════════════════════════════════════════
#  2. Met een grens vuurt hij alleen waar hij hoort
# ══════════════════════════════════════════════════════════════════════

def test_met_een_grens_blijft_een_plausibele_lokale_basislijn_staan():
    """Lokaal 88 ligt boven de grens 85 -> geen overname, meet tegen 88."""
    s = _spo2(88, 4)
    desat, nadir = get_desaturation(s, 200.0, 15.0, SF, global_spo2_baseline=92.0,
                                    global_baseline_min_local_pct=85.0)
    assert nadir == 84.0
    assert desat == pytest.approx(4.0, abs=0.5), (
        f"verwacht ~4 (88-84), kreeg {desat} -- de grens hield de overname niet tegen")


def test_een_implausibel_lage_lokale_basislijn_wordt_wel_overgenomen():
    """Lokaal 80 ligt onder de grens 85 -> overname, precies waarvoor hij is."""
    s = _spo2(80, 4)
    desat, _ = get_desaturation(s, 200.0, 15.0, SF, global_spo2_baseline=95.0,
                                global_baseline_min_local_pct=85.0)
    assert desat == pytest.approx(19.0, abs=1.0), (
        f"verwacht ~19 (95-76), kreeg {desat} -- de overname hoort hier juist wel")


def test_de_grens_kan_de_telling_alleen_verlagen():
    """Begrenzen kan een desaturatie kleiner maken, nooit groter."""
    for basis, glob in ((88, 92), (80, 95), (95, 96), (90, 90)):
        s = _spo2(basis, 4)
        zonder, _ = get_desaturation(s, 200.0, 15.0, SF, global_spo2_baseline=float(glob))
        met, _ = get_desaturation(s, 200.0, 15.0, SF, global_spo2_baseline=float(glob),
                                  global_baseline_min_local_pct=85.0)
        assert met <= zonder + 1e-9, f"basis {basis}: {met} > {zonder}"


# ══════════════════════════════════════════════════════════════════════
#  3. De bedrading -- de knop moet de detector BEREIKEN
# ══════════════════════════════════════════════════════════════════════

def _detector(extra, env=None, monkeypatch=None):
    from psgscoring.respiratory import detect_respiratory_events
    if env is not None:
        monkeypatch.setenv("PSGSCORING_DESAT_GLOBAL_BL_MIN_LOCAL_PCT", env)
    dur = 600
    t = np.arange(int(32.0 * dur)) / 32.0
    flow = np.sin(2 * np.pi * 0.25 * t)
    return detect_respiratory_events(
        flow_data=flow, thorax_data=None, abdomen_data=None,
        spo2_data=np.full(dur, 90.0), sf_flow=32.0, sf_spo2=1.0,
        hypno=["N2"] * (dur // 30), scoring_profile=extra)


def _chronische_desatureerder(gate):
    """Lokale basislijn 88 %, globaal 95e percentiel ~92 %, dips van 2,5 %.

    Zonder grens wordt elke dip tegen 92 gemeten (7,5 %, ruim boven de 3 %-eis)
    en scoort het event. Met een grens van 85 blijft de lokale 88 staan, wordt
    de dip 2,5 % en HAALT hij de eis niet meer. Dat is exact het klinische
    verschil: bij COPD/OHS telt de overname events mee die er geen zijn.

    De eerste hypopnee blijft in beide armen staan -- haar pre-event venster
    bevat nog het hoge stuk aan het begin, dus daar IS de lokale basislijn
    werkelijk hoog. Dat hoort zo en is geen fout in de fixture.
    """
    from psgscoring.respiratory import detect_respiratory_events
    dur, br = 900, 0.25
    t = np.arange(int(32.0 * dur)) / 32.0
    rng = np.random.default_rng(5)
    flow = np.sin(2 * np.pi * br * t) + rng.normal(0, 0.005, t.size)
    spo2 = np.full(dur, 88.0)
    spo2[0:120] = 93.0
    starts = list(range(200, dur - 120, 150))
    for s0 in starts:
        flow[int(s0 * 32.0):int((s0 + 20) * 32.0)] *= 0.45
        spo2[s0 + 12:s0 + 28] = 85.5
    prof = {"STABILITY_FILTER_CV": 0.0, "LOCAL_BL_MIN_REDUCTION_PCT": 10.0,
            "LOCAL_BL_STRICT_RED": 10.0}
    if gate is not None:
        prof["DESAT_GLOBAL_BL_MIN_LOCAL_PCT"] = gate
    r = detect_respiratory_events(
        flow_data=flow, thorax_data=None, abdomen_data=None, spo2_data=spo2,
        sf_flow=32.0, sf_spo2=1.0, hypno=["N2"] * (dur // 30),
        scoring_profile=prof)
    hyp = [e for e in r["events"] if "hypopnea" in str(e.get("type"))]
    return hyp, len(starts)


def test_de_grens_verandert_de_SCORING_niet_alleen_het_herkomstveld():
    """De knop moet de detector BEREIKEN, niet enkel in de uitvoer staan.

    Een eerdere versie van deze test controleerde alleen
    `scoring_thresholds[...]`, en die bleef groen toen de doorgifte naar
    `get_desaturation` verwijderd werd. Dat is precies het soort test dat
    niets bewaakt.
    """
    zonder, gepland = _chronische_desatureerder(None)
    met, _ = _chronische_desatureerder(85.0)
    assert len(zonder) == gepland, "fixture scoort zonder grens niet alles"
    assert len(met) < len(zonder), (
        f"de grens verandert de telling niet ({len(met)} tegen {len(zonder)}) "
        "-- de waarde bereikt get_desaturation dus niet")
    assert [e["desaturation_pct"] for e in zonder][1] > \
           max([e["desaturation_pct"] for e in met[1:]], default=0), (
        "de gemeten desaturaties veranderen niet")


def test_de_profielwaarde_bereikt_de_detector():
    r = _detector({"DESAT_GLOBAL_BL_MIN_LOCAL_PCT": 88.0})
    assert r["scoring_thresholds"]["desat_global_bl_min_local_pct"] == 88.0


def test_de_env_override_wint(monkeypatch):
    r = _detector({"DESAT_GLOBAL_BL_MIN_LOCAL_PCT": 88.0}, env="80", monkeypatch=monkeypatch)
    assert r["scoring_thresholds"]["desat_global_bl_min_local_pct"] == 80.0


def test_onleesbare_env_valt_terug_op_het_profiel(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="psgscoring.respiratory"):
        r = _detector({"DESAT_GLOBAL_BL_MIN_LOCAL_PCT": 88.0}, env="laag",
                      monkeypatch=monkeypatch)
    assert r["scoring_thresholds"]["desat_global_bl_min_local_pct"] == 88.0
    assert any("geen getal" in m for m in caplog.messages)


def test_ongezet_blijft_none_in_de_uitvoer():
    r = _detector({})
    assert r["scoring_thresholds"]["desat_global_bl_min_local_pct"] is None
