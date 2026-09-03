"""Gegradeerde apneu-subtypering alleen wáár periodieke ademhaling is.

DE AANLEIDING, GEMETEN (450-run, 375 opnames, per opname bewaard)
=================================================================
De gradering (s=0,25) is basiskansafhankelijk — wat vier gepoolde runs vier
verschillende oordelen gaf, bleek Simpson's paradox:

    prevalentie centrale apneus   +juist   +vals    ruil
    < 2 %   (58 opnames)              +4     +86   1:21,5
    2–5 %   ( 8 opnames)              +1     +15   1:15,0
    5–15 %  (20 opnames)             +11    +121   1:11,0
    > 15 %  (12 opnames)            +115     +57   1:0,5

Gepaard over opnames met beide klassen: Δκ +0,003, 11/22, p=0,29. De winst
bestaat alleen op nachten met periodieke ademhaling; de schade overal elders.

DE POORT
========
`shape_evidence_csr_gate` (default UIT = uitgerold gedrag): staat hij aan,
dan geldt de gradering alleen wanneer de bestaande CSR-detector op deze
opname periodieke ademhaling ziet. Een binaire contextpoort op een bestaande
detector — géén continu adaptief werkpunt (dat is bij de hypopneu-strictness
op zijn orakelplafond weerlegd).

De beslissing valt VÓÓR de classificatie, op een voorcontrole over hetzelfde
flow-envelope; de provenance staat in de summary zodat een rapportlezer kan
zien waaróm er wel of niet gegradeerd is.
"""
import mne
import numpy as np

import psgscoring
from psgscoring.constants import _profile_to_legacy_dict as _L
from psgscoring.profiles import PROFILES

SF = 32.0
N_S = 900.0


def _flow(csr: bool, seed=0):
    """Ademflow: 0,25 Hz teugen, bij CSR gemoduleerd met een 60 s-cyclus."""
    t = np.arange(int(N_S * SF)) / SF
    adem = np.sin(2 * np.pi * 0.25 * t)
    if csr:
        mod = 0.55 + 0.45 * np.sin(2 * np.pi * t / 60.0)
    else:
        mod = 1.0
    rng = np.random.default_rng(seed)
    return adem * mod + rng.normal(0, 0.02, len(t))


def _run(flow, gate: bool):
    info = mne.create_info(["Pres", "SpO2"], SF, ch_types="misc", verbose=False)
    spo2 = np.full(len(flow), 96.0)
    raw = mne.io.RawArray(np.vstack([flow, spo2]), info, verbose=False)
    hypno = ["N2"] * int(N_S // 30)
    import os
    os.environ["PSGSCORING_SHAPE_EVIDENCE_CSR_GATE"] = "1" if gate else "0"
    try:
        return psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                              scoring_profile="aasm_v3_rec")
    finally:
        os.environ.pop("PSGSCORING_SHAPE_EVIDENCE_CSR_GATE", None)


def test_default_staat_UIT_en_is_gedragsneutraal():
    d = _L(PROFILES["aasm_v3_rec"])
    assert d["SHAPE_EVIDENCE_CSR_GATE"] is False, (
        "default moet het uitgerolde gedrag zijn")
    uit = _run(_flow(csr=True), gate=False)
    s = (uit.get("respiratory") or {}).get("summary") or {}
    assert "shape_evidence_gate" not in s, (
        "zonder de vlag hoort er geen poortveld te bestaan -- golden-stabiel")


def test_poort_aan_CSR_aanwezig_dus_gegradeerd():
    uit = _run(_flow(csr=True), gate=True)
    g = ((uit.get("respiratory") or {}).get("summary") or {}).get(
        "shape_evidence_gate")
    assert g is not None, "provenance ontbreekt"
    assert g["csr_detected"] is True, (
        "een 60 s-crescendo-decrescendo hoort de CSR-detector te halen")
    assert g["graded"] is True


def test_poort_aan_GEEN_CSR_dus_niet_gegradeerd():
    uit = _run(_flow(csr=False), gate=True)
    g = ((uit.get("respiratory") or {}).get("summary") or {}).get(
        "shape_evidence_gate")
    assert g is not None
    assert g["csr_detected"] is False, (
        "regelmatige ademhaling mag de detector niet halen; anders is de "
        "poort een dode letter en geldt de gradering overal")
    assert g["graded"] is False


def test_de_poort_overleeft_de_summary_herberekening():
    """Stap 9 (CSR-detectie) herberekent de summary wanneer CSR gevonden is.

    Precies in dat geval — poort aan, CSR aanwezig — moet het poortveld er
    NA die herberekening nog staan. Leveringsoppervlak: een veld dat door een
    latere stap wordt weggegooid is niet geleverd. Dat is hier de gevaarlijke
    combinatie, want de herberekening draait alleen bij csr_detected=True.
    """
    uit = _run(_flow(csr=True), gate=True)
    s = (uit.get("respiratory") or {}).get("summary") or {}
    assert "shape_evidence_gate" in s
