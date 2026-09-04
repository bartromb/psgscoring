"""De tweepassagepoort: gradeer alleen waar passage 1 al veel centraal ziet.

HET BESLUIT (gebruiker, 2026-09-04)
===================================
Vijf subtyperingsruns lieten zien dat er op onselecte MESA geen instelling
bestaat die overal domineert; de keuze viel op de gerepliceerde middenweg:

  * passage 1 = classificatie ZONDER gradering (oud gedrag);
  * ziet die op deze opname > 15 % centrale apneus bij >= 5 apneus, dan gelden
    de GEGRADEERDE oordelen (s=0,25) -- de CSR/periodieke-ademhalingsnachten;
  * anders blijven de passage-1-oordelen staan.

Gemeten (450-run-afleiding + verse replicatie, per opname bewaard):

  s=0,25 overal   kappa 0,191   vals 344     poort   kappa 0,180   vals 271

De ruil -- 0,01 kappa voor ~20 % minder valse centrale apneus in de gewone
kliniek -- is een klinische keuze en is expliciet zo genomen.

De VLF-CSR-poort (`shape_evidence_csr_gate`) is hiervoor GEEN kandidaat: de
autocorrelatiepiek volgt de prevalentie niet (kappa vlak over elke drempel).
"""
import numpy as np

import mne

from psgscoring.constants import _profile_to_legacy_dict as _L
from psgscoring.profiles import PROFILES
from psgscoring.respiratory import apply_two_pass_gate


def _apneu(t_ongegradeerd, t_gegradeerd):
    return {"type": t_ongegradeerd, "onset_s": 0.0, "duration_s": 12.0,
            "confidence": 0.5, "classify_detail": {"pass": 1},
            "_graded_alt": (t_gegradeerd, 0.7, {"pass": 2})}


def test_default_wacht_op_de_herafleiding():
    """Het besluit vóór de poort staat, maar de gemeten drempel (0,15) geldt
    voor de fractie over MENSELIJK gekoppelde apneus; productie rekent over
    al onze apneus en die variabele verdunt (3012: 0,583 tegen 0,087). Tot de
    drempel op de productievariabele is herafgeleid en gerepliceerd blijft de
    default False -- anders rolt de poort de facto alles terug."""
    d = _L(PROFILES["aasm_v3_rec"])
    assert d["SHAPE_EVIDENCE_TWO_PASS"] is False
    assert d["TWO_PASS_CENTRAL_FRACTION"] == 0.15
    assert d["TWO_PASS_MIN_APNEAS"] == 5


def test_bevroren_profielen_blijven_erbuiten():
    for naam in ("mesa_shhs", "chicago_1999", "aasm_v1_rec", "aasm_v2_rec",
                 "cms_medicare"):
        assert _L(PROFILES[naam])["SHAPE_EVIDENCE_TWO_PASS"] is False, naam


def test_boven_de_drempel_gelden_de_gegradeerde_oordelen():
    ev = [_apneu("central", "central")] * 2 + [_apneu("obstructive", "central")] * 8
    uit, prov = apply_two_pass_gate([dict(e) for e in ev])
    assert prov["gated"] is True
    assert prov["pass1_central_fraction"] == 0.2
    assert sum(1 for e in uit if e["type"] == "central") == 10, (
        "boven de drempel horen de gegradeerde oordelen te gelden")
    assert all("_graded_alt" not in e for e in uit)


def test_onder_de_drempel_blijft_passage_1_staan():
    ev = [_apneu("central", "central")] + [_apneu("obstructive", "central")] * 9
    uit, prov = apply_two_pass_gate([dict(e) for e in ev])
    assert prov["gated"] is False and prov["pass1_central_fraction"] == 0.1
    assert sum(1 for e in uit if e["type"] == "central") == 1, (
        "onder de drempel mag de gradering NIET doorwerken")
    assert all("_graded_alt" not in e for e in uit)


def test_te_weinig_apneus_betekent_passage_1():
    """Vier centrale van vier is 100 %, maar vier apneus dragen geen oordeel
    over de nacht -- de afleiding gebruikte >= 5 en dat ligt vast."""
    ev = [_apneu("central", "central")] * 4
    uit, prov = apply_two_pass_gate([dict(e) for e in ev])
    assert prov["gated"] is False and prov["n_apneas"] == 4


def test_hypopneus_blijven_onaangeraakt():
    hyp = {"type": "hypopnea", "onset_s": 5.0, "duration_s": 15.0}
    ev = [dict(hyp)] + [_apneu("central", "central")] * 6
    uit, _ = apply_two_pass_gate([dict(e) for e in ev])
    assert uit[0] == hyp


def test_de_poort_bereikt_de_LEVERING():
    """summary["two_pass_gate"] moet er e2e staan (en overleeft stap 9)."""
    import os

    import psgscoring

    sf, n_s = 32.0, 600.0
    t = np.arange(int(sf * n_s)) / sf
    rng = np.random.default_rng(2)
    flow = np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.02, len(t))
    info = mne.create_info(["Pres", "SpO2"], sf, ch_types="misc", verbose=False)
    raw = mne.io.RawArray(np.vstack([flow, np.full(len(t), 96.0)]), info,
                          verbose=False)
    hypno = ["N2"] * int(n_s // 30)
    os.environ["PSGSCORING_SHAPE_EVIDENCE_TWO_PASS"] = "1"
    try:
        uit = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                             scoring_profile="aasm_v3_rec")
    finally:
        os.environ.pop("PSGSCORING_SHAPE_EVIDENCE_TWO_PASS", None)
    s = (uit.get("respiratory") or {}).get("summary") or {}
    assert "two_pass_gate" in s, "provenance ontbreekt in de summary"
    assert s["two_pass_gate"]["gated"] in (True, False)

    uit2 = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                          scoring_profile="aasm_v3_rec")
    s2 = (uit2.get("respiratory") or {}).get("summary") or {}
    assert "two_pass_gate" not in s2, "default (uit) hoort geen veld te dragen"
