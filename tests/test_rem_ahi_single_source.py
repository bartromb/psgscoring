"""Eén REM-AHI, niet twee die elkaar tegenspreken.

`respiratory._compute_summary` schrijft `ahi_rem`/`ahi_nrem` over DEZELFDE
events als `ahi_total` (de kale `uncertain`-apneus vallen eruit, zie de
toelichting daar). `pipeline._compute_rera_rdi` rekende daarnaast een eigen
`rem_ahi`/`nrem_ahi` over ALLE events, inclusief die `uncertain`. Twee
getallen onder bijna dezelfde naam, en `_compute_phenotypes` las het
verkeerde paar voor het REM-predominante fenotype.

YASAFlaskified leest `ahi_rem`/`ahi_nrem` in de tabellen en `rem_ahi`/
`nrem_ahi` alleen nog als terugval en in het fenotypeblok. De sleutels
blijven dus bestaan, maar als ALIAS van het consistente paar.
"""
from psgscoring.pipeline import _compute_phenotypes, _compute_rera_rdi
from psgscoring.respiratory import _compute_summary

# 10 min REM, 50 min N2.
HYPNO = ["R"] * 20 + ["N2"] * 100


def _ev(typ, onset, stage):
    ep = int(onset // 30)
    return {"type": typ, "onset_s": float(onset), "duration_s": 15.0,
            "stage": stage, "epoch": ep, "confidence": 0.9,
            "desaturation_pct": 4.0}


EVENTS = [
    _ev("obstructive", 30.0, "R"),
    _ev("uncertain", 120.0, "R"),        # kale uncertain in REM
    _ev("hypopnea", 700.0, "N2"),
    _ev("uncertain", 900.0, "N2"),       # en een in NREM
]


def _summary_after_rera():
    summ = _compute_summary(EVENTS, HYPNO)
    out = {"respiratory": {"success": True, "events": list(EVENTS),
                           "summary": summ, "rejected_hypopneas": [],
                           "_breaths": []}}
    _compute_rera_rdi(out, HYPNO, arousals=[], artifact_epochs=None)
    return out["respiratory"]["summary"]


def test_de_aliassen_zijn_gelijk_aan_het_consistente_paar():
    s = _summary_after_rera()
    assert s["rem_ahi"] == s["ahi_rem"]
    assert s["nrem_ahi"] == s["ahi_nrem"]


def test_de_rem_ahi_telt_de_kale_uncertain_niet_mee():
    """1 obstructieve apneu in 10 min REM = 6,0/u; de uncertain erbij zou
    12,0/u geven. Dezelfde telling als `ahi_total`."""
    s = _summary_after_rera()
    assert s["ahi_rem"] == 6.0
    assert s["rem_ahi"] == 6.0
    assert s["ahi_nrem"] == 1.2          # 1 hypopnee in 50 min
    assert s["nrem_ahi"] == 1.2


def test_het_fenotype_leest_het_consistente_paar():
    """Een samenvatting waarin de twee paren uiteenlopen: het fenotype moet
    `ahi_rem`/`ahi_nrem` volgen, niet het oude pipeline-paar."""
    out = {
        "respiratory": {"summary": {"ahi_total": 20,
                                    "ahi_rem": 40, "ahi_nrem": 10,
                                    "rem_ahi": 3, "nrem_ahi": 2}},
        "position": {"summary": {}},
    }
    _compute_phenotypes(out, ["R"] * 80 + ["N2"] * 400)
    ph = out["respiratory"]["summary"]["phenotypes"]
    assert ph["rem_predominant"]["flag"] is True
    assert ph["rem_predominant"]["rem_ahi"] == 40
    assert ph["rem_predominant"]["nrem_ahi"] == 10
