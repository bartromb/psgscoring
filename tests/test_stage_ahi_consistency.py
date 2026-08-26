"""Een stadium-AHI telt dezelfde events als de AHI zelf.

WAT ER MIS WAS
--------------
`ahi_rem` en `ahi_nrem` telden over de VOLLEDIGE eventlijst, inclusief de kale
`uncertain`-apneus die `ahi_total` juist uitsluit. Op een klinisch rapport van
26-08-2026 leverde dat een tegenspraak op die niemand kan rijmen:

    kop:            "Mild SAS — AHI 10,1/u"     (56 kwalificerende events)
    twee pagina's later:  "NREM-AHI 30,9/u"     (alle 128 events)

Beide klopten op zichzelf — 56/5,533 u = 10,1 en 128/5,533 u = 23,1 — maar ze
stonden naast elkaar alsof ze vergelijkbaar waren. Dat gebeurde omdat beide
RIP-banden waren uitgevallen, waardoor 72 van de 128 apneus niet getypeerd
konden worden.

DE INVARIANT die deze testen bewaken: de events onder `ahi_rem` en `ahi_nrem`
zijn samen precies de events onder `ahi_total`. Een stadium-index die andere
events telt dan zijn totaal, is geen deelverzameling maar een ander getal met
dezelfde naam.
"""
import pytest

from psgscoring.respiratory import _compute_summary

EPOCH = 30
UUR = 120          # epochs per uur


def _hypno(n_rem_epochs, n_nrem_epochs):
    return ["N2"] * n_nrem_epochs + ["R"] * n_rem_epochs


def _ev(onset, stage, typ):
    return {"onset_s": float(onset), "duration_s": 15.0, "stage": stage,
            "type": typ, "confidence": 0.9}


def _opzet():
    """Twee uur NREM + één uur REM, met ongetypeerde apneus in NREM.

    Dat is het klinische geval: effort-banden weg, dus apneus belanden als
    kale `uncertain` in de lijst.
    """
    hyp = _hypno(UUR, 2 * UUR)
    ev = []
    t = 60.0
    for _ in range(10):                       # 10 echte hypopneus in NREM
        ev.append(_ev(t, "N2", "hypopnea")); t += 120
    for _ in range(20):                       # 20 ONGETYPEERDE apneus in NREM
        ev.append(_ev(t, "N2", "uncertain")); t += 120
    t = 2 * 3600 + 60.0
    for _ in range(3):                        # 3 hypopneus in REM
        ev.append(_ev(t, "R", "hypopnea")); t += 120
    return ev, hyp


def test_de_stadium_ahi_s_tellen_niet_de_ongetypeerde_apneus():
    ev, hyp = _opzet()
    s = _compute_summary(ev, hyp)
    # NREM: 2 uur, 10 kwalificerende events -> 5,0/u. Met de 20 uncertain erbij
    # zou het 15,0/u zijn geweest, en dat was de fout.
    assert s["ahi_nrem"] == pytest.approx(5.0, abs=0.05), s["ahi_nrem"]
    assert s["ahi_rem"] == pytest.approx(3.0, abs=0.05), s["ahi_rem"]


def test_de_stadia_tellen_samen_op_tot_het_totaal():
    """De invariant: rem + nrem = totaal, in EVENTS, niet in indices."""
    ev, hyp = _opzet()
    s = _compute_summary(ev, hyp)
    rem_h, nrem_h = s["rem_min"] / 60, s["nrem_min"] / 60
    uit_stadia = s["ahi_rem"] * rem_h + s["ahi_nrem"] * nrem_h
    uit_totaal = s["ahi_total"] * s["index_denominator_h"]
    assert uit_stadia == pytest.approx(uit_totaal, abs=0.2), (
        f"stadia geven {uit_stadia:.1f} events, het totaal {uit_totaal:.1f}")


def test_de_incl_uncertain_variant_telt_ze_wel_en_klopt_ook():
    ev, hyp = _opzet()
    s = _compute_summary(ev, hyp)
    assert s["ahi_nrem_incl_uncertain"] == pytest.approx(15.0, abs=0.05)
    assert s["ahi_rem_incl_uncertain"] == pytest.approx(3.0, abs=0.05)
    rem_h, nrem_h = s["rem_min"] / 60, s["nrem_min"] / 60
    uit_stadia = (s["ahi_rem_incl_uncertain"] * rem_h
                  + s["ahi_nrem_incl_uncertain"] * nrem_h)
    uit_totaal = s["ahi_incl_uncertain"] * s["index_denominator_h"]
    assert uit_stadia == pytest.approx(uit_totaal, abs=0.2)


def test_het_aandeel_ongetypeerd_wordt_gerapporteerd():
    """Zonder dit getal kan een rapport niet zien dat de AHI onvolledig is."""
    ev, hyp = _opzet()
    s = _compute_summary(ev, hyp)
    # 20 ongetypeerd van 33 events totaal
    assert s["uncertain_fraction"] == pytest.approx(20 / 33, abs=0.01)
    assert s["n_uncertain_apnea"] == 20


def test_zonder_ongetypeerde_apneus_verandert_er_niets():
    """De reparatie mag een gewone opname niet aanraken."""
    hyp = _hypno(UUR, 2 * UUR)
    ev = [_ev(60 + 120 * i, "N2", "hypopnea") for i in range(10)]
    ev += [_ev(2 * 3600 + 60 + 120 * i, "R", "obstructive") for i in range(3)]
    s = _compute_summary(ev, hyp)
    assert s["ahi_nrem"] == s["ahi_nrem_incl_uncertain"]
    assert s["ahi_rem"] == s["ahi_rem_incl_uncertain"]
    assert s["uncertain_fraction"] == 0.0
