"""Fix 2 (SpO2-kruiscontaminatie) vergelijkt met het VORIGE event in de tijd.

WAT ER MIS WAS
--------------
`_spo2_cross_contaminated` keek naar het laatst TOEGEVOEGDE event in de lijst.
De hypopneedetector krijgt die lijst binnen met alle apneus er al in, en
voegt zijn eigen kandidaten achteraan toe. "Laatst toegevoegd" is dan niet
"vlak hiervoor": een hypopneu op 120 s werd vergeleken met een apneu op
20000 s (geen contaminatie, terecht) terwijl de apneu op 100 s wel degelijk
zijn SpO2-venster nog open had. En een hypopneu op 20005 s werd met de
laatst toegevoegde HYPOPNEU vergeleken in plaats van met de apneu op 20000 s.

De vlag is informatief (het rapport telt `n_spo2_cross_contaminated`), maar
een informatieve vlag die het verkeerde event noemt is misinformatie.
"""
from psgscoring.respiratory import _spo2_cross_contaminated


def _ev(onset, dur=10.0, type_="obstructive"):
    return {"type": type_, "onset_s": onset, "duration_s": dur}


def test_de_eerste_hypopneu_wordt_tegen_de_apneu_ervoor_getoetst():
    """Apneus op 100 s en 20000 s zitten al in de lijst; de hypopneu op 120 s
    ligt 10 s na het einde van de eerste -- binnen 15 s -> gevlagd."""
    bestaand = [_ev(100.0), _ev(20000.0)]
    assert _spo2_cross_contaminated(120.0, bestaand, 15.0) is True


def test_een_ver_verwijderde_hypopneu_wordt_niet_gevlagd_op_de_verkeerde_buur():
    """Na toevoeging van de hypopneu op 120 s is die het 'laatst toegevoegde'
    event. De hypopneu op 20005 s ligt 5 s na de apneu op 20000 s (einde
    20010 s, dus zelfs overlappend) -> gevlagd, en wel op DIE apneu."""
    bestaand = [_ev(100.0), _ev(20000.0), _ev(120.0, type_="hypopnea")]
    assert _spo2_cross_contaminated(20005.0, bestaand, 15.0) is True


def test_zonder_voorganger_binnen_het_venster_geen_vlag():
    bestaand = [_ev(100.0), _ev(20000.0), _ev(120.0, type_="hypopnea")]
    # 300 s: de dichtstbijzijnde voorganger eindigt op 130 s, 170 s eerder.
    assert _spo2_cross_contaminated(300.0, bestaand, 15.0) is False


def test_lege_lijst_en_nulvenster():
    assert _spo2_cross_contaminated(50.0, [], 15.0) is False
    assert _spo2_cross_contaminated(105.0, [_ev(100.0)], 0.0) is True   # overlapt
    assert _spo2_cross_contaminated(111.0, [_ev(100.0)], 0.0) is False


def test_een_later_event_mag_de_uitkomst_niet_bepalen():
    """Alleen events die VOOR de onset eindigen (of ermee overlappen) tellen;
    een event dat pas na de onset begint is geen voorganger."""
    bestaand = [_ev(500.0)]
    assert _spo2_cross_contaminated(120.0, bestaand, 15.0) is False
