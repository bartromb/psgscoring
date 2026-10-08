"""`aasm_v3_breath_dual_v2`: de zuivere vereniging.

`aasm_v3_breath_dual` zet vier schakelaars; twee ervan (referentiekanaal en
per-kanaal-poort) verplaatsen de primaire pas en veranderen RDI/RERA/FRI en de
ventilatoire last zonder dat de AHI dat nodig heeft (20 eigen PSG's, 07-10-2026).
De v2 mag ALLEEN de vereniging toevoegen: elk ander veld gelijk aan de ouder.
"""
from dataclasses import asdict

from psgscoring.profiles import PROFILES


def test_v2_differs_from_breath_only_in_the_union_switches():
    a = asdict(PROFILES["aasm_v3_breath"].post_processing)
    b = asdict(PROFILES["aasm_v3_breath_dual_v2"].post_processing)
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"dual_sensor_apnea", "dual_sensor_corroboration"} or diff == {"dual_sensor_apnea"}, diff
    assert b["dual_sensor_apnea"] is True and b["dual_sensor_corroboration"] is False
    assert b["flow_reference"] == a["flow_reference"]
    assert b["thermistor_gate"] == a["thermistor_gate"]
    for blok in ("hypopnea", "apnea", "spo2"):
        assert asdict(getattr(PROFILES["aasm_v3_breath"], blok)) == asdict(getattr(PROFILES["aasm_v3_breath_dual_v2"], blok))


def test_v1_still_carries_its_four_switches():
    """De oorspronkelijke variant blijft bevroren: de gepubliceerde cijfers hangen eraan."""
    pp = PROFILES["aasm_v3_breath_dual"].post_processing
    assert pp.dual_sensor_apnea and not pp.dual_sensor_corroboration
    assert pp.flow_reference == "hypopnea" and pp.thermistor_gate == "respiratory_band"


def test_v2_is_exploratory_and_not_shared_with_its_parent():
    p = PROFILES["aasm_v3_breath_dual_v2"]
    assert p.family == "exploratory"
    assert p.post_processing is not PROFILES["aasm_v3_breath"].post_processing
    assert p.hypopnea is not PROFILES["aasm_v3_breath"].hypopnea
