"""De paarpoort mag op de niet-schaalvrije tak niet crashen.

`compare_rip_pair` bouwt bij een energieratio boven 100 een melding met de
ademfractie van beide banden. Die sleutel (`breath_fraction`) zet
`assess_rip_channel` alleen op de schaalvrije tak; op de legacy-tak
(`scale_free=False`, de bevroren profielen `mesa_shhs` en `chicago_1999`) gaf
de melding een KeyError zodra beide banden hun eigen toets doorstonden. De
pijplijn ving dat op en liet de hele kwaliteitsstap leeg: geen poort, en een
foutmelding in plaats van een oordeel. De melding is op 19-08-2026
toegevoegd; daarvoor gaf deze tak gewoon `single-channel`.
"""
import numpy as np
import pytest

from psgscoring.signal_quality import compare_rip_pair

SF = 32.0
T = np.arange(0, 1800, 1 / SF)
ADEM = np.sin(2 * np.pi * 0.25 * T)


def test_de_legacy_tak_geeft_een_oordeel_in_plaats_van_een_keyerror():
    uit = compare_rip_pair(ADEM * 500.0, ADEM, SF)
    assert uit["recommended_mode"] == "single-channel"
    assert uit["working_channel"] == "thorax"
    assert uit.get("pair_gate_suspect") is True


def test_de_melding_noemt_de_ademfractie_van_beide_banden():
    uit = compare_rip_pair(ADEM * 500.0, ADEM, SF)
    tekst = " ".join(uit["warnings"])
    assert "ademfractie" in tekst and "thorax" in tekst and "abdomen" in tekst


def test_paarpoort_schaalvrij_op_de_legacy_kanaaltoets_crasht_ook_niet():
    uit = compare_rip_pair(ADEM * 500.0, ADEM, SF, pair_scale_free=True)
    assert uit["recommended_mode"] == "bilateral"


@pytest.mark.parametrize("kw", [{}, {"scale_free": True}])
def test_gelijke_banden_blijven_bilateraal(kw):
    assert compare_rip_pair(ADEM, ADEM * 1.5, SF, **kw)["recommended_mode"] == "bilateral"
