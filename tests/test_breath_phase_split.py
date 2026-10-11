"""`detect_breaths` splitst een ademcyclus op de nuldoorgang, niet in het midden.

WAT ER MIS WAS
--------------
Een cyclus loopt van nuldoorgang i naar nuldoorgang i+2; de nuldoorgang
ertussen (i+1) scheidt inspiratie van expiratie. De code deelde de cyclus
echter op de TEMPORELE helft (`dur_samp // 2`). Bij een I:E van 1:2 -- de
normale verhouding in rust -- zit die helft midden in de expiratie, zodat
`insp_segment` voor een derde uit expiratoire samples bestaat.

Dat segment voedt `compute_flattening_index` (fractie van |flow| boven 80 %
van de piek). Expiratoire samples met kleinere |flow| drukken die index
omlaag, en `avg_flattening` reist via de hypopnee-subtypering mee in de
classificatie. Daarom staat de reparatie achter het profielveld
`breath_phase_split`: default `zero_crossing` op de klinische profielen,
`midpoint` gepind op `mesa_shhs` en `chicago_1999`.
"""
import numpy as np
import pytest

from psgscoring.breath import detect_breaths

SF = 32.0
PERIODE = 4.0           # s
T_INSP = PERIODE / 3.0  # I:E = 1:2


def _flow_ie_1_2(n_cycli=40):
    """Positieve halve sinus van T_INSP s, dan negatieve van 2*T_INSP s."""
    t_i = np.arange(int(T_INSP * SF)) / SF
    t_e = np.arange(int(2 * T_INSP * SF)) / SF
    insp = np.sin(np.pi * t_i / T_INSP)
    exp_ = -0.5 * np.sin(np.pi * t_e / (2 * T_INSP))
    return np.tile(np.concatenate([insp, exp_]), n_cycli)


def _middelste(breaths):
    assert len(breaths) >= 10, len(breaths)
    return breaths[len(breaths) // 2]


def test_nuldoorgang_geeft_een_zuiver_inspiratoir_segment():
    b = _middelste(detect_breaths(_flow_ie_1_2(), SF, phase_split="zero_crossing"))
    seg = b["insp_segment"]
    assert len(seg) == pytest.approx(T_INSP * SF, abs=2), len(seg)
    assert np.all(seg >= -1e-9), "het inspiratoire segment bevat expiratoire samples"
    assert b["peak_insp"] == pytest.approx(1.0, abs=0.02)
    assert b["trough_exp"] == pytest.approx(-0.5, abs=0.02)


def test_midpoint_is_het_oude_gedrag_en_vervuilt_het_segment():
    """Gedocumenteerd, niet gerepareerd: dit is wat `midpoint` doet."""
    b = _middelste(detect_breaths(_flow_ie_1_2(), SF, phase_split="midpoint"))
    seg = b["insp_segment"]
    assert len(seg) == pytest.approx(PERIODE * SF / 2, abs=2), len(seg)
    assert np.any(seg < -1e-3), "bij I:E 1:2 hoort de helft expiratie te bevatten"


def test_zonder_argument_blijft_midpoint():
    """Directe aanroepers (signal.py, signal_quality.py) veranderen niet."""
    a = detect_breaths(_flow_ie_1_2(), SF)
    b = detect_breaths(_flow_ie_1_2(), SF, phase_split="midpoint")
    assert [len(x["insp_segment"]) for x in a] == [len(x["insp_segment"]) for x in b]


def test_de_afvlakkingsindex_verschuift_erdoor():
    from psgscoring.breath import compute_flattening_index
    zc = _middelste(detect_breaths(_flow_ie_1_2(), SF, phase_split="zero_crossing"))
    mp = _middelste(detect_breaths(_flow_ie_1_2(), SF, phase_split="midpoint"))
    assert compute_flattening_index(zc["insp_segment"]) > \
        compute_flattening_index(mp["insp_segment"])


def test_onbekende_waarde_wordt_geweigerd():
    with pytest.raises(ValueError):
        detect_breaths(_flow_ie_1_2(), SF, phase_split="halfway")


# ── het profielveld ──────────────────────────────────────────────────────

def test_veld_default_en_pins():
    from psgscoring.profiles import PROFILES, PostProcessingRules
    assert PostProcessingRules().breath_phase_split == "zero_crossing"
    for naam in ("mesa_shhs", "chicago_1999"):
        assert PROFILES[naam].post_processing.breath_phase_split == "midpoint", naam
    for naam in ("aasm_v3_rec", "aasm_v3_breath", "aasm_v3_breath_dual"):
        assert PROFILES[naam].post_processing.breath_phase_split == "zero_crossing", naam


def test_veld_bereikt_de_legacy_dict_en_de_ademteuganalyse():
    import inspect

    from psgscoring import respiratory
    from psgscoring.constants import SCORING_PROFILES
    assert SCORING_PROFILES["aasm_v3_rec"]["BREATH_PHASE_SPLIT"] == "zero_crossing"
    assert SCORING_PROFILES["mesa_shhs"]["BREATH_PHASE_SPLIT"] == "midpoint"
    bron = inspect.getsource(respiratory._run_breath_analysis)
    assert "phase_split=" in bron
