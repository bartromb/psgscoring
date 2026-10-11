"""
De TypedDicts in `_types.py` beschrijven wat `run_pneumo_analysis` teruggeeft.

Ze hebben geen runtime-effect, en dat is precies waarom ze stil verouderen:
`PneumoResults.arousals` (de sleutel heet `arousal`), tien top-level sleutels
die er nooit in stonden, `spo2_low_samplerate` gedocumenteerd in de
SpO2-samenvatting terwijl de pijplijn hem op `output["spo2"]` schrijft, en
positie- en PLM-sleutels die in geen enkele module voorkomen.

Dit draait de synthetische golden-case (`hypopnea_clean`, met
`split_night="auto"` zodat ook die sleutel verschijnt) en eist dat elke
gedocumenteerde top-level sleutel bestaat en dat geen gedocumenteerde
samenvattingssleutel ontbreekt. Voor PLM en positie, die op de synthetische
opname geen kanaal hebben, wordt tegen de bron van de module getoetst.
"""
from __future__ import annotations

import inspect
import re
import sys
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.test_golden_output import CASES, make_raw  # noqa: E402

PKG = REPO / "psgscoring"


@pytest.fixture(scope="module")
def output():
    import psgscoring
    cfg = dict(CASES["hypopnea_clean"])
    profiel = cfg.pop("profile")
    cfg.pop("arousals", None)
    raw, hypno, cmap = make_raw(**cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return psgscoring.run_pneumo_analysis(
            raw, hypno, channel_map=cmap, scoring_profile=profiel,
            split_night="auto")


def test_elke_gedocumenteerde_topsleutel_bestaat(output):
    from psgscoring._types import PneumoResults
    ontbreekt = set(PneumoResults.__annotations__) - set(output)
    assert not ontbreekt, f"gedocumenteerd maar niet in de uitvoer: {sorted(ontbreekt)}"


def test_geen_gedocumenteerde_samenvattingssleutel_ontbreekt(output):
    from psgscoring import _types
    for blok, td in (("respiratory", _types.ScoringSummary),
                     ("spo2", _types.SpO2Summary)):
        samenvatting = output[blok]["summary"]
        ontbreekt = set(td.__annotations__) - set(samenvatting)
        assert not ontbreekt, f"{td.__name__} documenteert {sorted(ontbreekt)}, " \
                              f"output[{blok!r}]['summary'] draagt ze niet"


def test_blokken_met_eigen_typeddict_kloppen(output):
    from psgscoring import _types
    for sleutel, td in (("ahi_interval", _types.AHIInterval),
                        ("hypoxic_burden", _types.HypoxicBurdenBlock),
                        ("postprocess", _types.PostprocessBlock),
                        ("cheyne_stokes", _types.CheyneStokesBlock),
                        ("anchor_baseline", _types.AnchorBaselineBlock),
                        ("channel_quality", _types.ChannelQualityBlock),
                        ("meta", _types.MetaBlock)):
        blok = output[sleutel]
        assert "error" not in blok, f"{sleutel} faalde op de golden case: {blok}"
        ontbreekt = {k for k in td.__annotations__ if k != "error"} - set(blok)
        assert not ontbreekt, f"{td.__name__} documenteert {sorted(ontbreekt)}"


def test_spo2_low_samplerate_staat_op_het_blok_niet_in_de_samenvatting():
    from psgscoring._types import SpO2Block, SpO2Summary
    assert "spo2_low_samplerate" not in SpO2Summary.__annotations__
    assert "spo2_low_samplerate" in SpO2Block.__annotations__
    bron = (PKG / "pipeline.py").read_text(encoding="utf-8")
    assert 'output["spo2"]["spo2_low_samplerate"]' in bron


def test_flow_reduction_blijft_als_ratio_gedocumenteerd():
    from psgscoring import _types
    bron = inspect.getsource(_types.RespiratoryEvent)
    regel = next(r for r in bron.splitlines() if r.strip().startswith("flow_reduction:"))
    assert "ratio" in regel, regel


def test_docstring_noemt_de_echte_modulenaam():
    from psgscoring import _types
    assert "psgscoring.types" not in (_types.__doc__ or "")
    assert "psgscoring._types" in (_types.__doc__ or "")


def test_pneumoresults_arousal_heet_arousal():
    from psgscoring._types import PneumoResults
    assert "arousal" in PneumoResults.__annotations__
    assert "arousals" not in PneumoResults.__annotations__


def test_plm_en_positie_sleutels_bestaan_in_hun_module():
    """Op de synthetische opname draaien deze stappen niet; toets de bron."""
    from psgscoring import _types
    plm = (PKG / "plm.py").read_text(encoding="utf-8")
    anc = (PKG / "ancillary.py").read_text(encoding="utf-8")
    for k in _types.PLMSummary.__annotations__:
        assert re.search(rf'"{k}"\s*:', plm), f"PLMSummary.{k} staat niet in plm.py"
    for k in _types.PositionSummary.__annotations__:
        assert re.search(rf'"{k}"\s*:', anc), f"PositionSummary.{k} staat niet in ancillary.py"


def test_eventsleutels_worden_ergens_geschreven():
    """Een gedocumenteerde eventsleutel die geen module ooit schrijft, is een
    belofte; `local_baseline_rejected` en `_label` waren er zo twee."""
    from psgscoring import _types
    bron = "\n".join((PKG / f).read_text(encoding="utf-8")
                     for f in ("respiratory.py", "pipeline.py", "postprocess.py",
                               "breath_scoring.py", "spo2.py", "classify.py"))
    for k in _types.RespiratoryEvent.__annotations__:
        assert f'"{k}"' in bron, f"RespiratoryEvent.{k} wordt nergens geschreven"
