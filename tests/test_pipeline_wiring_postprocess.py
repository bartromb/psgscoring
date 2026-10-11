"""`postprocess_respiratory_events` krijgt het profiel mee, zodat
`mixed_decomposition_on_envelope` ook via de pipeline zijn pin op de
bevroren profielen bereikt (review 11-10-2026)."""
from __future__ import annotations

import pytest

from psgscoring import pipeline, postprocess
from tests.test_golden_output import CASES, make_raw


@pytest.fixture(scope="module")
def _raw_en_hypno():
    cfg = dict(CASES["apnea_clean"])
    cfg.pop("profile"); cfg.pop("arousals", None)
    raw, hypno, channel_map = make_raw(**cfg)
    return raw, hypno, channel_map


@pytest.mark.parametrize("profiel", ["mesa_shhs", "aasm_v3_rec"])
def test_postprocess_krijgt_het_profiel(monkeypatch, _raw_en_hypno, profiel):
    raw, hypno, channel_map = _raw_en_hypno
    gezien: list = []
    # pipeline importeert de functie lokaal bij de aanroep: patch de bron.
    origineel = postprocess.postprocess_respiratory_events

    def _wrapper(*args, **kwargs):
        gezien.append(kwargs)
        return origineel(*args, **kwargs)

    monkeypatch.setattr(postprocess, "postprocess_respiratory_events", _wrapper)
    pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                 scoring_profile=profiel)
    assert gezien, "postprocess_respiratory_events niet aangeroepen"
    sp = gezien[0].get("scoring_profile")
    assert isinstance(sp, dict) and sp.get("_PROFILE_NAME") == profiel
