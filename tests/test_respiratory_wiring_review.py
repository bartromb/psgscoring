"""De profielvelden `spo2_nadir_time_axis` en `spectral_effort_resolution`
bereiken hun lezer via `detect_respiratory_events` (review 11-10-2026).

De lezers staan in `spo2` en `ecg_effort`; zonder deze bedrading geldt de
default ook op `mesa_shhs`/`chicago_1999` en pinnen hun snapshots niets.
"""
from __future__ import annotations

import pytest

from psgscoring import ecg_effort, pipeline, respiratory, spo2
from tests.test_golden_output import CASES, make_raw


@pytest.fixture(scope="module")
def _raw_en_hypno():
    cfg = dict(CASES["spo2_dropout"])
    cfg.pop("profile"); cfg.pop("arousals", None)
    raw, hypno, channel_map = make_raw(**cfg)
    return raw, hypno, channel_map


def _spy(monkeypatch, module, naam, bewaar):
    origineel = getattr(module, naam)

    def _wrapper(*args, **kwargs):
        bewaar.append(kwargs)
        return origineel(*args, **kwargs)

    monkeypatch.setattr(module, naam, _wrapper)


@pytest.mark.parametrize("profiel, verwacht", [("mesa_shhs", False), ("aasm_v3_rec", True)])
def test_get_desaturation_krijgt_de_tijdasvlag_van_het_profiel(
        monkeypatch, _raw_en_hypno, profiel, verwacht):
    raw, hypno, channel_map = _raw_en_hypno
    gezien: list = []
    # respiratory importeert de naam zelf (`from .spo2 import get_desaturation`).
    _spy(monkeypatch, respiratory, "get_desaturation", gezien)
    pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                 scoring_profile=profiel)
    assert gezien, "get_desaturation niet aangeroepen"
    # De eerste aanroepen komen uit de primaire pas; de AHI-intervalsweep
    # (strict/standard/sensitive) draait daarna onder zijn eigen profielen,
    # ook als het primaire profiel mesa_shhs is. Toets dus de primaire pas.
    assert gezien[0].get("nadir_time_axis") is verwacht
    if verwacht is False:
        assert any(k.get("nadir_time_axis") is False for k in gezien)
    else:
        assert all(k.get("nadir_time_axis") is True for k in gezien)


@pytest.mark.parametrize("profiel, verwacht", [("mesa_shhs", "4s"), ("aasm_v3_rec", "event")])
def test_ecg_effort_assessment_krijgt_de_resolutie_van_het_profiel(
        monkeypatch, profiel, verwacht):
    """`_get_ecg_assessment` geeft de profielresolutie door aan
    `ecg_effort_assessment` (importeert lokaal: patch de bron)."""
    import numpy as np
    from psgscoring import respiratory
    gezien: list = []
    _spy(monkeypatch, ecg_effort, "ecg_effort_assessment", gezien)
    from psgscoring.constants import SCORING_PROFILES
    sp = SCORING_PROFILES[profiel]
    from psgscoring.ecg_effort import spectral_resolution_from_profile
    res = spectral_resolution_from_profile(sp)
    assert res == verwacht
    ecg = np.zeros(2000); thorax = np.zeros(2000); abdomen = np.zeros(2000)
    respiratory._get_ecg_assessment(ecg, np.arange(0, 2000) / 100.0, np.array([10, 110, 210]),
                                    thorax, abdomen, 100.0, 100.0, 100, 600,
                                    spectral_resolution=res)
    assert gezien and gezien[0].get("spectral_resolution") == verwacht
