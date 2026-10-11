"""De docstring van `run_pneumo_analysis` noemde 8 van de 22 profielen, 10 van
de ~17 uitvoersleutels en de split-night-parameters helemaal niet. Een
docstring die achterloopt is erger dan geen: ze zegt met gezag iets
onvolledigs. Deze test houdt haar bij de registry en bij de uitvoer.
"""
import inspect

import pytest

from psgscoring.pipeline import run_pneumo_analysis
from psgscoring.profiles import list_profiles

DOC = inspect.getdoc(run_pneumo_analysis) or ""

#: De sleutels op het buitenste niveau van de uitvoer. `split_night` alleen
#: wanneer erom gevraagd is.
UITVOERSLEUTELS = (
    "meta", "channel_availability", "signal_quality", "respiratory",
    "ahi_interval", "channel_quality", "anchor_baseline", "spo2", "position",
    "heart_rate", "snore", "plm", "arousal", "cheyne_stokes", "hypoxic_burden",
    "postprocess", "split_night",
)


@pytest.mark.parametrize("naam", list_profiles())
def test_elk_profiel_staat_in_de_docstring(naam):
    assert naam in DOC, f"profiel {naam!r} ontbreekt in de docstring"


@pytest.mark.parametrize("sleutel", UITVOERSLEUTELS)
def test_elke_uitvoersleutel_staat_in_de_docstring(sleutel):
    assert sleutel in DOC, f"uitvoersleutel {sleutel!r} ontbreekt"


def test_de_split_night_parameters_zijn_gedocumenteerd():
    sig = inspect.signature(run_pneumo_analysis)
    for p in sig.parameters:
        assert p in DOC, f"parameter {p!r} ontbreekt in de docstring"
    assert "split_night_breakpoint_s" in DOC


def test_de_docstring_verwijst_naar_de_registry_als_bron():
    """Een opsomming veroudert; de verwijzing niet."""
    assert "list_profiles" in DOC
