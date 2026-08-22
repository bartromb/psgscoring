"""Het MESA-harnas telde nul apneus terwijl het er 2223 zag.

`n_apnea` zocht op de substring "apnea" in het eventtype. De bibliotheek
schrijft dat woord nergens: apneus heten `obstructive`, `central`, `mixed` en
-- niet onderverdeeld -- `uncertain`. De teller stond daardoor op nul op alle
50 MESA-opnames, ook op patienten met een AHI van 47.

Alleen boekhouding: `match`, `ahi` en `n_events` lopen niet langs dit veld,
dus eerdere F1- en bias-cijfers zijn ongemoeid. Wat wel fout was, is elke
uitspraak over apneu-AANTALLEN uit dit harnas.

Het harnas is een studie-artefact en hoort met dezelfde striktheid vast te
liggen als de bibliotheek; vandaar deze test.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _library_apnea_types():
    """Welke typen behandelt de bibliotheek zelf als apneu?"""
    src = (Path(__file__).resolve().parent.parent
           / "psgscoring" / "respiratory.py").read_text(encoding="utf-8")
    line = next(ln for ln in src.splitlines()
                if 'apneas' in ln and 'e["type"] in (' in ln)
    return {t.strip().strip('"\'')
            for t in line.split("in (")[1].split(")")[0].split(",") if t.strip()}


def test_the_harness_counts_the_types_the_library_actually_writes():
    mne = pytest.importorskip("mne")  # noqa: F841
    from validate_mesa import APNEA_TYPES

    lib = _library_apnea_types()
    assert lib, "kon de apneutypen niet uit respiratory.py lezen"
    ontbreekt = lib - set(APNEA_TYPES)
    assert not ontbreekt, (
        f"het harnas telt {sorted(ontbreekt)} niet als apneu, terwijl "
        f"respiratory.py dat wel doet")


def test_the_old_substring_rule_would_have_counted_none_of_them():
    """De fixture moet het defect kunnen tonen, anders meet deze test niets."""
    mne = pytest.importorskip("mne")  # noqa: F841
    from validate_mesa import APNEA_TYPES

    oud = [t for t in APNEA_TYPES if "apnea" in t and "hypo" not in t]
    assert not oud, (
        "de oude substring-regel zou deze typen wel gevangen hebben: "
        f"{oud} -- dan toont deze test het defect niet")


def test_hypopnea_subtypes_are_not_counted_as_apnea():
    """`hypopnea_central` bevat "central"; dat mag geen apneu worden."""
    mne = pytest.importorskip("mne")  # noqa: F841
    from validate_mesa import APNEA_TYPES

    for t in ("hypopnea", "hypopnea_central", "hypopnea_mixed",
              "hypopnea_uncertain"):
        assert t not in APNEA_TYPES, f"{t} wordt als apneu geteld"
