"""README-testaantal moet de werkelijke collectie zijn, niet een handtelling.

De 0.34.1-README claimde "1,474 unit tests" terwijl de suite er 1473
verzamelt en de releasecommit "1464 groen" meldde — drie handbijgehouden
cijfers, drie waarden. Op de regel naast de FDA-valutaclaim draagt zo'n
getal bewijskracht, dus het wordt hier tegen `pytest --collect-only`
gepind: wie een test toevoegt, werkt de README mee bij.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _readme_count() -> int:
    tekst = (REPO / "README.md").read_text(encoding="utf-8")
    m = re.search(r"([\d,]+)\s+unit tests", tekst)
    assert m, "README.md noemt geen 'N unit tests' meer — claim of test bijwerken"
    return int(m.group(1).replace(",", ""))


def _collected_count() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO / "tests"),
         "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=300, cwd=REPO,
    )
    m = re.search(r"(\d+) tests collected", proc.stdout)
    assert m, f"collectie-telling niet gevonden in:\n{proc.stdout[-500:]}"
    return int(m.group(1))


def test_readme_test_count_matches_collection():
    readme = _readme_count()
    echt = _collected_count()
    assert readme == echt, (
        f"README.md claimt {readme} unit tests, de suite verzamelt er "
        f"{echt}. Werk het getal in README.md (sectie Architecture) bij — "
        f"handtellingen op deze regel hebben al twee keer verschillende "
        f"waarden opgeleverd."
    )
