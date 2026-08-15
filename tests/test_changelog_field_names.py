"""
tests/test_changelog_field_names.py — the CHANGELOG names real fields.

The v0.18.0 overview table lists each new switch by its profile field name so
a reader can set it. Writing that table by hand produced
`limit_events_per_desaturation` for a field actually called
`max_events_per_desaturation` — a name that looks right, reads right, and
sends anyone who copies it into an AttributeError.

Nothing else catches this: the CHANGELOG is prose, and prose does not get
imported. This test treats the documented field names as an interface.
"""
import re
from dataclasses import fields
from pathlib import Path

from psgscoring.profiles import HypopneaRules, PostProcessingRules, Profile

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"

KNOWN = {
    f.name
    for cls in (Profile, HypopneaRules, PostProcessingRules)
    for f in fields(cls)
}

# `name="value"` or a bare `name` inside backticks in the overview table.
CITED = re.compile(r"`([a-z_][a-z0-9_]*)(?:=[^`]*)?`")


def _overview_table() -> str:
    """The v0.18.0 overview table only — later sections quote prose freely."""
    text = CHANGELOG.read_text(encoding="utf-8")
    start = text.index("# v0.18.0 — 2026-08-15 — overview")
    end = text.index("\n---\n", start)
    return text[start:end]


def test_every_field_cited_in_the_overview_exists():
    table = _overview_table()
    rows = [ln for ln in table.splitlines() if ln.startswith("|")]
    assert rows, "the v0.18.0 overview table is gone"

    cited = {m.group(1) for row in rows for m in CITED.finditer(row)}
    # Values and prose words also sit in backticks; only check things that
    # look like they are being offered as settable fields.
    plausible = {c for c in cited if "_" in c}

    unknown = sorted(c for c in plausible if c not in KNOWN)
    # Values such as breath_coherence / envelope_agreement are not field names.
    values = {"breath_coherence", "envelope_agreement", "respiratory_band"}
    unknown = [u for u in unknown if u not in values]

    assert not unknown, (
        f"the v0.18.0 overview cites profile fields that do not exist: {unknown}. "
        f"Anyone copying them gets an AttributeError."
    )


def test_the_overview_covers_every_new_switch():
    """A switch that ships undocumented is a switch nobody can find."""
    table = _overview_table()
    for field in (
        "thermistor_gate",
        "split_events_longer_than_s",
        "desat_low_baseline_relaxation",
        "event_boundaries",
        "max_events_per_desaturation",
    ):
        assert field in table, f"{field} ships in 0.18.0 but is absent from the overview table"
