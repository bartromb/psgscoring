"""
tests/test_readme_links.py — the README must survive being rendered by PyPI.

PyPI renders `README.md` as the project description, and it renders it at
`https://pypi.org/project/psgscoring/`. A relative markdown link like
`[LICENSE](LICENSE)` therefore resolves against *that* URL rather than against
the repository, and lands nowhere useful.

Two things make this worth a test rather than a habit:

  1. The failure is invisible from the repository. On GitHub every relative
     link works perfectly, so reviewing the README where it is written tells
     you nothing about where it is read.
  2. The published description is immutable. Fixing the link on `main` does
     NOT fix the published page — PyPI only re-renders on a new version, so
     each occurrence costs a release. It has now cost two (0.12.2 for one set
     of links, 0.19.1 for `LICENSE` and the interim-conclusion document).

The test therefore checks the *shape* of every link rather than any particular
one, so a link added later is caught before it ships instead of after.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/bartromb/psgscoring/blob/main/"

# [text](target) — the inline form. Reference-style links are not used in this
# README; if they ever are, `test_no_reference_style_links` fails and this
# pattern needs extending rather than the assertion being relaxed.
LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


@pytest.fixture(scope="module")
def readme() -> str:
    return (REPO / "README.md").read_text(encoding="utf-8")


def _links(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in LINK.finditer(text)]


def test_no_relative_links(readme):
    """
    Every link must be absolute, an in-page anchor, or a mailto. A relative
    path renders on GitHub and breaks on PyPI, which is the half nobody sees.
    """
    relative = [
        (text, url) for text, url in _links(readme)
        if not url.startswith(("http://", "https://", "#", "mailto:"))
    ]
    assert not relative, (
        "relative links in README.md resolve against pypi.org once published, "
        "and the published description cannot be corrected without a new "
        f"version: {relative}"
    )


def test_the_readme_has_links_at_all(readme):
    """
    Guards the guard. If the regex stops matching — a changed link style, a
    renamed file — `test_no_relative_links` passes on an empty list and reports
    success for having checked nothing.
    """
    assert len(_links(readme)) >= 8, (
        f"only {len(_links(readme))} links found; the link regex has probably "
        f"stopped matching, which would make the relative-link test vacuous")


def test_no_reference_style_links(readme):
    """
    `[text][ref]` with a `[ref]: url` definition elsewhere would slip past
    `LINK` entirely. Fail loudly rather than silently stop covering them.
    """
    ref_defs = re.findall(r"^\[[^\]]+\]:\s*\S+", readme, flags=re.MULTILINE)
    assert not ref_defs, (
        f"reference-style link definitions found ({ref_defs}); extend LINK to "
        f"cover them instead of leaving them unchecked")


def test_repo_relative_targets_point_at_files_that_exist(readme):
    """
    An absolute GitHub blob URL into this repository is only better than a
    relative one if the path behind it is real. A typo here produces a 404 that
    no local check would otherwise catch.
    """
    missing = []
    for _text, url in _links(readme):
        if not url.startswith(GITHUB):
            continue
        target = url[len(GITHUB):].split("#", 1)[0]
        if target and not (REPO / target).exists():
            missing.append(target)
    assert not missing, f"README links to paths absent from the repo: {missing}"
