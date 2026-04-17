"""
Smoke tests for psgscoring.

These tests verify that the package installs correctly and its public
API is importable across supported Python versions. They do NOT test
algorithmic correctness — for that, see the separate validation scripts
(validate_psgipa.py, validate_mesa.py, etc.) and the planned AZORG
prospective study.

Smoke tests should pass on every commit, every CI run. If a smoke test
fails, something fundamental is broken (missing module, unexpected
dependency change, version mismatch, etc.).
"""
from __future__ import annotations

import importlib
import re


# ═══════════════════════════════════════════════════════════════════
# 1. Package import and version
# ═══════════════════════════════════════════════════════════════════

def test_package_imports():
    """The package imports without errors."""
    import psgscoring  # noqa: F401


def test_version_exposed():
    """__version__ is present, non-empty, and follows PEP 440."""
    import psgscoring
    assert hasattr(psgscoring, "__version__"), \
        "psgscoring.__version__ must be defined"
    ver = psgscoring.__version__
    assert isinstance(ver, str), f"__version__ must be str, got {type(ver)}"
    assert len(ver) > 0, "__version__ must not be empty"
    # Loose PEP 440 check — allows 0.2.951 style
    assert re.match(r'^\d+\.\d+', ver), \
        f"__version__ '{ver}' should start with MAJOR.MINOR"


# ═══════════════════════════════════════════════════════════════════
# 2. Public API surface
# ═══════════════════════════════════════════════════════════════════

def test_main_pipeline_importable():
    """The main entry point exists and is callable."""
    from psgscoring import run_pneumo_analysis
    assert callable(run_pneumo_analysis)


def test_hypoxic_burden_api():
    """v0.2.92+ hypoxic burden API is exposed."""
    from psgscoring import compute_hypoxic_burden
    assert callable(compute_hypoxic_burden)


def test_postprocessing_api():
    """v0.2.92+ post-processing APIs are exposed."""
    from psgscoring import (
        postprocess_respiratory_events,
        reclassify_csr_events,
        decompose_mixed_apneas,
        compute_central_instability_index,
    )
    for fn in (postprocess_respiratory_events,
               reclassify_csr_events,
               decompose_mixed_apneas,
               compute_central_instability_index):
        assert callable(fn), f"{fn.__name__} must be callable"


# ═══════════════════════════════════════════════════════════════════
# 3. Optional / defensive checks
# ═══════════════════════════════════════════════════════════════════

def test_no_unexpected_hard_dependencies():
    """
    Importing psgscoring must not pull in optional dependencies.

    The [full] extra includes yasa and lightgbm, but the core package
    should work without them. This test catches regressions where a
    core module accidentally starts importing an optional library
    at module-load time.
    """
    import sys

    # Snapshot modules before import (the test runner has already
    # imported lots of things, so we check for specific optional libs)
    for optional in ("yasa", "lightgbm"):
        # If the user happens to have them installed, skip the check
        try:
            importlib.import_module(optional)
            # Already available — can't test that core doesn't pull it in
            continue
        except ImportError:
            pass

        # Not installed — confirm that importing psgscoring still works
        # and did NOT trigger an import attempt for this optional lib
        import psgscoring  # noqa: F401
        assert optional not in sys.modules, (
            f"psgscoring unexpectedly loaded optional dependency '{optional}'"
        )
