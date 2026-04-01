"""
conftest.py — pytest configuration for psgscoring tests.

Appends the project root to sys.path so the 'psgscoring' package can be
imported without installation.  The root is appended (not prepended) to
avoid the root-level signal.py shadowing Python's stdlib signal module.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
