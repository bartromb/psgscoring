"""conftest.py — pytest configuration for psgscoring."""
import sys
from pathlib import Path

# Ensure the package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
