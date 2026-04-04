"""
conftest.py
===========
Pytest root configuration for the cad_automation test suite.

Adds the project root to sys.path so all sub-packages resolve correctly
without needing a full ``pip install -e .`` for test runs.
"""

import sys
from pathlib import Path

# Project root = parent of this file
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
