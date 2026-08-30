"""Where the repository keeps its artifacts.

The entry points in `bin/` are run from wherever the user happens to be, so
their defaults resolve against the repository root rather than the cwd.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_POLICY = ROOT / "models" / "policy.pt"
DEFAULT_GLOBE = ROOT / "demo" / "globe.html"
