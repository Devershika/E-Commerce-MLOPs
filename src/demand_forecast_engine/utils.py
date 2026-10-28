"""Small helpers shared by the CLI scripts."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_params(path: str | Path) -> dict:
    """Read a YAML parameter file into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
