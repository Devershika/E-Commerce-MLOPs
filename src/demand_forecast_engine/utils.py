"""Helpers shared by the CLI scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

import yaml


class DataValidationError(ValueError):
    """Raised when the data or the parameters do not meet the pipeline's expectations."""


def load_params(path: str | Path) -> dict:
    """Read a YAML parameter file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cli(func: Callable[[], None], log: logging.Logger, verbose: bool = False) -> None:
    """Run a script's main function. Expected failures print one clear line and exit with 1.

    A non-zero exit code is what makes `dvc repro` stop and report the stage as failed.
    Use --verbose on the script to see the full traceback.
    """
    try:
        func()
    except DataValidationError as exc:
        log.error("%s", exc)
    except FileNotFoundError as exc:
        log.error("File not found: %s. If it is tracked by DVC, run `dvc pull` first.", exc)
    except KeyError as exc:
        log.error("Missing key %s. Check that params.yaml matches the expected layout.", exc)
    else:
        return
    if verbose:
        raise
    sys.exit(1)
