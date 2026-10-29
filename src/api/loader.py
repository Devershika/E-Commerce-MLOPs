"""
Model loader for the demand forecasting API.

Loading priority:
  1. Local file at MODEL_LOCAL_PATH (volume-mounted artifact or pre-downloaded).
  2. S3 download triggered when AWS credentials / S3_BUCKET env var are present.
  3. DVC pull (best-effort, requires dvc + remote config).
  4. Mock predictor as a safe fallback for dev / CI.

The loader is called once at application startup (lifespan context).
The resulting ModelBundle (or None) is stored in app.state.model_bundle
so every request handler can reuse it without re-loading.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
MODEL_LOCAL_PATH: Path = Path(os.getenv("MODEL_LOCAL_PATH", "models/holt_winters_model.joblib"))
S3_BUCKET: Optional[str] = os.getenv("S3_BUCKET")
S3_KEY: Optional[str] = os.getenv("S3_KEY", "models/holt_winters_model.joblib")

MODEL_NAME: str = "Holt-Winters (Fine-Tuned)"
MODEL_VERSION: str = "v1.0.0"


@dataclass
class ModelBundle:
    """
    Thin wrapper around a loaded HoltWintersForecaster (or any object
    exposing a .forecast(horizon) → iterable-of-floats interface).
    """

    model: Any
    model_name: str = MODEL_NAME
    version: str = MODEL_VERSION
    validation_mape: Optional[float] = field(default=None)

    def forecast(self, horizon: int) -> List[float]:
        """
        Delegate to the underlying model's forecast method and
        return a plain list of floats regardless of what the model returns.
        """
        result = self.model.forecast(horizon)
        # HoltWintersForecaster.forecast() returns a pd.Series
        try:
            return [float(v) for v in result]
        except TypeError:
            return list(result)


def _try_s3_download(dest: Path) -> bool:
    """Download the model artifact from S3. Returns True on success."""
    if not S3_BUCKET:
        return False
    try:
        import boto3  # type: ignore

        s3 = boto3.client("s3")
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading model from s3://%s/%s …", S3_BUCKET, S3_KEY)
        s3.download_file(S3_BUCKET, S3_KEY, str(dest))
        logger.info("S3 download complete → %s", dest)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("S3 download failed: %s", exc)
        return False


def _try_dvc_pull(dest: Path) -> bool:
    """Attempt a best-effort `dvc pull` for the model file. Returns True on success."""
    try:
        import subprocess

        result = subprocess.run(  # noqa: S603
            ["dvc", "pull", str(dest)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("DVC pull succeeded for %s", dest)
            return dest.exists()
        logger.warning("DVC pull returned non-zero: %s", result.stderr)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("DVC pull failed: %s", exc)
        return False


def load_model() -> Optional[ModelBundle]:
    """
    Attempt to load the Holt-Winters model bundle.

    Uses load_holt_winters_model() from the demand_forecast_engine package
    which reconstructs a full HoltWintersForecaster from the joblib bundle.

    Returns None if no artifact is available (API falls back to mock predictor).
    """
    artifact_path = MODEL_LOCAL_PATH

    # 1. Local path already exists
    if artifact_path.exists():
        logger.info("Model artifact found at: %s", artifact_path)

    # 2. Try S3 download
    elif _try_s3_download(artifact_path):
        pass

    # 3. Try DVC pull
    elif _try_dvc_pull(artifact_path):
        pass

    # 4. No artifact available — use mock
    else:
        logger.info(
            "No model artifact found at '%s'. API will use mock predictor.",
            artifact_path,
        )
        return None

    # Load via the shared loader from demand_forecast_engine
    try:
        from demand_forecast_engine.models.holt_winters import (
            load_holt_winters_model,
        )

        forecaster = load_holt_winters_model(str(artifact_path))
        logger.info("Model loaded successfully: %s %s", MODEL_NAME, MODEL_VERSION)
        return ModelBundle(model=forecaster, validation_mape=None)

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to load model from '%s': %s. Falling back to mock.",
            artifact_path,
            exc,
        )
        return None
