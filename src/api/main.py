"""
FastAPI prediction service for E-Commerce Demand Forecasting.
Endpoints:
  GET  /health   — liveness + model metadata
  POST /predict  — day-ahead demand forecast
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.inference import run_inference
from src.api.loader import MODEL_NAME, MODEL_VERSION, load_model
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan: load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model bundle at startup; release resources on shutdown."""
    logger.info("Starting up — attempting to load model artifact …")
    app.state.model_bundle = load_model()
    if app.state.model_bundle is None:
        logger.warning("No model artifact loaded. /predict will return mock forecasts.")
    else:
        logger.info("Model loaded: %s", app.state.model_bundle.model_name)
    yield
    logger.info("Shutting down.")
    app.state.model_bundle = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Demand Forecast API",
    description=(
        "Serves day-ahead SKU-level demand predictions "
        "from the Holt-Winters Exponential Smoothing champion model."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler — ensure JSON error bodies on unexpected failures
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception for %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service liveness and model metadata",
    tags=["Ops"],
)
async def health(request: Request) -> HealthResponse:
    """
    Returns service health and information about the currently loaded model.
    `model_loaded: false` means the API is running in mock mode.
    """
    bundle = request.app.state.model_bundle
    return HealthResponse(
        status="healthy",
        model=bundle.model_name if bundle else MODEL_NAME,
        version=bundle.version if bundle else MODEL_VERSION,
        model_loaded=bundle is not None,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Generate day-ahead demand forecast",
    status_code=200,
    tags=["Forecasting"],
)
async def predict(
    payload: PredictRequest,
    request: Request,
) -> PredictResponse:
    """
    Generate a demand forecast for a given Store × Product × horizon.
    - Returns HTTP 422 automatically if payload validation fails (Pydantic).
    - Returns HTTP 500 if inference raises an unexpected error.
    """
    bundle = request.app.state.model_bundle
    try:
        return run_inference(request=payload, model_bundle=bundle)
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Inference failed.") from exc
