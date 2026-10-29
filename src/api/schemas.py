"""
Pydantic request/response schemas for the demand forecasting API.
Matches Contract B defined in docs/MLOPS_IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    store_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        examples=["STORE_01"],
        description="Unique store identifier.",
    )
    product_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        examples=["PROD_42"],
        description="Unique product identifier (SKU).",
    )
    category: str = Field(
        ...,
        min_length=1,
        max_length=64,
        examples=["Electronics"],
        description="Product category.",
    )
    forecast_horizon_days: int = Field(
        ...,
        ge=1,
        le=90,
        examples=[14],
        description="Number of days to forecast ahead (1–90).",
    )

    @field_validator("store_id", "product_id", "category")
    @classmethod
    def no_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank or whitespace only.")
        return v.strip()


class DailyForecast(BaseModel):
    """A single day's predicted demand."""

    forecast_date: dt.date = Field(..., alias="date", description="Forecast date (YYYY-MM-DD).")
    predicted_demand: float = Field(
        ..., ge=0.0, description="Predicted demand units (non-negative)."
    )

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    store_id: str
    product_id: str
    forecast: List[DailyForecast]
    model_name: str = Field(..., description="Name/version of the model used.")
    validation_mape: Optional[float] = Field(
        None,
        ge=0.0,
        description="Validation MAPE from training, if available.",
    )

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    model: str
    version: str
    model_loaded: bool = Field(..., description="True when a real model artifact is in memory.")
