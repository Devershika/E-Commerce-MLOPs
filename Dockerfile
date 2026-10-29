# =============================================================================
# Multi-stage Dockerfile for the Demand Forecast FastAPI service
# Stage 1: builder — install all dependencies into a virtual environment
# Stage 2: runtime — minimal production image (non-root, Uvicorn)
# =============================================================================

# ---- Stage 1: builder -------------------------------------------------------
FROM python:3.11-slim AS builder

# Build tools needed for compiled wheels (numpy, statsmodels, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        gfortran \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for layer-cache efficiency
COPY requirements.txt requirements-api.txt pyproject.toml ./
COPY src/ ./src/

# Isolated venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip==24.0 && \
    # Install base project deps (pandas, numpy, statsmodels, etc.)
    pip install --no-cache-dir -r requirements.txt && \
    # Install API-specific deps
    pip install --no-cache-dir \
        "fastapi==0.111.0" \
        "uvicorn[standard]==0.29.0" \
        "pydantic==2.7.1" \
        "joblib==1.4.2" \
        "statsmodels>=0.14" \
        "httpx==0.27.0" \
    # Install the demand_forecast_engine package in editable mode
    && pip install --no-cache-dir -e .

# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime-only system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy venv from builder (keeps image lean)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application source and package manifest
COPY src/ ./src/
COPY pyproject.toml ./

# Pre-bake the model artifact into the image when available.
# Comment out if you prefer volume-mount or S3 loading at runtime.
COPY models/ ./models/

# Fix ownership so the non-root user can write temp files if needed
RUN chown -R appuser:appgroup /app

USER appuser
EXPOSE 8000

# Health check — polls /health every 30 s; 3 failures → unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production ASGI server: Uvicorn with 2 worker processes
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--log-level", "info"]
