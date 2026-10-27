import subprocess
import sys
import json

REPO = "Devershika/E-Commerce-MLOPs"

# GitHub Usernames for collaborators
# Track 1 -> Nakshatra-05
# Track 2 -> Devershika
# Track 3 -> Abhav (leave unassigned if no username matched, or assign collaborator)
# Track 4 -> khushik8605

issues = [
    # Track 1: Nakshatra
    {
        "title": "[Data & DVC] Setup DVC Remote Storage & Data Versioning",
        "body": """### Objective
Initialize DVC repository configuration and link it to remote cloud storage (AWS S3 bucket or Google Cloud Storage / DagsHub).

### Scope & Tasks
- [ ] Initialize DVC inside the repository (`dvc init`).
- [ ] Configure the remote storage endpoint (`dvc remote add -d <name> <url>`).
- [ ] Track `data/sales_data.csv` via DVC (`dvc add data/sales_data.csv`).
- [ ] Verify `dvc push` uploads data and `dvc pull` restores data on a clean directory.

### Acceptance Criteria
- `.dvc/` configuration committed to Git.
- `data/sales_data.csv.dvc` committed to Git without the large CSV.
- Remote upload/download successfully verified.""",
        "assignee": "Nakshatra-05"
    },
    {
        "title": "[Data & DVC] Modularize Preprocessing & Feature Engineering into DVC Stages",
        "body": """### Objective
Refactor preprocessing and feature engineering code from exploratory notebooks into reusable, CLI-runnable Python modules.

### Scope & Tasks
- [ ] Extract cleaning, log1p transformation, and temporal sorting into `src/demand_forecast_engine/preprocessing/`.
- [ ] Implement lag and calendar feature generation with strict temporal train/test splitting (zero data leakage).
- [ ] Create CLI scripts `preprocess.py` and `feature_engineer.py` that read raw data and output `data/processed/train.parquet` and `data/processed/test.parquet`.

### Acceptance Criteria
- Scripts execute from terminal with configurable CLI arguments or `params.yaml`.
- Output parquets are cleanly formatted and ready for model ingestion.""",
        "assignee": "Nakshatra-05"
    },
    {
        "title": "[Data & DVC] Create dvc.yaml Pipeline Configuration",
        "body": """### Objective
Build the end-to-end `dvc.yaml` pipeline connecting raw data, preprocessing, and feature engineering.

### Scope & Tasks
- [ ] Define the `preprocess` stage in `dvc.yaml` tracking input data dependencies and code.
- [ ] Define the `feature_engineer` stage in `dvc.yaml`.
- [ ] Test incremental pipeline reproducibility with `dvc repro`.

### Acceptance Criteria
- `dvc repro` executes all data stages sequentially.
- Running `dvc repro` twice results in `Data and pipelines are up to date`.""",
        "assignee": "Nakshatra-05"
    },
    {
        "title": "[Data & DVC] Parameterize Data Pipeline & Add Quality Checks",
        "body": """### Objective
Externalize pipeline hyperparameters into `params.yaml` and add data validation assertions.

### Scope & Tasks
- [ ] Create `params.yaml` with keys for `test_ratio`, `lag_days`, `seasonal_periods`, and categorical features.
- [ ] Add assertions in preprocessing scripts for null checks, timestamp continuity, and schema types.

### Acceptance Criteria
- Changing a parameter in `params.yaml` triggers only affected downstream stages in `dvc repro`.
- Pipeline fails gracefully with descriptive errors if required columns are missing.""",
        "assignee": "Nakshatra-05"
    },

    # Track 2: Devershika
    {
        "title": "[ML Model] Modularize Holt-Winters Exponential Smoothing Forecaster",
        "body": """### Objective
Refactor the champion Holt-Winters Exponential Smoothing logic from research notebooks into a production-ready model module.

### Scope & Tasks
- [ ] Create `src/demand_forecast_engine/models/holt_winters.py`.
- [ ] Implement `HoltWintersForecaster` class with `.fit(series)` and `.forecast(horizon)`.
- [ ] Support parameter selection for additive/multiplicative trend and seasonality with log1p demand transformation.
- [ ] Write unit tests verifying fitting and forecasting consistency across store-product slices.

### Acceptance Criteria
- Forecaster class can be imported and fitted independently on any Store-Product time series.
- Passing unit tests under `tests/test_holt_winters.py`.""",
        "assignee": "Devershika"
    },
    {
        "title": "[ML Model] Automated Evaluation & Metrics Generation",
        "body": """### Objective
Build an automated evaluation runner that evaluates the Holt-Winters forecaster across all Store–Product pairs.

### Scope & Tasks
- [ ] Create `evaluate.py` calculating MAPE, RMSE, and R2 against the test split.
- [ ] Generate aggregated and per-store metrics.
- [ ] Output metrics into `reports/metrics.json` compatible with DVC metrics tracking.

### Acceptance Criteria
- Script generates `reports/metrics.json` upon execution.
- Metrics match or exceed benchmark accuracy (target MAPE < 0.20).""",
        "assignee": "Devershika"
    },
    {
        "title": "[ML Model] Model Serialization & Bundling",
        "body": """### Objective
Create a serialization utility to bundle the fitted model parameters and transformers into a single deployable artifact.

### Scope & Tasks
- [ ] Implement `save_model.py` to package fitted Holt-Winters models, series maps, and log-transform metadata into `models/holt_winters_model.joblib`.
- [ ] Implement `load_model()` helper to restore the model in memory for inference.

### Acceptance Criteria
- Artifact is saved to `models/holt_winters_model.joblib`.
- Verified that loading the bundle in a separate session produces identical forecasts without requiring raw training data.""",
        "assignee": "Devershika"
    },
    {
        "title": "[ML Model] Integrate Holt-Winters Stage into DVC & Push to Remote Registry",
        "body": """### Objective
Incorporate the model training, evaluation, and packaging stages into `dvc.yaml` and remote registry.

### Scope & Tasks
- [ ] Add `train` and `evaluate` stages to `dvc.yaml`.
- [ ] Mark `models/holt_winters_model.joblib` as an output in `dvc.yaml`.
- [ ] Configure DVC to push the trained model artifact to remote storage upon `dvc push`.

### Acceptance Criteria
- `dvc repro` runs from raw data through model artifact creation.
- Model artifact is successfully pushed to the remote storage registry.""",
        "assignee": "Devershika"
    },

    # Track 3: Abhav
    {
        "title": "[Serving] Design FastAPI Prediction Service & Request Validation",
        "body": """### Objective
Build the production FastAPI inference microservice with strict request/response data contracts.

### Scope & Tasks
- [ ] Setup FastAPI project in `src/api/` with `main.py` and `schemas.py`.
- [ ] Define Pydantic request schema accepting `store_id`, `product_id`, `category`, and `forecast_horizon_days`.
- [ ] Implement `/health` and `/predict` endpoints with mock responses for early frontend integration.

### Acceptance Criteria
- `GET /health` returns status 200 with service metadata.
- `POST /predict` validates types and returns HTTP 422 for malformed requests.""",
        "assignee": None
    },
    {
        "title": "[Serving] Dynamic Model Loader from DVC Remote / S3",
        "body": """### Objective
Implement the runtime model loading mechanism to download and cache `holt_winters_model.joblib` from remote storage.

### Scope & Tasks
- [ ] Implement `src/api/loader.py` with automatic downloading from S3/DVC remote on startup if not present locally.
- [ ] Support local file path fallback for testing/development.
- [ ] Load model bundle into FastAPI application state on lifecycle startup.

### Acceptance Criteria
- Server boots up, downloads missing artifact from remote storage, and loads it into memory.
- `/health` endpoint exposes model loading status.""",
        "assignee": None
    },
    {
        "title": "[Serving] Implement End-to-End Inference Logic",
        "body": """### Objective
Connect the loaded Holt-Winters model bundle to the `/predict` API route to serve live predictions.

### Scope & Tasks
- [ ] Map incoming store and product requests to corresponding model series parameters.
- [ ] Compute forecast for the requested horizon and invert log1p transformation (`np.expm1`).
- [ ] Return forecast series with dates, predictions, and model metadata.

### Acceptance Criteria
- End-to-end integration test passes: sending valid JSON returns valid demand forecast numbers.""",
        "assignee": None
    },
    {
        "title": "[Serving] Dockerize FastAPI Service with Production ASGI",
        "body": """### Objective
Package the FastAPI service into an optimized, secure Docker container.

### Scope & Tasks
- [ ] Create multi-stage `Dockerfile` using Python 3.11 slim image.
- [ ] Install only runtime dependencies to minimize image size.
- [ ] Run with non-root user and Uvicorn/Gunicorn ASGI server.
- [ ] Add `docker-compose.yml` for local container testing.

### Acceptance Criteria
- `docker build` succeeds with small image footprint (<500MB).
- Container runs locally on port 8000 and responds to `/health` and `/predict`.""",
        "assignee": None
    },

    # Track 4: Khushi
    {
        "title": "[CI/CD & UI] Build Interactive Streamlit Frontend",
        "body": """### Objective
Modernize `apps/frontend_v01.py` into a full-featured Streamlit user interface connected to the FastAPI backend.

### Scope & Tasks
- [ ] Refactor `apps/frontend_v01.py` to `apps/app.py`.
- [ ] Add interactive sidebar controls for Category, Store ID, Product ID, and Forecast Horizon (days).
- [ ] Connect to backend API `http://<API_HOST>:8000/predict` with timeout and error handling.
- [ ] Render interactive Plotly charts showing historical sales demand alongside predicted forecast demand.

### Acceptance Criteria
- Intuitive, responsive dashboard UI.
- Graceful UI alerts if API is unreachable.
- Visual forecast graphs rendered accurately.""",
        "assignee": "khushik8605"
    },
    {
        "title": "[CI/CD & UI] Setup GitHub Actions CI Pipeline (Lint & Test)",
        "body": """### Objective
Implement automated continuous integration for code style and unit tests.

### Scope & Tasks
- [ ] Create `.github/workflows/ci.yaml`.
- [ ] Configure workflow to trigger on Pull Requests to `main`.
- [ ] Run lint checks (`flake8` / `black --check`).
- [ ] Run unit tests with `pytest`.

### Acceptance Criteria
- CI runs automatically on PRs.
- Passing tests are required before PR merges.""",
        "assignee": "khushik8605"
    },
    {
        "title": "[CI/CD & UI] Setup GitHub Actions CD Pipeline (Docker Registry Build & Push)",
        "body": """### Objective
Automate container image build and publishing to Container Registry on merges.

### Scope & Tasks
- [ ] Create `.github/workflows/cd.yaml` triggered on push to `main`.
- [ ] Setup authentication with AWS ECR or Docker Hub via GitHub Secrets.
- [ ] Build Docker image and tag with commit SHA and `latest`.
- [ ] Push image to container registry.

### Acceptance Criteria
- Merging to `main` builds and pushes container image to the registry automatically.""",
        "assignee": "khushik8605"
    },
    {
        "title": "[CI/CD & UI] AWS EC2 Provisioning, Deployment Script & Auto-Restart Policy",
        "body": """### Objective
Deploy the prediction service container to an AWS EC2 instance with high availability and auto-restart.

### Scope & Tasks
- [ ] Launch and configure AWS EC2 instance (Ubuntu, Docker installed, Security Group opening port 8000).
- [ ] Setup `docker-compose.prod.yml` with `restart: always` restart policy.
- [ ] Write deployment bash script `scripts/deploy.sh` to pull latest image and restart the container service.
- [ ] Validate external connectivity from Streamlit frontend to EC2 public IP.

### Acceptance Criteria
- Container automatically recovers if process crashes or instance reboots.
- Prediction API is reachable via public EC2 IP/DNS.""",
        "assignee": "khushik8605"
    }
]

def create_issues():
    for issue in issues:
        cmd = [
            "gh", "issue", "create",
            "--repo", REPO,
            "--title", issue["title"],
            "--body", issue["body"]
        ]
        if issue["assignee"]:
            cmd.extend(["--assignee", issue["assignee"]])
        
        print(f"Creating: {issue['title']} (Assignee: {issue['assignee']})...")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f" -> Created: {res.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f" -> Error creating issue: {e.stderr.strip()}")

if __name__ == "__main__":
    create_issues()
