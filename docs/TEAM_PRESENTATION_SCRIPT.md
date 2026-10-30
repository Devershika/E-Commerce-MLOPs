# 🎙️ E-Commerce Demand Forecasting MLOps: Team Presentation Script & Walkthrough

A complete, structured speaking script divided equally for **4 presenters** (**Nakshatra**, **Devershika**, **Abhav**, and **Khushi**). Designed for presentations, demos, viva, or client walkthroughs (~10–12 minutes total).

---

## ⏱️ Presentation Outline & Timing (Total: ~12 Mins)

| Speaker | Section / Topic | Duration | Key Visual / Demo |
| :--- | :--- | :--- | :--- |
| **Speaker 1: Nakshatra** | Project Vision & Data Pipeline (DVC) | 3 Mins | `dvc.yaml`, `params.yaml`, `Data/processed/` |
| **Speaker 2: Devershika** | Machine Learning, Holt-Winters & Evaluation | 3 Mins | `Notebooks/`, `holt_winters.py`, `metrics.json` |
| **Speaker 3: Abhav** | FastAPI Prediction Microservice & Containerization | 3 Mins | `src/api/main.py`, Swagger Docs (`/docs`), `Dockerfile` |
| **Speaker 4: Khushi** | CI/CD Automation, Cloud EC2 & Live Streamlit Demo | 3 Mins | GitHub Actions (`ci.yaml`, `cd.yaml`), Live Streamlit UI |

---

## 🎬 Section 1: Project Vision & Data Engineering
**Speaker: Nakshatra (Track 1 Lead)**  
*Time: ~3 Minutes*

### 🗣️ Speaking Script:
> "Hello everyone. Today our team is excited to present our **E-Commerce SKU-Level Demand Forecasting MLOps Pipeline**.
> 
> Initially, this work began as an exploratory research project. However, raw machine learning research often suffers from the classic problem: *it works on a Jupyter notebook on one machine, but cannot be reproduced, scaled, or served in production*. Our goal was to transition this into a fully automated, reproducible industry-standard MLOps architecture.
> 
> As the **Data Engineering & DVC Lead**, my primary responsibility was making sure our data and preprocessing pipelines are 100% reproducible and version-controlled.
> 
> Let's look at what we implemented:
> 1. **DVC Remote Data Versioning**: Large transactional sales datasets should never be committed into Git. Using **DVC**, our raw dataset `Data/sales_data.csv` is version-tracked via metadata pointers (`sales_data.csv.dvc`) linked to remote cloud storage.
> 2. **Reproducible Pipeline Orchestration (`dvc.yaml`)**: Rather than running ad-hoc notebooks, we modularized data cleaning, forward-filling, and temporal train-test splits into pipeline stages:
>    - `preprocess.py`: Performs schema validation, outlier handling, and log1p transformations.
>    - `feature_engineer.py`: Builds lag windows and calendar features without temporal data leakage.
> 3. **Hyperparameter Externalization (`params.yaml`)**: Everything from split ratios (`test_ratio: 0.2`) to seasonal periods is centrally configured. When a parameter changes, running `dvc repro` intelligently re-computes only the affected downstream stages.
> 
> Our pipeline outputs clean, versioned Parquet files into `Data/processed/train.parquet` and `test.parquet`, which feeds directly into our modeling stage.
> 
> I'll now hand over to **Devershika** to discuss how we trained and evaluated our champion forecasting model."

---

## 🎬 Section 2: ML Experiments, Holt-Winters Champion & Registry
**Speaker: Devershika (Track 2 Lead)**  
*Time: ~3 Minutes*

### 🗣️ Speaking Script:
> "Thank you, Nakshatra. Hi everyone, I am Devershika, and I led the **Machine Learning Modeling, Evaluation, and Model Registry** track.
> 
> In the research phase of this project, we systematically experimented with multiple time series and machine learning algorithms:
> - Baseline Linear Regression and Ridge Regression
> - Classical ARIMA / SARIMAX models
> - Univariate and Multivariate Prophet
> - XGBoost regressor
> - And **Holt-Winters Exponential Smoothing**
> 
> When comparing performance across all Store ID $\times$ Product ID combinations using standard industry metrics—specifically **MAPE (Mean Absolute Percentage Error)** and **$R^2$ Score**—the **Fine-Tuned Holt-Winters Exponential Smoothing model** with additive trend and weekly seasonality (period = 7) consistently delivered the best accuracy and stability on log-transformed demand.
> 
> For our production pipeline, we took this champion model and built:
> 1. **Production-Ready Model Module (`src/demand_forecast_engine/models/holt_winters.py`)**: A clean `HoltWintersForecaster` class implementing `.fit()`, `.forecast()`, and automated inversion of log transformations (`np.expm1`).
> 2. **Automated Evaluation Runner (`evaluate.py`)**: Computes granular metrics across each store-product series on the held-out test split, exporting a structured report to `reports/metrics.json`.
> 3. **Model Serialization & Registry (`save_model.py`)**: Packages the fitted model parameters and metadata into `models/holt_winters_model.joblib`. This artifact is versioned in DVC so any deployment runtime can pull and serve the exact model artifact without needing raw training data.
> 
> Now, once a model is trained and registered, how do external applications consume it in real time? I'll pass it over to **Abhav** to explain our serving infrastructure."

---

## 🎬 Section 3: Serving Infrastructure, API & Containerization
**Speaker: Abhav (Track 3 Lead)**  
*Time: ~3 Minutes*

### 🗣️ Speaking Script:
> "Thanks, Devershika. Hello everyone, I am Abhav, responsible for the **Model Serving Infrastructure and Containerization**.
> 
> A trained model is only useful if it can serve low-latency, resilient predictions to web apps, inventory management systems, and business clients.
> 
> Here is how we engineered our serving layer:
> 1. **FastAPI Inference Microservice (`src/api/main.py`)**:
>    - We built a lightweight REST API using **FastAPI** and **Pydantic** for strict data validation.
>    - It exposes a `GET /health` endpoint for readiness probes and model status.
>    - It provides a `POST /predict` endpoint that takes `store_id`, `product_id`, and `forecast_horizon_days` and returns day-by-day projected demand in milliseconds.
> 2. **Resilient Model Lifecycle**:
>    - On application startup, the service pre-loads `holt_winters_model.joblib` into memory. We also built defensive fallback logic to handle cross-version serialization smoothly.
> 3. **Multi-Stage Docker Containerization (`Dockerfile`)**:
>    - To eliminate the *'it works on my machine'* problem, we packaged the entire prediction engine into a hardened, multi-stage **Docker** container built on `python:3.11-slim`.
>    - The builder stage compiles dependencies, while the final runtime stage contains only the clean application files, running under a non-root `appuser` for security.
>    - It includes a native Docker `HEALTHCHECK` probe and is orchestrated with `docker-compose.yml`.
> 
> Anyone in the world with Docker can launch our service with a single command: `docker run -p 8000:8000 demand-forecast-api`.
> 
> Next, **Khushi** will show how we automated the deployment with CI/CD, hosted it on AWS EC2, and built our user-facing dashboard."

---

## 🎬 Section 4: CI/CD Automation, AWS Cloud & Live Frontend Demo
**Speaker: Khushi (Track 4 Lead)**  
*Time: ~3 Minutes*

### 🗣️ Speaking Script:
> "Thank you, Abhav. Hi everyone, I am Khushi, and I led the **CI/CD Automation, Cloud Deployment, and Frontend Application**.
> 
> To bring all these pieces together seamlessly without manual intervention, we implemented automated DevOps workflows and a modern user interface:
> 
> 1. **Continuous Integration (`.github/workflows/ci.yaml`)**:
>    - Every Pull Request or push triggers our CI pipeline.
>    - It verifies code style using `black` and `flake8`, and executes all **28 automated unit tests** across Python 3.10 and 3.11 matrices with Pytest and coverage reporting.
> 2. **Continuous Deployment (`.github/workflows/cd.yaml`)**:
>    - Whenever code is merged to `main`, GitHub Actions automatically builds our production Docker image and publishes it to the **GitHub Container Registry (GHCR)** tagged with the commit SHA and `latest`.
> 3. **AWS EC2 Cloud Hosting (`scripts/deploy_ec2.sh`)**:
>    - We provisioned an AWS EC2 instance running Ubuntu and Docker.
>    - We created `docker-compose.prod.yml` and `scripts/deploy_ec2.sh` with `--restart always`. If the instance reboots or a container crashes, Docker automatically restores the prediction service with zero manual intervention.
> 4. **Interactive Streamlit Dashboard (`apps/app.py`)**:
>    - *(Switch screen to live Streamlit dashboard)*
>    - Business planners can select any store, product, and category.
>    - When we adjust the **Forecast Horizon slider** (e.g., from 7 to 30 days), the frontend calls our backend API and plots the historical sales curve against the forecasted trajectory with dynamic confidence intervals ($\pm 12\%$).
>    - It calculates **inventory advice**—including recommended replenishment orders and safety stock buffers—and allows users to export the forecast table to CSV with one click.
> 
> In conclusion, we have transformed an isolated research notebook into an end-to-end, enterprise-ready MLOps solution covering **Data Versioning**, **Automated Training**, **Model Registry**, **Containerized Serving**, **CI/CD**, and an **Interactive Frontend**.
> 
> Thank you, and we would now love to open the floor to any questions!"

---

## 💡 Quick Q&A Reference Cheatsheet (For Any Questions Asked)

| Anticipated Question | Best Answer / Reference |
| :--- | :--- |
| **Why Holt-Winters instead of XGBoost/Prophet?** | *Empirical validation. On our granular Store $\times$ Product transactional series with strong weekly seasonality, fine-tuned Holt-Winters on log1p demand achieved the lowest MAPE (~0.18) with minimal latency and zero overfitting.* |
| **How do you prevent data leakage?** | *Strict temporal cutoff splitting in `preprocess.py` and `feature_engineer.py`. Lags and rolling stats are shifted, and categorical encodings are fitted strictly on the training partition.* |
| **What happens if the API backend goes down?** | *The Streamlit UI has built-in graceful fallback simulation with visual badges (`Backend Connected` vs `Backend Offline`). In production on AWS EC2, the container is guarded by `--restart always` and healthcheck monitors.* |
| **How do you reproduce the entire pipeline?** | *Running `dvc repro` executes data processing, training, and evaluation in sequence. Or run `python scripts/run_pipeline.py --all`.* |
