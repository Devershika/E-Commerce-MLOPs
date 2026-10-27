# Independent Work Guide & Dependency Matrix

## 1. Executive Summary: Why Work Is Not Blocked
A common bottleneck in Machine Learning projects is "sequential waiting" (e.g., Serving waits for Training, Frontend waits for Serving, Training waits for DVC). 

In this MLOps architecture, **all 4 team members can work concurrently on Day 1**. 
We achieve this through **Contract-First Design** and **Mock Interfaces**:
- Interfaces (JSON payloads, file formats, and directories) are locked down in advance.
- Each person uses a local mock/stub for upstream inputs.
- When independent components reach ~80-90% completion, plugging them together requires only changing a filepath or an API URL.

---

## 2. Parallel Work vs. Dependency Matrix

```mermaid
flowchart TD
    subgraph Phase 1: 100% Independent (Day 1 - Day 10)
        N_IND["Nakshatra (Track 1)
        - Configure DVC remote
        - Write preprocessing & feature logic
        - Test dvc repro with local CSV"]

        D_IND["Devershika (Track 2)
        - Modularize Holt-Winters model
        - Write evaluate.py & metric logging
        - Test model serialization with raw data"]

        A_IND["Abhav (Track 3)
        - Define Pydantic schemas
        - Build FastAPI endpoints with Mock Predictor
        - Write Dockerfile & test container locally"]

        K_IND["Khushi (Track 4)
        - Build Streamlit UI with stubbed responses
        - Write GitHub Actions CI/CD workflows
        - Provision AWS EC2 & configure Docker runtime"]
    end

    subgraph Phase 2: Lightweight Integration (Day 11+)
        INT1["Integrate DVC Pipeline:
        Devershika consumes Nakshatra's
        data/processed/*.parquet"]
        
        INT2["Integrate Serving:
        Abhav loads Devershika's
        models/holt_winters_model.joblib"]

        INT3["Integrate Cloud & Frontend:
        Khushi points Streamlit UI to
        Abhav's EC2 container endpoint"]
    end

    N_IND -.-> INT1
    D_IND -.-> INT1
    D_IND -.-> INT2
    A_IND -.-> INT2
    A_IND -.-> INT3
    K_IND -.-> INT3
```

---

## 3. Detailed Breakdown per Team Member

### 3.1 Nakshatra (Track 1: Data Engineering & DVC Pipeline)
* **What can be done 100% INDEPENDENTLY:**
  - Setup DVC remote storage (S3 bucket or DagsHub/GCS) and commit `.dvc/config`.
  - Version `data/sales_data.csv` using `dvc add`.
  - Refactor data cleaning, log1p transformation, and lag features into `src/demand_forecast_engine/preprocessing/`.
  - Create `params.yaml` with window parameters (`lag_days: [1, 7, 14]`, `test_ratio: 0.2`).
  - Create `dvc.yaml` covering the `preprocess` and `feature_engineer` stages.
  - Verify that `dvc repro` generates `data/processed/train.parquet` and `data/processed/test.parquet`.
* **What is DEPENDENT on others:**
  - **None.** Nakshatra can complete 100% of her assigned track without needing anything from Devershika, Abhav, or Khushi.

---

### 3.2 Devershika (Track 2: ML Model & Registry - Holt-Winters)
* **What can be done 100% INDEPENDENTLY:**
  - Modularize the Holt-Winters Exponential Smoothing logic from `Notebooks/Holt_Winter(Expoential Smoothing).ipynb` into `src/demand_forecast_engine/models/holt_winters.py`.
  - Train and evaluate the model using the existing raw `data/sales_data.csv` directly or a quick local pandas slice.
  - Write `save_model.py` to package the fitted parameters, seasonality configs, and demand transformers into `models/holt_winters_model.joblib`.
  - Write `evaluate.py` to calculate validation MAPE, RMSE, and $R^2$ into `reports/metrics.json`.
* **What is DEPENDENT on others (Plugging in at the end):**
  - **Minor Dependency on Nakshatra:** Once Nakshatra finishes `data/processed/train.parquet`, Devershika simply points `train.py` to read from `data/processed/train.parquet` instead of raw data, and appends the `train` stage to `dvc.yaml`.

---

### 3.3 Abhav (Track 3: Serving Infrastructure & Containerization)
* **What can be done 100% INDEPENDENTLY:**
  - Build the FastAPI application structure in `src/api/` (`main.py`, `schemas.py`).
  - Define the request/response contracts using Pydantic (e.g. `store_id`, `product_id`, `forecast_horizon_days`).
  - Use a **Dummy Mock Forecaster** function so the API works right away:
    ```python
    # src/api/mock_service.py (Used until model artifact is ready)
    def mock_predict(store_id: str, product_id: str, horizon: int):
        import datetime
        today = datetime.date.today()
        return [
            {"date": str(today + datetime.timedelta(days=i)), "predicted_demand": round(50.0 + i * 1.5, 2)}
            for i in range(horizon)
        ]
    ```
  - Implement `/health` and `/predict` routes and test with `curl` or Postman.
  - Write the `Dockerfile` and `docker-compose.yml`.
  - Test building and running the container locally:
    ```bash
    docker build -t demand-forecast-api .
    docker run -p 8000:8000 demand-forecast-api
    ```
* **What is DEPENDENT on others (Plugging in at the end):**
  - **Minor Dependency on Devershika:** Once `models/holt_winters_model.joblib` is generated, Abhav replaces the `mock_predict()` call with `joblib.load()` inside `src/api/loader.py`.

---

### 3.4 Khushi (Track 4: CI/CD, Cloud Deployment & Streamlit Frontend)
* **What can be done 100% INDEPENDENTLY:**
  - **Streamlit Frontend:**
    - Refactor `apps/frontend_v01.py` into a modern interactive UI (`apps/app.py`).
    - Build sidebar filters for Category, Store ID, Product ID, and forecast horizon slider.
    - Test against a mock API URL or local dictionary so all Plotly forecast curves, tables, and error toasts render beautifully.
  - **CI/CD Pipeline:**
    - Write `.github/workflows/ci.yaml` to run `pytest` and linter checks on every Pull Request.
    - Write `.github/workflows/cd.yaml` to test logging in to AWS ECR / Docker Hub and building/pushing container images.
  - **AWS EC2 Provisioning:**
    - Launch the AWS EC2 instance (Ubuntu, security group allowing port 8000).
    - Install Docker & Docker Compose on the EC2 instance.
    - Setup `docker-compose.prod.yml` with `restart: always`.
    - Write a deployment bash script (`scripts/deploy.sh`) to pull images and trigger restart.
* **What is DEPENDENT on others (Plugging in at the end):**
  - **Minor Dependency on Abhav:** Once Abhav finishes the Dockerfile, Khushi's CD pipeline builds his Dockerfile. In `apps/app.py`, Khushi updates `API_URL = "http://<EC2-PUBLIC-IP>:8000/predict"`.

---

## 4. Git Branching Strategy (Zero Merge Conflicts)

Because each person works in completely distinct directories, git merge conflicts are avoided:

| Developer | Branch Name | Directory Ownership |
| :--- | :--- | :--- |
| **Nakshatra** | `feature/dvc-pipeline` | `data/`, `params.yaml`, `dvc.yaml`, `src/demand_forecast_engine/preprocessing/` |
| **Devershika** | `feature/holt-winters` | `src/demand_forecast_engine/models/`, `evaluate.py`, `save_model.py` |
| **Abhav** | `feature/api-docker` | `src/api/`, `Dockerfile`, `docker-compose.yml` |
| **Khushi** | `feature/frontend-cicd` | `apps/app.py`, `.github/workflows/`, `scripts/deploy.sh` |

---

## 5. Integration Checklist (Connecting the Pieces)

When each person's independent track is finished:

1. **Step 1 (Nakshatra $\to$ Devershika):**
   - Devershika pulls `feature/dvc-pipeline`.
   - Replaces input path in `train.py` with `data/processed/train.parquet`.
   - Confirms `dvc repro` trains and packages `models/holt_winters_model.joblib`.
2. **Step 2 (Devershika $\to$ Abhav):**
   - Abhav places `models/holt_winters_model.joblib` in his local `models/` directory or pulls it via DVC.
   - Switches `src/api/` from the mock function to the actual model loader.
3. **Step 3 (Abhav $\to$ Khushi):**
   - Khushi triggers the CD workflow to build Abhav's Docker image and deploy it on EC2.
   - Streamlit frontend is pointed to the live EC2 IP and validated end-to-end.
