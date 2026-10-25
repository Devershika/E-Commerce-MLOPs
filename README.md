# Demand Forecasting using Time-Series & Machine Learning Algorithms

## Overview
This project focuses on demand forecasting at a granular level (Store ID × Product ID) using multiple time series and machine learning models. The goal is to build baseline models, systematically fine-tune them, and perform a fair comparison using standard evaluation metrics.
The pipeline is designed to be:
* Modular
* Interpretable
* Industry aligned


## Objectives
* Preserve store level and product level granularity
* Implement baseline forecasting models
* Perform step by step fine tuning for each model
* Compare baseline vs fine-tuned models
* Identify the best-performing model per Store–Product
* Provide a clean evaluation using MAPE and R²

## Repo status and notes
* Package code lives under src/demand_forecast_engine/.
* Notebooks for exploration and reproducible experiments are in Notebooks/.
* Data folders present: Data/ and data/ — 

## Quick start
* Clone the repo git clone https://github.com/Devershika/demand-forecast.git cd demand-forecast

* Install dependencies
  - Using pip: python -m venv venv source venv/bin/activate pip install -r requirements.txt
  - Recommended editable install (to import package modules in src/): pip install -e .
* Run notebooks
  - Launch Jupyter and open notebooks in the Notebooks/ directory: jupyter lab
  - Notebooks contain step-by-step exploration, preprocessing and modeling examples.

## Data Description
The dataset consists of time-stamped transactional demand data collected across multiple stores and products. Each record corresponds to the demand observed for a specific product in a particular store on a given date. Along with the target variable Demand, the dataset includes several explanatory features such as pricing information, promotional indicators, competitor pricing, and contextual attributes like region, category, and weather conditions. These additional variables enable both univariate and multivariate forecasting approaches. The temporal nature of the data makes it suitable for classical time series modeling, while the presence of external regressors allows the exploration of more advanced models such as SARIMAX and Prophet with regressors.


## Data Preprocessing

The preprocessing pipeline ensures that the data is clean, consistent and suitable for time series modeling. Key steps include:
* Conversion of the date column to datetime format and proper temporal sorting
* Handling missing values using forward filling within each Store–Product series
* Log transformation of demand (log1p) to stabilize variance

Feature engineering, including:
* Lag features (1 day, 7 day, 14 day)
* Time-based features (day, week, month, weekday)
* Encoding categorical variables using label encoding
* Ensuring no data leakage through strict temporal train–test splitting

## Models Implemented
### Baseline Models
The following baseline models are implemented to establish a performance reference:
* Linear Regression
* Holt-Winters Exponential Smoothing
* ARIMA (1,1,1)
* Prophet (Univariate)
* Prophet (Multivariate)
* XGBoost



### Fine-Tuned Models
Each model is fine-tuned independently, using interpretable and statistically grounded techniques:

1️. Linear Regression (Ridge-Regularized)
  - Lag-based feature engineering to capture temporal dependencies
  - Ridge regularization to mitigate multicollinearity
  - Hyperparameter tuning via cross-validated alpha selection

2️. Holt-Winters Exponential Smoothing
  - Selection of additive vs multiplicative trend and seasonality
  - Seasonality optimization (weekly/monthly patterns)
  - Evaluation of smoothing parameters

3️. ARIMA 
  - Stationarity testing using Augmented Dickey–Fuller (ADF) test
  - Manual differencing to ensure stationarity
  - Grid search over (p, d, q) using AIC minimization
  - Residual diagnostics to validate assumptions
  - Explicit baseline vs fine-tuned ARIMA comparison

4️. Prophet (Univariate)
  - Changepoint prior tuning to control trend flexibility
  - Selective seasonality activation
  - Noise reduction through regularization

5️. Prophet (Multivariate)
  - Correlation-based regressor selection
  - Lagged external regressors
  - Prior scale tuning for regressor influence

6 . XGBoost
  - Handles null values from rolling window and lag features automatically
  - Applies L1/L2 regularization without explicity adding penalties in the loss function
  - Industry standard for tabular data
 



## Evaluation Metrics

All models are evaluated using the following metrics:
1. MAPE (Mean Absolute Percentage Error)
Measures forecasting accuracy; lower values indicate better performance.
2. R² Score (Coefficient of Determination)
Measures how well the model explains demand variability.

Evaluation is conducted:
- At the Store,Product,Category,Region level
- Aggregated per model
- With baseline vs fine-tuned comparisons

## Usage examples
- Jupyter notebooks in Notebooks/ show end-to-end examples of preprocessing, feature creation, training and evaluation.
- Python REPL example (after pip install -e .): python -c "import demand_forecast_engine as dfe; print('package imported:', dfe)"
- To run a minimal preprocessing step (example pattern — adapt to available module APIs): python -c "from demand_forecast_engine.preprocessing.dataset import DatasetLoader; data=DataSetLoader(file_path,config_path),
df=data.read_data() "

## Contributing
Feel free to open issues or submit pull requests!

## Authors
This project was developed by:

- Devershika Mohane
- Harsh Dhiman
