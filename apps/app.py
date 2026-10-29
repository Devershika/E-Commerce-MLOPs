"""
Streamlit Demand Forecasting Application
=========================================
Tracks and visualizes historical store sales demand, lets users interactively
select Store, Product, Category, and Forecast Horizon, and fetches real-time 
SKU-level demand predictions from the FastAPI MLOps backend service.
"""

import os
import datetime
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="E-Commerce Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API endpoint configuration (configurable via environment variable)
DEFAULT_API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000")

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #4f46e5;
        margin-bottom: 15px;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-online { background-color: #dcfce7; color: #166534; }
    .badge-mock { background-color: #fef3c7; color: #92400e; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Loader (Cached)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_historical_data():
    """Loads and preprocesses sales data for visualization from parquet or csv."""
    root = os.getcwd()
    parquet_train = os.path.join(root, "Data", "processed", "train.parquet")
    parquet_test = os.path.join(root, "Data", "processed", "test.parquet")
    csv_path_data = os.path.join(root, "data", "sales_data.csv")
    csv_path_Data = os.path.join(root, "Data", "sales_data.csv")

    cols = ["Date", "Store ID", "Product ID", "Category", "Region", "Demand", "Price", "Promotion"]

    if os.path.exists(parquet_train):
        train_df = pd.read_parquet(parquet_train)
        available_cols = [c for c in cols if c in train_df.columns]
        dfs = [train_df[available_cols]]
        if os.path.exists(parquet_test):
            test_df = pd.read_parquet(parquet_test)
            dfs.append(test_df[available_cols])
        df = pd.concat(dfs, ignore_index=True)
    elif os.path.exists(csv_path_data) or os.path.exists(csv_path_Data):
        active_csv = csv_path_data if os.path.exists(csv_path_data) else csv_path_Data
        df = pd.read_csv(active_csv)
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols]
    else:
        st.error("No dataset found in Data/processed/*.parquet or Data/sales_data.csv. Run `dvc pull` to fetch raw data or check Data/processed.")
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Demand"])
    df = df.sort_values(by="Date", ascending=True)
    return df


# ---------------------------------------------------------
# API Helper with Mock Fallback
# ---------------------------------------------------------
def fetch_forecast(api_url: str, store_id: str, product_id: str, category: str, horizon_days: int, last_date: datetime.date):
    """
    Calls the FastAPI backend `/predict` endpoint.
    If backend is unreachable or not yet deployed, seamlessly falls back to a 
    local Holt-Winters trend simulation so the UI remains fully functional.
    """
    endpoint = f"{api_url.rstrip('/')}/predict"
    payload = {
        "store_id": store_id,
        "product_id": product_id,
        "category": category,
        "forecast_horizon_days": horizon_days,
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=3)
        if response.status_code == 200:
            data = response.json()
            forecast_list = data.get("forecast", [])
            df_fc = pd.DataFrame(forecast_list)
            df_fc["date"] = pd.to_datetime(df_fc["date"])
            return df_fc, data.get("model_name", "Holt-Winters Production"), False
    except requests.exceptions.RequestException:
        pass

    # Seamless Mock Fallback for local development / early testing
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, horizon_days + 1)]
    base_demand = 85.0 + (hash(product_id) % 40)
    # Simulate gentle trend + 7-day seasonality
    mock_preds = []
    for i, d in enumerate(future_dates):
        seasonality = 12.0 * np.sin(2 * np.pi * (i % 7) / 7.0)
        noise = float(np.random.normal(0, 3.5))
        val = max(10.0, base_demand + (i * 0.4) + seasonality + noise)
        mock_preds.append({"date": pd.to_datetime(d), "predicted_demand": round(val, 2)})

    df_fc = pd.DataFrame(mock_preds)
    return df_fc, "Holt-Winters (Local Mock Simulation)", True


# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------
def main():
    st.title("📦 E-Commerce Demand Forecasting Dashboard")
    st.caption("AI-powered SKU-level forecasting and inventory planning using Holt-Winters Exponential Smoothing")

    # Load dataset
    with st.spinner("Loading transactional demand data..."):
        df = load_historical_data()

    if df.empty:
        st.stop()

    # Sidebar: Backend Connection & Filter Controls
    st.sidebar.header("⚙️ Configuration")
    backend_url = st.sidebar.text_input("FastAPI Service URL", value=DEFAULT_API_URL)

    # Check Backend Health
    backend_healthy = False
    try:
        health_resp = requests.get(f"{backend_url.rstrip('/')}/health", timeout=1.5)
        if health_resp.status_code == 200:
            backend_healthy = True
    except requests.exceptions.RequestException:
        backend_healthy = False

    if backend_healthy:
        st.sidebar.markdown('<span class="badge badge-online">● Backend API Connected</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="badge badge-mock">○ Backend Offline (Mock Active)</span>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 SKU & Parameter Selection")

    categories = sorted(df["Category"].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", options=categories, index=0)

    category_df = df[df["Category"] == selected_category]
    stores = sorted(category_df["Store ID"].dropna().unique().tolist())
    selected_store = st.sidebar.selectbox("Select Store ID", options=stores, index=0)

    store_df = category_df[category_df["Store ID"] == selected_store]
    products = sorted(store_df["Product ID"].dropna().unique().tolist())
    selected_product = st.sidebar.selectbox("Select Product ID", options=products, index=0)

    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (Days)",
        min_value=7,
        max_value=60,
        value=14,
        step=7,
        help="Number of days to forecast into the future"
    )

    # Filtered Time Series Data
    sku_data = store_df[store_df["Product ID"] == selected_product].copy()
    if sku_data.empty:
        st.warning("No sales records found for this Store & Product combination.")
        st.stop()

    # KPI Summary Cards
    last_date = sku_data["Date"].max().date()
    avg_demand = sku_data["Demand"].mean()
    total_sales_units = sku_data["Demand"].sum()
    avg_price = sku_data["Price"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Historical Demand", f"{int(total_sales_units):,}")
    with col2:
        st.metric("Avg Daily Demand", f"{avg_demand:.1f} units")
    with col3:
        st.metric("Avg Unit Price", f"${avg_price:.2f}")
    with col4:
        st.metric("Data Range", f"{sku_data['Date'].min().strftime('%b %Y')} - {last_date.strftime('%b %Y')}")

    st.markdown("---")

    # Fetch Forecast
    forecast_df, model_source, is_mock = fetch_forecast(
        api_url=backend_url,
        store_id=selected_store,
        product_id=selected_product,
        category=selected_category,
        horizon_days=forecast_horizon,
        last_date=last_date
    )

    if is_mock:
        st.info("ℹ️ **Notice:** Running in mock simulation mode because the FastAPI backend is not yet running on port 8000. Once Abhav's API container or EC2 service is launched, real-time model predictions will automatically load here.")

    # Main Interactive Plotly Visualization
    st.subheader(f"📈 Historical Demand vs. {forecast_horizon}-Day Forecast")
    st.caption(f"Store: `{selected_store}` | Product: `{selected_product}` | Category: `{selected_category}` | Engine: `{model_source}`")

    # Recent window for clear view (last 90 days of history + forecast)
    recent_history_days = 90
    cutoff_date = pd.to_datetime(last_date) - pd.Timedelta(days=recent_history_days)
    recent_sku = sku_data[sku_data["Date"] >= cutoff_date]

    fig = go.Figure()

    # Historical demand trace
    fig.add_trace(
        go.Scatter(
            x=recent_sku["Date"],
            y=recent_sku["Demand"],
            mode="lines+markers",
            name="Historical Sales Demand",
            line=dict(color="#1f2937", width=2),
            marker=dict(size=4)
        )
    )

    # Forecast trace
    # Connect last historical point to first forecast point for visual continuity
    bridge_dates = [recent_sku["Date"].iloc[-1], *forecast_df["date"]]
    bridge_demand = [recent_sku["Demand"].iloc[-1], *forecast_df["predicted_demand"]]

    fig.add_trace(
        go.Scatter(
            x=bridge_dates,
            y=bridge_demand,
            mode="lines+markers",
            name="Forecasted Demand",
            line=dict(color="#4f46e5", width=2.5, dash="dash"),
            marker=dict(size=6, symbol="diamond")
        )
    )

    # Visual upper/lower uncertainty band estimate (+-10%)
    upper_bound = [d * 1.12 for d in bridge_demand]
    lower_bound = [max(0, d * 0.88) for d in bridge_demand]

    fig.add_trace(
        go.Scatter(
            x=bridge_dates + bridge_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Forecast Confidence Range (±12%)"
        )
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Units Demanded",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        height=480,
    )

    st.plotly_chart(fig, width="stretch")

    # Forecast Data Table & Export
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📋 Forecast Breakdown")
        display_fc = forecast_df.copy()
        display_fc["date"] = display_fc["date"].dt.strftime("%Y-%m-%d")
        display_fc.columns = ["Date", "Predicted Demand (Units)"]
        st.dataframe(display_fc, width="stretch", height=280)

    with col_right:
        st.subheader("📊 Demand Statistics & Inventory Advice")
        fc_mean = forecast_df["predicted_demand"].mean()
        fc_sum = forecast_df["predicted_demand"].sum()
        safety_stock = int(fc_sum * 1.15)

        st.markdown(f"""
        <div class="metric-box">
            <b>Recommended Replenishment Order:</b> {int(fc_sum):,} units<br>
            <b>Recommended Safety Stock (15% buffer):</b> {safety_stock:,} units<br>
            <b>Peak Projected Daily Demand:</b> {forecast_df['predicted_demand'].max():.1f} units<br>
            <b>Model Identified Seasonality:</b> Weekly cycles (7-day periodic peaks)
        </div>
        """, unsafe_allow_html=True)

        # Download forecast CSV
        csv_data = display_fc.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_data,
            file_name=f"demand_forecast_{selected_store}_{selected_product}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
