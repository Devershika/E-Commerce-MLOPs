import os
import demand_forecast_engine
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from demand_forecast_engine.preprocessing.dataset import DataSetLoader

file_path='data/sales_data.csv'
data=DataSetLoader(file_path)
df=data.read_data()
df=data.cleandata(df)
df=df.sort_values(by='Date',ascending=True)

st.title("Demand Forecasting Dashboard")
st.caption("Forecast SKU-level demand using historical sales & promotions")

st.sidebar.header("User selection for forecasting")

# Single select
category = st.sidebar.selectbox(
    "Select Category",
    options=sorted(df["Category"].dropna().unique())
)

# Multi-select
products = st.sidebar.multiselect(
    "Select Product IDs",
    options=sorted(df["Product ID"].dropna().unique()),
    default=sorted(df["Product ID"].dropna().unique())
)

filtered_df = df[
    (df["Category"] == category) &
    (df["Product ID"].isin(products))
]


if filtered_df.empty:
    st.warning("No data available for selected filters")
    st.stop()

fig = px.line(
    filtered_df,
    x="Date",
    y="Demand",
    color="Product ID",
    title="Demand Trend Over Time",
    markers=True
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Demand",
    legend_title="Product ID",
    template="plotly_white"
)

st.plotly_chart(fig, width='content')
