import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# --- App Config ---
st.set_page_config("📊 NetSage KPI Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Load & Prepare Data ---
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "Data.xlsx")
    df = pd.read_excel(data_path, sheet_name="Data", skiprows=1)
    df = df.reset_index().rename(columns={"index": "Time"})
    df["Time"] = pd.to_datetime(df["Time"])
    return df

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "Data.xlsx")
    
    # Load the Excel data, skipping header row
    df = pd.read_excel(data_path, sheet_name="Data", skiprows=1)

    # Rename first column as 'Time' explicitly
    if df.columns[0] != "Time":
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    
    # Ensure 'Time' is datetime type and drop invalid ones
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])
    df = df.sort_values("Time")

    return df


# --- KPI list ---
kpi_cols = [
    "AVE4GLTEDLTHRPUTALLKBITSSECFL1",
    "DLTRAFFICVOLUMEGB",
    "ULTRAFFICVOLUMEGB",
    "RRCSETUPSUCCESSRATE",
    "LTE_CALL_SETUP_SUCCESS_RATE",
    "CALLDROPRATE",
    "AVERAGECQI",
    "DLPRBUTILISATION"
]
# Drop null timestamps
df = df.dropna(subset=["Time"])

# Ensure Time is datetime and sorted
df["Time"] = pd.to_datetime(df["Time"])
df = df.sort_values("Time")

# Handle edge case: not enough unique dates
unique_dates = df["Time"].dt.date.unique()
if len(unique_dates) < 2:
    st.error("📅 Not enough date data to generate a slider. Please check your Excel file.")
    st.stop()

# Safe default range
default_start = unique_dates[0]
default_end = unique_dates[-1]

date_range = st.sidebar.slider(
    "📅 Select Date Range",
    min_value=default_start,
    max_value=default_end,
    value=(default_start, default_end),
    format="YYYY-MM-DD"
)

if auto_refresh:
    st.experimental_rerun()

# --- Filter Data ---
filtered_df = df[(df["Time"].dt.date >= date_range[0]) & (df["Time"].dt.date <= date_range[1])]

# --- Title ---
st.title("📡 NetSage Advanced KPI Dashboard")

# --- KPI Metric Cards ---
latest_row = filtered_df.iloc[-1]
previous_row = filtered_df.iloc[-2]

st.markdown("### 📌 Latest KPI Snapshot")
cols = st.columns(len(kpi_cols))

for idx, kpi in enumerate(kpi_cols):
    change = latest_row[kpi] - previous_row[kpi]
    delta_color = "normal" if change >= 0 else "inverse"
    cols[idx].metric(
        label=kpi.replace("_", " "),
        value=f"{latest_row[kpi]:,.2f}",
        delta=f"{change:+.2f}",
        delta_color=delta_color
    )

# --- Line Chart for Selected KPI ---
st.markdown(f"### 📈 {selected_kpi.replace('_', ' ')} Over Time")
fig1 = px.line(filtered_df, x="Time", y=selected_kpi, title=f"{selected_kpi.replace('_', ' ')} Over Time", markers=True, line_shape='spline', template='plotly_white', color_discrete_sequence=["#636EFA"])
fig1.update_layout(xaxis_title="Time", yaxis_title=selected_kpi, hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)

# --- Multi-KPI Chart ---
if selected_multi:
    st.markdown("### 📊 Multi-KPI Comparison")
    fig2 = go.Figure()
    for kpi in selected_multi:
        fig2.add_trace(go.Scatter(x=filtered_df["Time"], y=filtered_df[kpi], mode='lines+markers', name=kpi, line_shape='spline'))
    fig2.update_layout(title="Multi-KPI Trends", xaxis_title="Time", yaxis_title="Value", template='plotly_dark', hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

# --- Correlation Heatmap ---
st.markdown("### 🧠 Correlation Heatmap")
if len(selected_multi) >= 2:
    corr_df = filtered_df[selected_multi].dropna()
    corr_matrix = corr_df.corr()
    fig3 = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto", title="Correlation Between Selected KPIs")
    fig3.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Select at least two KPIs to view correlation heatmap.")

# --- Footer ---
st.markdown("---")
st.markdown("<center>📍 Built with ❤️ by NetSage • Last Updated: " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "</center>", unsafe_allow_html=True)

