import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Britannia Stock Forecast Dashboard",
    layout="wide"
)

st.title("📈 Britannia Stock Price Forecasting Dashboard")
st.markdown("### Stock Price Forecasting Using ARMA(1,1) Model")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("BRITANNIA.NS_stock_data.csv")

uploaded_file = st.file_uploader(
    "Upload CSV File (Optional)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# ==========================================
# DATA CLEANING
# ==========================================

df.columns = df.columns.str.lower()

if "unnamed: 0" in df.columns:
    df.rename(columns={"unnamed: 0": "date"}, inplace=True)

df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# ==========================================
# DATASET PREVIEW
# ==========================================

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ==========================================
# DATASET STATISTICS
# ==========================================

st.subheader("📋 Dataset Statistics")
st.dataframe(df.describe())

# ==========================================
# LATEST CLOSE PRICE
# ==========================================

st.subheader("💰 Latest Close Price")

st.metric(
    label="Latest Close Price",
    value=f"{df['close'].iloc[-1]:.2f}"
)

# ==========================================
# STOCK PRICE TREND
# ==========================================

st.subheader("📈 Stock Price Trend")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df.index,
    df["close"]
)

ax.set_title("Britannia Closing Price Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price")
ax.grid(True)

st.pyplot(fig)

# ==========================================
# ROLLING MEAN & STANDARD DEVIATION
# ==========================================

st.subheader("📉 Rolling Mean & Standard Deviation")

rolling_mean = df["close"].rolling(window=12).mean()
rolling_std = df["close"].rolling(window=12).std()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df["close"], label="Original")
ax.plot(rolling_mean, label="Rolling Mean")
ax.plot(rolling_std, label="Rolling Std")

ax.legend()
ax.grid(True)

st.pyplot(fig)

# ==========================================
# ADF TEST
# ==========================================

st.subheader("🧪 ADF Stationarity Test")

result = adfuller(df["close"].dropna())

st.write("ADF Statistic:", round(result[0], 4))
st.write("p-value:", round(result[1], 6))

if result[1] < 0.05:
    st.success("✅ Series is Stationary")
else:
    st.error("❌ Series is Non-Stationary")

# ==========================================
# ACF PLOT
# ==========================================

st.subheader("📊 ACF Plot")

fig, ax = plt.subplots(figsize=(10, 4))

plot_acf(
    df["close"].dropna(),
    lags=30,
    ax=ax
)

st.pyplot(fig)

# ==========================================
# PACF PLOT
# ==========================================

st.subheader("📊 PACF Plot")

fig, ax = plt.subplots(figsize=(10, 4))

plot_pacf(
    df["close"].dropna(),
    lags=30,
    ax=ax
)

st.pyplot(fig)

# ==========================================
# ARMA(1,1) FORECAST
# ==========================================

st.subheader("🔮 Britannia Next 4-Day Forecast (ARMA(1,1))")

try:

    arma_model = ARIMA(
        df["close"],
        order=(1, 0, 1)
    )

    arma_fit = arma_model.fit()

    future_forecast = arma_fit.forecast(
        steps=4
    )

    forecast_df = pd.DataFrame({
        "Day": [
            "Day 1",
            "Day 2",
            "Day 3",
            "Day 4"
        ],
        "Forecasted Close Price":
        future_forecast.values
    })

    st.dataframe(forecast_df)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        forecast_df["Day"],
        forecast_df["Forecasted Close Price"],
        marker="o",
        linewidth=2
    )

    ax.set_title(
        "Britannia Next 4-Day Forecast (ARMA(1,1))"
    )

    ax.set_xlabel("Future Days")
    ax.set_ylabel("Forecasted Close Price")
    ax.grid(True)

    st.pyplot(fig)

except Exception as e:

    st.error(f"Forecast Error: {e}")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("### Developed by Sugumar Ranganathan")
