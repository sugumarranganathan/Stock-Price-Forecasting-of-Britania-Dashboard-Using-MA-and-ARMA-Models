import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Britannia Stock Forecast Dashboard",
    layout="wide"
)

st.title("📈 Britannia Stock Price Forecasting Dashboard")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("BRITANNIA.NS_stock_data.csv")

uploaded_file = st.file_uploader(
    "Upload CSV File (Optional)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------

df.columns = df.columns.str.lower()

if "unnamed: 0" in df.columns:
    df.rename(columns={"unnamed: 0": "date"}, inplace=True)

df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Dataset Statistics
# --------------------------------------------------

st.subheader("📋 Dataset Statistics")
st.dataframe(df.describe())

# --------------------------------------------------
# Latest Close Price
# --------------------------------------------------

st.subheader("💰 Latest Close Price")

st.metric(
    label="Latest Close Price",
    value=f"{df['close'].iloc[-1]:.2f}"
)

# --------------------------------------------------
# Stock Price Trend
# --------------------------------------------------

st.subheader("📈 Stock Price Trend")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df.index, df["close"])

ax.set_title("Britannia Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(True)

st.pyplot(fig)

# --------------------------------------------------
# Rolling Mean & Std
# --------------------------------------------------

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

# --------------------------------------------------
# ADF Test
# --------------------------------------------------

st.subheader("🧪 ADF Stationarity Test")

result = adfuller(df["close"].dropna())

st.write("ADF Statistic:", round(result[0], 4))
st.write("p-value:", round(result[1], 6))

if result[1] < 0.05:
    st.success("✅ Series is Stationary")
else:
    st.error("❌ Series is Non-Stationary")

# --------------------------------------------------
# ACF Plot
# --------------------------------------------------

st.subheader("📊 ACF Plot")

fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(df["close"].dropna(), lags=30, ax=ax)

st.pyplot(fig)

# --------------------------------------------------
# PACF Plot
# --------------------------------------------------

st.subheader("📊 PACF Plot")

fig, ax = plt.subplots(figsize=(10, 4))
plot_pacf(df["close"].dropna(), lags=30, ax=ax)

st.pyplot(fig)

# --------------------------------------------------
# Moving Average Forecast
# --------------------------------------------------

st.subheader("🔮 Next 4 Days Forecast")

close_prices = df["close"]

forecast = []

last_values = close_prices.tail(3).values

for i in range(4):
    next_value = np.mean(last_values)

    forecast.append(next_value)

    last_values = np.append(
        last_values[1:],
        next_value
    )

forecast_df = pd.DataFrame({
    "Day": [1, 2, 3, 4],
    "Forecast Price": forecast
})

st.dataframe(forecast_df)

# --------------------------------------------------
# Forecast Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    forecast_df["Day"],
    forecast_df["Forecast Price"],
    marker="o"
)

ax.set_title("Next 4 Days Forecast")
ax.set_xlabel("Day")
ax.set_ylabel("Forecast Price")
ax.grid(True)

st.pyplot(fig)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")
st.markdown(
    "### Developed by Sugumar Ranganathan"
)
