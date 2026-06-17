import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Page Configuration
st.set_page_config(
    page_title="Britannia Stock Forecast Dashboard",
    layout="wide"
)

# Title
st.title("📈 Britannia Stock Price Forecasting Dashboard")

# Load Default Dataset
df = pd.read_csv("BRITANNIA.NS_stock_data.csv")

# Optional CSV Upload
uploaded_file = st.file_uploader(
    "Upload Britannia CSV File (Optional)",
    type=["csv"]
)

# If user uploads another CSV, replace default dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# Dataset Preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Date Processing
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Dataset Information
st.subheader("📋 Dataset Information")
st.write(df.describe())

# Latest Close Price
st.metric(
    label="Latest Close Price",
    value=round(df['Close'].iloc[-1], 2)
)

# Stock Price Trend
st.subheader("📈 Stock Closing Price Trend")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Close'])
ax.set_title("Britannia Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(True)

st.pyplot(fig)

# Rolling Mean and Std
st.subheader("📉 Rolling Mean & Standard Deviation")

rolling_mean = df['Close'].rolling(window=12).mean()
rolling_std = df['Close'].rolling(window=12).std()

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df['Close'], label='Original')
ax.plot(rolling_mean, label='Rolling Mean')
ax.plot(rolling_std, label='Rolling Std')

ax.legend()
ax.grid(True)

st.pyplot(fig)

# ADF Test
st.subheader("🧪 ADF Stationarity Test")

result = adfuller(df['Close'].dropna())

st.write("ADF Statistic:", round(result[0], 4))
st.write("p-value:", round(result[1], 6))

if result[1] < 0.05:
    st.success("✅ Series is Stationary")
else:
    st.error("❌ Series is Non-Stationary")

# ACF Plot
st.subheader("📊 ACF Plot")

fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(df['Close'].dropna(), lags=30, ax=ax)

st.pyplot(fig)

# PACF Plot
st.subheader("📊 PACF Plot")

fig, ax = plt.subplots(figsize=(10, 4))
plot_pacf(df['Close'].dropna(), lags=30, ax=ax)

st.pyplot(fig)

# Forecast Section
st.subheader("🔮 Next 4 Days Forecast (Moving Average)")

close = df['Close']

forecast = []

last_values = close.tail(3).values

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

# Forecast Chart
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

# Footer
st.markdown("---")
st.markdown("### Developed by Sugumar Ranganathan")
