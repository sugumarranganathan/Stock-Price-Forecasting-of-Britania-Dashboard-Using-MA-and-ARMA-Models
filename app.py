import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.set_page_config(
    page_title="Britannia Stock Forecast Dashboard",
    layout="wide"
)

st.title("📈 Britannia Stock Price Forecasting Dashboard")

# Upload CSV
df = pd.read_csv("BRITANNIA.NS_stock_data.csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # Date Column
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    st.subheader("Dataset Information")

    st.write(df.describe())

    # Closing Price Trend
    st.subheader("Stock Closing Price Trend")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['Close'])
    ax.set_title("Britannia Closing Price")
    st.pyplot(fig)

    # Rolling Mean
    st.subheader("Rolling Mean & Standard Deviation")

    rolling_mean = df['Close'].rolling(12).mean()
    rolling_std = df['Close'].rolling(12).std()

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(df['Close'], label='Original')
    ax.plot(rolling_mean, label='Rolling Mean')
    ax.plot(rolling_std, label='Rolling Std')

    ax.legend()

    st.pyplot(fig)

    # ADF Test
    st.subheader("ADF Stationarity Test")

    result = adfuller(df['Close'].dropna())

    st.write("ADF Statistic:", result[0])
    st.write("p-value:", result[1])

    if result[1] < 0.05:
        st.success("Series is Stationary")
    else:
        st.error("Series is Non-Stationary")

    # ACF Plot
    st.subheader("ACF Plot")

    fig = plt.figure(figsize=(10,4))
    plot_acf(df['Close'].dropna(), lags=30)

    st.pyplot(fig)

    # PACF Plot
    st.subheader("PACF Plot")

    fig = plt.figure(figsize=(10,4))
    plot_pacf(df['Close'].dropna(), lags=30)

    st.pyplot(fig)
