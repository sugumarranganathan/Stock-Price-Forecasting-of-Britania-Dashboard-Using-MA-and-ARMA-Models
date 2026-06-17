import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Britannia Stock Forecast Dashboard",
    layout="wide"
)

st.title("📈 Britannia Stock Price Forecasting Dashboard")
st.markdown("### Stock Price Forecasting Using MA and ARMA Models")

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv("BRITANNIA.NS_stock_data.csv")

uploaded_file = st.file_uploader(
    "Upload CSV File (Optional)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# ==================================================
# DATA CLEANING
# ==================================================

df.columns = df.columns.str.lower()

if "unnamed: 0" in df.columns:
    df.rename(
        columns={"unnamed: 0": "date"},
        inplace=True
    )

df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

series = df["close"]

# ==================================================
# DATASET PREVIEW
# ==================================================

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ==================================================
# DATASET STATISTICS
# ==================================================

st.subheader("📋 Dataset Statistics")
st.dataframe(df.describe())

# ==================================================
# LATEST CLOSE PRICE
# ==================================================

st.subheader("💰 Latest Close Price")

st.metric(
    label="Current Close Price",
    value=f"₹ {series.iloc[-1]:.2f}"
)

# ==================================================
# STOCK PRICE TREND
# ==================================================

st.subheader("📈 Stock Price Trend")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(series.index, series.values)

ax.set_title("Britannia Closing Price Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.grid(True)

st.pyplot(fig)

# ==================================================
# ROLLING MEAN & STD
# ==================================================

st.subheader("📉 Rolling Mean & Standard Deviation")

rolling_mean = series.rolling(12).mean()
rolling_std = series.rolling(12).std()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(series, label="Original")
ax.plot(rolling_mean, label="Rolling Mean")
ax.plot(rolling_std, label="Rolling Std")

ax.legend()
ax.grid(True)

st.pyplot(fig)

# ==================================================
# ADF TEST
# ==================================================

st.subheader("🧪 ADF Stationarity Test")

adf_result = adfuller(series.dropna())

st.write("ADF Statistic:", round(adf_result[0], 4))
st.write("p-value:", round(adf_result[1], 6))

if adf_result[1] < 0.05:
    st.success("✅ Series is Stationary")
else:
    st.error("❌ Series is Non-Stationary")

# ==================================================
# DIFFERENCING
# ==================================================

series_diff = series.diff().dropna()

# ==================================================
# ACF PLOT
# ==================================================

st.subheader("📊 ACF Plot")

fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(series_diff, lags=30, ax=ax)

st.pyplot(fig)

# ==================================================
# PACF PLOT
# ==================================================

st.subheader("📊 PACF Plot")

fig, ax = plt.subplots(figsize=(10, 4))
plot_pacf(series_diff, lags=30, ax=ax)

st.pyplot(fig)

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

train_size = int(len(series_diff) * 0.80)

train = series_diff[:train_size]
test = series_diff[train_size:]

# ==================================================
# ARMA(1,1) FORECAST
# ==================================================

st.subheader("🔮 Britannia Next 4-Day Forecast (ARMA(1,1))")

try:

    arma_model = ARIMA(
        train,
        order=(1, 0, 1)
    )

    arma_fit = arma_model.fit()

    future_diff = arma_fit.forecast(
        steps=4
    )

    future_price = []

    current_price = series.iloc[-1]

    for diff in future_diff:

        current_price = current_price + diff

        future_price.append(current_price)

    forecast_df = pd.DataFrame({

        "Day": [
            "Day 1",
            "Day 2",
            "Day 3",
            "Day 4"
        ],

        "Forecasted Close Price (₹)": [
            round(x, 2)
            for x in future_price
        ]

    })

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    # Forecast Graph

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        forecast_df["Day"],
        forecast_df["Forecasted Close Price (₹)"],
        marker="o",
        linewidth=2
    )

    for i, value in enumerate(
        forecast_df["Forecasted Close Price (₹)"]
    ):

        ax.annotate(
            f"{value:.2f}",
            (i, value),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center"
        )

    ax.set_title(
        "Britannia Next 4-Day Forecast (ARMA(1,1))"
    )

    ax.set_xlabel("Future Days")
    ax.set_ylabel("Forecasted Close Price (₹)")
    ax.grid(True)

    plt.tight_layout()

    st.pyplot(fig)

except Exception as e:

    st.error(
        f"Forecast Error: {e}"
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")
st.markdown(
    "### Developed by Sugumar Ranganathan"
)
