import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.linear_model import LinearRegression

st.title("Electricity Usage vs. Prices Dashboard")

# Ensure merged.csv exists
if not os.path.exists("data/merged.csv"):
    import preprocessing  # runs preprocessing.py

# Load data
df = pd.read_csv("data/merged.csv", parse_dates=["date"])

# Train models if not present
def train_and_save_models(df):
    X = df["date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    usage_y = df["usage_kwh"].values
    price_y = df["price_per_kwh"].values
    cost_y = df["daily_cost"].values

    usage_model = LinearRegression().fit(X, usage_y)
    price_model = LinearRegression().fit(X, price_y)
    cost_model = LinearRegression().fit(X, cost_y)

    os.makedirs("models", exist_ok=True)
    with open("models/usage_model.pkl", "wb") as f:
        pickle.dump(usage_model, f)
    with open("models/price_model.pkl", "wb") as f:
        pickle.dump(price_model, f)
    with open("models/cost_model.pkl", "wb") as f:
        pickle.dump(cost_model, f)

if not (os.path.exists("models/usage_model.pkl") and os.path.exists("models/price_model.pkl") and os.path.exists("models/cost_model.pkl")):
    train_and_save_models(df)

# Charts for historical data
st.subheader("Electricity Usage (kWh)")
st.line_chart(df.set_index("date")["usage_kwh"])

st.subheader("Electricity Price ($/kWh)")
st.line_chart(df.set_index("date")["price_per_kwh"])

st.subheader("Daily Cost ($)")
st.line_chart(df.set_index("date")["daily_cost"])

# Forecast section
st.header("AI Forecasts (Next 90 Days)")

# Load models
with open("models/usage_model.pkl", "rb") as f:
    usage_model = pickle.load(f)
with open("models/price_model.pkl", "rb") as f:
    price_model = pickle.load(f)
with open("models/cost_model.pkl", "rb") as f:
    cost_model = pickle.load(f)

# Generate future dates
future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=90, freq="D")
X_future = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

forecast_usage = usage_model.predict(X_future)
forecast_price = price_model.predict(X_future)
forecast_cost = cost_model.predict(X_future)

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast_usage": forecast_usage,
    "forecast_price": forecast_price,
    "forecast_cost": forecast_cost
})

st.subheader("Forecasted Electricity Usage (kWh)")
st.line_chart(forecast_df.set_index("date")["forecast_usage"])

st.subheader("Forecasted Electricity Price ($/kWh)")
st.line_chart(forecast_df.set_index("date")["forecast_price"])

st.subheader("Forecasted Daily Cost ($)")
st.line_chart(forecast_df.set_index("date")["forecast_cost"])

st.subheader("Data Preview")
st.write(df.tail())
