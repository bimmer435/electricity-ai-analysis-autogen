import pandas as pd
import streamlit as st
import altair as alt

def show_seasonality(df):
    st.header("Seasonality Analysis")

    # Extract month names
    df["month"] = df["date"].dt.strftime("%B")

    # Compute average usage and cost per month
    monthly_usage = df.groupby("month")["usage_kwh"].mean()
    monthly_cost = df.groupby("month")["daily_cost"].mean()

    # Reorder months Jan → Dec
    months_order = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
    monthly_usage = monthly_usage.reindex(months_order)
    monthly_cost = monthly_cost.reindex(months_order)

    # Combined DataFrame
    monthly_usage_cost = pd.DataFrame({
        "Usage (kWh)": monthly_usage,
        "Cost ($)": monthly_cost
    }, index=months_order)

    # Toggle for static vs interactive
    view_option = st.radio(
        "Choose seasonality chart type:",
        ["Static (Streamlit)", "Interactive (Altair)"]
    )

    if view_option == "Static (Streamlit)":
        st.subheader("Grouped Seasonality: Usage vs. Cost (Static)")
        st.bar_chart(monthly_usage_cost)

    else:
        st.subheader("Grouped Seasonality: Usage vs. Cost (Interactive)")
        monthly_usage_cost_reset = monthly_usage_cost.reset_index().melt(
            id_vars="index", var_name="Metric", value_name="Value"
        )
        monthly_usage_cost_reset.rename(columns={"index": "Month"}, inplace=True)

        chart = alt.Chart(monthly_usage_cost_reset).mark_bar().encode(
            x=alt.X("Month:N", sort=months_order, title="Month"),
            y=alt.Y("Value:Q", title="Average"),
            color="Metric:N",
            tooltip=["Month", "Metric", "Value"]
        ).properties(width=700)

        st.altair_chart(chart, use_container_width=True)

    # CSV Download button
    st.subheader("Download Seasonality Data")
    seasonality_csv = monthly_usage_cost.reset_index()
    seasonality_csv.rename(columns={"index": "Month"}, inplace=True)
    st.download_button(
        label="📥 Download Seasonality Data as CSV",
        data=seasonality_csv.to_csv(index=False).encode("utf-8"),
        file_name="seasonality_usage_cost.csv",
        mime="text/csv"
    )
