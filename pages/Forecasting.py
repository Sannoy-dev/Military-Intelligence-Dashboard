import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

from utils.data_loader import load_data
from utils.ui import load_css


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Forecasting",
    layout="wide"
)

load_css()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="page-header">
        <div>
            <h1>Terrorism Forecasting</h1>
            <p>
                Analyze historical attack patterns and generate
                short-term activity projections using machine learning.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "iyear",
    "country_txt"
]


# ============================================================
# LOAD ACTIVE DATASET
# ============================================================

with st.spinner("Preparing forecasting data..."):

    try:
        df = load_data()

    except Exception as e:

        st.error(
            "Unable to load the active dataset."
        )

        st.exception(e)

        st.stop()

if df is None:
    st.error("No active dataset found.")
    st.info("Please upload and process a dataset first.")
    st.stop()

# ============================================================
# DATA VALIDATION
# ============================================================

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The active dataset cannot be used for forecasting."
    )

    st.markdown(
        f"""
        **Missing required columns:**

        `{", ".join(missing_columns)}`

        Please map the corresponding columns in the
        Custom Dataset Mapping section.
        """
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

with st.spinner("Preparing historical records..."):

    forecast_df = df[
        ["iyear", "country_txt"]
    ].copy()

    forecast_df["iyear"] = pd.to_numeric(
        forecast_df["iyear"],
        errors="coerce"
    )

    forecast_df["country_txt"] = (
        forecast_df["country_txt"]
        .astype(str)
        .str.strip()
    )

    forecast_df = forecast_df.dropna(
        subset=["iyear", "country_txt"]
    )

    forecast_df["iyear"] = (
        forecast_df["iyear"]
        .astype(int)
    )


# ============================================================
# CHECK DATA
# ============================================================

if forecast_df.empty:

    st.warning(
        "No valid historical records are available for forecasting."
    )

    st.stop()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.markdown(
    "###Forecast Settings"
)

countries = sorted(
    forecast_df["country_txt"]
    .dropna()
    .unique()
    .tolist()
)

if not countries:

    st.error(
        "No countries are available in the active dataset."
    )

    st.stop()


country = st.sidebar.selectbox(
    "Country",
    countries,
    index=0,
    label_visibility="collapsed"
)


forecast_years = st.sidebar.slider(
    "Forecast Period",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)


# ============================================================
# COUNTRY DATA
# ============================================================

with st.spinner(
    f"Analyzing historical activity for {country}..."
):

    country_df = forecast_df[
        forecast_df["country_txt"] == country
    ].copy()

    yearly = (
        country_df
        .groupby("iyear")
        .size()
        .reset_index(name="Attacks")
        .sort_values("iyear")
    )


# ============================================================
# DATA AVAILABILITY CHECK
# ============================================================

if yearly.empty:

    st.warning(
        f"No historical attack records were found for {country}."
    )

    st.stop()


if len(yearly) < 5:

    st.warning(
        f"""
        Only **{len(yearly)} historical years** are available
        for **{country}**.

        At least 5 years of historical data are recommended
        for this forecasting module.
        """
    )

    st.dataframe(
        yearly,
        use_container_width=True,
        hide_index=True
    )

    st.stop()


# ============================================================
# HISTORICAL STATISTICS
# ============================================================

historical_years = len(yearly)

first_year = int(
    yearly["iyear"].min()
)

last_year = int(
    yearly["iyear"].max()
)

total_attacks = int(
    yearly["Attacks"].sum()
)

average_attacks = float(
    yearly["Attacks"].mean()
)

maximum_attacks = int(
    yearly["Attacks"].max()
)

minimum_attacks = int(
    yearly["Attacks"].min()
)

current_attacks = int(
    yearly.iloc[-1]["Attacks"]
)


# ============================================================
# MODEL TRAINING
# ============================================================

with st.spinner(
    "Training local forecasting model..."
):

    X = yearly[
        ["iyear"]
    ]

    y = yearly[
        "Attacks"
    ]

    model = LinearRegression()

    model.fit(
        X,
        y
    )


# ============================================================
# GENERATE FORECAST
# ============================================================

with st.spinner(
    "Generating future attack projections..."
):

    future_years = np.arange(
        last_year + 1,
        last_year + forecast_years + 1
    )

    future_df = pd.DataFrame(
        {
            "iyear": future_years
        }
    )

    predictions = model.predict(
        future_df
    )

    # Attack counts cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    forecast = pd.DataFrame(
        {
            "Year": future_years,
            "Forecasted Attacks":
                np.round(predictions).astype(int)
        }
    )


# ============================================================
# FORECAST STATISTICS
# ============================================================

forecast_last = int(
    forecast.iloc[-1]["Forecasted Attacks"]
)

growth = (
    (forecast_last - current_attacks)
    / max(current_attacks, 1)
) * 100


# ============================================================
# TREND CLASSIFICATION
# ============================================================

if growth > 15:

    trend = "Increasing"
    trend_icon = "🔴"

elif growth >= 0:

    trend = "Stable"
    trend_icon = "🟡"

else:

    trend = "Decreasing"
    trend_icon = "🟢"


# ============================================================
# AI FORECAST SUMMARY
# ============================================================

st.markdown(
    "Forecast Summary"
)

st.markdown(
    f"""
    <div class="report-card">

    <h3>{country} — Forecast Assessment</h3>

    <p>
    Historical records from <b>{first_year}</b> to
    <b>{last_year}</b> were analyzed to estimate future
    attack activity.
    </p>

    <div class="assessment-grid">

    <div>
        <strong>Historical Years</strong><br>
        {historical_years}
    </div>

    <div>
        <strong>Average Annual Attacks</strong><br>
        {average_attacks:.1f}
    </div>

    <div>
        <strong>Current Activity</strong><br>
        {current_attacks:,}
    </div>

    <div>
        <strong>Forecast Activity</strong><br>
        {forecast_last:,}
    </div>

    <div>
        <strong>Projected Change</strong><br>
        {growth:.2f}%
    </div>

    <div>
        <strong>Trend</strong><br>
        {trend_icon} {trend}
    </div>

    </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI DASHBOARD
# ============================================================

st.markdown(
    "Forecast Indicators"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Historical Attacks",
        f"{total_attacks:,}"
    )

with c2:

    st.metric(
        "Current Year",
        f"{current_attacks:,}"
    )

with c3:

    st.metric(
        f"Forecast ({forecast_years}Y)",
        f"{forecast_last:,}"
    )

with c4:

    st.metric(
        "Projected Change",
        f"{growth:.2f}%"
    )


# ============================================================
# FORECAST CHART
# ============================================================

st.markdown(
    "Historical vs Forecast Activity"
)

fig = go.Figure()


# Historical line
fig.add_trace(
    go.Scatter(
        x=yearly["iyear"],
        y=yearly["Attacks"],
        mode="lines+markers",
        name="Historical",
        line=dict(
            width=3
        ),
        marker=dict(
            size=7
        )
    )
)


# Forecast line
fig.add_trace(
    go.Scatter(
        x=forecast["Year"],
        y=forecast["Forecasted Attacks"],
        mode="lines+markers",
        name="Forecast",
        line=dict(
            width=3,
            dash="dash"
        ),
        marker=dict(
            size=8
        )
    )
)


fig.update_layout(
    title=f"Attack Activity Forecast — {country}",
    xaxis_title="Year",
    yaxis_title="Number of Attacks",

    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="white"
    ),

    height=550,

    hovermode="x unified",

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    ),

    legend=dict(
        bgcolor="rgba(0,0,0,0)"
    )
)


with st.spinner(
    "Rendering forecasting visualization..."
):

    st.markdown(
        '<div class="chart-card">',
        unsafe_allow_html=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# FORECAST TABLE
# ============================================================

st.markdown(
    "Forecast Results"
)

display_forecast = forecast.copy()

display_forecast[
    "Forecasted Attacks"
] = display_forecast[
    "Forecasted Attacks"
].map(
    lambda x: f"{x:,}"
)

st.dataframe(
    display_forecast,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

st.markdown(
    "Historical Activity Analysis"
)

h1, h2, h3 = st.columns(3)

with h1:

    st.metric(
        "Highest Annual Activity",
        f"{maximum_attacks:,}"
    )

with h2:

    st.metric(
        "Lowest Annual Activity",
        f"{minimum_attacks:,}"
    )

with h3:

    st.metric(
        "Average Annual Activity",
        f"{average_attacks:.1f}"
    )


# ============================================================
# THREAT OUTLOOK
# ============================================================

st.markdown(
    "Forecast Outlook"
)

if growth < 0:

    st.success(
        f"""
        ### 🟢 Decreasing Trend

        The model projects a decrease in attack activity
        over the selected forecast period.

        Projected change: **{growth:.2f}%**
        """
    )

elif growth < 15:

    st.warning(
        f"""
        ### 🟡 Relatively Stable Trend

        The model projects relatively stable activity
        over the selected forecast period.

        Projected change: **{growth:.2f}%**
        """
    )

else:

    st.error(
        f"""
        ### 🔴 Increasing Trend

        The model projects an increase in attack activity
        over the selected forecast period.

        Projected change: **{growth:.2f}%**
        """
    )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "ℹ️ Forecasting Methodology"
):

    st.markdown(
        """
        **Method used:** Linear Regression

        The model uses historical yearly incident counts
        as the target variable and year as the explanatory
        variable.

        **Process:**

        1. Load the active dataset.
        2. Filter records for the selected country.
        3. Aggregate incidents by year.
        4. Train a Linear Regression model locally.
        5. Generate future yearly projections.
        6. Prevent negative predicted attack counts.
        7. Compare projected activity with the latest
           historical activity.

        Forecasts are analytical projections based on
        historical patterns. They should not be interpreted
        as definitive predictions of future events.
        """
    )


# ============================================================
# EXPORT
# ============================================================

st.markdown(
    "Export Forecast"
)

csv = forecast.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name=f"{country}_forecast.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Forecasting uses the active dataset configured by the application. "
    "Results are intended for analytical and decision-support purposes."
)