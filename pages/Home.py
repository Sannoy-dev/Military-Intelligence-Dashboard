import streamlit as st
import plotly.express as px

from utils.data_loader import load_data
from utils.ui import load_css

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Military Intelligence Dashboard",
    layout="wide"
)

load_css()

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
        <div>
            <h1>Military Intelligence Dashboard</h1>
            <p>
              Analyze the currently active intelligence dataset
                              using interactive analytics, machine learning,
                              forecasting, and reporting tools.
            </p>
        </div>
   
    """,
    unsafe_allow_html=True
)


# =====================================================
# LOAD ACTIVE DATASET
# =====================================================

try:

    with st.spinner(
        "Loading active intelligence dataset..."
    ):

        df = load_data()

except Exception as error:

    st.error(
        f"Unable to load the active dataset: {error}"
    )

    st.stop()

# =====================================================
# DATASET AVAILABILITY
# =====================================================

if df is None or df.empty:

    st.warning(
        """
        ### No Active Dataset

        No processed dataset is currently available.

        Upload a CSV dataset from the main dashboard and
        complete the column-mapping process before using
        the intelligence modules.
        """
    )

    st.info(
        """
        **Workflow**

        1. Upload your CSV dataset.
        2. Map the dataset columns.
        3. Process the dataset.
        4. Train the available machine-learning models.
        5. Return to this dashboard for analysis.
        """
    )

    st.stop()

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def numeric_sum(
    dataframe,
    column
):

    if column not in dataframe.columns:

        return 0

    return int(
        dataframe[column]
        .fillna(0)
        .sum()
    )


def unique_count(
    dataframe,
    column
):

    if column not in dataframe.columns:

        return 0

    return int(
        dataframe[column]
        .dropna()
        .nunique()
    )


def most_common(
    dataframe,
    column
):

    if column not in dataframe.columns:

        return "N/A"

    values = (
        dataframe[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    if values.empty:

        return "N/A"

    return values.value_counts().idxmax()

# =====================================================
# DATASET STATISTICS
# =====================================================

total_incidents = len(df)

fatalities = numeric_sum(
    df,
    "nkill"
)

injuries = numeric_sum(
    df,
    "nwound"
)

countries = unique_count(
    df,
    "country_txt"
)

attack_types = unique_count(
    df,
    "attacktype1_txt"
)

groups = unique_count(
    df,
    "gname"
)

# =====================================================
# DATASET OVERVIEW
# =====================================================

st.markdown(
    "Active Dataset Overview"
)

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Total Incidents",
        f"{total_incidents:,}"
    )

with c2:

    st.metric(
        "Fatalities",
        f"{fatalities:,}"
    )

with c3:

    st.metric(
        "Injuries",
        f"{injuries:,}"
    )


c4, c5, c6 = st.columns(3)

with c4:

    st.metric(
        "Countries",
        countries
    )

with c5:

    st.metric(
        "Attack Types",
        attack_types
    )

with c6:

    st.metric(
        "Organizations",
        groups
    )

# =====================================================
# DATASET STATUS
# =====================================================

st.caption(
    f"Active dataset contains {len(df):,} records "
    f"and {len(df.columns):,} columns."
)

# =====================================================
# AI EXECUTIVE SUMMARY
# =====================================================

st.markdown(
    "Intelligence Summary"
)

top_country = most_common(
    df,
    "country_txt"
)

top_attack = most_common(
    df,
    "attacktype1_txt"
)

top_group = most_common(
    df,
    "gname"
)

top_weapon = most_common(
    df,
    "weaptype1_txt"
)

st.markdown(
    f"""
    <div class="assessment-card">

    <h3>Dataset Assessment</h3>

    <p>
            <strong>Total Recorded Incidents:</strong>
            {total_incidents:,}
        </p>

    <p>
            <strong>Countries Represented:</strong>
            {countries:,}
        </p>

    <p>
            <strong>Total Fatalities:</strong>
            {fatalities:,}
        </p>

    <p>
            <strong>Total Injuries:</strong>
            {injuries:,}
    </p>

    <p>
            <strong>Highest Activity Country:</strong>
            {top_country}
        </p>

    <p>
            <strong>Most Common Attack Type:</strong>
            {top_attack}
        </p>

    <p>
            <strong>Most Frequently Used Weapon:</strong>
            {top_weapon}
        </p>

    <p>
            <strong>Most Active Organization:</strong>
            {top_group}
        </p>

    <p>
            The dashboard generates analytical summaries from
            the currently active uploaded dataset. Results depend
            on the fields available in the processed dataset.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# ATTACK TREND
# =====================================================

if "iyear" in df.columns:

    st.markdown(
        "Incident Trend Over Time"
    )

    try:

        trend_df = df.copy()

        trend_df["iyear"] = (
            trend_df["iyear"]
            .astype(str)
            .str.extract(
                r"(\d{4})"
            )[0]
        )

        trend_df["iyear"] = (
            trend_df["iyear"]
            .astype(float)
        )

        trend_df = (
            trend_df
            .dropna(
                subset=["iyear"]
            )
            .groupby("iyear")
            .size()
            .reset_index(
                name="Incidents"
            )
        )

        if not trend_df.empty:

            trend_df["iyear"] = (
                trend_df["iyear"]
                .astype(int)
            )

            fig = px.line(
                trend_df,
                x="iyear",
                y="Incidents",
                markers=True,
                title="Historical Incident Activity"
            )

            fig.update_layout(

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                font_color="white",

                height=500,

                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

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

        else:

            st.info(
                "No valid year information is available "
                "for trend analysis."
            )

    except Exception as error:

        st.warning(
            f"Unable to generate the historical trend: {error}"
        )

else:

    st.info(
        """
        Historical trend analysis is unavailable because
        the active dataset does not contain an `iyear` field.
        """
    )

# =====================================================
# TOP COUNTRIES
# =====================================================

if "country_txt" in df.columns:

    st.markdown(
        "Top 10 Countries by Incidents"
    )

    country_chart = (
        df["country_txt"]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
        .head(10)
        .reset_index()
    )

    country_chart.columns = [
        "Country",
        "Incidents"
    ]

    if not country_chart.empty:

        fig = px.bar(
            country_chart,
            x="Country",
            y="Incidents",
            color="Incidents",
            color_continuous_scale="Reds"
        )

        fig.update_layout(

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            font_color="white",

            height=500,

            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            )
        )

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

# =====================================================
# DATASET SCHEMA
# =====================================================

st.markdown(
    "Active Dataset Schema"
)

with st.expander(
    "View available columns"
):

    columns = list(
        df.columns
    )

    st.write(
        columns
    )

# =====================================================
# QUICK NAVIGATION
# =====================================================

st.markdown(
    "Intelligence Modules"
)

st.info(
    """
    ### Available Modules

    🗺 **Global Threat Map**

    Explore incidents geographically when valid
    latitude and longitude fields are available.

    **Country Analysis**

    Analyze activity and trends for individual
    countries when country information is available.

    **Attack Prediction**

    Predict attack types using the locally trained
    machine-learning model.

    **Threat Level Prediction**

    Estimate threat severity using the locally
    trained threat-level model.

    **Forecasting**

    Analyze historical patterns and generate forecasts.

    **AI Intelligence Report**

    Generate analytical summaries from the active dataset.

    **Data Explorer**

    Search, filter, analyze, and export the active dataset.
    """
)

# =====================================================
# FOOTER
# =====================================================

st.caption(
    "AI Military Intelligence Dashboard • "
    "Powered by the active user-uploaded dataset"
)