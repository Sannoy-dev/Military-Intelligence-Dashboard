import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.ui import load_css


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Intelligence Report",
    layout="wide"
)

load_css()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Intelligence Report</h1>
        <p>
            Generate a structured intelligence assessment from
            the currently active dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD ACTIVE DATASET
# ============================================================

df = load_data()

if df is None:
    st.error("No active dataset found.")
    st.info("Please upload and process a dataset first.")
    st.stop()

if df.empty:
    st.error("The active dataset is empty.")
    st.stop()

# ============================================================
# PREPARE NUMERIC COLUMNS
# ============================================================

if "nkill" in df.columns:
    df["nkill"] = pd.to_numeric(
        df["nkill"],
        errors="coerce"
    ).fillna(0)

else:
    df["nkill"] = 0


if "nwound" in df.columns:
    df["nwound"] = pd.to_numeric(
        df["nwound"],
        errors="coerce"
    ).fillna(0)

else:
    df["nwound"] = 0


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("## ⚙ Report Filters")


if "iyear" in df.columns:

    years = sorted(
        pd.to_numeric(
            df["iyear"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    selected_year = st.sidebar.selectbox(
        "Year",
        ["All"] + years
    )

else:

    selected_year = "All"


# ============================================================
# APPLY FILTER
# ============================================================

filtered_df = df.copy()


if selected_year != "All":

    filtered_df = filtered_df[
        pd.to_numeric(
            filtered_df["iyear"],
            errors="coerce"
        ) == selected_year
    ]


if filtered_df.empty:

    st.warning(
        "No records are available for the selected year."
    )

    st.stop()


# ============================================================
# BASIC INTELLIGENCE STATISTICS
# ============================================================

total_incidents = len(filtered_df)

total_fatalities = int(
    filtered_df["nkill"].sum()
)

total_injuries = int(
    filtered_df["nwound"].sum()
)


if "country_txt" in filtered_df.columns:

    countries = filtered_df[
        "country_txt"
    ].nunique()

else:

    countries = 0


if "gname" in filtered_df.columns:

    organizations = filtered_df[
        "gname"
    ].nunique()

else:

    organizations = 0


# ============================================================
# TOP INTELLIGENCE INDICATORS
# ============================================================

def get_top_value(dataframe, column, default="Unknown"):

    if column not in dataframe.columns:
        return default

    values = dataframe[column].dropna()

    if values.empty:
        return default

    return values.value_counts().idxmax()


def get_top_count(dataframe, column):

    if column not in dataframe.columns:
        return 0

    values = dataframe[column].dropna()

    if values.empty:
        return 0

    return int(values.value_counts().iloc[0])


top_country = get_top_value(
    filtered_df,
    "country_txt"
)

top_group = get_top_value(
    filtered_df,
    "gname"
)

top_attack = get_top_value(
    filtered_df,
    "attacktype1_txt"
)

top_weapon = get_top_value(
    filtered_df,
    "weaptype1_txt"
)


top_country_count = get_top_count(
    filtered_df,
    "country_txt"
)

top_group_count = get_top_count(
    filtered_df,
    "gname"
)


# ============================================================
# THREAT INDICATOR
# ============================================================

impact = (
    filtered_df["nkill"]
    + filtered_df["nwound"]
)

average_impact = impact.mean()


if average_impact <= 2:

    threat_level = "LOW"
    threat_symbol = "🟢"

elif average_impact <= 10:

    threat_level = "MEDIUM"
    threat_symbol = "🟡"

else:

    threat_level = "HIGH"
    threat_symbol = "🔴"


# ============================================================
# REPORT HEADER
# ============================================================

st.markdown("Intelligence Overview")


c1, c2, c3, c4, c5 = st.columns(5)


with c1:

    st.metric(
        "Incidents",
        f"{total_incidents:,}"
    )


with c2:

    st.metric(
        "Fatalities",
        f"{total_fatalities:,}"
    )


with c3:

    st.metric(
        "Injuries",
        f"{total_injuries:,}"
    )


with c4:

    st.metric(
        "Countries",
        countries
    )


with c5:

    st.metric(
        "Threat Level",
        f"{threat_symbol} {threat_level}"
    )


# ============================================================
# EXECUTIVE ASSESSMENT
# ============================================================

st.markdown("Executive Intelligence Assessment")
st.markdown(
    f"""
    <div class="report-card">

    <h3>Current Geographic Assessment</h3>

    <p>
    The selected dataset contains
                        <strong>{total_incidents:,}</strong> recorded incidents
                        across <strong>{countries}</strong> countries.
    </p>

    <p>
    The recorded incidents resulted in
                        <strong>{total_fatalities:,}</strong> fatalities and
                        <strong>{total_injuries:,}</strong> injuries.
    </p>

    <p>
    <strong>{top_country}</strong> has the highest number
                        of recorded incidents in the selected dataset.
    </p>

    <p>
    The most frequently recorded attack category is
                        <strong>{top_attack}</strong>, while
                        <strong>{top_weapon}</strong> is the most common
                        weapon category.
    </p>
    <p>
       The most frequently associated organization is
                           <strong>{top_group}</strong>.
    </p>
    <p>
            Based on the average recorded casualties per incident,
                                the calculated dataset-level threat indicator is
                                <strong>{threat_symbol} {threat_level}</strong>.
    </p>
     <small>
                        This assessment is derived from historical dataset
                        statistics and should be treated as analytical
                        decision-support information rather than a real-time
                        operational threat assessment.
                    </small>

    </div>
    """,
    unsafe_allow_html=True
)



# ============================================================
# TOP COUNTRIES
# ============================================================

st.markdown("Geographic Activity")


left, right = st.columns(2)


with left:

    if "country_txt" in filtered_df.columns:

        country_counts = (
            filtered_df["country_txt"]
            .dropna()
            .value_counts()
            .head(10)
            .reset_index()
        )

        country_counts.columns = [
            "Country",
            "Incidents"
        ]

        fig = px.bar(
            country_counts,
            x="Incidents",
            y="Country",
            orientation="h",
            title="Top 10 Countries by Recorded Incidents"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=450,
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


# ============================================================
# ATTACK TYPE DISTRIBUTION
# ============================================================

with right:

    if "attacktype1_txt" in filtered_df.columns:

        attack_counts = (
            filtered_df["attacktype1_txt"]
            .dropna()
            .value_counts()
            .head(10)
            .reset_index()
        )

        attack_counts.columns = [
            "Attack Type",
            "Incidents"
        ]

        fig = px.bar(
            attack_counts,
            x="Incidents",
            y="Attack Type",
            orientation="h",
            title="Most Common Attack Types"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=450,
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


# ============================================================
# ORGANIZATION ANALYSIS
# ============================================================

st.markdown("Organization Activity")


if "gname" in filtered_df.columns:

    group_counts = (
        filtered_df["gname"]
        .dropna()
        .value_counts()
    )

    # Remove generic/unknown organization records
    group_counts = group_counts[
        ~group_counts.index.isin(
            [
                "Unknown",
                "unknown",
                "Unknown Group"
            ]
        )
    ]

    group_counts = (
        group_counts
        .head(10)
        .reset_index()
    )

    group_counts.columns = [
        "Organization",
        "Incidents"
    ]


    if not group_counts.empty:

        fig = px.bar(
            group_counts,
            x="Incidents",
            y="Organization",
            orientation="h",
            title="Most Frequently Recorded Organizations"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=450
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


# ============================================================
# WEAPON ANALYSIS
# ============================================================

st.markdown("Weapon Distribution")


if "weaptype1_txt" in filtered_df.columns:

    weapon_counts = (
        filtered_df["weaptype1_txt"]
        .dropna()
        .value_counts()
        .head(10)
        .reset_index()
    )

    weapon_counts.columns = [
        "Weapon Type",
        "Incidents"
    ]

    fig = px.bar(
        weapon_counts,
        x="Weapon Type",
        y="Incidents",
        title="Most Common Weapon Categories"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=450
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


# ============================================================
# STRATEGIC OBSERVATIONS
# ============================================================

st.markdown("Key Intelligence Observations")


observation_col1, observation_col2 = st.columns(2)


with observation_col1:

    st.markdown(
        f"""
        <div class="chart-card">

        <h3>Primary Activity</h3>

        <p>
        <strong>Country:</strong> {top_country}
        </p>

        <p>
        <strong>Recorded Incidents:</strong>
        {top_country_count:,}
        </p>

        <p>
        <strong>Dominant Attack Type:</strong>
        {top_attack}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with observation_col2:

    st.markdown(
        f"""
        <div class="chart-card">

        <h3>Organization & Weapon Pattern</h3>

        <p>
        <strong>Most Recorded Organization:</strong>
        {top_group}
        </p>

        <p>
        <strong>Associated Records:</strong>
        {top_group_count:,}
        </p>

        <p>
        <strong>Common Weapon Category:</strong>
        {top_weapon}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# REPORT GENERATION
# ============================================================

report = f"""
AI MILITARY INTELLIGENCE REPORT
==============================================

Dataset Scope
----------------------------------------------
Year Selected        : {selected_year}
Records Analyzed     : {total_incidents:,}
Countries Covered    : {countries}
Organizations        : {organizations}

Impact Indicators
----------------------------------------------
Fatalities           : {total_fatalities:,}
Injuries             : {total_injuries:,}
Average Impact       : {average_impact:.2f}

Threat Indicator
----------------------------------------------
Threat Level         : {threat_level}

Key Findings
----------------------------------------------
Highest Activity Country :
{top_country}

Recorded Incidents :
{top_country_count:,}

Most Active Organization :
{top_group}

Organization Records :
{top_group_count:,}

Most Common Attack Type :
{top_attack}

Most Common Weapon Type :
{top_weapon}

Analytical Assessment
----------------------------------------------
The selected dataset contains {total_incidents:,}
recorded incidents across {countries} countries.

The highest concentration of recorded activity is
associated with {top_country}.

The most frequently recorded attack category is
{top_attack}, while {top_weapon} represents the
most common weapon category.

The calculated dataset-level threat indicator is
{threat_level}, based on the average recorded impact
of incidents.

This report is generated from historical dataset
records and is intended for analytical and
decision-support purposes. It should not be treated
as a real-time operational intelligence assessment.

==============================================
"""


# ============================================================
# EXPORT
# ============================================================

st.markdown("Export Intelligence Report")


st.download_button(
    label="Download Intelligence Report",
    data=report,
    file_name="AI_Intelligence_Report.txt",
    mime="text/plain",
    use_container_width=True
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown("Report Information")


info1, info2, info3 = st.columns(3)


with info1:

    st.metric(
        "Analysis Period",
        str(selected_year)
    )


with info2:

    st.metric(
        "Records Analyzed",
        f"{total_incidents:,}"
    )


with info3:

    st.metric(
        "Average Impact",
        f"{average_impact:.2f}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Military Intelligence Dashboard • "
    "Historical Data Analysis • "
    "Decision-Support System"
)

