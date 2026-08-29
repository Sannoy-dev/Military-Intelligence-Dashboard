import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.ui import load_css


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Country Intelligence Analysis",
    layout="wide"
)

load_css()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>Country Intelligence Analysis</h1>
        <p>
            Analyze historical terrorism activity, operational patterns,
            organizations, weapons, casualties and incident locations
            for a selected country.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD ACTIVE DATASET
# =========================================================

with st.spinner("Loading intelligence data..."):
    df = load_data()

if df is None:
    st.error("No active dataset found.")
    st.info("Please upload and process a dataset first.")
    st.stop()
# =========================================================
# REQUIRED / OPTIONAL COLUMNS
# =========================================================

required_columns = [
    "country_txt"
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:

    st.error(
        "The active dataset cannot be analyzed because the following "
        f"required column(s) are missing: {', '.join(missing_required)}"
    )

    st.info(
        "Use the dataset mapping section to map your uploaded dataset "
        "to the required standard columns."
    )

    st.stop()


# =========================================================
# NORMALIZE NUMERIC COLUMNS
# =========================================================

for column in ["nkill", "nwound", "iyear", "latitude", "longitude"]:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ⚙ Country Filters")

countries = sorted(
    df["country_txt"]
    .dropna()
    .astype(str)
    .unique()
)

if not countries:
    st.warning("No countries are available in the active dataset.")
    st.stop()


country = st.sidebar.selectbox(
    "Country",
    countries
)


# =========================================================
# FILTER COUNTRY
# =========================================================

with st.spinner(f"Analyzing {country}..."):

    country_df = df[
        df["country_txt"].astype(str) == country
    ].copy()


if country_df.empty:

    st.warning(
        f"No incidents were found for {country}."
    )

    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_sum(dataframe, column):

    if column not in dataframe.columns:
        return 0

    return int(
        pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


def safe_unique(dataframe, column):

    if column not in dataframe.columns:
        return 0

    return dataframe[column].dropna().nunique()


def safe_mode(dataframe, column):

    if column not in dataframe.columns:
        return "Not available"

    values = dataframe[column].dropna()

    if values.empty:
        return "Not available"

    return str(values.mode().iloc[0])


# =========================================================
# INTELLIGENCE SUMMARY
# =========================================================

total_incidents = len(country_df)

fatalities = safe_sum(
    country_df,
    "nkill"
)

injuries = safe_sum(
    country_df,
    "nwound"
)

organizations = safe_unique(
    country_df,
    "gname"
)

top_group = safe_mode(
    country_df,
    "gname"
)

top_attack = safe_mode(
    country_df,
    "attacktype1_txt"
)

top_weapon = safe_mode(
    country_df,
    "weaptype1_txt"
)

primary_target = safe_mode(
    country_df,
    "targtype1_txt"
)


# =========================================================
# COUNTRY HEADER
# =========================================================

st.markdown(
    f"""
    <div class="report-card">

    <h2>{country}</h2>

    <p>
    Historical intelligence profile generated from the
                currently active dataset.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# KPI DASHBOARD
# =========================================================

st.markdown("Key Intelligence Indicators")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Incidents",
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

with c4:
    st.metric(
        "Organizations",
        f"{organizations:,}"
    )


# =========================================================
# AI / ANALYTICAL SUMMARY
# =========================================================

st.markdown("Intelligence Summary")

st.info(
    f"""
**Country Assessment**

- **Recorded incidents:** {total_incidents:,}
- **Total fatalities:** {fatalities:,}
- **Total injuries:** {injuries:,}
- **Organizations identified:** {organizations:,}
- **Most active organization:** {top_group}
- **Most common attack type:** {top_attack}
- **Most common weapon type:** {top_weapon}
- **Most frequently targeted category:** {primary_target}

These findings summarize historical patterns in the active dataset.
They should be interpreted as analytical decision-support information
rather than definitive operational intelligence.
"""
)


# =========================================================
# ATTACKS OVER TIME
# =========================================================

if "iyear" in country_df.columns:

    st.markdown("Historical Attack Activity")

    with st.spinner("Analyzing historical attack trends..."):

        yearly = (
            country_df
            .dropna(subset=["iyear"])
            .groupby("iyear")
            .size()
            .reset_index(name="Attacks")
            .sort_values("iyear")
        )

    if not yearly.empty:

        fig = px.line(
            yearly,
            x="iyear",
            y="Attacks",
            markers=True,
            title=f"Attack Activity Over Time — {country}"
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

    else:

        st.warning(
            "Year information is not available for this dataset."
        )


# =========================================================
# ATTACK TYPE + TARGET TYPE
# =========================================================

left, right = st.columns(2)


# ---------------------------------------------------------
# Attack Types
# ---------------------------------------------------------

with left:

    st.markdown("Attack Type Distribution")

    if "attacktype1_txt" in country_df.columns:

        with st.spinner("Analyzing attack types..."):

            attack_df = (
                country_df["attacktype1_txt"]
                .dropna()
                .value_counts()
                .reset_index()
            )

            attack_df.columns = [
                "Attack Type",
                "Count"
            ]

        if not attack_df.empty:

            fig = px.pie(
                attack_df,
                names="Attack Type",
                values="Count",
                hole=0.45
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=430,
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10
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
            st.warning("Attack type information unavailable.")

    else:
        st.warning(
            "Attack type column is not available."
        )


# ---------------------------------------------------------
# Target Types
# ---------------------------------------------------------

with right:

    st.markdown("Target Type Distribution")

    if "targtype1_txt" in country_df.columns:

        with st.spinner("Analyzing target patterns..."):

            target_df = (
                country_df["targtype1_txt"]
                .dropna()
                .value_counts()
                .head(10)
                .reset_index()
            )

            target_df.columns = [
                "Target Type",
                "Count"
            ]

        if not target_df.empty:

            fig = px.bar(
                target_df,
                x="Count",
                y="Target Type",
                orientation="h",
                color="Count",
                color_continuous_scale="Blues"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=430,
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
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
            st.warning("Target information unavailable.")

    else:
        st.warning(
            "Target type column is not available."
        )


# =========================================================
# ORGANIZATIONS + WEAPONS
# =========================================================

st.markdown("Operational Patterns")

left, right = st.columns(2)


# ---------------------------------------------------------
# Organizations
# ---------------------------------------------------------

with left:

    st.markdown("Most Active Organizations")

    if "gname" in country_df.columns:

        with st.spinner("Analyzing organizations..."):

            groups_df = (
                country_df["gname"]
                .dropna()
                .value_counts()
                .head(10)
                .reset_index()
            )

            groups_df.columns = [
                "Organization",
                "Attacks"
            ]

        if not groups_df.empty:

            fig = px.bar(
                groups_df,
                x="Attacks",
                y="Organization",
                orientation="h",
                color="Attacks",
                color_continuous_scale="Reds"
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

        else:
            st.warning(
                "Organization information unavailable."
            )

    else:
        st.warning(
            "Organization column is not available."
        )


# ---------------------------------------------------------
# Weapons
# ---------------------------------------------------------

with right:

    st.markdown("Weapon Distribution")

    if "weaptype1_txt" in country_df.columns:

        with st.spinner("Analyzing weapon patterns..."):

            weapon_df = (
                country_df["weaptype1_txt"]
                .dropna()
                .value_counts()
                .head(10)
                .reset_index()
            )

            weapon_df.columns = [
                "Weapon Type",
                "Count"
            ]

        if not weapon_df.empty:

            fig = px.bar(
                weapon_df,
                x="Weapon Type",
                y="Count",
                color="Count",
                color_continuous_scale="Blues"
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

        else:
            st.warning(
                "Weapon information unavailable."
            )

    else:
        st.warning(
            "Weapon type column is not available."
        )


# =========================================================
# INCIDENT MAP
# =========================================================

st.markdown("Incident Location Analysis")

if (
    "latitude" in country_df.columns
    and "longitude" in country_df.columns
):

    map_df = country_df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    if not map_df.empty:

        original_map_count = len(map_df)

        # Prevent huge Plotly payloads
        MAX_MAP_POINTS = 5000

        if len(map_df) > MAX_MAP_POINTS:

            map_df = map_df.sample(
                MAX_MAP_POINTS,
                random_state=42
            )

            st.caption(
                f"Showing {MAX_MAP_POINTS:,} sampled incidents "
                f"out of {original_map_count:,} incidents with "
                f"valid geographic coordinates."
            )

        with st.spinner("Rendering geographic intelligence..."):

            hover_data = {}

            optional_hover_columns = [
                "iyear",
                "city",
                "attacktype1_txt",
                "gname",
                "nkill",
                "nwound"
            ]

            for column in optional_hover_columns:

                if column in map_df.columns:
                    hover_data[column] = True

            fig = px.scatter_geo(
                map_df,
                lat="latitude",
                lon="longitude",
                hover_name=(
                    "city"
                    if "city" in map_df.columns
                    else None
                ),
                hover_data=hover_data,
                color=(
                    "attacktype1_txt"
                    if "attacktype1_txt" in map_df.columns
                    else None
                ),
                projection="natural earth",
                height=600,
                title=f"Incident Locations — {country}"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(
                    l=0,
                    r=0,
                    t=50,
                    b=0
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

        st.warning(
            "No valid latitude/longitude information is available "
            "for this country."
        )

else:

    st.warning(
        "Geographic coordinates are not available in the active dataset."
    )


# =========================================================
# INCIDENT DATABASE
# =========================================================

st.markdown("Incident Database")

display_columns = [
    "iyear",
    "city",
    "attacktype1_txt",
    "targtype1_txt",
    "weaptype1_txt",
    "gname",
    "nkill",
    "nwound"
]

available_columns = [
    column
    for column in display_columns
    if column in country_df.columns
]

if available_columns:

    st.dataframe(
        country_df[available_columns],
        use_container_width=True,
        hide_index=True,
        height=500
    )

else:

    st.warning(
        "No standard incident-detail columns are available."
    )


# =========================================================
# EXPORT
# =========================================================

st.markdown("Export Country Intelligence")

csv_data = country_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Country Intelligence Data",
    data=csv_data,
    file_name=f"{country}_intelligence.csv",
    mime="text/csv",
    use_container_width=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="bottom-banner">
        Country analysis is based on the currently active dataset.
        Use the results as analytical decision-support information.
    </div>
    """,
    unsafe_allow_html=True
)