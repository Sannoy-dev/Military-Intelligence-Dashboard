import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data
from utils.ui import load_css


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Global Intelligence Data Explorer",
    layout="wide",
)

load_css()


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="hero">

    <h1>
            Global Intelligence Data Explorer
        </h1>

    <p>
            Explore terrorism incidents through interactive intelligence
            analytics, geospatial visualization, operational trends,
            terrorist organizations, weapons, casualties and advanced
            filtering.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# LOAD ACTIVE DATASET
# ==========================================================

with st.spinner("Loading intelligence dataset..."):

    df = load_data()

if df is None:

    st.error(
        "No active dataset available."
    )

    st.info(
        "Upload and standardize a dataset first."
    )

    st.stop()


# ==========================================================
# REQUIRED COLUMNS
# ==========================================================

required_columns = [

    "iyear",
    "country_txt",
    "region_txt",
    "city",
    "attacktype1_txt",
    "weaptype1_txt",
    "gname",
    "latitude",
    "longitude",
    "nkill",
    "nwound",

]

missing = [

    c
    for c in required_columns
    if c not in df.columns

]

if missing:

    st.error(
        "Required columns missing:\n\n"
        + ", ".join(missing)
    )

    st.stop()


# ==========================================================
# NORMALIZE NUMERIC COLUMNS
# ==========================================================

numeric_columns = [

    "iyear",
    "imonth",
    "iday",
    "latitude",
    "longitude",
    "nkill",
    "nwound",

]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("# Intelligence Filters")

st.sidebar.markdown("---")


# ==========================================================
# YEAR
# ==========================================================

years = sorted(

    df["iyear"]
    .dropna()
    .astype(int)
    .unique()

)

selected_years = st.sidebar.multiselect(

    "Year",

    years,

)


# ==========================================================
# REGION
# ==========================================================

regions = sorted(

    df["region_txt"]
    .dropna()
    .astype(str)
    .unique()

)

selected_regions = st.sidebar.multiselect(

    "Region",

    regions,

)


# ==========================================================
# COUNTRY
# ==========================================================

countries = sorted(

    df["country_txt"]
    .dropna()
    .astype(str)
    .unique()

)

selected_countries = st.sidebar.multiselect(

    "Country",

    countries,

)


# ==========================================================
# ATTACK TYPE
# ==========================================================

attack_types = sorted(

    df["attacktype1_txt"]
    .dropna()
    .astype(str)
    .unique()

)

selected_attack_types = st.sidebar.multiselect(

    "Attack Type",

    attack_types,

)


# ==========================================================
# WEAPON TYPE
# ==========================================================

weapon_types = sorted(

    df["weaptype1_txt"]
    .dropna()
    .astype(str)
    .unique()

)

selected_weapon_types = st.sidebar.multiselect(

    "Weapon Type",

    weapon_types,

)


# ==========================================================
# TARGET TYPE
# ==========================================================

if "targtype1_txt" in df.columns:

    target_types = sorted(

        df["targtype1_txt"]
        .dropna()
        .astype(str)
        .unique()

    )

else:

    target_types = []

selected_target_types = st.sidebar.multiselect(

    "Target Type",

    target_types,

)


# ==========================================================
# TERRORIST GROUP
# ==========================================================

groups = sorted(

    df["gname"]
    .dropna()
    .astype(str)
    .unique()

)

selected_groups = st.sidebar.multiselect(

    "Terrorist Group",

    groups,

)


# ==========================================================
# SUCCESS
# ==========================================================

if "success" in df.columns:

    success_filter = st.sidebar.selectbox(

        "Successful Attack",

        [

            "All",

            "Successful",

            "Failed",

        ],

    )

else:

    success_filter = "All"


# ==========================================================
# RESET BUTTON
# ==========================================================

if st.sidebar.button(

    "Reset All Filters",

    use_container_width=True,

):

    st.rerun()


# ==========================================================
# GLOBAL SEARCH
# ==========================================================

search = st.text_input(

    "Global Intelligence Search",

    placeholder=(
        "Search city, country, group, "
        "weapon, target or summary..."
    ),

)


# ==========================================================
# APPLY FILTERS
# ==========================================================

filtered_df = df.copy()


if selected_years:

    filtered_df = filtered_df[

        filtered_df["iyear"].isin(

            selected_years

        )

    ]


if selected_regions:

    filtered_df = filtered_df[

        filtered_df["region_txt"].isin(

            selected_regions

        )

    ]


if selected_countries:

    filtered_df = filtered_df[

        filtered_df["country_txt"].isin(

            selected_countries

        )

    ]


if selected_attack_types:

    filtered_df = filtered_df[

        filtered_df["attacktype1_txt"].isin(

            selected_attack_types

        )

    ]


if selected_weapon_types:

    filtered_df = filtered_df[

        filtered_df["weaptype1_txt"].isin(

            selected_weapon_types

        )

    ]


if selected_target_types:

    filtered_df = filtered_df[

        filtered_df["targtype1_txt"].isin(

            selected_target_types

        )

    ]


if selected_groups:

    filtered_df = filtered_df[

        filtered_df["gname"].isin(

            selected_groups

        )

    ]


if success_filter == "Successful":

    filtered_df = filtered_df[

        filtered_df["success"] == 1

    ]


elif success_filter == "Failed":

    filtered_df = filtered_df[

        filtered_df["success"] == 0

    ]


# ==========================================================
# GLOBAL SEARCH
# ==========================================================

if search:

    search = search.lower()

    search_columns = [

        "country_txt",
        "city",
        "region_txt",
        "gname",
        "attacktype1_txt",
        "weaptype1_txt",
        "targtype1_txt",
        "summary",

    ]

    mask = False

    for column in search_columns:

        if column in filtered_df.columns:

            current = (

                filtered_df[column]

                .fillna("")

                .astype(str)

                .str.lower()

                .str.contains(
                    search,
                    regex=False,
                )

            )

            mask = current if isinstance(mask, bool) else (mask | current)

    if not isinstance(mask, bool):

        filtered_df = filtered_df[mask]

# ==========================================================
# EXECUTIVE INTELLIGENCE DASHBOARD
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div class="section-title">
        Executive Intelligence Overview
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# KPI CALCULATIONS
# ==========================================================

total_incidents = len(filtered_df)

total_countries = (

    filtered_df["country_txt"]

    .nunique()

)

total_regions = (

    filtered_df["region_txt"]

    .nunique()

)

total_groups = (

    filtered_df["gname"]

    .replace("Unknown", np.nan)

    .dropna()

    .nunique()

)

total_attack_types = (

    filtered_df["attacktype1_txt"]

    .nunique()

)

total_weapon_types = (

    filtered_df["weaptype1_txt"]

    .nunique()

)

total_fatalities = int(

    filtered_df["nkill"]

    .fillna(0)

    .sum()

)

total_injuries = int(

    filtered_df["nwound"]

    .fillna(0)

    .sum()

)

successful_attacks = 0

if "success" in filtered_df.columns:

    successful_attacks = int(

        (filtered_df["success"] == 1)

        .sum()

    )

success_rate = (

    successful_attacks / total_incidents * 100

    if total_incidents

    else 0

)

average_fatalities = (

    round(

        total_fatalities / total_incidents,

        2,

    )

    if total_incidents

    else 0

)

average_injuries = (

    round(

        total_injuries / total_incidents,

        2,

    )

    if total_incidents

    else 0

)


# ==========================================================
# KPI ROW 1
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(

        "Incidents",

        f"{total_incidents:,}",

    )

with c2:

    st.metric(

        "Countries",

        f"{total_countries}",

    )

with c3:

    st.metric(

        "Fatalities",

        f"{total_fatalities:,}",

    )

with c4:

    st.metric(

        "Injuries",

        f"{total_injuries:,}",

    )


# ==========================================================
# KPI ROW 2
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(

        "Terror Groups",

        total_groups,

    )

with c2:

    st.metric(

        "Attack Types",

        total_attack_types,

    )

with c3:

    st.metric(

        "Weapons",

        total_weapon_types,

    )

with c4:

    st.metric(

        "Success Rate",

        f"{success_rate:.1f}%",

    )


# ==========================================================
# KPI ROW 3
# ==========================================================

c1, c2 = st.columns(2)

with c1:

    st.metric(

        "Average Fatalities / Incident",

        average_fatalities,

    )

with c2:

    st.metric(

        "Average Injuries / Incident",

        average_injuries,

    )


# ==========================================================
# TOP INTELLIGENCE
# ==========================================================

st.markdown("---")

left, right = st.columns([1.1, 1])


# ==========================================================
# TOP THREATS
# ==========================================================

with left:

    st.subheader("Top Intelligence Indicators")

    top_country = (

        filtered_df["country_txt"]

        .value_counts()

        .head(10)

        .reset_index()

    )

    top_country.columns = [

        "Country",

        "Incidents",

    ]

    fig = px.bar(

        top_country,

        x="Incidents",

        y="Country",

        orientation="h",

        color="Incidents",

        title="Most Affected Countries",

    )

    fig.update_layout(

        height=420,

        yaxis=dict(

            categoryorder="total ascending"

        ),

    )

    st.plotly_chart(

        fig,

        use_container_width=True,

    )


# ==========================================================
# REGION DISTRIBUTION
# ==========================================================

with right:

    st.subheader("Regional Distribution")

    region_df = (

        filtered_df["region_txt"]

        .value_counts()

        .reset_index()

    )

    region_df.columns = [

        "Region",

        "Incidents",

    ]

    fig = px.pie(

        region_df,

        names="Region",

        values="Incidents",

        hole=.55,

    )

    fig.update_layout(

        height=420,

    )

    st.plotly_chart(

        fig,

        use_container_width=True,

    )


# ==========================================================
# AI EXECUTIVE SUMMARY
# ==========================================================

st.markdown("---")

top_country_name = "N/A"

if len(filtered_df):

    top_country_name = (

        filtered_df["country_txt"]

        .mode()

        .iloc[0]

    )

top_group = "Unknown"

if len(filtered_df):

    top_group = (

        filtered_df["gname"]

        .mode()

        .iloc[0]

    )

top_weapon = "Unknown"

if len(filtered_df):

    top_weapon = (

        filtered_df["weaptype1_txt"]

        .mode()

        .iloc[0]

    )

top_attack = "Unknown"

if len(filtered_df):

    top_attack = (

        filtered_df["attacktype1_txt"]

        .mode()

        .iloc[0]

    )

st.markdown(

    f"""
<div class="report-card">

<h3>Executive Intelligence Summary</h3>

<b>Total Recorded Incidents</b><br>
{total_incidents:,}

<br>

<b>Most Affected Country</b><br>
{top_country_name}

<br>

<b>Dominant Terrorist Organization</b><br>
{top_group}

<br>

<b>Most Common Attack Type</b><br>
{top_attack}

<br>

<b>Most Frequently Used Weapon</b><br>
{top_weapon}

<br>

<b>Total Fatalities</b><br>
{total_fatalities:,}

<br>

<b>Total Injuries</b><br>
{total_injuries:,}

</div>

""",

    unsafe_allow_html=True,

)


# ==========================================================
# QUICK STATISTICS
# ==========================================================

st.markdown("---")

st.subheader("Dataset Statistics")

col1, col2, col3 = st.columns(3)

with col1:

    st.write("Records")

    st.code(f"{filtered_df.shape[0]:,}")

    st.write("Columns")

    st.code(filtered_df.shape[1])

with col2:

    st.write("Memory Usage")

    memory = round(

        filtered_df.memory_usage(

            deep=True

        ).sum() / 1024**2,

        2,

    )

    st.code(f"{memory} MB")

    st.write("Duplicate Records")

    st.code(

        filtered_df.duplicated().sum()

    )

with col3:

    missing = int(

        filtered_df.isna()

        .sum()

        .sum()

    )

    st.write("Missing Values")

    st.code(missing)

    st.write("Regions")

    st.code(total_regions)

# =========================================================
# INTELLIGENCE ANALYTICS
# =========================================================

st.markdown("---")
st.subheader("Intelligence Analytics")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Timeline",
        "Attack Patterns",
        "Organizations",
        "Geographic"
    ]
)

# =========================================================
# TIMELINE
# =========================================================

with tab1:

    if "iyear" in filtered_df.columns:

        timeline = (
            filtered_df
            .groupby("iyear")
            .size()
            .reset_index(name="Incidents")
            .sort_values("iyear")
        )

        fig = px.line(
            timeline,
            x="iyear",
            y="Incidents",
            markers=True,
            title="Incident Trend"
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Incidents",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info("Year information unavailable.")

# =========================================================
# ATTACK PATTERNS
# =========================================================

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        if "attacktype1_txt" in filtered_df.columns:

            attack_chart = (
                filtered_df["attacktype1_txt"]
                .fillna("Unknown")
                .value_counts()
                .head(12)
                .reset_index()
            )

            attack_chart.columns = [
                "Attack",
                "Incidents"
            ]

            fig = px.bar(
                attack_chart,
                x="Attack",
                y="Incidents",
                color="Incidents",
                title="Most Frequent Attack Types"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        if "weaptype1_txt" in filtered_df.columns:

            weapon_chart = (
                filtered_df["weaptype1_txt"]
                .fillna("Unknown")
                .value_counts()
                .head(12)
                .reset_index()
            )

            weapon_chart.columns = [
                "Weapon",
                "Incidents"
            ]

            fig = px.bar(
                weapon_chart,
                x="Weapon",
                y="Incidents",
                color="Incidents",
                title="Weapon Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# =========================================================
# TERRORIST ORGANIZATIONS
# =========================================================

with tab3:

    if "gname" in filtered_df.columns:

        group_chart = (
            filtered_df["gname"]
            .fillna("Unknown")
            .value_counts()
            .head(15)
            .reset_index()
        )

        group_chart.columns = [
            "Organization",
            "Incidents"
        ]

        fig = px.bar(
            group_chart,
            x="Incidents",
            y="Organization",
            orientation="h",
            color="Incidents",
            title="Most Active Organizations"
        )

        fig.update_layout(height=650)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =========================================================
# GEOGRAPHIC DISTRIBUTION
# =========================================================

with tab4:

    if (
        "latitude" in filtered_df.columns
        and
        "longitude" in filtered_df.columns
    ):

        geo_df = filtered_df.dropna(
            subset=[
                "latitude",
                "longitude"
            ]
        )

        if len(geo_df):

            fig = px.scatter_mapbox(

                geo_df,

                lat="latitude",

                lon="longitude",

                hover_name=(
                    "country_txt"
                    if "country_txt" in geo_df.columns
                    else None
                ),

                hover_data=[
                    c
                    for c in [
                        "city",
                        "attacktype1_txt",
                        "gname",
                        "nkill"
                    ]
                    if c in geo_df.columns
                ],

                zoom=1,

                height=700,

                color=(
                    "attacktype1_txt"
                    if "attacktype1_txt" in geo_df.columns
                    else None
                )
            )

            fig.update_layout(
                mapbox_style="carto-darkmatter",
                margin=dict(
                    l=0,
                    r=0,
                    t=0,
                    b=0
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("No coordinates available.")

    else:

        st.info("Latitude / Longitude not found.")

# =========================================================
# INCIDENT SEARCH
# =========================================================

st.markdown("---")
st.subheader("Incident Search")

search = st.text_input(
    "Search by city, terrorist group, summary or target"
)

if search:

    search_columns = []

    for col in [
        "city",
        "country_txt",
        "gname",
        "summary",
        "target1",
        "attacktype1_txt",
    ]:

        if col in filtered_df.columns:
            search_columns.append(col)

    mask = False

    for col in search_columns:

        current = (
            filtered_df[col]
            .fillna("")
            .astype(str)
            .str.contains(
                search,
                case=False,
                regex=False,
            )
        )

        mask = current if isinstance(mask, bool) else (mask | current)

    results = filtered_df.loc[mask]

    st.success(
        f"{len(results):,} matching incidents found."
    )

    st.dataframe(
        results,
        use_container_width=True,
        height=450,
    )

else:

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=500,
    )

# =========================================================
# DATA QUALITY
# =========================================================

st.markdown("---")
st.subheader("Dataset Quality")

col1, col2 = st.columns(2)

with col1:

    missing = (
        filtered_df
        .isna()
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    missing.columns = [
        "Column",
        "Missing Values",
    ]

    st.dataframe(
        missing,
        use_container_width=True,
        height=400,
    )

with col2:

    completeness = pd.DataFrame({

        "Column": filtered_df.columns,

        "Completeness (%)":
        (
            (
                1
                - filtered_df.isna().mean()
            ) * 100
        ).round(2)

    })

    fig = px.bar(

        completeness.sort_values(
            "Completeness (%)"
        ),

        x="Completeness (%)",

        y="Column",

        orientation="h",

        color="Completeness (%)",

        title="Column Completeness"

    )

    fig.update_layout(height=700)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# DATASET INFORMATION
# =========================================================

st.markdown("---")
st.subheader("Dataset Information")

memory = (
    filtered_df
    .memory_usage(deep=True)
    .sum()
    / 1024**2
)

dtype_df = pd.DataFrame({

    "Column": filtered_df.columns,

    "Data Type":
    filtered_df.dtypes.astype(str),

    "Missing":
    filtered_df.isna().sum(),

    "Unique":
    filtered_df.nunique(),

})

left, right = st.columns([1,2])

with left:

    st.metric(
        "Rows",
        f"{len(filtered_df):,}"
    )

    st.metric(
        "Columns",
        filtered_df.shape[1]
    )

    st.metric(
        "Memory",
        f"{memory:.2f} MB"
    )

    st.metric(
        "Duplicates",
        int(filtered_df.duplicated().sum())
    )

with right:

    st.dataframe(
        dtype_df,
        use_container_width=True,
        height=420,
    )

# =========================================================
# EXPORT CENTER
# =========================================================

st.markdown("---")
st.subheader("Export Center")

csv = filtered_df.to_csv(
    index=False
)

col1, col2 = st.columns(2)

with col1:

    st.download_button(

        "Download Filtered Dataset",

        csv,

        file_name="Filtered_Intelligence_Data.csv",

        mime="text/csv",

        use_container_width=True,

    )

with col2:

    st.download_button(

        "Download Current Search Results",

        csv,

        file_name="Current_View.csv",

        mime="text/csv",

        use_container_width=True,

    )

# =========================================================
# INTELLIGENCE SUMMARY
# =========================================================

st.markdown("---")
st.subheader("Executive Intelligence Summary")

summary = []

summary.append(
    f"Total incidents analysed: {len(filtered_df):,}"
)

if "country_txt" in filtered_df.columns:

    summary.append(
        f"Countries represented: {filtered_df['country_txt'].nunique()}"
    )

if "nkill" in filtered_df.columns:

    summary.append(
        f"Confirmed fatalities: {int(filtered_df['nkill'].fillna(0).sum()):,}"
    )

if "nwound" in filtered_df.columns:

    summary.append(
        f"Reported injuries: {int(filtered_df['nwound'].fillna(0).sum()):,}"
    )

if (
    "attacktype1_txt" in filtered_df.columns
    and len(filtered_df)
):

    attack = (
        filtered_df["attacktype1_txt"]
        .fillna("Unknown")
        .mode()[0]
    )

    summary.append(
        f"Most common attack type: {attack}"
    )

if (
    "gname" in filtered_df.columns
    and len(filtered_df)
):

    group = (
        filtered_df["gname"]
        .fillna("Unknown")
        .mode()[0]
    )

    summary.append(
        f"Most active organization: {group}"
    )

st.info(
    "\n\n".join(summary)
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Military Intelligence Dashboard • Data Explorer • Active Dataset Analysis"
)