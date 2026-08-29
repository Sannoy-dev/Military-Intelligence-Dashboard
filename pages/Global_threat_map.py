import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data
from utils.ui import load_css


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Global Threat Map",
    layout="wide"
)

load_css()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="page-header">
        <div class="page-header-icon"></div>
        <div>
            <h1>Global Threat Map</h1>
            <p>
              Explore the geographic distribution of recorded
              incidents and identify major patterns of activity.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD ACTIVE DATASET
# ============================================================

with st.spinner("Preparing global threat intelligence..."):

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
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "iyear",
    "country_txt",
    "latitude",
    "longitude"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "The active dataset cannot be used by the Global Threat Map."
    )

    st.markdown(
        f"""
        **Missing required columns:**

        `{", ".join(missing_columns)}`

        Please map these columns in the Custom Dataset
        Mapping section.
        """
    )

    st.stop()


# ============================================================
# PREPARE MAP DATA
# ============================================================

with st.spinner("Preparing geographic records..."):

    map_df = df.copy()

    map_df["iyear"] = pd.to_numeric(
        map_df["iyear"],
        errors="coerce"
    )

    map_df["latitude"] = pd.to_numeric(
        map_df["latitude"],
        errors="coerce"
    )

    map_df["longitude"] = pd.to_numeric(
        map_df["longitude"],
        errors="coerce"
    )

    map_df = map_df.dropna(
        subset=[
            "iyear",
            "latitude",
            "longitude"
        ]
    )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown(
    "### ⚙ Map Filters"
)


# -------------------------
# Year
# -------------------------

years = sorted(
    map_df["iyear"]
    .astype(int)
    .unique()
    .tolist()
)


selected_year = st.sidebar.selectbox(
    "Year",
    ["All"] + years,
    label_visibility="collapsed"
)


# -------------------------
# Country
# -------------------------

if "country_txt" in map_df.columns:

    country_options = sorted(
        map_df["country_txt"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    country_options = []


selected_country = st.sidebar.selectbox(
    "Country",
    ["All"] + country_options,
    label_visibility="collapsed"
)


# -------------------------
# Attack Type
# -------------------------

if "attacktype1_txt" in map_df.columns:

    attack_options = sorted(
        map_df["attacktype1_txt"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    attack_options = []


selected_attack = st.sidebar.selectbox(
    "Attack Type",
    ["All"] + attack_options,
    label_visibility="collapsed"
)


# ============================================================
# APPLY FILTERS
# ============================================================

with st.spinner("Filtering threat intelligence..."):

    filtered_df = map_df.copy()

    if selected_year != "All":

        filtered_df = filtered_df[
            filtered_df["iyear"] == selected_year
        ]

    if selected_country != "All":

        filtered_df = filtered_df[
            filtered_df["country_txt"]
            == selected_country
        ]

    if selected_attack != "All":

        filtered_df = filtered_df[
            filtered_df["attacktype1_txt"]
            == selected_attack
        ]


# ============================================================
# CHECK FILTER RESULTS
# ============================================================

if filtered_df.empty:

    st.warning(
        "No geographic incidents match the selected filters."
    )

    st.stop()


# ============================================================
# INTELLIGENCE STATISTICS
# ============================================================

total_incidents = len(
    filtered_df
)

country_count = filtered_df[
    "country_txt"
].nunique()

if "nkill" in filtered_df.columns:

    fatalities = int(
        pd.to_numeric(
            filtered_df["nkill"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )

else:

    fatalities = 0


if "attacktype1_txt" in filtered_df.columns:

    attack_count = filtered_df[
        "attacktype1_txt"
    ].nunique()

else:

    attack_count = 0


# ============================================================
# KPI SECTION
# ============================================================

st.markdown(
    "Global Threat Overview"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Incidents",
        f"{total_incidents:,}"
    )


with c2:

    st.metric(
        "Countries",
        country_count
    )


with c3:

    st.metric(
        "Fatalities",
        f"{fatalities:,}"
    )


with c4:

    st.metric(
        "Attack Types",
        attack_count
    )


# ============================================================
# INTELLIGENCE SUMMARY
# ============================================================

st.markdown(
    "Geographic Intelligence Summary"
)


top_country = "Not available"
top_attack = "Not available"


if "country_txt" in filtered_df.columns:

    country_counts = (
        filtered_df["country_txt"]
        .dropna()
        .value_counts()
    )

    if not country_counts.empty:

        top_country = country_counts.index[0]


if "attacktype1_txt" in filtered_df.columns:

    attack_counts = (
        filtered_df["attacktype1_txt"]
        .dropna()
        .value_counts()
    )

    if not attack_counts.empty:

        top_attack = attack_counts.index[0]


st.markdown(
    f"""
    <div class="report-card">

    <h3>Current Geographic Assessment</h3>

    <p>
    The current filtered dataset contains
    <strong>{total_incidents:,}</strong> geographically
    identifiable incidents across
    <strong>{country_count}</strong> countries.
    </p>

    <p>
    The country with the highest number of recorded
    incidents in the current selection is
    <strong>{top_country}</strong>.
    </p>

    <p>
    The most frequently recorded attack category is
    <strong>{top_attack}</strong>.
    </p>

    <p>
    These statistics describe historical records in the
    selected dataset and should be interpreted as
    analytical information rather than predictions.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GLOBAL MAP
# ============================================================

st.markdown(
    "Interactive Threat Map"
)


with st.spinner("Rendering geographic intelligence..."):

    hover_columns = {}

    optional_hover = [
        "city",
        "gname",
        "iyear",
        "attacktype1_txt",
        "weaptype1_txt",
        "nkill",
        "nwound"
    ]

    for column in optional_hover:

        if column in filtered_df.columns:

            hover_columns[column] = True


    hover_columns["latitude"] = False
    hover_columns["longitude"] = False


    if "attacktype1_txt" in filtered_df.columns:

        color_column = "attacktype1_txt"

    else:

        color_column = None


    fig = px.scatter_geo(
        filtered_df,

        lat="latitude",

        lon="longitude",

        color=color_column,

        hover_name="country_txt",

        hover_data=hover_columns,

        projection="natural earth",

        height=650
    )


    fig.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="white"
        ),

        margin=dict(
            l=0,
            r=0,
            t=20,
            b=0
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )


    fig.update_geos(
        showland=True,
        showcountries=True,
        showcoastlines=True
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
# ATTACK DISTRIBUTION
# ============================================================

if "attacktype1_txt" in filtered_df.columns:

    st.markdown(
        "Attack Distribution"
    )

    attack_distribution = (
        filtered_df["attacktype1_txt"]
        .dropna()
        .value_counts()
        .reset_index()
    )

    attack_distribution.columns = [
        "Attack Type",
        "Incidents"
    ]


    left, right = st.columns(2)


    with left:

        fig = px.bar(
            attack_distribution.head(10),
            x="Incidents",
            y="Attack Type",
            orientation="h",
            title="Top Attack Types"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
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


    with right:

        fig = px.pie(
            attack_distribution,
            names="Attack Type",
            values="Incidents",
            hole=0.45,
            title="Attack Type Distribution"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
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
# INCIDENT DATABASE
# ============================================================

st.markdown(
    "Incident Records"
)


display_columns = [
    "iyear",
    "country_txt",
    "city",
    "attacktype1_txt",
    "weaptype1_txt",
    "gname",
    "nkill",
    "nwound"
]


available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]


# Limit the number of rows sent to the browser.
# This prevents Streamlit's message-size problem
# when the dataset is very large.

MAX_DISPLAY_ROWS = 5000

table_df = filtered_df[
    available_columns
].head(MAX_DISPLAY_ROWS)


st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    height=450
)


if len(filtered_df) > MAX_DISPLAY_ROWS:

    st.caption(
        f"""
        Showing the first {MAX_DISPLAY_ROWS:,} records
        out of {len(filtered_df):,}. Download the filtered
        dataset below to access the complete result.
        """
    )


# ============================================================
# EXPORT
# ============================================================

st.markdown(
    "Export Intelligence"
)


csv = filtered_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Filtered Intelligence",
    data=csv,
    file_name="Global_Threat_Map_Data.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "ℹ️ Map Methodology"
):

    st.markdown(
        """
        The Global Threat Map uses the geographic coordinates
        available in the active dataset.

        **Processing:**

        1. Load the active dataset through `load_data()`.
        2. Validate the required geographic columns.
        3. Convert latitude and longitude to numeric values.
        4. Remove records without valid coordinates.
        5. Apply the selected year, country and attack-type filters.
        6. Display the resulting incidents geographically.
        7. Provide aggregated statistics and downloadable data.

        Extra columns in a custom dataset are not required by
        this module and can be ignored by the dataset mapping
        system.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "The map visualizes historical records from the active dataset. "
    "Geographic visualization is intended for analytical purposes."
)