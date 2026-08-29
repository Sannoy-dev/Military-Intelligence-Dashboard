import json
import os
import streamlit as st

from utils.ui import load_css
from utils.data_loader import (
    load_data,
    get_dataset_info,
    dataset_exists
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Dashboard Settings",
    layout="wide"
)

load_css()

# =====================================================
# SETTINGS FILE
# =====================================================

SETTINGS_DIR = "data"
SETTINGS_FILE = os.path.join(
    SETTINGS_DIR,
    "dashboard_settings.json"
)

DEFAULT_SETTINGS = {
    "theme": "Dark (Recommended)",
    "layout": "Wide",
    "default_country": "",
    "forecast_years": 5,
    "minimum_confidence": 80,
    "show_probability": True,
    "show_ai_summary": True,
    "animations": True,
    "report_format": "PDF",
    "include_charts": True,
    "include_tables": True,
    "include_recommendations": True,
    "forecast_model": "Linear Regression",
    "forecast_period": 5
}

# =====================================================
# LOAD SETTINGS
# =====================================================

def load_settings():

    if not os.path.exists(SETTINGS_FILE):
        return DEFAULT_SETTINGS.copy()

    try:

        with open(
            SETTINGS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            saved_settings = json.load(file)

        settings = DEFAULT_SETTINGS.copy()

        settings.update(
            saved_settings
        )

        return settings

    except Exception:

        return DEFAULT_SETTINGS.copy()


# =====================================================
# SAVE SETTINGS
# =====================================================

def save_settings(settings):

    os.makedirs(
        SETTINGS_DIR,
        exist_ok=True
    )

    with open(
        SETTINGS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            settings,
            file,
            indent=4
        )


# =====================================================
# LOAD CURRENT SETTINGS
# =====================================================

settings = load_settings()

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
        <div>
            <h1>Dashboard Settings</h1>
            <p>
              Configure dashboard preferences, reports,
            forecasting behavior, and display options.
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
        "Loading dashboard configuration..."
    ):

        df = load_data()

except Exception as error:

    df = None

    st.error(
        f"Unable to load the active dataset: {error}"
    )

# =====================================================
# DASHBOARD PREFERENCES
# =====================================================

st.markdown(
    "Dashboard Preferences"
)

left, right = st.columns(2)

with left:

    st.markdown(
        '<div class="chart-card">',
        unsafe_allow_html=True
    )

    theme = st.selectbox(
        "Theme",
        [
            "Dark (Recommended)",
            "Light"
        ],
        index=(
            0
            if settings["theme"]
            == "Dark (Recommended)"
            else 1
        )
    )

    layout = st.selectbox(
        "Layout",
        [
            "Wide",
            "Centered"
        ],
        index=(
            0
            if settings["layout"]
            == "Wide"
            else 1
        )
    )

    default_country = st.text_input(
        "Default Country",
        value=settings[
            "default_country"
        ]
    )

    forecast_years = st.slider(
        "Default Forecast Years",
        min_value=1,
        max_value=10,
        value=settings[
            "forecast_years"
        ]
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


with right:

    st.markdown(
        '<div class="chart-card">',
        unsafe_allow_html=True
    )

    confidence = st.slider(
        "Minimum Prediction Confidence (%)",
        min_value=50,
        max_value=100,
        value=settings[
            "minimum_confidence"
        ]
    )

    show_probability = st.checkbox(
        "Show Prediction Confidence",
        value=settings[
            "show_probability"
        ]
    )

    show_ai_summary = st.checkbox(
        "Show AI Executive Summary",
        value=settings[
            "show_ai_summary"
        ]
    )

    animations = st.checkbox(
        "Enable Animations",
        value=settings[
            "animations"
        ]
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

# =====================================================
# REPORT SETTINGS
# =====================================================

st.markdown(
    "Intelligence Report Settings"
)

report_format = st.selectbox(
    "Report Format",
    [
        "PDF",
        "Text",
        "Word"
    ],
    index=[
        "PDF",
        "Text",
        "Word"
    ].index(
        settings["report_format"]
    )
)

include_charts = st.checkbox(
    "Include Charts",
    value=settings[
        "include_charts"
    ]
)

include_tables = st.checkbox(
    "Include Tables",
    value=settings[
        "include_tables"
    ]
)

include_recommendations = st.checkbox(
    "Include AI Recommendations",
    value=settings[
        "include_recommendations"
    ]
)

# =====================================================
# FORECAST SETTINGS
# =====================================================

st.markdown(
    "Forecast Configuration"
)

forecast_models = [
    "Linear Regression",
    "ARIMA",
    "Prophet"
]

forecast_model = st.selectbox(
    "Forecasting Model",
    forecast_models,
    index=forecast_models.index(
        settings["forecast_model"]
    )
)

forecast_period = st.slider(
    "Forecast Horizon (Years)",
    min_value=1,
    max_value=10,
    value=settings[
        "forecast_period"
    ]
)

# =====================================================
# DATASET INFORMATION
# =====================================================

st.markdown(
    "Active Dataset Information"
)

if df is not None and not df.empty:

    rows = len(df)

    columns = len(df.columns)

    if "country_txt" in df.columns:

        countries = (
            df["country_txt"]
            .dropna()
            .nunique()
        )

    else:

        countries = "N/A"

    memory = round(
        df.memory_usage(
            deep=True
        ).sum()
        / 1024 ** 2,
        2
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Records",
        f"{rows:,}"
    )

    c2.metric(
        "Columns",
        columns
    )

    c3.metric(
        "Countries",
        countries
    )

    c4.metric(
        "Memory",
        f"{memory} MB"
    )

    st.success(
        "Active uploaded dataset is loaded successfully."
    )

    with st.expander(
        "View Dataset Columns"
    ):

        st.write(
            list(df.columns)
        )

else:

    st.warning(
        """
        No active dataset is currently available.

        Upload and process a dataset from the main dashboard
        to view dataset information.
        """
    )

# =====================================================
# DATASET STATUS
# =====================================================

st.markdown(
    "Dataset Status"
)

dataset_info = get_dataset_info()

status_1, status_2, status_3 = st.columns(3)

status_1.metric(
    "Dataset Status",
    (
        "Loaded"
        if dataset_info["loaded"]
        else "Not Available"
    )
)

status_2.metric(
    "Records",
    f"{dataset_info['rows']:,}"
)

status_3.metric(
    "Columns",
    dataset_info["columns"]
)

# =====================================================
# DASHBOARD INFORMATION
# =====================================================

st.markdown(
    "Dashboard Information"
)

st.markdown(
    f"""
    <div class="report-card">
   
    <h3>Custom Intelligence Dashboard</h3>

    <p>
    This dashboard operates using the currently active
                uploaded dataset.
    </p>

    <p>
    It does not depend on a permanently hardcoded
                default dataset for analytics or machine-learning
                modules.
    </p>

    <p>
    Dataset compatibility depends on the columns
                available after upload and column mapping.
    </p>

    <h4>Available Capabilities</h4>
    
    <ul>
                <li>Dataset exploration</li>
                <li>Interactive analytics</li>
                <li>Local machine-learning model training</li>
                <li>Attack-type prediction</li>
                <li>Threat-level prediction</li>
                <li>Forecasting</li>
                <li>Intelligence report generation</li>
    </ul>
    
    </div>
    """,
    unsafe_allow_html=True
)
# =====================================================
# SAVE SETTINGS
# =====================================================

st.markdown(
    "Configuration"
)

left, right = st.columns(2)

with left:

    if st.button(
        "Save Settings",
        use_container_width=True
    ):

        new_settings = {

            "theme":
                theme,

            "layout":
                layout,

            "default_country":
                default_country,

            "forecast_years":
                forecast_years,

            "minimum_confidence":
                confidence,

            "show_probability":
                show_probability,

            "show_ai_summary":
                show_ai_summary,

            "animations":
                animations,

            "report_format":
                report_format,

            "include_charts":
                include_charts,

            "include_tables":
                include_tables,

            "include_recommendations":
                include_recommendations,

            "forecast_model":
                forecast_model,

            "forecast_period":
                forecast_period
        }

        try:

            save_settings(
                new_settings
            )

            st.success(
                "Settings saved successfully."
            )

        except Exception as error:

            st.error(
                f"Unable to save settings: {error}"
            )


with right:

    if st.button(
        "Restore Defaults",
        use_container_width=True
    ):

        try:

            save_settings(
                DEFAULT_SETTINGS
            )

            st.success(
                "Default settings restored."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to restore defaults: {error}"
            )

# =====================================================
# FOOTER
# =====================================================

st.caption(
    "AI Intelligence Dashboard • Configuration Panel"
)