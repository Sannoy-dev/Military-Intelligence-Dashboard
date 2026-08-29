import streamlit as st
import pandas as pd
import traceback
from utils.ui import load_css
from utils.data_loader import load_data
from utils.model_trainer import (
    load_threat_model,
    threat_model_exists,
    predict_threat,
    load_threat_metadata,
    THREAT_FEATURE_COLUMNS,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Threat Level Prediction",
    layout="wide"
)

load_css()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """

    <div>
        <h1>AI Threat Level Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ARCHITECTURE NOTICE
# ============================================================

st.info(
    """
    **Local Model Architecture**

    This module uses only the Threat Level model trained from
    the currently active uploaded dataset.

    No default or pre-trained threat model is used.
    """
)

# ============================================================
# CHECK ACTIVE DATASET
# ============================================================

try:

    with st.spinner("Loading active intelligence dataset..."):
        df = load_data()

except Exception as error:

    st.error(
        f"Unable to load the active dataset: {error}"
    )

    st.stop()

if df is None:
    st.error("No dataset loaded.")
    st.stop()

if df.empty:
    st.warning("Dataset is empty.")
    st.stop()
# ============================================================
# NO DATASET
# ============================================================

if df is None or df.empty:

    st.warning(
        """
        No active dataset is available.

        Upload and process a compatible CSV dataset from the
        main dashboard before using Threat Level Prediction.
        """
    )

    st.stop()

# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown("Active Dataset")

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Records",
        f"{len(df):,}"
    )

with c2:

    st.metric(
        "Columns",
        len(df.columns)
    )

with c3:

    st.metric(
        "Threat Model",
        "Available"
        if threat_model_exists()
        else "Not Trained"
    )

# ============================================================
# REQUIRED DATASET FEATURES
# ============================================================

missing_features = [
    column
    for column in THREAT_FEATURE_COLUMNS
    if column not in df.columns
]

if missing_features:

    st.error(
        "This dataset cannot currently support Threat Level "
        "Prediction."
    )

    st.markdown("Missing Training Features")

    st.code(
        "\n".join(missing_features),unsafe_allow_html=True
    )

    st.info(
        """
        Return to the main dashboard and make sure the uploaded
        dataset has been mapped to the required standard fields.

        Threat Level Prediction requires:

        - Country
        - Region
        - Attack Type
        - Weapon Type
        - Target Type
        - Fatalities
        - Injuries
        """
    )

    st.stop()

# ============================================================
# MODEL STATUS
# ============================================================

if not threat_model_exists():

    st.warning(
        """
        **Threat Level Prediction model has not been trained.**

        The active dataset contains the required fields, but a
        locally trained Threat Level model is not currently
        available.

        Go to the main dashboard and use:

        **Train Models Locally**
        """
    )

    st.markdown("Required Workflow")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="chart-card">

                <h3>1. Upload Dataset</h3>

                <p>
                    Upload your CSV intelligence dataset from
                    the main dashboard.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="chart-card">

                <h3>2. Map Columns</h3>

                <p>
                    Map uploaded columns to the standardized
                    application fields.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="chart-card">

                <h3>3. Train Locally</h3>

                <p>
                    Train the Threat Level model using the
                    active dataset.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.stop()

# ============================================================
# LOAD THREAT MODEL
# ============================================================

try:
    with st.spinner(
        "Loading trained model..."
    ):

     model_data = model = load_threat_model()

except Exception as error:

    st.error(
        f"Unable to load the Threat Level model: {error}"
    )

    st.stop()

# ============================================================
# INVALID / CORRUPTED MODEL
# ============================================================

if model_data is None:

    st.error(
        """
        The Threat Level model files are incomplete,
        unavailable, or corrupted.

        Retrain the Threat Level model from the main dashboard.
        """
    )

    st.stop()

# ============================================================
# MODEL METADATA
# ============================================================

metadata = load_threat_metadata()

st.success(
    "🟢 Locally trained Threat Level model is active."
)

# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("Model Information")

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "Model",
        "Random Forest"
    )

with m2:

    if metadata:

        st.metric(
            "Training Records",
            f"{metadata.get('training_records', 0):,}"
        )

    else:

        st.metric(
            "Training Records",
            "N/A"
        )

with m3:

    if metadata:

        accuracy = metadata.get(
            "accuracy"
        )

        if accuracy is not None:

            st.metric(
                "Test Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        else:

            st.metric(
                "Test Accuracy",
                "N/A"
            )

    else:

        st.metric(
            "Test Accuracy",
            "N/A"
        )

with m4:

    if metadata:

        st.metric(
            "Classes",
            metadata.get(
                "number_of_classes",
                "N/A"
            )
        )

    else:

        st.metric(
            "Classes",
            "N/A"
        )

# ============================================================
# PREPARE DISPLAY DATA
# ============================================================

display_df = df.copy()

categorical_columns = [
    "country_txt",
    "region_txt",
    "attacktype1_txt",
    "weaptype1_txt",
    "targtype1_txt"
]

for column in categorical_columns:

    display_df[column] = (
        display_df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

# ------------------------------------------------------------
# Numeric fields
# ------------------------------------------------------------

for column in [
    "nkill",
    "nwound"
]:

    display_df[column] = pd.to_numeric(
        display_df[column],
        errors="coerce"
    ).fillna(0)

# ============================================================
# INCIDENT PARAMETERS
# ============================================================

st.markdown("Incident Parameters")

st.caption(
    """
    Provide the known characteristics of an incident to
    generate a Threat Level classification.
    """
)

with st.form(
    "threat_prediction_form"
):

    left, right = st.columns(2)

    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with left:

        country_options = sorted(
            display_df[
                "country_txt"
            ].dropna().unique().tolist()
        )

        region_options = sorted(
            display_df[
                "region_txt"
            ].dropna().unique().tolist()
        )

        attack_options = sorted(
            display_df[
                "attacktype1_txt"
            ].dropna().unique().tolist()
        )

        country = st.selectbox(
            "Country",
            country_options
        )

        region = st.selectbox(
            "Region",
            region_options
        )

        attack = st.selectbox(
            "Attack Type",
            attack_options
        )

    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with right:

        weapon_options = sorted(
            display_df[
                "weaptype1_txt"
            ].dropna().unique().tolist()
        )

        target_options = sorted(
            display_df[
                "targtype1_txt"
            ].dropna().unique().tolist()
        )

        weapon = st.selectbox(
            "Weapon Type",
            weapon_options
        )

        target = st.selectbox(
            "Target Type",
            target_options
        )

        nkill = st.number_input(
            "Fatalities",
            min_value=0,
            max_value=100000,
            value=0,
            step=1
        )

        nwound = st.number_input(
            "Injuries",
            min_value=0,
            max_value=100000,
            value=0,
            step=1
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    predict_button = st.form_submit_button(
        "Generate Threat Assessment",
        use_container_width=True
    )

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = {

        "country_txt":
            country,

        "region_txt":
            region,

        "attacktype1_txt":
            attack,

        "weaptype1_txt":
            weapon,

        "targtype1_txt":
            target,

        "nkill":
            nkill,

        "nwound":
            nwound
    }

    try:

        with st.spinner(
            "Running AI threat assessment..."
        ):

            result = predict_threat(
                          model,
                        input_data
                          )

    except Exception as error:

        st.error(
            f"Threat assessment failed: {error}"
        )

        st.stop()

    # ========================================================
    # RESULT DATA
    # ========================================================

    threat_level = result.get(
        "prediction",
        "UNKNOWN"
    )

    confidence = result.get(
        "confidence"
    )

    probabilities = result.get(
        "probabilities"
    )

    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.markdown(
        "<div class='result-divider'></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="report-card">

            <div class="report-label">
                AI THREAT ASSESSMENT
            </div>

            <h2>Prediction Result</h2>

            <p>
                The locally trained machine-learning model has
                completed the threat-level classification.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # THREAT CLASSIFICATION
    # ========================================================

    if threat_level == "LOW":

        threat_class = "low"

        threat_title = "🟢 LOW THREAT"

        threat_description = (
            "The model classified the incident as a "
            "relatively low-impact threat level."
        )

    elif threat_level == "MEDIUM":

        threat_class = "medium"

        threat_title = "🟡 MEDIUM THREAT"

        threat_description = (
            "The model classified the incident as a "
            "moderate-impact threat level."
        )

    elif threat_level == "HIGH":

        threat_class = "high"

        threat_title = "🔴 HIGH THREAT"

        threat_description = (
            "The model classified the incident as a "
            "high-impact threat level."
        )

    else:

        threat_class = "medium"

        threat_title = "⚪ UNKNOWN THREAT"

        threat_description = (
            "The model returned an unrecognized threat "
            "classification."
        )

    # ========================================================
    # THREAT CARD
    # ========================================================

    st.markdown(
        f"""
        <div class="threat-result {threat_class}">

            <div class="threat-title">
                {threat_title}
            </div>

            <div class="threat-description">
                {threat_description}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # ASSESSMENT METRICS
    # ========================================================

    st.markdown(
        "### Assessment Metrics"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Threat Level",
        threat_level
    )

    if confidence is not None:

        c2.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    else:

        c2.metric(
            "Confidence",
            "N/A"
        )

    c3.metric(
        "Fatalities",
        f"{nkill:,}"
    )

    c4.metric(
        "Injuries",
        f"{nwound:,}"
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    if confidence is not None:

        st.markdown(
            "### Confidence Score"
        )

        st.progress(
            min(
                max(
                    confidence / 100,
                    0
                ),
                1
            )
        )

        st.caption(
            f"Model confidence: {confidence:.2f}%"
        )

    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    if probabilities:

        st.markdown(
            "### Threat Probability Distribution"
        )

        probability_df = pd.DataFrame(
            {
                "Threat Level":
                    list(probabilities.keys()),

                "Probability":
                    list(probabilities.values())
            }
        )

        st.dataframe(
            probability_df.style.format(
                {
                    "Probability":
                        "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # INCIDENT SUMMARY
    # ========================================================

    st.markdown(
        "### Incident Summary"
    )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Country",
        country
    )

    s2.metric(
        "Attack Type",
        attack
    )

    s3.metric(
        "Weapon Type",
        weapon
    )

    s4.metric(
        "Target Type",
        target
    )

    # ========================================================
    # CASUALTY SUMMARY
    # ========================================================

    st.markdown(
        "### Impact Summary"
    )

    impact = nkill + nwound

    i1, i2, i3 = st.columns(3)

    i1.metric(
        "Fatalities",
        f"{nkill:,}"
    )

    i2.metric(
        "Injuries",
        f"{nwound:,}"
    )

    i3.metric(
        "Total Impact",
        f"{impact:,}"
    )

    # ========================================================
    # AI ASSESSMENT
    # ========================================================

    st.markdown(
        "### AI Assessment"
    )

    confidence_text = (
        f"{confidence:.2f}%"
        if confidence is not None
        else "N/A"
    )

    st.markdown(
        f"""
        <div class="assessment-card">

            <h3>Threat Classification</h3>

            <p>
                The locally trained model classified the
                provided incident as
                <strong>{threat_level}</strong> threat.
            </p>

            <p>
                The estimated model confidence is
                <strong>{confidence_text}</strong>.
            </p>

            <p>
                The classification is based on the incident
                characteristics and historical patterns
                available in the uploaded dataset used during
                model training.
            </p>

            <p>
                This output is intended for analytical
                decision support and should be interpreted
                together with verified intelligence and
                human assessment.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# MODEL REQUIREMENTS
# ============================================================

st.markdown("Model Requirements")

st.caption(
    """
    The currently trained Threat Level model expects the
    following standardized input features:
    """
)

st.write(
    THREAT_FEATURE_COLUMNS
)

if metadata:

    threat_definition = metadata.get(
        "threat_definition"
    )

    if threat_definition:

        st.markdown(
            "### Threat-Level Definition"
        )

        st.info(
            """
            **LOW:** Fatalities + Injuries ≤ 2

            **MEDIUM:** Fatalities + Injuries between 3 and 10

            **HIGH:** Fatalities + Injuries > 10
            """
        )
