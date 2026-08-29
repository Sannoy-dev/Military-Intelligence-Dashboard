import os
import traceback
import pandas as pd
import streamlit as st
from utils.ui import load_css
from utils.model_trainer import (
    load_custom_model,
    load_model_metadata,
    custom_model_exists,
    predict_attack,
    get_missing_columns,
    ATTACK_FEATURE_COLUMNS,
    ATTACK_TARGET_COLUMN
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Attack Prediction",
    layout="wide"
)

load_css()


# ============================================================
# PATH CONFIGURATION
# ============================================================

DATA_DIR = "data"

ACTIVE_DATASET = os.path.join(
    DATA_DIR,
    "custom_dataset.csv"
)


# ============================================================
# ATTACK MODEL REQUIREMENTS
# ============================================================

ATTACK_FEATURE_COLUMNS = [
    "country_txt",
    "region_txt",
    "weaptype1_txt",
    "targtype1_txt",
    "gname",
    "success",
    "suicide",
    "nkill",
    "nwound"
]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Attack Type Prediction</h1>
        <p>
              Locally trained machine-learning classification
                                            using the currently active intelligence dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ARCHITECTURE NOTICE
# ============================================================

st.info(
    """
    **Local Dataset Architecture**

    This module uses the dataset prepared through the main
    dashboard. Predictions are available only after a compatible
    dataset has been mapped, saved, and used to train the local
    Attack Prediction model.
    """
)


# ============================================================
# LOAD ACTIVE DATASET
# ============================================================

if not os.path.isfile(ACTIVE_DATASET):

    st.warning(
        """
        No active dataset is available.

        Upload and process a CSV dataset from the main dashboard
        before using Attack Prediction.
        """
    )

    st.stop()


try:

    with st.spinner(
        "Loading active intelligence dataset..."
    ):

        df = pd.read_csv(
            ACTIVE_DATASET,
            low_memory=False
        )

except Exception as error:

    st.error(
        f"Unable to load the active dataset: {error}"
    )

    st.stop()


# ============================================================
# NO DATASET
# ============================================================

if df is None or df.empty:

    st.warning(
        """
        The active dataset contains no records.

        Upload and save a valid dataset from the main dashboard.
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
        "Prediction Model",
        "Available"
        if custom_model_exists()
        else "Not Trained"
    )


# ============================================================
# DATASET COMPATIBILITY
# ============================================================

missing_columns = [
    column
    for column in ATTACK_FEATURE_COLUMNS
    if column not in df.columns
]


if missing_columns:

    st.error(
        "The active dataset cannot currently support "
        "Attack Prediction."
    )

    st.markdown(
        "### Missing Training Columns"
    )

    st.write(
        """
        The standardized dataset is missing one or more
        fields required by the Attack Prediction model.
        """
    )

    missing_display = "\n".join(
        missing_columns
    )

    st.code(
        missing_display,
        language="text",
        unsafe_allow_html=True
    )

    st.info(
        """
        Return to the main dashboard and verify the column
        mapping. The missing fields must be available before
        the Attack Prediction model can be trained.
        """
    )

    st.stop()


# ============================================================
# MODEL AVAILABILITY
# ============================================================

if not custom_model_exists():

    st.warning(
        """
        The Attack Prediction model has not been trained.

        The active dataset contains the required fields, but
        no complete locally trained model is currently available.
        """
    )

    st.markdown("Required Workflow")

    workflow_1, workflow_2, workflow_3 = st.columns(
        3,
        gap="large"
    )

    with workflow_1:

        st.markdown(
            """
            <div class="chart-card workflow-card">

                <h3>01</h3>

                <h4>Upload Dataset</h4>

                <p>
                    Upload a compatible CSV dataset through
                    the main dashboard.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with workflow_2:

        st.markdown(
            """
            <div class="chart-card workflow-card">

                <h3>02</h3>

                <h4>Map Columns</h4>

                <p>
                    Map uploaded columns to the application's
                    standardized fields.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with workflow_3:

        st.markdown(
            """
            <div class="chart-card workflow-card">

                <h3>03</h3>

                <h4>Train Locally</h4>

                <p>
                    Train the Attack Prediction model using
                    the processed dataset.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.stop()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:
    with st.spinner(
        "Loading trained model..."
    ):

     model_data = load_custom_model()

except Exception as error:

    st.error(
        f"Unable to load the trained model: {error}"
    )

    st.stop()


if model_data is None:

    st.error(
        """
        The Attack Prediction model files are incomplete,
        missing, or corrupted.

        Retrain the model from the main dashboard.
        """
    )

    st.stop()


# ============================================================
# LOAD MODEL METADATA
# ============================================================
with st.spinner(
        "Loading model metadata..."
    ):
      metadata = load_model_metadata()


st.success(
    "Locally trained Attack Prediction model is active."
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

        training_records = metadata.get(
            "training_records",
            0
        )

        st.metric(
            "Training Records",
            f"{training_records:,}"
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
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        else:

            st.metric(
                "Accuracy",
                "N/A"
            )

    else:

        st.metric(
            "Accuracy",
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
with st.spinner(
        "Preparing to display data..."
    ):
      prediction_df = df.copy()

categorical_columns = [
    "country_txt",
    "region_txt",
    "weaptype1_txt",
    "targtype1_txt",
    "gname"
]


for column in categorical_columns:

    if column in prediction_df.columns:

        prediction_df[column] = (
            prediction_df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )


# ============================================================
# SELECTBOX OPTIONS
# ============================================================

def get_options(column):

    if column not in prediction_df.columns:

        return ["Unknown"]

    values = (
        prediction_df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    values = sorted(
        values.unique()
    )

    if not values:

        return ["Unknown"]

    return values


# ============================================================
# PREDICTION FORM
# ============================================================

st.markdown("Incident Parameters")

st.caption(
    "Enter incident characteristics to classify the probable "
    "attack type using the locally trained model."
)


with st.form(
    "attack_prediction_form"
):

    left, right = st.columns(
        2,
        gap="large"
    )

    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with left:

        country = st.selectbox(
            "Country",
            get_options(
                "country_txt"
            )
        )

        region = st.selectbox(
            "Region",
            get_options(
                "region_txt"
            )
        )

        weapon = st.selectbox(
            "Weapon Type",
            get_options(
                "weaptype1_txt"
            )
        )

        target = st.selectbox(
            "Target Type",
            get_options(
                "targtype1_txt"
            )
        )

    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with right:

        group = st.selectbox(
            "Terrorist Organization",
            get_options(
                "gname"
            )
        )

        success = st.selectbox(
            "Successful Attack",
            [0, 1],
            format_func=lambda value:
                "Yes" if value == 1 else "No"
        )

        suicide = st.selectbox(
            "Suicide Attack",
            [0, 1],
            format_func=lambda value:
                "Yes" if value == 1 else "No"
        )

        nkill = st.number_input(
            "Fatalities",
            min_value=0,
            value=0,
            step=1
        )

        nwound = st.number_input(
            "Injuries",
            min_value=0,
            value=0,
            step=1
        )


    predict_button = st.form_submit_button(
        "Generate Attack Prediction",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = {

        "country_txt": country,

        "region_txt": region,

        "weaptype1_txt": weapon,

        "targtype1_txt": target,

        "gname": group,

        "success": success,

        "suicide": suicide,

        "nkill": nkill,

        "nwound": nwound
    }
    result = None


    try:

        with st.spinner(
            "Analyzing incident characteristics..."
        ):

            result = predict_attack(
                model_data,
                input_data
            )

    except Exception:

         st.exception(traceback.format_exc())
    # ========================================================
    # RESULT DATA
    # ========================================================
    if result:
    
     attack_type = result.get(
        "prediction",
        "Unknown"
    )

    confidence = result.get(
        "confidence"
    )

    probabilities = result.get(
        "probabilities"
    )


    # ========================================================
    # RESULT SECTION
    # ========================================================

    st.markdown(
        "<div class='result-divider'></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "## Prediction Result"
    )


    result_left, result_right = st.columns(
        [2, 1],
        gap="large"
    )


    with result_left:

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    PREDICTED ATTACK TYPE
                </div>

                <div class="prediction-value">
                    {attack_type}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with result_right:

        if confidence is not None:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        else:

            st.metric(
                "Confidence",
                "N/A"
            )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    if confidence is not None:

        st.markdown(
            "### Prediction Confidence"
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


    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    if probabilities:

        st.markdown(
            "### Attack-Type Probability Distribution"
        )

        probability_df = pd.DataFrame(
            {
                "Attack Type":
                    list(
                        probabilities.keys()
                    ),

                "Probability (%)":
                    list(
                        probabilities.values()
                    )
            }
        )

        probability_df = (
            probability_df
            .sort_values(
                "Probability (%)",
                ascending=False
            )
            .reset_index(drop=True)
        )

        st.dataframe(
            probability_df,
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

    with s1:

        st.metric(
            "Country",
            country
        )

    with s2:

        st.metric(
            "Weapon",
            weapon
        )

    with s3:

        st.metric(
            "Fatalities",
            f"{nkill:,}"
        )

    with s4:

        st.metric(
            "Injuries",
            f"{nwound:,}"
        )


    # ========================================================
    # AI ASSESSMENT
    # ========================================================

    st.markdown(
        "### AI Assessment"
    )

    st.markdown(
        f"""
        <div class="assessment-card">

            <h3>Classification Summary</h3>

            <p>
                The locally trained Random Forest model
                classified the provided incident as
                <strong>{attack_type}</strong>.
            </p>

            <p>
                The classification is based on the features
                available in the active uploaded dataset and
                the data used during local model training.
            </p>

            <p>
                This prediction is analytical decision-support
                information and should not be treated as a
                definitive assessment without independent
                verification and human analysis.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATASET REQUIREMENTS
# ============================================================

st.markdown(
    "## Model Requirements"
)

st.caption(
    "The currently trained Attack Prediction model expects "
    "the following input features:"
)

st.code(
    "\n".join(
        ATTACK_FEATURE_COLUMNS
    ),
    language="text",
)

st.caption(
    "The prediction target is attacktype1_txt. "
    "A future uploaded dataset must contain the required "
    "training fields before a new Attack Prediction model "
    "can be trained."
)

