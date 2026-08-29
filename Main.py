# ============================================================
# main.py
#
# AI Military Intelligence Dashboard
#
# Responsibilities:
#   • Dataset upload
#   • Dataset mapping UI
#   • Dataset standardization
#   • Dataset validation
#   • Model compatibility
#   • Local model training
#
# Column mapping / validation logic is handled by:
#   utils.data_mapper
#
# Model training is handled by:
#   utils.model_trainer
#
# UI styling is handled by:
#   utils.ui
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import traceback
import pandas as pd
import streamlit as st

from utils.ui import load_css

from utils.data_mapper import (
    STANDARD_COLUMNS,
    REQUIRED_ANALYSIS_COLUMNS,
    ATTACK_PREDICTION_COLUMNS,
    THREAT_LEVEL_COLUMNS,

    auto_map_columns,
    get_mapping_table,
    get_mapped_columns,
    get_unmapped_columns,
    get_missing_standard_columns,

    get_required_columns,
    get_missing_training_columns,
    check_training_eligibility,

    apply_mapping,
    prepare_dataset,

    validate_dataset,
    validate_model_dataset,

    get_dataset_summary,
    get_mapping_statistics,
    get_dataset_report,
    get_model_readiness,
)

from utils.model_trainer import (
    train_all_models
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Military Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()


# ============================================================
# PATH CONFIGURATION
# ============================================================

DATA_DIR = "data"

ACTIVE_DATASET = os.path.join(
    DATA_DIR,
    "custom_dataset.csv",
)

CUSTOM_MODEL_DIR = os.path.join(
    "models",
    "custom",
)


# ============================================================
# DISPLAY NAMES
#
# These are UI labels only.
#
# The actual internal application schema comes from
# utils.data_mapper.STANDARD_COLUMNS.
# ============================================================

COLUMN_DISPLAY_NAMES = {

    "iyear":
        "Year",

    "country_txt":
        "Country",

    "region_txt":
        "Region",

    "city":
        "City",

    "latitude":
        "Latitude",

    "longitude":
        "Longitude",

    "attacktype1_txt":
        "Attack Type",

    "weaptype1_txt":
        "Weapon Type",

    "targtype1_txt":
        "Target Type",

    "gname":
        "Terrorist Organization",

    "success":
        "Successful Attack",

    "suicide":
        "Suicide Attack",

    "nkill":
        "Fatalities",

    "nwound":
        "Injuries",
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION_STATE = {

    "uploaded_filename":
        None,

    "uploaded_df":
        None,

    "mapping":
        None,

    "mapping_saved":
        False,

    "training_results":
        None,
}


for key, default_value in DEFAULT_SESSION_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = default_value


# ============================================================
# SAFE CSV READER
# ============================================================

def read_csv_safely(uploaded_file):
    """
    Read a CSV file using several common encodings.

    The previous implementation attempted multiple encodings
    but always passed utf-8 to pandas. This implementation
    correctly tries each encoding.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:

        try:

            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding=encoding,
                low_memory=False,
            )

        except UnicodeDecodeError as error:

            last_error = error

        except pd.errors.ParserError:

            raise

    raise ValueError(
        "Unable to decode the CSV file. "
        "The file may use an unsupported character encoding."
    ) from last_error


# ============================================================
# REMOVE OLD CUSTOM MODELS
# ============================================================

def remove_old_models():
    """
    Remove previously generated custom models.

    Only files inside models/custom are removed.
    """

    if not os.path.exists(
        CUSTOM_MODEL_DIR
    ):

        return

    for filename in os.listdir(
        CUSTOM_MODEL_DIR
    ):

        filepath = os.path.join(
            CUSTOM_MODEL_DIR,
            filename,
        )

        if os.path.isfile(
            filepath
        ):

            try:

                os.remove(
                    filepath
                )

            except OSError:

                pass


# ============================================================
# GET DISPLAY NAME
# ============================================================

def get_display_name(column):
    """
    Convert an internal standard column name into a
    user-friendly display name.
    """

    return COLUMN_DISPLAY_NAMES.get(
        column,
        column.replace(
            "_",
            " ",
        ).title(),
    )


# ============================================================
# GET UPLOADED COLUMN OPTIONS
# ============================================================

def get_column_options(df):
    """
    Return column options used by the mapping interface.
    """

    if df is None:

        return [
            "Unmapped"
        ]

    return [
        "Unmapped",
        *list(df.columns),
    ]


# ============================================================
# BUILD UI MAPPING
# ============================================================

def build_ui_mapping(
    dataframe,
    automatic_mapping,
):
    """
    Display the mapping interface and return the user's
    selected mapping.

    Internal format:

        {
            uploaded_column: standard_column
        }

    This format is directly compatible with
    apply_mapping().
    """

    if dataframe is None:

        return {}

    options = get_column_options(
        dataframe
    )

    selected_mapping = {}

    used_uploaded_columns = set()

    # --------------------------------------------------------
    # Mapping controls
    # --------------------------------------------------------

    for standard_column in STANDARD_COLUMNS:

        display_name = get_display_name(
            standard_column
        )

        # ----------------------------------------------------
        # Find automatically suggested uploaded column
        # ----------------------------------------------------

        automatic_uploaded_column = None

        if automatic_mapping:

            for (
                uploaded_column,
                mapping_info,
            ) in automatic_mapping.items():

                if (
                    mapping_info.get(
                        "mapped_to"
                    )
                    == standard_column
                ):

                    automatic_uploaded_column = (
                        uploaded_column
                    )

                    break

        # ----------------------------------------------------
        # Prevent duplicate defaults
        # ----------------------------------------------------

        if (
            automatic_uploaded_column
            in used_uploaded_columns
        ):

            automatic_uploaded_column = None

        if (
            automatic_uploaded_column
            in options
        ):

            default_index = options.index(
                automatic_uploaded_column
            )

        else:

            default_index = 0

        # ----------------------------------------------------
        # Selectbox
        # ----------------------------------------------------

        selected = st.selectbox(
            display_name,
            options,
            index=default_index,
            key=(
                f"column_mapping_"
                f"{standard_column}"
            ),
        )

        # ----------------------------------------------------
        # Store mapping in uploaded -> standard format
        # ----------------------------------------------------

        if selected != "Unmapped":

            if selected in used_uploaded_columns:

                st.warning(
                    f"'{selected}' is already assigned "
                    f"to another standard field."
                )

            else:

                selected_mapping[
                    selected
                ] = standard_column

                used_uploaded_columns.add(
                    selected
                )

    return selected_mapping


# ============================================================
# SHOW MAPPING SUMMARY
# ============================================================

def show_mapping_summary(
    dataframe,
    mapping,
):
    """
    Display mapping statistics and unmapped columns.
    """

    if dataframe is None:

        return

    total_columns = len(
        dataframe.columns
    )

    mapped_count = len(
        mapping
    )

    unmapped_count = (
        total_columns
        - mapped_count
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Uploaded Columns",
            total_columns,
        )

    with col2:

        st.metric(
            "Mapped Columns",
            mapped_count,
        )

    with col3:

        st.metric(
            "Unmapped Columns",
            unmapped_count,
        )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    progress = (
        mapped_count / total_columns
        if total_columns > 0
        else 0
    )

    st.progress(
        progress
    )

    st.caption(
        f"{mapped_count} of "
        f"{total_columns} uploaded columns "
        f"are currently mapped."
    )

    # --------------------------------------------------------
    # Additional columns
    # --------------------------------------------------------

    extra_columns = [
        column
        for column in dataframe.columns
        if column not in mapping
    ]

    if extra_columns:

        with st.expander(
            f"Additional / Unmapped Columns "
            f"({len(extra_columns)})"
        ):

            st.write(
                extra_columns
            )

            st.caption(
                "These columns are retained in the uploaded "
                "dataset but are not part of the application's "
                "standardized schema."
            )


# ============================================================
# SHOW MODEL COMPATIBILITY
# ============================================================

def show_model_compatibility(
    dataframe,
):
    """
    Display compatibility information for every supported
    machine-learning model.
    """

    if dataframe is None:

        return

    st.markdown(
        "## Model Compatibility"
    )

    # --------------------------------------------------------
    # Attack Prediction
    # --------------------------------------------------------

    attack_missing = (
        get_missing_training_columns(
            dataframe,
            "attack_prediction",
        )
    )

    if not attack_missing:

        attack_valid, attack_issues = (
            validate_model_dataset(
                dataframe,
                "attack_prediction",
            )
        )

        if attack_valid:

            st.success(
                "Attack Prediction: "
                "Dataset is ready for training."
            )

        else:

            st.warning(
                "Attack Prediction: "
                "Required columns exist, but the dataset "
                "has validation issues."
            )

            for issue in attack_issues:

                st.caption(
                    str(issue)
                )

    else:

        st.error(
            "Attack Prediction: "
            "Required fields are missing."
        )

        missing_display = [
            get_display_name(
                column
            )
            for column in attack_missing
        ]

        st.caption(
            "Missing: "
            + ", ".join(
                missing_display
            )
        )

    # --------------------------------------------------------
    # Threat Level
    # --------------------------------------------------------

    threat_missing = (
        get_missing_training_columns(
            dataframe,
            "threat_level",
        )
    )

    if not threat_missing:

        threat_valid, threat_issues = (
            validate_model_dataset(
                dataframe,
                "threat_level",
            )
        )

        if threat_valid:

            st.success(
                "Threat Level Prediction: "
                "Dataset is ready for training."
            )

        else:

            st.warning(
                "Threat Level Prediction: "
                "Required columns exist, but the dataset "
                "has validation issues."
            )

            for issue in threat_issues:

                st.caption(
                    str(issue)
                )

    else:

        st.error(
            "Threat Level Prediction: "
            "Required fields are missing."
        )

        missing_display = [
            get_display_name(
                column
            )
            for column in threat_missing
        ]

        st.caption(
            "Missing: "
            + ", ".join(
                missing_display
            )
        )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Military Intelligence Dashboard</h1>
        <p>
            Upload an intelligence dataset, map its
                                fields to the standardized schema, train
                                compatible machine-learning models locally,
                                and perform analytical exploration.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DATASET NOTICE
# ============================================================

st.info(
    """
    **Custom Dataset Mode**

    This dashboard operates on the dataset uploaded by the
    user. Dataset mapping, cleaning, standardization and
    machine-learning preparation are performed locally.
    """
)


# ============================================================
# DATASET UPLOAD
# ============================================================

st.markdown(
    "## Upload Dataset"
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help=(
        "Upload the dataset you want to analyze. "
        "The application will automatically detect "
        "compatible columns."
    ),
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    if (
        st.session_state.uploaded_filename
        != uploaded_file.name
    ):

        try:

            with st.spinner(
                "Reading uploaded dataset..."
            ):

                uploaded_df = (
                    read_csv_safely(
                        uploaded_file
                    )
                )

            # ------------------------------------------------
            # Save uploaded dataset in session state
            # ------------------------------------------------

            st.session_state.uploaded_filename = (
                uploaded_file.name
            )

            st.session_state.uploaded_df = (
                uploaded_df
            )

            # ------------------------------------------------
            # Generate automatic mapping using the new
            # centralized mapper.
            #
            # Format:
            #
            # {
            #     "Year": {
            #         "mapped_to": "iyear",
            #         ...
            #     }
            # }
            # ------------------------------------------------

            st.session_state.mapping = (
                auto_map_columns(
                    uploaded_df
                )
            )

            st.session_state.mapping_saved = (
                False
            )

            st.session_state.training_results = (
                None
            )

            st.success(
                "Dataset loaded successfully: "
                f"{len(uploaded_df):,} rows × "
                f"{len(uploaded_df.columns)} columns"
            )

        except Exception as error:

            st.error(
                f"Unable to read the CSV file: {error}"
            )

            st.stop()


# ============================================================
# MAPPING INTERFACE
# ============================================================

if st.session_state.uploaded_df is not None:

    uploaded_df = (
        st.session_state.uploaded_df
    )

    st.markdown(
        "## Column Mapping"
    )

    st.caption(
        """
        The application automatically detects compatible
        columns using the centralized dataset mapper.
        Review the suggested mapping before saving the
        standardized dataset.
        """
    )

    # --------------------------------------------------------
    # Get automatic mapping
    # --------------------------------------------------------

    automatic_mapping = (
        st.session_state.mapping
        or auto_map_columns(
            uploaded_df
        )
    )

    # --------------------------------------------------------
    # UI mapping
    #
    # uploaded column -> standard column
    # --------------------------------------------------------

    mapping = build_ui_mapping(
        uploaded_df,
        automatic_mapping,
    )

    # --------------------------------------------------------
    # Mapping statistics
    # --------------------------------------------------------

    show_mapping_summary(
        uploaded_df,
        mapping,
    )

    # --------------------------------------------------------
    # Preview mapping
    # --------------------------------------------------------

    if mapping:

        with st.expander(
            "Preview Selected Mapping"
        ):

            preview_rows = []

            for (
                uploaded_column,
                standard_column,
            ) in mapping.items():

                preview_rows.append(
                    {
                        "Uploaded Column":
                            uploaded_column,

                        "Mapped To":
                            get_display_name(
                                standard_column
                            ),

                        "Standard Name":
                            standard_column,
                    }
                )

            mapping_preview = pd.DataFrame(
                preview_rows
            )

            st.dataframe(
                mapping_preview,
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # DATASET PREPARATION
    # ========================================================

    st.markdown(
        "### Dataset Preparation"
    )

    if st.button(
        "Apply Mapping and Save Dataset",
        use_container_width=True,
        type="primary",
    ):

        if not mapping:

            st.error(
                "No columns have been mapped. "
                "Map at least the required dataset fields "
                "before saving."
            )

        else:

            try:

                with st.spinner(
                    "Standardizing dataset..."
                ):

                    # ----------------------------------------
                    # Convert uploaded -> standard mapping
                    # using centralized mapper.
                    # ----------------------------------------
                    st.write("1")

                    standardized_df = (
                        apply_mapping(
                            uploaded_df,
                            mapping,
                        )
                    )

                    # ----------------------------------------
                    # Perform centralized cleaning.
                    #
                    # prepare_dataset() automatically:
                    #   • keeps recognized columns
                    #   • converts numeric values
                    #   • handles missing impact values
                    #   • cleans categorical values
                    # ----------------------------------------
                    st.write("2")
                    st.write(uploaded_df.columns.tolist())
                    standardized_df = (
                        prepare_dataset(
                            uploaded_df,
                            mapping,
                        )
                    )

                # ------------------------------------------------
                # Validate general dashboard requirements
                # ------------------------------------------------
                st.write("3")

                analysis_valid, analysis_issues = (
                    validate_dataset(
                        standardized_df
                    )
                )

                # ------------------------------------------------
                # Create data directory
                # ------------------------------------------------

                os.makedirs(
                    DATA_DIR,
                    exist_ok=True,
                )

                # ------------------------------------------------
                # Save standardized dataset
                # ------------------------------------------------
                st.write("4")

                standardized_df.to_csv(
                    ACTIVE_DATASET,
                    index=False,
                    encoding="utf-8",
                )

                # ------------------------------------------------
                # New dataset means old custom models are
                # no longer guaranteed to be compatible.
                # ------------------------------------------------

                remove_old_models()

                # ------------------------------------------------
                # Update session state
                # ------------------------------------------------

                st.session_state.mapping = (
                    auto_map_columns(
                        uploaded_df
                    )
                )

                st.session_state.mapping_saved = (
                    True
                )

                st.session_state.training_results = (
                    None
                )

                # ------------------------------------------------
                # Clear Streamlit caches
                # ------------------------------------------------

                try:

                    st.cache_data.clear()

                except Exception:

                    pass

                try:

                    st.cache_resource.clear()

                except Exception:

                    pass

                # ------------------------------------------------
                # Success
                # ------------------------------------------------

                st.success(
                    "Dataset successfully standardized "
                    "and saved."
                )

                if not analysis_valid:

                    st.warning(
                        "The dataset was saved, but it does "
                        "not currently satisfy all general "
                        "dashboard requirements."
                    )

                    for issue in analysis_issues:

                        st.caption(
                            str(issue)
                        )

                st.rerun()

            except Exception as error:
                st.exception(error)


# ============================================================
# ACTIVE DATASET
# ============================================================

if os.path.exists(
    ACTIVE_DATASET
):

    try:

        active_df = pd.read_csv(
            ACTIVE_DATASET,
            encoding="utf-8",
            low_memory=False,
        )

        st.markdown(
            "## Active Dataset"
        )

        # ----------------------------------------------------
        # Dataset metrics
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Records",
                f"{len(active_df):,}",
            )

        with c2:

            st.metric(
                "Columns",
                len(active_df.columns),
            )

        with c3:

            st.metric(
                "Mapped Fields",
                len(
                    get_mapped_columns(
                        active_df
                    )
                ),
            )

        with c4:

            st.metric(
                "Dataset Type",
                "Custom",
            )

        source_name = (
            st.session_state.uploaded_filename
            or "Saved dataset"
        )

        st.caption(
            f"Source file: {source_name}"
        )

        # ----------------------------------------------------
        # Dataset preview
        # ----------------------------------------------------

        with st.expander(
            "Preview Active Dataset"
        ):

            st.dataframe(
                active_df.head(100),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:

        st.error(
            f"Unable to load active dataset: {error}"
        )

        active_df = None

else:

    active_df = None


# ============================================================
# DATASET VALIDATION
# ============================================================

if active_df is not None:

    analysis_valid, analysis_issues = (
        validate_dataset(
            active_df
        )
    )

    if analysis_valid:

        st.success(
            "Dataset satisfies the minimum "
            "dashboard requirements."
        )

    else:

        st.warning(
            "Dataset does not currently satisfy all "
            "general dashboard requirements."
        )

        if analysis_issues:

            st.caption(
                "Issues: "
                + ", ".join(
                    str(issue)
                    for issue in analysis_issues
                )
            )


# ============================================================
# DATASET SUMMARY
# ============================================================

if active_df is not None:

    with st.expander(
        "Dataset Mapping Summary"
    ):

        summary = (
            get_dataset_summary(
                active_df
            )
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:

            st.metric(
                "Rows",
                f"{summary['rows']:,}",
            )

        with s2:

            st.metric(
                "Uploaded Columns",
                summary[
                    "uploaded_columns"
                ],
            )

        with s3:

            st.metric(
                "Mapped Columns",
                summary[
                    "mapped_columns"
                ],
            )

        with s4:

            st.metric(
                "Mapping Coverage",
                (
                    f"{summary['mapping_percentage']:.1f}%"
                ),
            )

        # ----------------------------------------------------
        # Missing standard columns
        # ----------------------------------------------------

        missing_standard = (
            summary[
                "missing_standard_columns"
            ]
        )

        if missing_standard:

            st.markdown(
                "**Optional Standard Fields Missing**"
            )

            st.write(
                [
                    get_display_name(
                        column
                    )
                    for column in missing_standard
                ]
            )


# ============================================================
# MODEL COMPATIBILITY
# ============================================================

if active_df is not None:

    show_model_compatibility(
        active_df
    )


# ============================================================
# LOCAL MODEL TRAINING
# ============================================================

st.markdown(
    "## Local Model Training"
)

st.markdown(
    """
    <div class="report-card">

    <h3>Train on This Computer</h3>

    <p>
        Training is performed locally using the active
                standardized dataset. Generated custom models
                are stored inside the
                <strong>models/custom</strong> directory.
    </p>

    <p>
       
            Attack Prediction and Threat Level Prediction
            are trained independently. If one model is not
            compatible with the dataset, the other model can
            still be trained.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TRAINING BUTTON
# ============================================================

if active_df is None:

    st.warning(
        "Upload and save a dataset before training."
    )

else:

    if st.button(
        "Train Available Models Locally",
        use_container_width=True,
        type="primary",
    ):

        try:

            progress = st.progress(
                0
            )

            status_text = st.empty()

            status_text.info(
                "Starting local model training..."
            )

            # ------------------------------------------------
            # Train all supported models
            # ------------------------------------------------

            results = train_all_models(
                ACTIVE_DATASET
            )

            progress.progress(
                100
            )

            status_text.empty()

            # ------------------------------------------------
            # Validate trainer response
            # ------------------------------------------------

            if not isinstance(
                results,
                dict,
            ):

                st.error(
                    "The model trainer returned "
                    "an invalid result."
                )

                st.stop()

            # ------------------------------------------------
            # Save results
            # ------------------------------------------------

            st.session_state.training_results = (
                results
            )

            overall_success = results.get(
                "success",
                False,
            )

            overall_message = results.get(
                "message",
                "Training process completed.",
            )

            # ------------------------------------------------
            # Results section
            # ------------------------------------------------

            st.markdown(
                "### Training Results"
            )

            attack_result = results.get(
                "attack_prediction"
            )

            threat_result = results.get(
                "threat_level"
            )

            # =================================================
            # ATTACK PREDICTION RESULT
            # =================================================

            if isinstance(
                attack_result,
                dict,
            ):

                if attack_result.get(
                    "success",
                    False,
                ):

                    st.success(
                        "Attack Prediction model "
                        "trained successfully."
                    )

                    accuracy = (
                        attack_result.get(
                            "accuracy"
                        )
                    )

                    if accuracy is not None:

                        st.caption(
                            f"Accuracy: "
                            f"{accuracy * 100:.2f}%"
                        )

                else:

                    st.warning(
                        "Attack Prediction model "
                        "was not trained."
                    )

                    problems = (
                        attack_result.get(
                            "problems",
                            [],
                        )
                    )

                    if problems:

                        for problem in problems:

                            st.caption(
                                str(problem)
                            )

                    else:

                        st.caption(
                            attack_result.get(
                                "message",
                                "Training requirements "
                                "were not satisfied.",
                            )
                        )

            # =================================================
            # THREAT LEVEL RESULT
            # =================================================

            if isinstance(
                threat_result,
                dict,
            ):

                if threat_result.get(
                    "success",
                    False,
                ):

                    st.success(
                        "Threat Level model "
                        "trained successfully."
                    )

                    accuracy = (
                        threat_result.get(
                            "accuracy"
                        )
                    )

                    if accuracy is not None:

                        st.caption(
                            f"Accuracy: "
                            f"{accuracy * 100:.2f}%"
                        )

                else:

                    st.warning(
                        "Threat Level model "
                        "was not trained."
                    )

                    problems = (
                        threat_result.get(
                            "problems",
                            [],
                        )
                    )

                    if problems:

                        for problem in problems:

                            st.caption(
                                str(problem)
                            )

                    else:

                        st.caption(
                            threat_result.get(
                                "message",
                                "Training requirements "
                                "were not satisfied.",
                            )
                        )

            # =================================================
            # OVERALL STATUS
            # =================================================

            if overall_success:

                st.info(
                    overall_message
                )

            else:

                st.warning(
                    overall_message
                )

            # ------------------------------------------------
            # Clear cached model resources
            # ------------------------------------------------

            try:

                st.cache_resource.clear()

            except Exception:

                pass

        except ImportError:

            st.error(
                "The model trainer could not be imported. "
                "Check utils/model_trainer.py and ensure "
                "train_all_models() is available."
            )

        except Exception as error:

            st.error(
                f"Training process failed: {error}"
            )


# ============================================================
# TRAINING RESULT PERSISTENCE
# ============================================================

if (
    st.session_state.training_results
    is not None
):

    results = (
        st.session_state.training_results
    )

    with st.expander(
        "View Last Training Response"
    ):

        st.json(
            results
        )


# ============================================================
# WORKFLOW
# ============================================================

st.markdown(
    "## Workflow"
)

w1, w2, w3, w4 = st.columns(4)

workflow_cards = [

    (
        w1,
        "01",
        "Upload",
        "Upload the CSV dataset you want to analyze.",
    ),

    (
        w2,
        "02",
        "Map",
        "Map uploaded fields to the standardized schema.",
    ),

    (
        w3,
        "03",
        "Train",
        "Train compatible machine-learning models locally.",
    ),

    (
        w4,
        "04",
        "Analyze",
        "Use predictions, maps, reports and data exploration.",
    ),
]


for (
    column,
    number,
    title,
    description,
) in workflow_cards:

    with column:
        st.markdown(
         f"""
         <div class="report-card">
         <div>
                 {number}
                  </div>
         <h4>{title}</h4>

         <p>
          {description}
         </p>


         


         </div>
         """,
         unsafe_allow_html=True
        )



# ============================================================
# SYSTEM NOTICE
# ============================================================

st.divider()

st.caption(
    "AI Military Intelligence Dashboard | "
    "Custom Dataset Mode | "
    "Centralized Dataset Mapping | "
    "Local Processing | "
    "Local Machine Learning"
)