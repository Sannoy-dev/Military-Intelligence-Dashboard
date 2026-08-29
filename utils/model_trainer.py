"""
utils/model_trainer.py

Central machine-learning utilities for the
AI Military Intelligence Dashboard.

Responsibilities
----------------
• Attack Prediction model
• Threat Level Prediction model
• Dataset preparation and validation
• Model preprocessing
• Model training
• Model persistence
• Model loading
• Prediction utilities
• Prediction probabilities
• Model metadata
• Model status

The module expects the dataset to use the standardized
schema produced by utils.dataset_mapper.

The module can also work with a dataframe that already
contains the required standardized columns.

Author:
    AI Military Intelligence Dashboard

Version:
    3.0
"""


# ============================================================
# STANDARD LIBRARY
# ============================================================
import streamlit as st
import json
import os
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# THIRD-PARTY LIBRARIES
# ============================================================

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
)


# ============================================================
# OPTIONAL DATASET MAPPER IMPORT
# ============================================================

try:
    from utils.data_mapper import STANDARD_COLUMNS
except ImportError:
    STANDARD_COLUMNS = []


# ============================================================
# MODEL DIRECTORY CONFIGURATION
# ============================================================

BASE_MODEL_DIRECTORY = os.path.join(
    "models",
    "custom",
)

os.makedirs(
    BASE_MODEL_DIRECTORY,
    exist_ok=True,
)


# ============================================================
# ATTACK MODEL FILES
# ============================================================

ATTACK_MODEL_FILE = os.path.join(
    BASE_MODEL_DIRECTORY,
    "attack_prediction_model.joblib",
)

ATTACK_LABEL_ENCODER_FILE = os.path.join(
    BASE_MODEL_DIRECTORY,
    "attack_prediction_label_encoder.joblib",
)

ATTACK_METADATA_FILE = os.path.join(
    BASE_MODEL_DIRECTORY,
    "attack_prediction_metadata.json",
)


# ============================================================
# THREAT MODEL FILES
# ============================================================

THREAT_MODEL_FILE = os.path.join(
    BASE_MODEL_DIRECTORY,
    "threat_level_model.joblib",
)

THREAT_METADATA_FILE = os.path.join(
    BASE_MODEL_DIRECTORY,
    "threat_level_metadata.json",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_VERSION = "3.0"

RANDOM_STATE = 42

ATTACK_N_ESTIMATORS = 250

THREAT_N_ESTIMATORS = 250

TEST_SIZE = 0.20

MINIMUM_TRAINING_RECORDS = 20

MINIMUM_TARGET_CLASSES = 2

MINIMUM_CLASS_RECORDS = 2


# ============================================================
# ATTACK PREDICTION FEATURES
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
    "nwound",
]

ATTACK_TARGET_COLUMN = "attacktype1_txt"


ATTACK_CATEGORICAL_COLUMNS = [
    "country_txt",
    "region_txt",
    "weaptype1_txt",
    "targtype1_txt",
    "gname",
]


ATTACK_NUMERIC_COLUMNS = [
    "success",
    "suicide",
    "nkill",
    "nwound",
]


# ============================================================
# THREAT LEVEL FEATURES
# ============================================================

THREAT_FEATURE_COLUMNS = [
    "country_txt",
    "region_txt",
    "attacktype1_txt",
    "weaptype1_txt",
    "targtype1_txt",
    "nkill",
    "nwound",
]

THREAT_TARGET_COLUMN = "Threat_Level"


THREAT_CATEGORICAL_COLUMNS = [
    "country_txt",
    "region_txt",
    "attacktype1_txt",
    "weaptype1_txt",
    "targtype1_txt",
]


THREAT_NUMERIC_COLUMNS = [
    "nkill",
    "nwound",
]


# ============================================================
# HELPER: ONE-HOT ENCODER
# ============================================================

def _create_one_hot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder compatible with multiple
    scikit-learn versions.

    Newer versions use:
        sparse_output=True

    Older versions use:
        sparse=True
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


# ============================================================
# MODEL EXISTENCE CHECKS
# ============================================================

def custom_model_exists() -> bool:
    """
    Check whether the Attack Prediction model and
    its label encoder are available.
    """

    return (
        os.path.isfile(ATTACK_MODEL_FILE)
        and
        os.path.isfile(ATTACK_LABEL_ENCODER_FILE)
    )


def threat_model_exists() -> bool:
    """
    Check whether the Threat Level Prediction model
    is available.
    """

    return os.path.isfile(
        THREAT_MODEL_FILE
    )


def attack_metadata_exists() -> bool:
    """
    Check whether Attack Prediction metadata exists.
    """

    return os.path.isfile(
        ATTACK_METADATA_FILE
    )


def threat_metadata_exists() -> bool:
    """
    Check whether Threat Level metadata exists.
    """

    return os.path.isfile(
        THREAT_METADATA_FILE
    )


# ============================================================
# COLUMN VALIDATION
# ============================================================

def get_missing_columns(
    dataframe: Optional[pd.DataFrame],
    required_columns: List[str],
) -> List[str]:
    """
    Return columns required by a model but missing
    from the supplied dataframe.
    """

    if dataframe is None:
        return list(required_columns)

    return [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]


def validate_model_columns(
    dataframe: Optional[pd.DataFrame],
    feature_columns: List[str],
) -> Dict[str, Any]:
    """
    Validate whether a dataframe contains all
    required model features.
    """

    missing = get_missing_columns(
        dataframe,
        feature_columns,
    )

    if missing:
        return {
            "valid": False,
            "missing": missing,
            "message": (
                "Dataset is missing required model columns."
            ),
        }

    return {
        "valid": True,
        "missing": [],
        "message": (
            "All required model columns are available."
        ),
    }


# ============================================================
# GENERAL DATAFRAME VALIDATION
# ============================================================

def validate_training_dataframe(
    dataframe: Optional[pd.DataFrame],
    feature_columns: List[str],
    target_column: str,
) -> Dict[str, Any]:
    """
    Validate a dataframe before model training.
    """

    if dataframe is None:
        return {
            "valid": False,
            "message": "Training dataframe is None.",
            "problems": [
                "No dataframe was supplied."
            ],
        }

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return {
            "valid": False,
            "message": "Invalid training data.",
            "problems": [
                "Input is not a pandas DataFrame."
            ],
        }

    if dataframe.empty:
        return {
            "valid": False,
            "message": "Training dataframe is empty.",
            "problems": [
                "No records are available."
            ],
        }

    required_columns = (
        list(feature_columns)
        +
        [target_column]
    )

    missing = get_missing_columns(
        dataframe,
        required_columns,
    )

    if missing:
        return {
            "valid": False,
            "message": (
                "Training dataframe is missing "
                "required columns."
            ),
            "problems": missing,
        }

    if len(dataframe) < MINIMUM_TRAINING_RECORDS:
        return {
            "valid": False,
            "message": (
                "Dataset contains too few usable records."
            ),
            "problems": [
                (
                    f"At least "
                    f"{MINIMUM_TRAINING_RECORDS} "
                    f"records are required."
                )
            ],
        }

    target = dataframe[target_column]

    if target.nunique(
        dropna=True
    ) < MINIMUM_TARGET_CLASSES:
        return {
            "valid": False,
            "message": (
                "Target column must contain at least "
                "two classes."
            ),
            "problems": [
                (
                    f"Target '{target_column}' "
                    "contains fewer than two classes."
                )
            ],
        }

    return {
        "valid": True,
        "message": (
            "Training dataframe is valid."
        ),
        "problems": [],
    }


# ============================================================
# METADATA SAVE
# ============================================================

def save_metadata(
    metadata: Dict[str, Any],
    filepath: str,
) -> bool:
    """
    Save model metadata as JSON.
    """

    try:
        directory = os.path.dirname(
            filepath
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False,
            )

        return True

    except Exception:
        return False


# ============================================================
# METADATA LOAD
# ============================================================

def load_metadata(
    filepath: str,
) -> Optional[Dict[str, Any]]:
    """
    Load metadata from a JSON file.
    """

    if not os.path.isfile(
        filepath
    ):
        return None

    try:
        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(
                file
            )

        if not isinstance(
            metadata,
            dict,
        ):
            return None

        return metadata

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# PUBLIC METADATA API
# ============================================================

def load_model_metadata() -> Optional[Dict[str, Any]]:
    """
    Load Attack Prediction metadata.
    """

    return load_metadata(
        ATTACK_METADATA_FILE
    )


def load_threat_metadata() -> Optional[Dict[str, Any]]:
    """
    Load Threat Level Prediction metadata.
    """

    return load_metadata(
        THREAT_METADATA_FILE
    )


# ============================================================
# ATTACK MODEL LOADER
# ============================================================

def load_custom_model() -> Optional[Dict[str, Any]]:
    """
    Load the trained Attack Prediction model.

    Returns
    -------
    dict | None

        {
            "model": Pipeline,
            "label_encoder": LabelEncoder
        }
    """

    if not custom_model_exists():
        return None

    try:
        model = joblib.load(
            ATTACK_MODEL_FILE
        )

        label_encoder = joblib.load(
            ATTACK_LABEL_ENCODER_FILE
        )

        if model is None:
            return None

        if label_encoder is None:
            return None

        return {
            "model": model,
            "label_encoder": label_encoder,
        }

    except Exception:
        return None


# ============================================================
# THREAT MODEL LOADER
# ============================================================

def load_threat_model() -> Optional[Any]:
    """
    Load the trained Threat Level Prediction model.
    """

    if not threat_model_exists():
        return None

    try:
        model = joblib.load(
            THREAT_MODEL_FILE
        )

        if model is None:
            return None

        return model

    except Exception:
        return None


# ============================================================
# MODEL STATUS
# ============================================================

def get_model_status() -> Dict[str, Any]:
    """
    Return the current status of all custom models.
    """

    attack_metadata = (
        load_model_metadata()
    )

    threat_metadata = (
        load_threat_metadata()
    )

    return {
        "attack_prediction": {
            "available": (
                custom_model_exists()
            ),
            "metadata_available": (
                attack_metadata is not None
            ),
            "model_file": (
                ATTACK_MODEL_FILE
            ),
            "label_encoder_file": (
                ATTACK_LABEL_ENCODER_FILE
            ),
            "metadata_file": (
                ATTACK_METADATA_FILE
            ),
            "accuracy": (
                attack_metadata.get("accuracy")
                if attack_metadata
                else None
            ),
        },

        "threat_level": {
            "available": (
                threat_model_exists()
            ),
            "metadata_available": (
                threat_metadata is not None
            ),
            "model_file": (
                THREAT_MODEL_FILE
            ),
            "metadata_file": (
                THREAT_METADATA_FILE
            ),
            "accuracy": (
                threat_metadata.get("accuracy")
                if threat_metadata
                else None
            ),
        },

        "model_version": MODEL_VERSION,
    }


# ============================================================
# DATA CLEANING HELPERS
# ============================================================

def _clean_categorical_columns(
    dataframe: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Clean categorical columns.

    Missing and empty values become "Unknown".
    """

    dataframe = dataframe.copy()

    for column in columns:

        if column not in dataframe.columns:
            continue

        dataframe[column] = (
            dataframe[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        dataframe[column] = (
            dataframe[column]
            .replace("", "Unknown")
            .replace(
                {
                    "nan": "Unknown",
                    "None": "Unknown",
                    "NaN": "Unknown",
                }
            )
        )

    return dataframe


def _clean_numeric_columns(
    dataframe: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Convert numeric columns safely.

    Invalid values become zero and negative
    casualty values are clipped.
    """

    dataframe = dataframe.copy()

    for column in columns:

        if column not in dataframe.columns:
            continue

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        dataframe[column] = (
            dataframe[column]
            .fillna(0)
        )

    return dataframe


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_attack_dataset(
    dataframe: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Prepare a dataframe for Attack Prediction.

    The dataframe must already contain the standardized
    columns produced by dataset_mapper.py.

    Returns
    -------
    dataframe, messages
    """

    messages: List[str] = []

    if dataframe is None:
        return None, [
            "Dataset is None."
        ]

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return None, [
            "Input is not a pandas DataFrame."
        ]

    if dataframe.empty:
        return None, [
            "Dataset is empty."
        ]

    required_columns = (
        ATTACK_FEATURE_COLUMNS
        +
        [ATTACK_TARGET_COLUMN]
    )

    missing = get_missing_columns(
        dataframe,
        required_columns,
    )

    if missing:
        return None, [
            (
                "Missing required standardized columns: "
                +
                ", ".join(missing)
            )
        ]

    dataframe = dataframe[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # Clean categorical data
    # --------------------------------------------------------

    dataframe = _clean_categorical_columns(
        dataframe,
        ATTACK_CATEGORICAL_COLUMNS
        +
        [ATTACK_TARGET_COLUMN],
    )

    # --------------------------------------------------------
    # Clean numeric data
    # --------------------------------------------------------

    dataframe = _clean_numeric_columns(
        dataframe,
        ATTACK_NUMERIC_COLUMNS,
    )

    # --------------------------------------------------------
    # Remove invalid target values
    # --------------------------------------------------------

    invalid_target_values = [
        "",
        "Unknown",
        "nan",
        "None",
        "NaN",
        "-1",
        "-99",
    ]

    before_count = len(
        dataframe
    )

    dataframe = dataframe[
        ~dataframe[
            ATTACK_TARGET_COLUMN
        ].isin(
            invalid_target_values
        )
    ].copy()

    removed_count = (
        before_count
        -
        len(dataframe)
    )

    if removed_count > 0:
        messages.append(
            (
                f"Removed {removed_count} "
                "records with invalid attack types."
            )
        )

    # --------------------------------------------------------
    # Check remaining data
    # --------------------------------------------------------

    if dataframe.empty:
        return None, [
            "No usable records remain after cleaning."
        ]

    # --------------------------------------------------------
    # Remove classes with only one record
    #
    # Required for stratified train/test splitting.
    # --------------------------------------------------------

    target_counts = (
        dataframe[
            ATTACK_TARGET_COLUMN
        ]
        .value_counts()
    )

    rare_classes = (
        target_counts[
            target_counts < MINIMUM_CLASS_RECORDS
        ]
        .index
        .tolist()
    )

    if rare_classes:

        dataframe = dataframe[
            ~dataframe[
                ATTACK_TARGET_COLUMN
            ].isin(rare_classes)
        ].copy()

        messages.append(
            (
                f"Removed {len(rare_classes)} "
                "rare attack classes containing fewer "
                "than two records."
            )
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if len(dataframe) < MINIMUM_TRAINING_RECORDS:
        return None, [
            (
                f"Only {len(dataframe)} usable records "
                f"remain. Minimum required is "
                f"{MINIMUM_TRAINING_RECORDS}."
            )
        ]

    if (
        dataframe[
            ATTACK_TARGET_COLUMN
        ].nunique()
        <
        MINIMUM_TARGET_CLASSES
    ):
        return None, [
            (
                "Attack target contains fewer than "
                "two usable classes."
            )
        ]

    dataframe = dataframe.reset_index(
        drop=True
    )

    return dataframe, messages


# ============================================================
# PREPROCESSING PIPELINE HELPERS
# ============================================================

def _build_categorical_pipeline() -> Pipeline:
    """
    Build preprocessing pipeline for categorical features.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "encoder",
                _create_one_hot_encoder(),
            ),
        ]
    )


def _build_numeric_pipeline() -> Pipeline:
    """
    Build preprocessing pipeline for numerical features.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0,
                ),
            ),
        ]
    )


# ============================================================
# ATTACK PREPROCESSOR
# ============================================================

def build_attack_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline for Attack Prediction.
    """

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                _build_categorical_pipeline(),
                ATTACK_CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                _build_numeric_pipeline(),
                ATTACK_NUMERIC_COLUMNS,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# ATTACK MODEL PIPELINE
# ============================================================

def build_attack_pipeline() -> Pipeline:
    """
    Build complete Attack Prediction pipeline.

    Raw data
        ↓
    Missing-value handling
        ↓
    One-hot encoding
        ↓
    Random Forest
    """

    preprocessor = (
        build_attack_preprocessor()
    )

    classifier = RandomForestClassifier(
        n_estimators=ATTACK_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# THREAT PREPROCESSOR
# ============================================================

def build_threat_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline for Threat Level Prediction.
    """

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                _build_categorical_pipeline(),
                THREAT_CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                _build_numeric_pipeline(),
                THREAT_NUMERIC_COLUMNS,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# THREAT MODEL PIPELINE
# ============================================================

def build_threat_pipeline() -> Pipeline:
    """
    Build complete Threat Level Prediction pipeline.

    Raw data
        ↓
    Missing-value handling
        ↓
    One-hot encoding
        ↓
    Random Forest
        ↓
    Threat Level
    """

    preprocessor = (
        build_threat_preprocessor()
    )

    classifier = RandomForestClassifier(
        n_estimators=THREAT_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# CREATE THREAT LEVEL LABELS
# ============================================================

def create_threat_labels(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create Threat_Level target labels.

    Rules
    -----

    HIGH
        nkill >= 10
        OR
        total casualties >= 20

    MEDIUM
        total casualties >= 5

    LOW
        otherwise

    Total casualties =
        nkill + nwound
    """

    if dataframe is None:
        raise ValueError(
            "Dataframe cannot be None."
        )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    dataframe = dataframe.copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    dataframe["nkill"] = pd.to_numeric(
        dataframe["nkill"],
        errors="coerce",
    ).fillna(0)

    dataframe["nwound"] = pd.to_numeric(
        dataframe["nwound"],
        errors="coerce",
    ).fillna(0)

    # --------------------------------------------------------
    # Prevent negative values
    # --------------------------------------------------------

    dataframe["nkill"] = (
        dataframe["nkill"]
        .clip(lower=0)
    )

    dataframe["nwound"] = (
        dataframe["nwound"]
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # Total casualties
    # --------------------------------------------------------

    total_casualties = (
        dataframe["nkill"]
        +
        dataframe["nwound"]
    )

    # --------------------------------------------------------
    # Threat classification
    # --------------------------------------------------------

    dataframe[
        THREAT_TARGET_COLUMN
    ] = np.select(
        [
            (
                (dataframe["nkill"] >= 10)
                |
                (total_casualties >= 20)
            ),
            (
                total_casualties >= 5
            ),
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )

    return dataframe


# ============================================================
# VALIDATE THREAT DATASET
# ============================================================

def validate_threat_dataset(
    dataframe: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Validate dataset before Threat Level training.
    """

    result = {
        "valid": False,
        "problems": [],
        "records": 0,
        "classes": [],
        "class_distribution": {},
    }

    if dataframe is None:
        result["problems"].append(
            "Dataset is None."
        )
        return result

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        result["problems"].append(
            "Input is not a pandas DataFrame."
        )
        return result

    if dataframe.empty:
        result["problems"].append(
            "Dataset contains no records."
        )
        return result

    missing = get_missing_columns(
        dataframe,
        THREAT_FEATURE_COLUMNS,
    )

    if missing:
        result["problems"].append(
            (
                "Missing required columns: "
                +
                ", ".join(missing)
            )
        )
        return result

    try:
        labelled_dataframe = (
            create_threat_labels(
                dataframe
            )
        )

    except Exception as error:
        result["problems"].append(
            (
                "Unable to create threat labels: "
                +
                str(error)
            )
        )
        return result

    classes = (
        labelled_dataframe[
            THREAT_TARGET_COLUMN
        ]
        .dropna()
        .unique()
        .tolist()
    )

    result["records"] = int(
        len(labelled_dataframe)
    )

    result["classes"] = sorted(
        [
            str(value)
            for value in classes
        ]
    )

    class_distribution = (
        labelled_dataframe[
            THREAT_TARGET_COLUMN
        ]
        .value_counts()
        .to_dict()
    )

    result["class_distribution"] = {
        str(key): int(value)
        for key, value
        in class_distribution.items()
    }

    if len(classes) < MINIMUM_TARGET_CLASSES:
        result["problems"].append(
            (
                "Threat dataset must contain "
                "at least two threat classes."
            )
        )
        return result

    if len(labelled_dataframe) < MINIMUM_TRAINING_RECORDS:
        result["problems"].append(
            (
                f"At least "
                f"{MINIMUM_TRAINING_RECORDS} "
                "records are required."
            )
        )
        return result

    insufficient_classes = [
        str(label)
        for label, count
        in class_distribution.items()
        if count < MINIMUM_CLASS_RECORDS
    ]

    if insufficient_classes:
        result["problems"].append(
            (
                "Some threat classes contain fewer "
                "than two records: "
                +
                ", ".join(
                    insufficient_classes
                )
            )
        )
        return result

    result["valid"] = True

    return result


# ============================================================
# PREPARE THREAT DATASET
# ============================================================

def prepare_threat_dataset(
    dataframe: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Prepare a standardized dataframe for Threat Level
    Prediction.
    """

    messages: List[str] = []

    if dataframe is None:
        return None, [
            "Dataset is None."
        ]

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return None, [
            "Input is not a pandas DataFrame."
        ]

    required_columns = (
        THREAT_FEATURE_COLUMNS
    )

    missing = get_missing_columns(
        dataframe,
        required_columns,
    )

    if missing:
        return None, [
            (
                "Missing required standardized columns: "
                +
                ", ".join(missing)
            )
        ]

    dataframe = dataframe[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # Clean categorical columns
    # --------------------------------------------------------

    dataframe = _clean_categorical_columns(
        dataframe,
        THREAT_CATEGORICAL_COLUMNS,
    )

    # --------------------------------------------------------
    # Clean numeric columns
    # --------------------------------------------------------

    dataframe = _clean_numeric_columns(
        dataframe,
        THREAT_NUMERIC_COLUMNS,
    )

    # --------------------------------------------------------
    # Generate labels
    # --------------------------------------------------------

    dataframe = create_threat_labels(
        dataframe
    )

    # --------------------------------------------------------
    # Minimum records
    # --------------------------------------------------------

    if len(dataframe) < MINIMUM_TRAINING_RECORDS:
        return None, [
            (
                f"Only {len(dataframe)} records "
                "are available. "
                f"Minimum required is "
                f"{MINIMUM_TRAINING_RECORDS}."
            )
        ]

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_counts = (
        dataframe[
            THREAT_TARGET_COLUMN
        ]
        .value_counts()
    )

    rare_classes = (
        class_counts[
            class_counts < MINIMUM_CLASS_RECORDS
        ]
        .index
        .tolist()
    )

    if rare_classes:
        messages.append(
            (
                f"{len(rare_classes)} threat class(es) "
                "contain fewer than two records."
            )
        )

    if (
        dataframe[
            THREAT_TARGET_COLUMN
        ].nunique()
        <
        MINIMUM_TARGET_CLASSES
    ):
        return None, [
            (
                "Threat Level target contains fewer "
                "than two classes."
            )
        ]

    dataframe = dataframe.reset_index(
        drop=True
    )

    return dataframe, messages


# ============================================================
# TRAIN ATTACK PREDICTION MODEL
# ============================================================

def train_attack_model(
    dataset_path: str,
) -> Dict[str, Any]:
    """
    Train and persist the Attack Prediction model.
    """

    result = {
        "success": False,
        "accuracy": None,
        "message": "",
        "problems": [],
        "training_records": 0,
        "test_records": 0,
        "total_records": 0,
        "number_of_classes": 0,
    }

    # --------------------------------------------------------
    # Validate path
    # --------------------------------------------------------

    if not dataset_path:
        result["message"] = (
            "No dataset path was provided."
        )
        return result

    if not isinstance(
        dataset_path,
        (str, os.PathLike),
    ):
        result["message"] = (
            "Invalid dataset path."
        )
        return result

    dataset_path = os.fspath(
        dataset_path
    )

    if not os.path.isfile(
        dataset_path
    ):
        result["message"] = (
            "Dataset not found."
        )
        result["problems"] = [
            dataset_path
        ]
        return result

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    try:
        dataframe = pd.read_csv(
            dataset_path,
            low_memory=False,
        )

    except Exception as error:
        result["message"] = (
            "Unable to read dataset."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Prepare dataset
    # --------------------------------------------------------

    try:
        dataframe, messages = (
            prepare_attack_dataset(
                dataframe
            )
        )

    except Exception as error:
        result["message"] = (
            "Dataset preparation failed."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    if dataframe is None:
        result["message"] = (
            "Dataset validation failed."
        )
        result["problems"] = messages
        return result

    if messages:
        result["problems"].extend(
            messages
        )

    result["total_records"] = int(
        len(dataframe)
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_counts = (
        dataframe[
            ATTACK_TARGET_COLUMN
        ]
        .value_counts()
    )

    if len(target_counts) < 2:
        result["message"] = (
            "Attack target must contain "
            "at least two classes."
        )
        return result

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = dataframe[
        ATTACK_FEATURE_COLUMNS
    ].copy()

    y_raw = dataframe[
        ATTACK_TARGET_COLUMN
    ].copy()

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    try:
        y = label_encoder.fit_transform(
            y_raw
        )

    except Exception as error:
        result["message"] = (
            "Unable to encode attack target."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    except ValueError as error:
        result["message"] = (
            "Unable to split dataset "
            "for training and testing."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    try:
        pipeline = (
            build_attack_pipeline()
        )

    except Exception as error:
        result["message"] = (
            "Unable to build Attack "
            "Prediction model."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    try:
        print("\n========== TRAIN DTYPES ==========")
        print(X_train.dtypes)
        print("===============================\n")
        pipeline.fit(
            X_train,
            y_train,
        )

    except Exception as error:
        result["message"] = (
            "Attack Prediction model "
            "training failed."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    try:
        predictions = (
            pipeline.predict(
                X_test
            )
        )

    except Exception as error:
        result["message"] = (
            "Attack model evaluation failed."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    try:
        report = classification_report(
            y_test,
            predictions,
            labels=np.arange(
                len(
                    label_encoder.classes_
                )
            ),
            target_names=(
                label_encoder.classes_
            ),
            output_dict=True,
            zero_division=0,
        )

    except Exception:
        report = {}

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    try:
        joblib.dump(
            pipeline,
            ATTACK_MODEL_FILE,
        )

        joblib.dump(
            label_encoder,
            ATTACK_LABEL_ENCODER_FILE,
        )

    except Exception as error:
        result["message"] = (
            "Model trained but could "
            "not be saved."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "model": (
            "RandomForestClassifier"
        ),
        "model_type": (
            "Attack Prediction"
        ),
        "version": MODEL_VERSION,
        "accuracy": float(
            accuracy
        ),
        "training_records": int(
            len(X_train)
        ),
        "test_records": int(
            len(X_test)
        ),
        "total_records": int(
            len(dataframe)
        ),
        "number_of_classes": int(
            len(
                label_encoder.classes_
            )
        ),
        "classes": (
            label_encoder
            .classes_
            .tolist()
        ),
        "features": (
            ATTACK_FEATURE_COLUMNS
        ),
        "categorical_features": (
            ATTACK_CATEGORICAL_COLUMNS
        ),
        "numeric_features": (
            ATTACK_NUMERIC_COLUMNS
        ),
        "target": (
            ATTACK_TARGET_COLUMN
        ),
        "classification_report": (
            report
        ),
        "random_state": RANDOM_STATE,
        "n_estimators": (
            ATTACK_N_ESTIMATORS
        ),
    }

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    if not save_metadata(
        metadata,
        ATTACK_METADATA_FILE,
    ):
        result["message"] = (
            "Model trained and saved, "
            "but metadata could not be saved."
        )
        result["problems"] = [
            "Metadata save operation failed."
        ]
        return result

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    result["success"] = True

    result["accuracy"] = float(
        accuracy
    )

    result["training_records"] = int(
        len(X_train)
    )

    result["test_records"] = int(
        len(X_test)
    )

    result["number_of_classes"] = int(
        len(
            label_encoder.classes_
        )
    )

    result["message"] = (
        "Attack Prediction model "
        "trained successfully."
    )

    return result


# ============================================================
# TRAIN THREAT LEVEL MODEL
# ============================================================

def train_threat_model(
    dataset_path: str,
) -> Dict[str, Any]:
    """
    Train and persist the Threat Level Prediction model.
    """

    result = {
        "success": False,
        "accuracy": None,
        "message": "",
        "problems": [],
        "training_records": 0,
        "test_records": 0,
        "total_records": 0,
        "number_of_classes": 0,
        "class_distribution": {},
    }

    # --------------------------------------------------------
    # Validate path
    # --------------------------------------------------------

    if not dataset_path:
        result["message"] = (
            "No dataset path was provided."
        )
        return result

    if not isinstance(
        dataset_path,
        (str, os.PathLike),
    ):
        result["message"] = (
            "Invalid dataset path."
        )
        return result

    dataset_path = os.fspath(
        dataset_path
    )

    if not os.path.isfile(
        dataset_path
    ):
        result["message"] = (
            "Dataset not found."
        )
        result["problems"] = [
            dataset_path
        ]
        return result

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    try:
        dataframe = pd.read_csv(
            dataset_path,
            low_memory=False,
        )

    except Exception as error:
        result["message"] = (
            "Unable to read dataset."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Prepare dataset
    # --------------------------------------------------------

    try:
        dataframe, messages = (
            prepare_threat_dataset(
                dataframe
            )
        )

    except Exception as error:
        result["message"] = (
            "Threat dataset preparation failed."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    if dataframe is None:
        result["message"] = (
            "Threat dataset validation failed."
        )
        result["problems"] = messages
        return result

    if messages:
        result["problems"].extend(
            messages
        )

    result["total_records"] = int(
        len(dataframe)
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_distribution = (
        dataframe[
            THREAT_TARGET_COLUMN
        ]
        .value_counts()
        .to_dict()
    )

    result["class_distribution"] = {
        str(key): int(value)
        for key, value
        in class_distribution.items()
    }

    unique_classes = (
        dataframe[
            THREAT_TARGET_COLUMN
        ]
        .unique()
        .tolist()
    )

    if len(unique_classes) < 2:
        result["message"] = (
            "Threat Level target must "
            "contain at least two classes."
        )
        result["problems"] = [
            "Only one threat class was generated."
        ]
        return result

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X = dataframe[
        THREAT_FEATURE_COLUMNS
    ].copy()

    y = dataframe[
        THREAT_TARGET_COLUMN
    ].copy()

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    except ValueError as error:
        result["message"] = (
            "Unable to split threat dataset "
            "for training and testing."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    try:
        pipeline = (
            build_threat_pipeline()
        )

    except Exception as error:
        result["message"] = (
            "Unable to build Threat "
            "Level model."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    try:
        print("\n========== TRAIN DTYPES ==========")
        print(X_train.dtypes)
        print("===============================\n")
        pipeline.fit(
            X_train,
            y_train,
        )

    except Exception as error:
        result["message"] = (
            "Threat Level model "
            "training failed."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    try:
        predictions = (
            pipeline.predict(
                X_test
            )
        )

    except Exception as error:
        result["message"] = (
            "Threat Level model "
            "evaluation failed."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    try:
        report = classification_report(
            y_test,
            predictions,
            labels=sorted(
                unique_classes
            ),
            target_names=sorted(
                [
                    str(value)
                    for value in unique_classes
                ]
            ),
            output_dict=True,
            zero_division=0,
        )

    except Exception:
        report = {}

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    try:
        joblib.dump(
            pipeline,
            THREAT_MODEL_FILE,
        )

    except Exception as error:
        result["message"] = (
            "Threat Level model trained "
            "but could not be saved."
        )
        result["problems"] = [
            str(error)
        ]
        return result

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "model": (
            "RandomForestClassifier"
        ),
        "model_type": (
            "Threat Level Prediction"
        ),
        "version": MODEL_VERSION,
        "accuracy": float(
            accuracy
        ),
        "training_records": int(
            len(X_train)
        ),
        "test_records": int(
            len(X_test)
        ),
        "total_records": int(
            len(dataframe)
        ),
        "number_of_classes": int(
            len(unique_classes)
        ),
        "classes": sorted(
            [
                str(value)
                for value in unique_classes
            ]
        ),
        "class_distribution": {
            str(key): int(value)
            for key, value
            in class_distribution.items()
        },
        "features": (
            THREAT_FEATURE_COLUMNS
        ),
        "categorical_features": (
            THREAT_CATEGORICAL_COLUMNS
        ),
        "numeric_features": (
            THREAT_NUMERIC_COLUMNS
        ),
        "target": (
            THREAT_TARGET_COLUMN
        ),
        "threat_rules": {
            "high": (
                "nkill >= 10 OR "
                "total_casualties >= 20"
            ),
            "medium": (
                "total_casualties >= 5"
            ),
            "low": (
                "otherwise"
            ),
        },
        "classification_report": (
            report
        ),
        "random_state": RANDOM_STATE,
        "n_estimators": (
            THREAT_N_ESTIMATORS
        ),
    }

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    if not save_metadata(
        metadata,
        THREAT_METADATA_FILE,
    ):
        result["message"] = (
            "Threat Level model trained "
            "and saved, but metadata "
            "could not be saved."
        )
        result["problems"] = [
            "Metadata save operation failed."
        ]
        return result

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    result["success"] = True

    result["accuracy"] = float(
        accuracy
    )

    result["training_records"] = int(
        len(X_train)
    )

    result["test_records"] = int(
        len(X_test)
    )

    result["number_of_classes"] = int(
        len(unique_classes)
    )

    result["message"] = (
        "Threat Level model "
        "trained successfully."
    )

    return result


# ============================================================
# ATTACK PREDICTION
# ============================================================

def predict_attack(
    model_data: Dict[str, Any],
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict the most likely attack type.

    Parameters
    ----------
    model_data:
        Result returned by load_custom_model().

    input_data:
        Dictionary containing attack features.

    Returns
    -------
    dict
        {
            "prediction": str,
            "confidence": float,
            "probabilities": dict
        }
    """

    if not model_data:
        raise ValueError(
            "Attack prediction model "
            "is not loaded."
        )

    if "model" not in model_data:
        raise ValueError(
            "Invalid attack model package."
        )

    if "label_encoder" not in model_data:
        raise ValueError(
            "Attack label encoder is missing."
        )

    model = model_data[
        "model"
    ]

    label_encoder = model_data[
        "label_encoder"
    ]

    if not isinstance(
        input_data,
        dict,
    ):
        raise TypeError(
            "input_data must be a dictionary."
        )

    # --------------------------------------------------------
    # Build sample
    # --------------------------------------------------------

    sample = pd.DataFrame(
        [input_data]
    )

    # --------------------------------------------------------
    # Ensure expected columns
    # --------------------------------------------------------

    for column in ATTACK_FEATURE_COLUMNS:

        if column not in sample.columns:
            sample[column] = np.nan

    sample = sample[ATTACK_FEATURE_COLUMNS].copy()

# categorical
    for col in ATTACK_CATEGORICAL_COLUMNS:
     sample[col] = sample[col].astype("string")

# numeric
    for col in ATTACK_NUMERIC_COLUMNS:
     sample[col] = (
        pd.Series(sample[col], dtype="float64")
    )
    for column in sample.columns:

       sample[column] = pd.to_numeric(
        sample[column],
        errors="coerce",
    )

       if pd.api.types.is_numeric_dtype(sample[column]):
        sample[column] = sample[column].astype(np.float64)

    st.write("===== SAMPLE DTYPES =====")
    st.write(sample.dtypes)

    st.write("===== SAMPLE =====")
    st.write(sample)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    print("\n========== PREDICT DTYPES ==========")
    print(sample.dtypes)
    print("===============================\n")

    prediction = model.predict(
        sample
    )[0]

    # --------------------------------------------------------
    # Decode target
    # --------------------------------------------------------

    try:
        label = (
            label_encoder
            .inverse_transform(
                [prediction]
            )[0]
        )

    except Exception as error:
        raise ValueError(
            "Unable to decode attack prediction."
        ) from error

    result = {
        "prediction": label
    }

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model.predict_proba(
                sample
            )[0]
        )

        model_classes = (
            model.classes_
        )

        probability_dict = {}

        for (
            encoded_class,
            probability,
        ) in zip(
            model_classes,
            probabilities,
        ):

            try:
                decoded_class = (
                    label_encoder
                    .inverse_transform(
                        [encoded_class]
                    )[0]
                )

            except Exception:
                decoded_class = str(
                    encoded_class
                )

            probability_dict[
                str(decoded_class)
            ] = round(
                float(
                    probability * 100
                ),
                2,
            )

        result["confidence"] = round(
            float(
                np.max(
                    probabilities
                )
                *
                100
            ),
            2,
        )

        result["probabilities"] = (
            probability_dict
        )

    return result


# ============================================================
# THREAT LEVEL PREDICTION
# ============================================================

def predict_threat(
    model: Any,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict the threat level.
    """

    if model is None:
        raise ValueError(
            "Threat prediction model "
            "is not loaded."
        )

    if not isinstance(
        input_data,
        dict,
    ):
        raise TypeError(
            "input_data must be a dictionary."
        )

    # --------------------------------------------------------
    # Build sample
    # --------------------------------------------------------

    sample = pd.DataFrame(
        [input_data]
    )

    # --------------------------------------------------------
    # Ensure expected columns
    # --------------------------------------------------------

    for column in THREAT_FEATURE_COLUMNS:

        if column not in sample.columns:
            sample[column] = np.nan

    sample = sample[
        THREAT_FEATURE_COLUMNS
    ]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    print("\n========== PREDICT DTYPES ==========")
    print(sample.dtypes)
    print("===============================\n")

    prediction = model.predict(
        sample
    )[0]

    result = {
        "prediction": prediction
    }

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model.predict_proba(
                sample
            )[0]
        )

        probability_dict = {}

        for (
            cls,
            probability,
        ) in zip(
            model.classes_,
            probabilities,
        ):

            probability_dict[
                str(cls)
            ] = round(
                float(
                    probability * 100
                ),
                2,
            )

        result["confidence"] = round(
            float(
                np.max(
                    probabilities
                )
                *
                100
            ),
            2,
        )

        result["probabilities"] = (
            probability_dict
        )

    return result


# ============================================================
# TRAIN ALL MODELS
# ============================================================

def train_all_models(
    dataset_path: str,
) -> Dict[str, Any]:
    """
    Train both supported models.

    Models
    ------
    1. Attack Prediction
    2. Threat Level Prediction
    """

    results = {
        "success": False,
        "message": "",
        "attack_prediction": {
            "success": False,
            "accuracy": None,
            "message": "",
            "problems": [],
        },
        "threat_level": {
            "success": False,
            "accuracy": None,
            "message": "",
            "problems": [],
        },
    }

    # --------------------------------------------------------
    # Validate path
    # --------------------------------------------------------

    if not dataset_path:
        results["message"] = (
            "No dataset path was provided."
        )
        return results

    if not isinstance(
        dataset_path,
        (str, os.PathLike),
    ):
        results["message"] = (
            "Invalid dataset path."
        )
        return results

    dataset_path = os.fspath(
        dataset_path
    )

    if not os.path.isfile(
        dataset_path
    ):
        results["message"] = (
            "Dataset not found."
        )
        return results

    # --------------------------------------------------------
    # Attack Prediction
    # --------------------------------------------------------

    try:
        attack_result = (
            train_attack_model(
                dataset_path
            )
        )

    except Exception as error:
        attack_result = {
            "success": False,
            "accuracy": None,
            "message": (
                "Attack Prediction "
                "training failed."
            ),
            "problems": [
                str(error)
            ],
        }

    results[
        "attack_prediction"
    ] = attack_result

    # --------------------------------------------------------
    # Threat Level
    # --------------------------------------------------------

    try:
        threat_result = (
            train_threat_model(
                dataset_path
            )
        )

    except Exception as error:
        threat_result = {
            "success": False,
            "accuracy": None,
            "message": (
                "Threat Level training "
                "failed."
            ),
            "problems": [
                str(error)
            ],
        }

    results[
        "threat_level"
    ] = threat_result

    # --------------------------------------------------------
    # Determine overall success
    # --------------------------------------------------------

    attack_success = bool(
        attack_result.get(
            "success",
            False,
        )
    )

    threat_success = bool(
        threat_result.get(
            "success",
            False,
        )
    )

    results["success"] = (
        attack_success
        or
        threat_success
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if (
        attack_success
        and
        threat_success
    ):

        results["message"] = (
            "Attack Prediction and "
            "Threat Level models "
            "trained successfully."
        )

    elif attack_success:

        results["message"] = (
            "Attack Prediction model "
            "trained successfully, "
            "but Threat Level model "
            "could not be trained."
        )

    elif threat_success:

        results["message"] = (
            "Threat Level model "
            "trained successfully, "
            "but Attack Prediction "
            "model could not be trained."
        )

    else:

        results["message"] = (
            "No models could be trained."
        )

    return results


# ============================================================
# DELETE MODEL FILES
# ============================================================

def delete_custom_models() -> Dict[str, Any]:
    """
    Delete all custom model files and metadata.

    Useful from the Settings/Admin section.
    """

    files_to_delete = [
        ATTACK_MODEL_FILE,
        ATTACK_LABEL_ENCODER_FILE,
        ATTACK_METADATA_FILE,
        THREAT_MODEL_FILE,
        THREAT_METADATA_FILE,
    ]

    deleted = []
    failed = []

    for filepath in files_to_delete:

        if not os.path.isfile(filepath):
            continue

        try:
            os.remove(filepath)
            deleted.append(filepath)

        except OSError:
            failed.append(filepath)

    return {
        "success": len(failed) == 0,
        "deleted": deleted,
        "failed": failed,
        "message": (
            "Custom model cleanup completed."
            if not failed
            else
            "Some model files could not be deleted."
        ),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_attack_model_info() -> Dict[str, Any]:
    """
    Return detailed Attack Prediction model information.
    """

    metadata = load_model_metadata()

    if metadata is None:
        return {
            "available": False,
            "metadata": None,
        }

    return {
        "available": custom_model_exists(),
        "metadata": metadata,
    }


def get_threat_model_info() -> Dict[str, Any]:
    """
    Return detailed Threat Level model information.
    """

    metadata = load_threat_metadata()

    if metadata is None:
        return {
            "available": False,
            "metadata": None,
        }

    return {
        "available": threat_model_exists(),
        "metadata": metadata,
    }


# ============================================================
# QUICK MODEL VALIDATION
# ============================================================

def validate_saved_models() -> Dict[str, Any]:
    """
    Verify that saved model files can actually be loaded.
    """

    result = {
        "attack_prediction": {
            "exists": custom_model_exists(),
            "loadable": False,
        },
        "threat_level": {
            "exists": threat_model_exists(),
            "loadable": False,
        },
        "valid": False,
    }

    # --------------------------------------------------------
    # Attack model
    # --------------------------------------------------------

    if custom_model_exists():

        attack_model = (
            load_custom_model()
        )

        result[
            "attack_prediction"
        ][
            "loadable"
        ] = (
            attack_model is not None
        )

    # --------------------------------------------------------
    # Threat model
    # --------------------------------------------------------

    if threat_model_exists():

        threat_model = (
            load_threat_model()
        )

        result[
            "threat_level"
        ][
            "loadable"
        ] = (
            threat_model is not None
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    result["valid"] = (
        result[
            "attack_prediction"
        ]["loadable"]
        or
        result[
            "threat_level"
        ]["loadable"]
    )

    return result


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [

    # Configuration
    "MODEL_VERSION",
    "BASE_MODEL_DIRECTORY",

    # Files
    "ATTACK_MODEL_FILE",
    "ATTACK_LABEL_ENCODER_FILE",
    "ATTACK_METADATA_FILE",
    "THREAT_MODEL_FILE",
    "THREAT_METADATA_FILE",

    # Attack configuration
    "ATTACK_FEATURE_COLUMNS",
    "ATTACK_TARGET_COLUMN",
    "ATTACK_CATEGORICAL_COLUMNS",
    "ATTACK_NUMERIC_COLUMNS",

    # Threat configuration
    "THREAT_FEATURE_COLUMNS",
    "THREAT_TARGET_COLUMN",
    "THREAT_CATEGORICAL_COLUMNS",
    "THREAT_NUMERIC_COLUMNS",

    # Validation
    "get_missing_columns",
    "validate_model_columns",
    "validate_training_dataframe",
    "validate_threat_dataset",

    # Dataset preparation
    "prepare_attack_dataset",
    "prepare_threat_dataset",
    "create_threat_labels",

    # Pipelines
    "build_attack_preprocessor",
    "build_attack_pipeline",
    "build_threat_preprocessor",
    "build_threat_pipeline",

    # Training
    "train_attack_model",
    "train_threat_model",
    "train_all_models",

    # Loading
    "load_custom_model",
    "load_threat_model",

    # Metadata
    "save_metadata",
    "load_metadata",
    "load_model_metadata",
    "load_threat_metadata",

    # Status
    "custom_model_exists",
    "threat_model_exists",
    "attack_metadata_exists",
    "threat_metadata_exists",
    "get_model_status",
    "get_attack_model_info",
    "get_threat_model_info",
    "validate_saved_models",

    # Prediction
    "predict_attack",
    "predict_threat",

    # Cleanup
    "delete_custom_models",
]