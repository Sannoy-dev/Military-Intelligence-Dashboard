# utils/data_loader.py

import os
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

CUSTOM_DATASET = "data/custom_dataset.csv"


# ============================================================
# LOAD ACTIVE DATASET
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    """
    Load the currently active standardized dataset.

    The application does not use a default GTD dataset.

    The active dataset must be uploaded and processed
    through main.py and saved as:

        data/custom_dataset.csv

    Returns:
        pandas.DataFrame:
            If an active dataset exists.

        None:
            If no dataset has been uploaded.
    """

    # --------------------------------------------------------
    # Check dataset availability
    # --------------------------------------------------------

    if not os.path.isfile(CUSTOM_DATASET):

        return None

    # --------------------------------------------------------
    # Read dataset
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            CUSTOM_DATASET,
            low_memory=False
        )

    except Exception as error:

        raise RuntimeError(
            f"Unable to read the active dataset: {error}"
        )

    # --------------------------------------------------------
    # Normalize column names
    #
    # The dataset should already have been mapped by
    # data_mapper.py, but this protects against accidental
    # whitespace/case differences.
    # --------------------------------------------------------

    df.columns = [
        str(column)
        .strip()
        .lower()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    df = (
        df
        .dropna(
            how="all"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "iyear",
        "latitude",
        "longitude",
        "success",
        "suicide",
        "nkill",
        "nwound"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Fill impact values
    #
    # Missing casualty values are treated as zero.
    # --------------------------------------------------------

    if "nkill" in df.columns:

        df["nkill"] = (
            df["nkill"]
            .fillna(0)
        )

    if "nwound" in df.columns:

        df["nwound"] = (
            df["nwound"]
            .fillna(0)
        )

    # --------------------------------------------------------
    # Fill binary fields
    # --------------------------------------------------------

    if "success" in df.columns:

        df["success"] = (
            df["success"]
            .fillna(0)
        )

    if "suicide" in df.columns:

        df["suicide"] = (
            df["suicide"]
            .fillna(0)
        )

    # --------------------------------------------------------
    # Clean categorical columns
    # --------------------------------------------------------

    categorical_columns = [
        "country_txt",
        "region_txt",
        "city",
        "attacktype1_txt",
        "weaptype1_txt",
        "targtype1_txt",
        "gname"
    ]

    for column in categorical_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

    # --------------------------------------------------------
    # Return active standardized dataset
    # --------------------------------------------------------

    return df


# ============================================================
# ACTIVE DATASET ALIAS
# ============================================================

def load_active_data():
    """
    Return the currently active standardized dataset.

    This is an alias for load_data().
    """

    return load_data()


# ============================================================
# CHECK DATASET AVAILABILITY
# ============================================================

def dataset_exists():
    """
    Check whether an active dataset exists.
    """

    return os.path.isfile(
        CUSTOM_DATASET
    )


# ============================================================
# GET DATASET INFORMATION
# ============================================================

def get_dataset_info():
    """
    Return basic information about the active dataset.
    """

    df = load_data()

    # --------------------------------------------------------
    # No dataset
    # --------------------------------------------------------

    if df is None:

        return {

            "loaded": False,

            "rows": 0,

            "columns": 0,

            "column_names": [],

            "path": CUSTOM_DATASET
        }

    # --------------------------------------------------------
    # Dataset available
    # --------------------------------------------------------

    return {

        "loaded": True,

        "rows": len(df),

        "columns": len(df.columns),

        "column_names":
            list(df.columns),

        "path":
            CUSTOM_DATASET
    }


# ============================================================
# GET DATASET ROW COUNT
# ============================================================

def get_dataset_row_count():
    """
    Return the number of records in the active dataset.
    """

    df = load_data()

    if df is None:

        return 0

    return len(df)


# ============================================================
# GET DATASET COLUMNS
# ============================================================

def get_dataset_columns():
    """
    Return the columns available in the active dataset.
    """

    df = load_data()

    if df is None:

        return []

    return list(
        df.columns
    )


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

def has_required_columns(
    required_columns
):
    """
    Check whether the active dataset contains
    all specified columns.

    Returns:
        True / False
    """

    df = load_data()

    if df is None:

        return False

    return all(
        column in df.columns
        for column in required_columns
    )


# ============================================================
# GET MISSING COLUMNS
# ============================================================

def get_missing_columns(
    required_columns
):
    """
    Return columns required by a feature/model
    that are missing from the active dataset.
    """

    df = load_data()

    if df is None:

        return list(
            required_columns
        )

    return [
        column
        for column in required_columns
        if column not in df.columns
    ]


# ============================================================
# CLEAR DATASET CACHE
# ============================================================

def clear_dataset_cache():
    """
    Clear the Streamlit dataset cache.

    Use this after:

        - Uploading a new dataset
        - Replacing the active dataset
        - Re-mapping columns
        - Deleting the active dataset
    """

    load_data.clear()


# ============================================================
# REMOVE ACTIVE DATASET
# ============================================================

def remove_dataset():
    """
    Delete the currently active dataset.

    This does NOT delete trained models.
    """

    if os.path.isfile(
        CUSTOM_DATASET
    ):

        try:

            os.remove(
                CUSTOM_DATASET
            )

        except OSError as error:

            raise RuntimeError(
                f"Unable to remove the active dataset: {error}"
            )

    clear_dataset_cache()


# ============================================================
# SAVE ACTIVE DATASET
# ============================================================

def save_active_dataset(df):
    """
    Save a standardized DataFrame as the active dataset.

    This should normally be called after data_mapper.py
    has completed column mapping and preparation.
    """

    if df is None:

        raise ValueError(
            "Cannot save an empty dataset."
        )

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "df must be a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Create data directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            CUSTOM_DATASET
        ),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save standardized dataset
    # --------------------------------------------------------

    df.to_csv(
        CUSTOM_DATASET,
        index=False
    )

    # --------------------------------------------------------
    # Clear cached old dataset
    # --------------------------------------------------------

    clear_dataset_cache()

    return CUSTOM_DATASET