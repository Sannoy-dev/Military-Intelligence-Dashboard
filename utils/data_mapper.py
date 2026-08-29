# ============================================================
# utils/data_mapper.py
#
# Part 1A
#
# Includes:
#   • Imports
#   • Standard column definitions
#   • Model column requirements
#   • Column aliases
#   • Normalization utilities
#   • Cached lookup dictionaries
#
# Compatible with:
#   Part 1B
#   Part 2
#   Part 3
#   Part 4
#   Part 5
# ============================================================

from __future__ import annotations
from pprint import pprint
import re
import difflib
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import pandas as pd

# ============================================================
# STANDARD APPLICATION COLUMNS
# ============================================================

STANDARD_COLUMNS: List[str] = [
    "iyear",
    "country_txt",
    "region_txt",
    "city",
    "latitude",
    "longitude",
    "attacktype1_txt",
    "weaptype1_txt",
    "targtype1_txt",
    "gname",
    "success",
    "suicide",
    "nkill",
    "nwound",
]

# ============================================================
# REQUIRED DASHBOARD COLUMNS
# ============================================================

REQUIRED_ANALYSIS_COLUMNS: List[str] = [
    "iyear",
    "country_txt",
]

# ============================================================
# ATTACK PREDICTION MODEL
# ============================================================

ATTACK_PREDICTION_COLUMNS: List[str] = [
    "country_txt",
    "region_txt",
    "weaptype1_txt",
    "targtype1_txt",
    "gname",
    "success",
    "suicide",
    "nkill",
    "nwound",
    "attacktype1_txt",
]

# ============================================================
# THREAT LEVEL MODEL
# ============================================================

THREAT_LEVEL_COLUMNS: List[str] = [
    "country_txt",
    "region_txt",
    "attacktype1_txt",
    "weaptype1_txt",
    "targtype1_txt",
    "nkill",
    "nwound",
]

# ============================================================
# FUZZY MATCHING CONFIGURATION
# ============================================================

FUZZY_THRESHOLD: float = 0.85

# ============================================================
# COLUMN ALIASES
# ============================================================

COLUMN_ALIASES: Dict[str, List[str]] = {

    "iyear": [
        "year",
        "attack_year",
        "incident_year",
        "date_year",
        "incident year",
        "incident date year",
    ],

    "country_txt": [
        "country",
        "country_name",
        "country name",
        "countryname",
        "nation",
    ],

    "region_txt": [
        "region",
        "region_name",
        "region name",
        "area",
    ],

    "city": [
        "city_name",
        "city name",
        "town",
        "location",
    ],

    "latitude": [
        "lat",
        "latitude_value",
        "latitude value",
    ],

    "longitude": [
        "lon",
        "lng",
        "long",
        "longitude_value",
        "longitude value",
    ],

    "attacktype1_txt": [
        "attack",
        "attack_type",
        "attack type",
        "attacktype",
        "attacktype1",
        "attack type 1",
        "incident_type",
        "incident type",
    ],

    "weaptype1_txt": [
        "weapon",
        "weapon_name",
        "weapon_type",
        "weapon type",
        "weapon type 1",
        "weaptype",
        "weaptype1",
    ],

    "targtype1_txt": [
        "target",
        "target_type",
        "target type",
        "target type 1",
        "targettype",
        "targtype1",
    ],

    "gname": [
        "group",
        "group_name",
        "group name",
        "organisation",
        "organization",
        "organisation name",
        "organization name",
        "terrorist_group",
        "terrorist group",
        "terrorist organisation",
        "terrorist organization",
    ],

    "success": [
        "successful",
        "successful attack",
        "successful_attack",
        "attack_success",
        "is_successful",
    ],

    "suicide": [
        "suicide_attack",
        "suicide attack",
        "suicide_flag",
        "suicide flag",
        "is_suicide",
    ],

    "nkill": [
        "killed",
        "deaths",
        "fatalities",
        "people killed",
        "number killed",
        "number_killed",
        "num_killed",
    ],

    "nwound": [
        "wounded",
        "injured",
        "injuries",
        "people wounded",
        "number wounded",
        "number_wounded",
        "num_wounded",
    ],
}

# ============================================================
# COLUMN NAME NORMALIZATION
# ============================================================

def normalize_column_name(column: str) -> str:
    """
    Normalize a column name.

    Examples
    --------
    Country Name
        -> country_name

    COUNTRY-NAME
        -> country_name

    attack type
        -> attack_type
    """

    column = str(column).strip().lower()

    column = re.sub(
        r"[^a-z0-9]+",
        "_",
        column,
    )

    column = re.sub(
        r"_+",
        "_",
        column,
    )

    return column.strip("_")


# ============================================================
# BUILD CACHED LOOKUP TABLES
# ============================================================

@lru_cache(maxsize=1)
def get_column_candidates() -> Dict[str, List[str]]:
    """
    Returns every standard column together with all aliases.

    Cached because this never changes during runtime.
    """

    candidates: Dict[str, List[str]] = {}

    for standard in STANDARD_COLUMNS:

        aliases = COLUMN_ALIASES.get(
            standard,
            [],
        )

        candidates[standard] = [
            standard,
            *aliases,
        ]

    return candidates


# ============================================================
# NORMALIZED LOOKUP TABLE
# ============================================================

@lru_cache(maxsize=1)
def get_normalized_lookup() -> Dict[str, str]:
    """
    Creates a normalized lookup table.

    Example

    country name
            ↓

    country_name
            ↓

    country_txt
    """

    lookup: Dict[str, str] = {}

    for standard, aliases in get_column_candidates().items():

        lookup[
            normalize_column_name(standard)
        ] = standard

        for alias in aliases:

            lookup[
                normalize_column_name(alias)
            ] = standard

    return lookup


# ============================================================
# NORMALIZED CANDIDATE LIST
# ============================================================

@lru_cache(maxsize=1)
def get_normalized_candidates() -> List[Tuple[str, str]]:
    """
    Returns

        [
            (
                normalized_alias,
                standard_column
            ),
            ...
        ]

    Used for fuzzy matching.

    Cached for speed.
    """

    normalized: List[Tuple[str, str]] = []

    for standard, aliases in get_column_candidates().items():

        normalized.append(
            (
                normalize_column_name(standard),
                standard,
            )
        )

        for alias in aliases:

            normalized.append(
                (
                    normalize_column_name(alias),
                    standard,
                )
            )

    return normalized

# ============================================================
# End of Part 1A
#
# Next:
# Part 1B
#   • find_column_match()
#   • auto_map_columns()
# ============================================================

# ============================================================
# PART 1B-1
#
# Includes:
#   • find_column_match()
#
# Requires:
#   • Part 1A
#
# ============================================================

def find_column_match(
    uploaded_column: str,
) -> Tuple[Optional[str], str, float]:
    """
    Find the best matching application column.

    Parameters
    ----------
    uploaded_column : str
        Column name from the uploaded dataset.

    Returns
    -------
    (
        mapped_column,
        match_type,
        confidence
    )

    Match Types
    -----------
    Exact Match
    Alias Match
    Fuzzy Match
    Unmapped
    """

    normalized = normalize_column_name(uploaded_column)

    lookup = get_normalized_lookup()

    # --------------------------------------------------------
    # Exact standard-column match
    # --------------------------------------------------------

    if normalized in lookup:

        mapped = lookup[normalized]

        if normalized == normalize_column_name(mapped):

            return (
                mapped,
                "Exact Match",
                1.0,
            )

        return (
            mapped,
            "Alias Match",
            1.0,
        )

    # --------------------------------------------------------
    # Fuzzy matching
    # --------------------------------------------------------

    best_column: Optional[str] = None
    best_score: float = 0.0

    for candidate, standard in get_normalized_candidates():

        score = difflib.SequenceMatcher(
            None,
            normalized,
            candidate,
        ).ratio()

        if score > best_score:

            best_score = score
            best_column = standard

    # --------------------------------------------------------
    # Accept fuzzy match
    # --------------------------------------------------------

    if (
        best_column is not None
        and best_score >= FUZZY_THRESHOLD
    ):

        return (
            best_column,
            "Fuzzy Match",
            round(best_score, 3),
        )

    # --------------------------------------------------------
    # No match found
    # --------------------------------------------------------

    return (
        None,
        "Unmapped",
        round(best_score, 3),
    )


# ============================================================
# Optional helper
#
# Useful for debugging or displaying mappings.
# Not required by existing code but helpful.
# ============================================================

def explain_column_match(
    uploaded_column: str,
) -> Dict[str, object]:
    """
    Return detailed mapping information.

    Example
    -------
    >>> explain_column_match("Country Name")

    {
        "uploaded_column": "Country Name",
        "normalized": "country_name",
        "mapped_to": "country_txt",
        "match_type": "Alias Match",
        "confidence": 1.0
    }
    """

    mapped_to, match_type, confidence = find_column_match(
        uploaded_column
    )

    return {
        "uploaded_column": uploaded_column,
        "normalized": normalize_column_name(uploaded_column),
        "mapped_to": mapped_to,
        "match_type": match_type,
        "confidence": confidence,
    }


# ============================================================
# End of Part 1B-1
#
# Next:
# Part 1B-2
#
# Includes:
#   • auto_map_columns()
# ============================================================
# ============================================================
# PART 1B-2
#
# Includes:
#   • auto_map_columns()
#
# Requires:
#   • Part 1A
#   • Part 1B-1
#
# ============================================================

def auto_map_columns(
    df: Optional[pd.DataFrame],
) -> Dict[str, Dict[str, object]]:
    """
    Automatically map uploaded dataset columns to the
    application's standard column names.

    Each uploaded column is matched against the standard
    schema using:
        1. Exact Match
        2. Alias Match
        3. Fuzzy Match

    Duplicate mappings are prevented.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict

    Example
    -------
    {
        "Year": {
            "mapped_to": "iyear",
            "status": "Mapped",
            "match_type": "Alias Match",
            "confidence": 1.0
        },

        "Country Name": {
            "mapped_to": "country_txt",
            "status": "Mapped",
            "match_type": "Alias Match",
            "confidence": 1.0
        }
    }
    """

    if df is None or df.empty:
        return {}

    mapping: Dict[str, Dict[str, object]] = {}

    # Prevent multiple uploaded columns mapping
    # to the same standard column.
    used_standard_columns = set()

    for uploaded_column in df.columns:

        mapped_column, match_type, confidence = find_column_match(
            uploaded_column
        )

        # ----------------------------------------------------
        # No valid match
        # ----------------------------------------------------

        if mapped_column is None:

            mapping[uploaded_column] = {
                "mapped_to": None,
                "status": "Unmapped",
                "match_type": "Unmapped",
                "confidence": confidence,
            }

            continue

        # ----------------------------------------------------
        # Duplicate mapping
        # ----------------------------------------------------

        if mapped_column in used_standard_columns:

            mapping[uploaded_column] = {
                "mapped_to": None,
                "status": "Unmapped",
                "match_type": "Duplicate Match",
                "confidence": confidence,
            }

            continue

        # ----------------------------------------------------
        # Successful mapping
        # ----------------------------------------------------

        mapping[uploaded_column] = {
            "mapped_to": mapped_column,
            "status": "Mapped",
            "match_type": match_type,
            "confidence": confidence,
        }

        used_standard_columns.add(mapped_column)

    return mapping


# ============================================================
# Convenience Helpers
# ============================================================

def get_mapped_columns(
    df: Optional[pd.DataFrame],
) -> List[str]:
    """
    Return all successfully mapped standard columns.
    """

    mapping = auto_map_columns(df)

    return [
        info["mapped_to"]
        for info in mapping.values()
        if info["mapped_to"] is not None
    ]


def get_unmapped_columns(
    df: Optional[pd.DataFrame],
) -> List[str]:
    """
    Return uploaded columns that could not be mapped.
    """

    mapping = auto_map_columns(df)

    return [
        column
        for column, info in mapping.items()
        if info["status"] != "Mapped"
    ]


def get_missing_standard_columns(
    df: Optional[pd.DataFrame],
) -> List[str]:
    """
    Return application standard columns that are missing
    after automatic mapping.
    """

    mapped = set(get_mapped_columns(df))

    return [
        column
        for column in STANDARD_COLUMNS
        if column not in mapped
    ]


def get_mapping_table(
    df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Return the automatic mapping results as a DataFrame.

    Useful for displaying in Streamlit.
    """

    mapping = auto_map_columns(df)

    rows = []

    for uploaded_column, info in mapping.items():

        confidence = info.get("confidence", 0)

        rows.append(
            {
                "Uploaded Column": uploaded_column,
                "Mapped To": (
                    info["mapped_to"]
                    if info["mapped_to"]
                    else "Unmapped"
                ),
                "Status": info["status"],
                "Match Type": info["match_type"],
                "Confidence": (
                    f"{confidence * 100:.1f}%"
                    if confidence > 0
                    else "-"
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# End of Part 1B-2
#
# Part 1 is now complete.
#
# Next:
# Part 2
#
# Includes:
#   • get_required_columns()
#   • get_missing_training_columns()
#   • check_training_eligibility()
# ============================================================
# ============================================================
# PART 2
#
# Includes:
#   • get_required_columns()
#   • get_missing_training_columns()
#   • check_training_eligibility()
#
# Requires:
#   • Part 1A
#   • Part 1B-1
#   • Part 1B-2
#
# ============================================================


# ============================================================
# GET REQUIRED COLUMNS
# ============================================================

def get_required_columns(
    training_type: str,
) -> List[str]:
    """
    Return the columns required by a specific ML model.

    Supported training types:
        • attack_prediction
        • threat_level

    Parameters
    ----------
    training_type : str
        Name of the model/training pipeline.

    Returns
    -------
    List[str]
        Required standardized columns.

    Raises
    ------
    ValueError
        If an unsupported training type is supplied.
    """

    training_type = str(
        training_type
    ).strip().lower()

    requirements = {
        "attack_prediction": ATTACK_PREDICTION_COLUMNS,
        "threat_level": THREAT_LEVEL_COLUMNS,
    }

    if training_type not in requirements:

        raise ValueError(
            f"Unknown training type: {training_type}. "
            f"Supported types: "
            f"{', '.join(requirements.keys())}"
        )

    return requirements[
        training_type
    ].copy()


# ============================================================
# GET MISSING TRAINING COLUMNS
# ============================================================

def get_missing_training_columns(
    df: Optional[pd.DataFrame],
    training_type: str = "attack_prediction",
) -> List[str]:
    """
    Determine which columns are missing for a specific
    machine-learning model.

    The uploaded dataset is first passed through the
    automatic column mapper.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded dataset.

    training_type : str
        Model type.

    Returns
    -------
    List[str]
        Standardized columns required by the model but
        unavailable in the uploaded dataset.
    """

    required_columns = get_required_columns(
        training_type
    )

    # --------------------------------------------------------
    # No dataset
    # --------------------------------------------------------

    if df is None or df.empty:

        return required_columns

    # --------------------------------------------------------
    # Determine successfully mapped columns
    # --------------------------------------------------------

    mapped_columns = set(
        get_mapped_columns(df)
    )

    # --------------------------------------------------------
    # Find missing requirements
    # --------------------------------------------------------

    return [
        column
        for column in required_columns
        if column not in mapped_columns
    ]


# ============================================================
# CHECK TRAINING ELIGIBILITY
# ============================================================

def check_training_eligibility(
    df: Optional[pd.DataFrame],
    training_type: str = "attack_prediction",
) -> Tuple[bool, List[str]]:
    """
    Check whether a dataset contains all columns required
    for a particular ML model.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded dataset.

    training_type : str
        Model type.

    Returns
    -------
    Tuple[bool, List[str]]

    Example
    -------
    (
        True,
        []
    )

    or

    (
        False,
        [
            "gname",
            "nwound"
        ]
    )
    """

    missing_columns = get_missing_training_columns(
        df,
        training_type,
    )

    eligible = len(
        missing_columns
    ) == 0

    return (
        eligible,
        missing_columns,
    )


# ============================================================
# CHECK WHETHER A MODEL IS SUPPORTED
# ============================================================

def is_supported_training_type(
    training_type: str,
) -> bool:
    """
    Return True when the supplied training type is supported.

    This helper prevents callers from having to catch a
    ValueError just to check whether a model exists.
    """

    if not isinstance(
        training_type,
        str,
    ):
        return False

    return (
        training_type.strip().lower()
        in {
            "attack_prediction",
            "threat_level",
        }
    )


# ============================================================
# GET ALL MODEL REQUIREMENTS
# ============================================================

def get_model_requirements() -> Dict[str, List[str]]:
    """
    Return the complete model requirement configuration.

    Returns
    -------
    dict

    Example
    -------
    {
        "attack_prediction": [...],
        "threat_level": [...]
    }

    A copy is returned so callers cannot accidentally
    modify the global configuration.
    """

    return {
        "attack_prediction":
            ATTACK_PREDICTION_COLUMNS.copy(),

        "threat_level":
            THREAT_LEVEL_COLUMNS.copy(),
    }


# ============================================================
# GET MODEL READINESS SUMMARY
# ============================================================

def get_model_readiness(
    df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    """
    Return a compact readiness report for all supported
    machine-learning models.

    This is useful for the Streamlit Settings/Data Explorer
    pages.

    Example
    -------
    {
        "attack_prediction": {
            "ready": True,
            "missing": []
        },
        "threat_level": {
            "ready": False,
            "missing": ["nwound"]
        }
    }
    """

    results: Dict[str, object] = {}

    for training_type in (
        "attack_prediction",
        "threat_level",
    ):

        eligible, missing = check_training_eligibility(
            df,
            training_type,
        )

        results[training_type] = {
            "ready": eligible,
            "missing": missing,
        }

    return results


# ============================================================
# END OF PART 2
#
# Next:
# Part 3
#
# Includes:
#   • apply_mapping()
#   • prepare_dataset()
#   • numeric cleaning
#   • categorical cleaning
#   • dataset standardization
# ============================================================
# ============================================================
# PART 3
#
# Includes:
#   • apply_mapping()
#   • prepare_dataset()
#   • Numeric cleaning
#   • Binary-field cleaning
#   • Categorical-field cleaning
#   • Basic dataset standardization
#
# Requires:
#   • Part 1A
#   • Part 1B-1
#   • Part 1B-2
#   • Part 2
#
# ============================================================


# ============================================================
# APPLY COLUMN MAPPING
# ============================================================

def apply_mapping(
    df: Optional[pd.DataFrame],
    mapping: Optional[Dict[str, object]],
) -> pd.DataFrame:
    """
    Rename uploaded dataset columns to application-standard
    column names.

    Both mapping formats are supported:

    Dictionary format
    -----------------
    {
        "Year": {
            "mapped_to": "iyear"
        }
    }

    Simple format
    -------------
    {
        "Year": "iyear"
    }

    Parameters
    ----------
    df : pd.DataFrame
        Original uploaded dataset.

    mapping : dict
        Column mapping information.

    Returns
    -------
    pd.DataFrame
        Dataset with mapped column names.

    Notes
    -----
    Unmapped columns are retained at this stage.

    Only valid mappings to STANDARD_COLUMNS are applied.
    """

    # --------------------------------------------------------
    # Handle missing dataset
    # --------------------------------------------------------

    if df is None:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Work on a copy
    #
    # This prevents accidental modification of the original
    # uploaded DataFrame.
    # --------------------------------------------------------

    mapped_df = df.copy()

    # --------------------------------------------------------
    # No mapping supplied
    # --------------------------------------------------------

    if not mapping:

        return mapped_df

    rename_dict: Dict[str, str] = {}

    # --------------------------------------------------------
    # Process every mapping
    # --------------------------------------------------------

    for uploaded_column, mapping_info in mapping.items():

        # Uploaded column must actually exist.
        if uploaded_column not in mapped_df.columns:
            continue

        standard_column: Optional[str] = None

        # ----------------------------------------------------
        # Dictionary mapping
        #
        # {
        #     "Year": {
        #         "mapped_to": "iyear"
        #     }
        # }
        # ----------------------------------------------------

        if isinstance(
            mapping_info,
            dict,
        ):

            standard_column = mapping_info.get(
                "mapped_to"
            )

        # ----------------------------------------------------
        # Simple mapping
        #
        # {
        #     "Year": "iyear"
        # }
        # ----------------------------------------------------

        elif isinstance(
            mapping_info,
            str,
        ):

            standard_column = mapping_info

        # ----------------------------------------------------
        # Validate destination
        # ----------------------------------------------------

        if not isinstance(
            standard_column,
            str,
        ):
            continue

        standard_column = standard_column.strip()

        if standard_column not in STANDARD_COLUMNS:
            continue

        # ----------------------------------------------------
        # Prevent two uploaded columns from being renamed
        # to the same standard column.
        # ----------------------------------------------------
        if (standard_column in mapped_df.columns
             and uploaded_column != standard_column
        ):
         continue

        if standard_column in rename_dict.values():
            continue

        rename_dict[
            uploaded_column
        ] = standard_column

    # --------------------------------------------------------
    # Apply rename
    # --------------------------------------------------------

    if rename_dict:

        mapped_df = mapped_df.rename(
            columns=rename_dict
        )

    mapped_df = mapped_df.loc[:, ~mapped_df.columns.duplicated()]

    return mapped_df


# ============================================================
# NUMERIC COLUMN CONFIGURATION
# ============================================================

NUMERIC_COLUMNS: List[str] = [
    "iyear",
    "latitude",
    "longitude",
    "success",
    "suicide",
    "nkill",
    "nwound",
]


# ============================================================
# IMPACT COLUMNS
# ============================================================

IMPACT_COLUMNS: List[str] = [
    "nkill",
    "nwound",
]


# ============================================================
# BINARY COLUMNS
# ============================================================

BINARY_COLUMNS: List[str] = [
    "success",
    "suicide",
]


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

CATEGORICAL_COLUMNS: List[str] = [
    "country_txt",
    "region_txt",
    "city",
    "attacktype1_txt",
    "weaptype1_txt",
    "targtype1_txt",
    "gname",
]


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

def _clean_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert recognized numeric columns to numeric dtype.

    Invalid values are converted to NaN.
    """

    cleaned_df = df.copy()

    for column in NUMERIC_COLUMNS:

        if column not in cleaned_df.columns:
            continue

        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column],
            errors="coerce",
        )

    return cleaned_df


# ============================================================
# CLEAN IMPACT COLUMNS
# ============================================================

def _clean_impact_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean fatalities and injuries.

    Missing impact values are interpreted as zero.

    This is appropriate for the dashboard's impact
    calculations because missing values should not propagate
    as NaN during:

        fatalities = sum(nkill)

        injuries = sum(nwound)
    """

    cleaned_df = df.copy()

    for column in IMPACT_COLUMNS:

        if column not in cleaned_df.columns:
            continue

        cleaned_df[column] = (
            cleaned_df[column]
            .fillna(0)
        )

    return cleaned_df


# ============================================================
# CLEAN BINARY COLUMNS
# ============================================================

def _clean_binary_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean binary fields such as:

        success
        suicide

    Missing values are represented as 0.

    Existing numeric values are preserved.
    """

    cleaned_df = df.copy()

    for column in BINARY_COLUMNS:

        if column not in cleaned_df.columns:
            continue

        cleaned_df[column] = (
            cleaned_df[column]
            .fillna(0)
        )

    return cleaned_df


# ============================================================
# CLEAN CATEGORICAL COLUMNS
# ============================================================

# def _clean_categorical_columns(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Clean recognized categorical columns.

#     Operations:
#         • Replace missing values with "Unknown"
#         • Convert to string
#         • Remove leading/trailing whitespace
#     """
    
#     cleaned_df = df.copy()

#     for column in CATEGORICAL_COLUMNS:

#         if column not in cleaned_df.columns:
#             continue
#         print(column, type(cleaned_df[column]))
#         cleaned_df[column] = (
#             cleaned_df[column]
#             .fillna("Unknown")
#             .astype(str)
#             .str.strip()
#         )

#         # ----------------------------------------------------
#         # Empty strings should also be treated as Unknown.
#         # ----------------------------------------------------

#         cleaned_df[column] = (
#             cleaned_df[column]
#             .replace(
#                 "",
#                 "Unknown",
#             )
#         )

#     return cleaned_df

def _clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:

    cleaned_df = df.copy()

    for column in CATEGORICAL_COLUMNS:

        if column not in cleaned_df.columns:
            continue

        obj = cleaned_df[column]

        print(f"\nColumn: {column}")
        print("Type:", type(obj))

        if isinstance(obj, pd.DataFrame):
            raise Exception(
                f"'{column}' appears {obj.shape[1]} times in the DataFrame. "
                f"Duplicate column names: {obj.columns.tolist()}"
            )

        cleaned_df[column] = (
            obj.fillna("Unknown")
               .astype(str)
               .str.strip()
               .replace("", "Unknown")
        )

    return cleaned_df
# ============================================================
# REMOVE COMPLETELY EMPTY ROWS
# ============================================================

def _remove_empty_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove rows where every recognized column is empty.
    """

    if df.empty:
        return df.copy()

    return (
        df
        .dropna(
            how="all",
        )
        .reset_index(
            drop=True,
        )
    )


# ============================================================
# KEEP STANDARD COLUMNS
# ============================================================

def _keep_standard_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only columns recognized by the application.

    Column ordering follows STANDARD_COLUMNS.
    """

    available_columns = [
        column
        for column in STANDARD_COLUMNS
        if column in df.columns
    ]

    return df[
        available_columns
    ].copy()


# ============================================================
# PREPARE STANDARDIZED DATASET
# ============================================================

def prepare_dataset(
    df: Optional[pd.DataFrame],
    mapping: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Convert an uploaded dataset into the application's
    standardized internal format.

    Processing pipeline
    -------------------
    1. Validate input
    2. Generate automatic mapping if required
    3. Apply mapping
    4. Keep recognized columns
    5. Remove completely empty rows
    6. Convert numeric fields
    7. Clean impact fields
    8. Clean binary fields
    9. Clean categorical fields
    10. Reset index

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded dataset.

    mapping : dict, optional
        User-confirmed or automatically generated mapping.

    Returns
    -------
    pd.DataFrame
        Standardized dataset.

    Example
    -------
    Uploaded:

        Year
        Country
        Fatalities

    becomes:

        iyear
        country_txt
        nkill
    """

    # --------------------------------------------------------
    # Handle missing input
    # --------------------------------------------------------

    if df is None:

        return pd.DataFrame()

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "prepare_dataset() expects a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Work with a copy
    # --------------------------------------------------------

    source_df = df.copy()

    # --------------------------------------------------------
    # Generate automatic mapping
    # --------------------------------------------------------

    if mapping is None:

        mapping = auto_map_columns(
            source_df
        )

    # --------------------------------------------------------
    # Apply mapping
    # --------------------------------------------------------
    pprint(mapping)
    mapped_df = apply_mapping(
      source_df,
      mapping,
    )
    if mapped_df.columns.duplicated().any():

     raise Exception(
        f"Duplicate columns found: "
        f"{mapped_df.columns[mapped_df.columns.duplicated()].tolist()}"
    )

# DEBUG
    print("Columns after mapping:")
    print(mapped_df.columns.tolist())

    print("Duplicate columns:")
    print(mapped_df.columns[mapped_df.columns.duplicated()].tolist())

# or, if you prefer seeing it in Streamlit:
# st.write(mapped_df.columns.tolist())
# st.write(mapped_df.columns[mapped_df.columns.duplicated()].tolist())

# --------------------------------------------------------
# Keep only recognized application columns
# --------------------------------------------------------

    mapped_df = _keep_standard_columns(
      mapped_df
)

    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    mapped_df = _remove_empty_rows(
        mapped_df
    )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    mapped_df = _clean_numeric_columns(
        mapped_df
    )

    # --------------------------------------------------------
    # Clean fatalities and injuries
    # --------------------------------------------------------

    mapped_df = _clean_impact_columns(
        mapped_df
    )

    # --------------------------------------------------------
    # Clean binary fields
    # --------------------------------------------------------

    mapped_df = _clean_binary_columns(
        mapped_df
    )

    # --------------------------------------------------------
    # Clean categorical fields
    # --------------------------------------------------------

    mapped_df = _clean_categorical_columns(
        mapped_df
    )

    # --------------------------------------------------------
    # Final index reset
    # --------------------------------------------------------

    mapped_df = (
        mapped_df
        .reset_index(
            drop=True
        )
    )

    return mapped_df


# ============================================================
# END OF PART 3
#
# Next:
# Part 4
#
# Includes:
#   • validate_dataset()
#   • validate_model_dataset()
#   • model-specific validation
# ============================================================
# ============================================================
# PART 4
#
# Includes:
#   • validate_dataset()
#   • validate_model_dataset()
#   • Attack prediction target validation
#   • Generic dataset validation helpers
#
# Requires:
#   • Part 1A
#   • Part 1B-1
#   • Part 1B-2
#   • Part 2
#   • Part 3
#
# ============================================================


# ============================================================
# VALIDATE GENERAL ANALYSIS DATASET
# ============================================================

def validate_dataset(
    df: Optional[pd.DataFrame],
) -> Tuple[bool, List[str]]:
    """
    Validate whether a standardized dataset can be used
    by the general dashboard.

    Required columns:

        iyear
        country_txt

    Parameters
    ----------
    df : pd.DataFrame
        Standardized dataset.

    Returns
    -------
    Tuple[bool, List[str]]

    Examples
    --------
    Valid:

        (True, [])

    Missing columns:

        (
            False,
            ["country_txt"]
        )

    Empty required values:

        (
            False,
            ["iyear"]
        )
    """

    # --------------------------------------------------------
    # Handle missing dataset
    # --------------------------------------------------------

    if df is None:

        return (
            False,
            REQUIRED_ANALYSIS_COLUMNS.copy(),
        )

    # --------------------------------------------------------
    # Validate DataFrame type
    # --------------------------------------------------------

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        return (
            False,
            REQUIRED_ANALYSIS_COLUMNS.copy(),
        )

    # --------------------------------------------------------
    # Empty dataset
    # --------------------------------------------------------

    if df.empty:

        return (
            False,
            [
                "Dataset contains no rows."
            ],
        )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_ANALYSIS_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        return (
            False,
            missing_columns,
        )

    # --------------------------------------------------------
    # Check whether required columns contain usable values
    # --------------------------------------------------------

    invalid_columns: List[str] = []

    for column in REQUIRED_ANALYSIS_COLUMNS:

        series = df[column]

        # Remove NaN
        valid_values = series.dropna()

        # Remove empty strings
        if valid_values.dtype == "object":

            valid_values = (
                valid_values
                .astype(str)
                .str.strip()
            )

            valid_values = valid_values[
                valid_values != ""
            ]

            valid_values = valid_values[
                valid_values.str.lower()
                != "unknown"
            ]

        if valid_values.empty:

            invalid_columns.append(
                column
            )

    if invalid_columns:

        return (
            False,
            invalid_columns,
        )

    return (
        True,
        [],
    )


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

def _get_missing_columns(
    df: Optional[pd.DataFrame],
    required_columns: List[str],
) -> List[str]:
    """
    Return columns that do not exist in the DataFrame.

    Internal helper used by validation functions.
    """

    if df is None:

        return required_columns.copy()

    return [
        column
        for column in required_columns
        if column not in df.columns
    ]


# ============================================================
# CHECK COLUMN HAS USABLE VALUES
# ============================================================

def _has_usable_values(
    series: pd.Series,
) -> bool:
    """
    Determine whether a Series contains at least one
    meaningful value.

    Empty strings and placeholder "Unknown" values are
    considered unusable.
    """

    if series is None:

        return False

    values = series.dropna()

    if values.empty:

        return False

    # --------------------------------------------------------
    # Handle text columns
    # --------------------------------------------------------

    if (
        pd.api.types.is_object_dtype(values)
        or pd.api.types.is_string_dtype(values)
    ):

        values = (
            values
            .astype(str)
            .str.strip()
        )

        values = values[
            values != ""
        ]

        values = values[
            values.str.lower()
            != "unknown"
        ]

    return not values.empty


# ============================================================
# VALIDATE REQUIRED COLUMN VALUES
# ============================================================

def _validate_column_values(
    df: pd.DataFrame,
    columns: List[str],
) -> List[str]:
    """
    Return required columns that exist but contain no
    usable values.
    """

    invalid_columns: List[str] = []

    for column in columns:

        if column not in df.columns:
            continue

        if not _has_usable_values(
            df[column]
        ):

            invalid_columns.append(
                column
            )

    return invalid_columns


# ============================================================
# VALIDATE ATTACK PREDICTION TARGET
# ============================================================

def _validate_attack_prediction_target(
    df: pd.DataFrame,
) -> Optional[str]:
    """
    Validate the attack prediction target column.

    The attack prediction model needs at least two distinct
    attack types to perform classification.

    Returns
    -------
    None
        Target is valid.

    str
        Error message when invalid.
    """

    target_column = "attacktype1_txt"

    # --------------------------------------------------------
    # Column must exist
    # --------------------------------------------------------

    if target_column not in df.columns:

        return (
            "Missing target column: "
            "attacktype1_txt."
        )

    target = (
        df[target_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove empty values
    # --------------------------------------------------------

    target = target[
        target != ""
    ]

    # --------------------------------------------------------
    # Remove placeholder values
    # --------------------------------------------------------

    target = target[
        target.str.lower()
        != "unknown"
    ]

    # --------------------------------------------------------
    # Check whether any target values remain
    # --------------------------------------------------------

    if target.empty:

        return (
            "attacktype1_txt does not contain "
            "any usable attack type values."
        )

    # --------------------------------------------------------
    # Classification requires at least two classes
    # --------------------------------------------------------

    if target.nunique() < 2:

        return (
            "attacktype1_txt must contain "
            "at least two different attack types."
        )

    return None


# ============================================================
# VALIDATE NUMERIC MODEL COLUMNS
# ============================================================

def _validate_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> List[str]:
    """
    Check numeric columns for usable numeric values.

    Returns columns that contain no usable numeric values.
    """

    invalid_columns: List[str] = []

    for column in columns:

        if column not in df.columns:
            continue

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if numeric_values.empty:

            invalid_columns.append(
                column
            )

    return invalid_columns


# ============================================================
# VALIDATE MODEL DATASET
# ============================================================

def validate_model_dataset(
    df: Optional[pd.DataFrame],
    training_type: str = "attack_prediction",
) -> Tuple[bool, List[str]]:
    """
    Validate whether a standardized dataset contains all
    columns and usable values required by a specific model.

    Supported models:

        attack_prediction
        threat_level

    Parameters
    ----------
    df : pd.DataFrame
        Standardized dataset.

    training_type : str
        Model to validate.

    Returns
    -------
    Tuple[bool, List[str]]

    The second value contains missing/invalid requirements.
    """

    # --------------------------------------------------------
    # Validate training type
    # --------------------------------------------------------

    try:

        required_columns = get_required_columns(
            training_type
        )

    except ValueError as error:

        return (
            False,
            [str(error)],
        )

    # --------------------------------------------------------
    # Validate dataset existence
    # --------------------------------------------------------

    if df is None:

        return (
            False,
            required_columns,
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        return (
            False,
            required_columns,
        )

    # --------------------------------------------------------
    # Empty dataset
    # --------------------------------------------------------

    if df.empty:

        return (
            False,
            [
                "Dataset contains no rows."
            ],
        )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = _get_missing_columns(
        df,
        required_columns,
    )

    if missing_columns:

        return (
            False,
            missing_columns,
        )

    # --------------------------------------------------------
    # Check columns for usable values
    # --------------------------------------------------------

    invalid_columns = _validate_column_values(
        df,
        required_columns,
    )

    if invalid_columns:

        return (
            False,
            invalid_columns,
        )

    # --------------------------------------------------------
    # Attack prediction-specific validation
    # --------------------------------------------------------

    if training_type == "attack_prediction":

        target_error = (
            _validate_attack_prediction_target(
                df
            )
        )

        if target_error is not None:

            return (
                False,
                [target_error],
            )

    # --------------------------------------------------------
    # Threat-level-specific validation
    # --------------------------------------------------------

    elif training_type == "threat_level":

        numeric_required = [
            "nkill",
            "nwound",
        ]

        invalid_numeric = _validate_numeric_columns(
            df,
            numeric_required,
        )

        if invalid_numeric:

            return (
                False,
                invalid_numeric,
            )

    return (
        True,
        [],
    )


# ============================================================
# VALIDATE ALL MODELS
# ============================================================

def validate_all_models(
    df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    """
    Validate the dataset against every supported model.

    Returns a dictionary containing the validation result
    for each model.
    """

    results: Dict[str, object] = {}

    for training_type in (
        "attack_prediction",
        "threat_level",
    ):

        valid, issues = validate_model_dataset(
            df,
            training_type,
        )

        results[training_type] = {
            "valid": valid,
            "issues": issues,
        }

    return results


# ============================================================
# GET VALIDATION STATUS
# ============================================================

def get_validation_status(
    df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    """
    Return an overall validation status for the dataset.

    Useful for displaying dataset health in Streamlit.
    """

    analysis_valid, analysis_issues = (
        validate_dataset(df)
    )

    model_results = validate_all_models(
        df
    )

    return {
        "analysis": {
            "valid": analysis_valid,
            "issues": analysis_issues,
        },
        "models": model_results,
        "overall_valid": (
            analysis_valid
            or any(
                result["valid"]
                for result in model_results.values()
            )
        ),
    }


# ============================================================
# END OF PART 4
#
# Next:
# Part 5
#
# Includes:
#   • get_dataset_summary()
#   • final helper functions
#   • final data_mapper.py integration
# ============================================================

# ============================================================
# PART 5
#
# Includes:
#   • get_dataset_summary()
#   • Mapping statistics
#   • Model readiness statistics
#   • Dataset information helpers
#
# Requires:
#   • Parts 1A, 1B-1, 1B-2, 2, 3 and 4
#
# ============================================================


# ============================================================
# DATASET SUMMARY
# ============================================================

def get_dataset_summary(
    df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    """
    Return a complete summary of an uploaded dataset.

    The summary contains:

        • Number of rows
        • Number of uploaded columns
        • Number of mapped columns
        • Number of unmapped columns
        • Missing standard columns
        • Missing attack-prediction columns
        • Missing threat-level columns
        • Attack-prediction readiness
        • Threat-level readiness

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded dataset.

    Returns
    -------
    dict
    """

    # --------------------------------------------------------
    # Empty / missing dataset
    # --------------------------------------------------------

    if df is None:

        return {
            "rows": 0,

            "uploaded_columns": 0,

            "mapped_columns": 0,

            "unmapped_columns": 0,

            "missing_standard_columns":
                STANDARD_COLUMNS.copy(),

            "attack_prediction_missing":
                ATTACK_PREDICTION_COLUMNS.copy(),

            "threat_level_missing":
                THREAT_LEVEL_COLUMNS.copy(),

            "attack_prediction_ready":
                False,

            "threat_level_ready":
                False,

            "analysis_ready":
                False,

            "mapping_percentage":
                0.0,
        }

    # --------------------------------------------------------
    # Validate DataFrame
    # --------------------------------------------------------

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "get_dataset_summary() expects "
            "a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Generate mapping once
    #
    # This is important because automatic mapping can be
    # relatively expensive for datasets with many columns.
    # --------------------------------------------------------

    mapping = auto_map_columns(
        df
    )

    # --------------------------------------------------------
    # Mapping statistics
    # --------------------------------------------------------

    mapped_columns = [
        info
        for info in mapping.values()
        if info.get("mapped_to") is not None
    ]

    unmapped_columns = [
        column
        for column, info in mapping.items()
        if info.get("mapped_to") is None
    ]

    # --------------------------------------------------------
    # Missing standard columns
    # --------------------------------------------------------

    mapped_standard_columns = {
        info["mapped_to"]
        for info in mapping.values()
        if info.get("mapped_to") is not None
    }

    missing_standard_columns = [
        column
        for column in STANDARD_COLUMNS
        if column not in mapped_standard_columns
    ]

    # --------------------------------------------------------
    # Model-specific missing columns
    # --------------------------------------------------------

    attack_prediction_missing = (
        get_missing_training_columns(
            df,
            "attack_prediction",
        )
    )

    threat_level_missing = (
        get_missing_training_columns(
            df,
            "threat_level",
        )
    )

    # --------------------------------------------------------
    # Model validation
    # --------------------------------------------------------

    attack_prediction_valid, attack_prediction_issues = (
        validate_model_dataset(
            df,
            "attack_prediction",
        )
    )

    threat_level_valid, threat_level_issues = (
        validate_model_dataset(
            df,
            "threat_level",
        )
    )

    # --------------------------------------------------------
    # General dashboard validation
    # --------------------------------------------------------

    analysis_ready, analysis_issues = (
        validate_dataset(
            df
        )
    )

    # --------------------------------------------------------
    # Mapping percentage
    # --------------------------------------------------------

    uploaded_column_count = len(
        df.columns
    )

    if uploaded_column_count > 0:

        mapping_percentage = (
            len(mapped_columns)
            / uploaded_column_count
        ) * 100

    else:

        mapping_percentage = 0.0

    # --------------------------------------------------------
    # Return summary
    # --------------------------------------------------------

    return {
        "rows":
            len(df),

        "uploaded_columns":
            uploaded_column_count,

        "mapped_columns":
            len(mapped_columns),

        "unmapped_columns":
            len(unmapped_columns),

        "mapping_percentage":
            round(
                mapping_percentage,
                2,
            ),

        "missing_standard_columns":
            missing_standard_columns,

        "attack_prediction_missing":
            attack_prediction_missing,

        "threat_level_missing":
            threat_level_missing,

        "attack_prediction_ready":
            attack_prediction_valid,

        "threat_level_ready":
            threat_level_valid,

        "analysis_ready":
            analysis_ready,

        "analysis_issues":
            analysis_issues,

        "attack_prediction_issues":
            attack_prediction_issues,

        "threat_level_issues":
            threat_level_issues,
    }


# ============================================================
# GET MAPPING STATISTICS
# ============================================================

def get_mapping_statistics(
    df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    """
    Return detailed statistics about automatic column mapping.

    Useful for the dataset-mapping UI.
    """

    if df is None:

        return {
            "total": 0,
            "mapped": 0,
            "unmapped": 0,
            "exact": 0,
            "alias": 0,
            "fuzzy": 0,
            "duplicate": 0,
            "mapping_percentage": 0.0,
        }

    mapping = auto_map_columns(
        df
    )

    total = len(
        mapping
    )

    mapped = sum(
        1
        for info in mapping.values()
        if info.get("status") == "Mapped"
    )

    unmapped = sum(
        1
        for info in mapping.values()
        if info.get("status") != "Mapped"
    )

    exact = sum(
        1
        for info in mapping.values()
        if info.get("match_type") == "Exact Match"
    )

    alias = sum(
        1
        for info in mapping.values()
        if info.get("match_type") == "Alias Match"
    )

    fuzzy = sum(
        1
        for info in mapping.values()
        if info.get("match_type") == "Fuzzy Match"
    )

    duplicate = sum(
        1
        for info in mapping.values()
        if info.get("match_type") == "Duplicate Match"
    )

    mapping_percentage = (
        mapped / total * 100
        if total > 0
        else 0.0
    )

    return {
        "total": total,

        "mapped": mapped,

        "unmapped": unmapped,

        "exact": exact,

        "alias": alias,

        "fuzzy": fuzzy,

        "duplicate": duplicate,

        "mapping_percentage":
            round(
                mapping_percentage,
                2,
            ),
    }


# ============================================================
# GET DATASET DIMENSIONS
# ============================================================

def get_dataset_dimensions(
    df: Optional[pd.DataFrame],
) -> Tuple[int, int]:
    """
    Return:

        (rows, columns)

    for the supplied dataset.
    """

    if df is None:

        return (
            0,
            0,
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        return (
            0,
            0,
        )

    return (
        len(df),
        len(df.columns),
    )


# ============================================================
# GET COLUMN MAPPING DICTIONARY
# ============================================================

def get_simple_mapping(
    df: Optional[pd.DataFrame],
) -> Dict[str, str]:
    """
    Return only successful column mappings.

    Example
    -------
    {
        "Year": "iyear",
        "Country": "country_txt",
        "Fatalities": "nkill"
    }

    Unlike auto_map_columns(), this function removes all
    metadata and returns a simple rename dictionary.

    Useful when passing mappings directly to pandas.
    """

    mapping = auto_map_columns(
        df
    )

    return {
        uploaded_column: info["mapped_to"]
        for uploaded_column, info in mapping.items()
        if info.get("mapped_to") is not None
    }


# ============================================================
# GET STANDARDIZED COLUMN ORDER
# ============================================================

def get_standard_column_order(
    df: Optional[pd.DataFrame],
) -> List[str]:
    """
    Return available standard columns in the application's
    canonical order.
    """

    if df is None:

        return []

    return [
        column
        for column in STANDARD_COLUMNS
        if column in df.columns
    ]


# ============================================================
# CHECK WHETHER DATASET IS MAPPED
# ============================================================

def is_dataset_mapped(
    df: Optional[pd.DataFrame],
) -> bool:
    """
    Return True if at least one uploaded column can be
    mapped to an application-standard column.
    """

    if df is None or df.empty:

        return False

    return len(
        get_mapped_columns(df)
    ) > 0


# ============================================================
# CHECK WHETHER DATASET IS READY FOR DASHBOARD
# ============================================================

def is_dataset_ready(
    df: Optional[pd.DataFrame],
) -> bool:
    """
    Return True if the dataset satisfies the minimum
    requirements for the general dashboard.
    """

    valid, _ = validate_dataset(
        df
    )

    return valid


# ============================================================
# CHECK WHETHER DATASET IS READY FOR MODEL
# ============================================================

def is_model_ready(
    df: Optional[pd.DataFrame],
    training_type: str = "attack_prediction",
) -> bool:
    """
    Return True when a dataset is valid for the selected
    machine-learning model.
    """

    valid, _ = validate_model_dataset(
        df,
        training_type,
    )

    return valid


# ============================================================
# COMPLETE DATASET REPORT
# ============================================================

def get_dataset_report(
    df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    """
    Generate a complete dataset report.

    This combines:

        • Dataset summary
        • Mapping statistics
        • Model readiness
        • Dataset dimensions

    Useful for the Streamlit Settings/Data Explorer pages.
    """

    if df is None:

        return {
            "dimensions": {
                "rows": 0,
                "columns": 0,
            },

            "summary":
                get_dataset_summary(None),

            "mapping":
                get_mapping_statistics(None),

            "models":
                get_model_readiness(None),
        }

    rows, columns = (
        get_dataset_dimensions(df)
    )

    return {
        "dimensions": {
            "rows": rows,
            "columns": columns,
        },

        "summary":
            get_dataset_summary(df),

        "mapping":
            get_mapping_statistics(df),

        "models":
            get_model_readiness(df),
    }


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # --------------------------------------------------------
    # Constants
    # --------------------------------------------------------

    "STANDARD_COLUMNS",
    "REQUIRED_ANALYSIS_COLUMNS",
    "ATTACK_PREDICTION_COLUMNS",
    "THREAT_LEVEL_COLUMNS",
    "COLUMN_ALIASES",
    "FUZZY_THRESHOLD",

    # --------------------------------------------------------
    # Mapping
    # --------------------------------------------------------

    "normalize_column_name",
    "get_column_candidates",
    "get_normalized_lookup",
    "get_normalized_candidates",
    "find_column_match",
    "explain_column_match",
    "auto_map_columns",
    "get_mapping_table",
    "get_mapped_columns",
    "get_unmapped_columns",
    "get_missing_standard_columns",
    "get_simple_mapping",

    # --------------------------------------------------------
    # Model requirements
    # --------------------------------------------------------

    "get_required_columns",
    "get_missing_training_columns",
    "check_training_eligibility",
    "is_supported_training_type",
    "get_model_requirements",
    "get_model_readiness",

    # --------------------------------------------------------
    # Dataset preparation
    # --------------------------------------------------------

    "apply_mapping",
    "prepare_dataset",

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    "validate_dataset",
    "validate_model_dataset",
    "validate_all_models",
    "get_validation_status",

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    "get_dataset_summary",
    "get_mapping_statistics",
    "get_dataset_dimensions",
    "get_standard_column_order",
    "is_dataset_mapped",
    "is_dataset_ready",
    "is_model_ready",
    "get_dataset_report",
]


# ============================================================
# END OF utils/data_mapper.py
# ============================================================