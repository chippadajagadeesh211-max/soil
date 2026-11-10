"""
SlopeGuard AI — Preprocessing & Feature Engineering Module
Safely transforms, cleans, and casts features into model-ready tensors.
"""

import pandas as pd
import numpy as np
from config import FEATURE_COLUMNS, CSV_ALIASES


# Baseline statistical moments computed directly from landslide_modelling_dataset_12409.csv
SCALER_BASELINE = {
    'aspect_deg': {'mean': 178.1878, 'std': 98.7455},
    'elevation_m': {'mean': 1780.1436, 'std': 1263.6101},
    'slope_deg': {'mean': 26.7938, 'std': 13.4340}
}


def prepare_feature_tensor(data_input, feature_cols=None) -> pd.DataFrame:
    """
    Constructs a model-ready 2D DataFrame with guaranteed numeric float64 dtypes.
    Accepts:
      - pd.Series (a single dataset row)
      - dict (from manual inputs or JSON)
      - pd.DataFrame (batch uploaded rows)

    Guarantees:
      - All columns in feature_cols exist
      - Exactly in the expected order
      - Every column has float64 dtype (prevents XGBoost object dtype errors)
      - No NaNs (imputed safely with baseline medians)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    if isinstance(data_input, pd.Series):
        df = data_input.to_frame().T
    elif isinstance(data_input, dict):
        df = pd.DataFrame([data_input])
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise TypeError(f"Unsupported data_input type: {type(data_input)}")

    # Auto-compute z-scores if only raw features are supplied
    for raw_col, z_col in [('aspect_deg', 'Aspect_z'), ('elevation_m', 'Elevation_z'), ('slope_deg', 'Slope_z')]:
        if raw_col in df.columns and z_col not in df.columns:
            mean = SCALER_BASELINE[raw_col]['mean']
            std = SCALER_BASELINE[raw_col]['std']
            df[z_col] = (pd.to_numeric(df[raw_col], errors='coerce') - mean) / std

    # Handle one-hot LULC if only lulc_code is supplied
    if 'lulc_code' in df.columns:
        for i in range(10, 110, 10):
            col_name = f'LULC_{i}'
            if col_name not in df.columns:
                df[col_name] = (pd.to_numeric(df['lulc_code'], errors='coerce') == i).astype(float)

    # Ensure all required feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            if col == 'LULC_10':
                df[col] = 1.0  # Default to majority Tree cover in HP
            elif col.startswith('LULC_'):
                df[col] = 0.0
            else:
                df[col] = 0.0

    # Extract exact columns in order
    X = df[feature_cols].copy()

    # STRICT NUMERIC CONVERSION: Fixes "ValueError: DataFrame.dtypes must be int, float, bool or category"
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float64)

    return X


def remap_uploaded_csv_columns(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently maps user-uploaded CSV columns to standard SlopeGuard AI features
    using case-insensitive alias dictionaries.
    """
    col_map = {}
    canonical_columns = set(CSV_ALIASES) | set(FEATURE_COLUMNS) | {'class', 'city'}
    canonical_lookup = {name.lower(): name for name in canonical_columns}
    alias_lookup = {
        alias.lower(): canonical
        for canonical, aliases in CSV_ALIASES.items()
        for alias in aliases
    }

    for column in uploaded_df.columns:
        normalized = str(column).strip().lower()
        canonical = canonical_lookup.get(normalized, alias_lookup.get(normalized))
        if canonical is not None:
            col_map[column] = canonical

    remapped_df = uploaded_df.rename(columns=col_map).copy()
    return remapped_df


def validate_uploaded_dataset(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize and validate uploaded rows before they feed dashboard inference."""
    if uploaded_df.empty:
        raise ValueError("The CSV contains no data rows.")

    normalized_columns = [str(column).strip().lower() for column in uploaded_df.columns]
    if any(not column for column in normalized_columns):
        raise ValueError("The CSV contains a blank column name.")
    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError("The CSV contains duplicate column names (ignoring case and whitespace).")

    df = remap_uploaded_csv_columns(uploaded_df)
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Multiple CSV columns map to the same field: {', '.join(duplicates)}.")

    missing_coordinates = [column for column in ('latitude', 'longitude') if column not in df]
    if missing_coordinates:
        raise ValueError(
            "Missing required map columns: "
            f"{', '.join(missing_coordinates)}. Provide latitude/longitude or lat/lon."
        )

    for raw_column, z_column in (
        ('aspect_deg', 'Aspect_z'),
        ('elevation_m', 'Elevation_z'),
        ('slope_deg', 'Slope_z')
    ):
        if z_column not in df and raw_column in df:
            values = pd.to_numeric(df[raw_column], errors='coerce')
            baseline = SCALER_BASELINE[raw_column]
            df[z_column] = (values - baseline['mean']) / baseline['std']

    if 'lulc_code' in df:
        codes = pd.to_numeric(df['lulc_code'], errors='coerce')
        valid_codes = set(range(10, 110, 10))
        if codes.isna().any():
            raise ValueError("Column 'lulc_code' contains missing or non-numeric value(s).")
        invalid_codes = codes.notna() & ~codes.isin(valid_codes)
        if invalid_codes.any():
            raise ValueError("lulc_code must contain one of the model's supported values: 10, 20, ..., 100.")
        df['lulc_code'] = codes
        for code in sorted(valid_codes):
            column = f'LULC_{code}'
            if column not in df:
                df[column] = (codes == code).astype(float)

    missing_features = [column for column in FEATURE_COLUMNS if column not in df]
    if missing_features:
        recognized_columns = ", ".join(str(column) for column in df.columns)
        raise ValueError(
            "Missing model input columns: "
            f"{', '.join(missing_features)}. Include the model feature columns, "
            "or provide aspect_deg, elevation_m, and slope_deg for their Z-scores "
            "and lulc_code instead of all LULC_10...LULC_100 columns. "
            "Latitude and longitude are required to update the GIS map. "
            f"Detected columns: {recognized_columns}."
        )

    numeric_columns = ['latitude', 'longitude', *FEATURE_COLUMNS]
    if 'class' in df:
        numeric_columns.append('class')
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors='coerce')
        invalid = values.isna() | ~np.isfinite(values)
        if invalid.any():
            raise ValueError(
                f"Column '{column}' contains {int(invalid.sum())} missing or non-numeric value(s). "
                "Correct or remove those rows before uploading."
            )
        df[column] = values

    invalid_latitude = ~df['latitude'].between(-90, 90)
    invalid_longitude = ~df['longitude'].between(-180, 180)
    if invalid_latitude.any() or invalid_longitude.any():
        raise ValueError("Latitude must be within -90..90 and longitude within -180..180.")

    if 'class' in df and not df['class'].isin([0, 1]).all():
        raise ValueError("If supplied, the 'class' column must contain only 0 or 1.")

    land_cover_columns = [f'LULC_{code}' for code in range(10, 110, 10)]
    land_cover_values = df[land_cover_columns].to_numpy()
    if not np.isin(land_cover_values, [0, 1]).all():
        raise ValueError("LULC_10 through LULC_100 must contain only 0 or 1.")
    if not np.equal(land_cover_values.sum(axis=1), 1).all():
        raise ValueError(
            "Each row must have exactly one active land-cover class across "
            "LULC_10 through LULC_100 (or a valid lulc_code)."
        )

    return df.reset_index(drop=True)
