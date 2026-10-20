"""ML feature preprocessing and pipeline construction."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Numeric feature columns
NUMERIC_FEATURES = [
    "ear_avg",
    "ear_left",
    "ear_right",
    "mar",
    "face_quality",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "shoulder_tilt_deg",
    "torso_upright_ratio",
    "head_shoulder_offset",
    "phone_confidence",
]

# Boolean / Binary indicators
BINARY_FEATURES = [
    "face_detected",
    "head_pose_valid",
    "pose_detected",
    "wrists_near_head",
    "wrists_near_table",
    "phone_detected",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "observability_level",
]


def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for robust preprocessing."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNOBSERVABLE")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """Separates feature columns, ground truth target labels, and participant grouping IDs."""
    available_cols = [c for c in (NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES) if c in df.columns]
    X = df[available_cols].copy()

    # Fill binary missing values with 0
    for col in BINARY_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(0).astype(int)

    y = df["human_behavior_label"] if "human_behavior_label" in df.columns else None
    groups = df["participant_id"] if "participant_id" in df.columns else None

    return X, y, groups
