"""Data loading, feature definitions and the preprocessing pipeline for MedPredict.

Dataset: UCI Heart Disease - Cleveland Clinic Foundation (processed.cleveland.data)
https://archive.ics.uci.edu/dataset/45/heart+disease
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Feature groups (drive the ColumnTransformer)
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]  # ca = 0-3 vessel count (ordinal)
BINARY = ["sex", "fbs", "exang"]
NOMINAL = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC + BINARY + NOMINAL

# Human-readable names / value labels (used in plots and tables)
FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thallium stress-test result",
}

VALUE_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "<= 120 mg/dl", 1: "> 120 mg/dl"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the Cleveland data set.

    Returns a DataFrame with the 13 input features, ``num`` (original 0-4 label)
    and ``target`` (binary: 0 = no disease, 1 = disease present).
    Missing values (marked ``?`` in the source file) are returned as NaN.

    If the file is not available locally it is downloaded from UCI.
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = pd.read_csv(DATA_URL, header=None, names=COLUMNS, na_values="?")
        except Exception as exc:  # pragma: no cover
            raise FileNotFoundError(
                f"{path} not found and download failed ({exc}). Download "
                f"'processed.cleveland.data' from {DATA_URL} into {path.parent}."
            ) from exc
        raw.to_csv(path, header=False, index=False, na_rep="?")

    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a loaded DataFrame into features and the binary target."""
    return df[FEATURES].copy(), df["target"].copy()


def get_split(test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train/test split. Every notebook and script uses this one split."""
    X, y = get_X_y(load_data())
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# --------------------------------------------------------------------------- #
# Preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Imputation + scaling + encoding.

    All statistics (medians, modes, means, category lists) are learned from the
    training data only, because this transformer lives inside the model Pipeline
    and is therefore re-fitted within every cross-validation fold.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("bin", binary, BINARY),
            ("nom", nominal, NOMINAL),
        ],
        verbose_feature_names_out=False,
    )


def pretty_feature_names(names) -> list[str]:
    """'cp_4.0' -> 'cp_4' (one-hot column names from the encoder)."""
    return [str(n).replace(".0", "") for n in names]
