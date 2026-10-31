"""Model definitions, hyper-parameter grids, nested cross-validation and training.

Run the whole training stage from the command line:

    python -m src.train            # 5-fold x 3-repeat nested CV (default)
    python -m src.train --repeats 1

Protocol
--------
1. A stratified 80/20 split is made once (``preprocessing.get_split``). The 20 % test set
   is NOT touched during training, tuning or model selection.
2. Model comparison uses *nested* cross-validation on the training set:
       outer loop  -> estimates generalisation performance
       inner loop  -> GridSearchCV tunes hyper-parameters (scored by ROC-AUC)
   so the reported CV scores are not inflated by the tuning itself.
3. Each model is then tuned once more on the full training set and saved.
4. The "best" model is the one with the highest nested-CV ROC-AUC (chosen before
   looking at the test set).
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import MODELS_DIR, RANDOM_STATE, RESULTS_DIR, build_preprocessor, get_split

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
METRICS = list(SCORING)

# --------------------------------------------------------------------------- #
# Models and search grids (kept deliberately small: 303 samples)
# --------------------------------------------------------------------------- #
def get_models() -> dict[str, tuple[object, dict]]:
    """name -> (estimator, param_grid). Grid keys are prefixed with ``clf__``."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
            {
                "clf__max_depth": [3, 5, None],
                "clf__min_samples_leaf": [1, 3, 5],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [50, 100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3, 4],
            },
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    """Preprocessing + classifier in one object (no leakage across CV folds)."""
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


def make_search(estimator, grid: dict, n_jobs: int = 1) -> GridSearchCV:
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(
        make_pipeline(estimator), grid, scoring="roc_auc", cv=inner, n_jobs=n_jobs, refit=True
    )


# --------------------------------------------------------------------------- #
# Nested CV
# --------------------------------------------------------------------------- #
def nested_cv(estimator, grid: dict, X, y, n_splits: int = 5, n_repeats: int = 3, n_jobs: int = 1):
    """Nested CV for a tuned model. Returns (per-fold scores DataFrame, best params per fold)."""
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    res = cross_validate(
        make_search(estimator, grid),
        X, y, cv=outer, scoring=SCORING, n_jobs=n_jobs, return_estimator=True,
    )
    folds = pd.DataFrame({m: res[f"test_{m}"] for m in METRICS})
    best_params = [est.best_params_ for est in res["estimator"]]
    return folds, best_params


def baseline_cv(X, y, n_splits: int = 5, n_repeats: int = 3) -> pd.DataFrame:
    """Majority-class baseline, evaluated with the same outer CV (reference row)."""
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    res = cross_validate(
        DummyClassifier(strategy="most_frequent"), X, y, cv=outer, scoring=SCORING, error_score=0
    )
    return pd.DataFrame({m: res[f"test_{m}"] for m in METRICS})


def summarise(folds: pd.DataFrame) -> dict:
    out = {}
    for m in METRICS:
        out[f"{m}_mean"] = folds[m].mean()
        out[f"{m}_std"] = folds[m].std()
    return out


# --------------------------------------------------------------------------- #
# Final fit + persistence
# --------------------------------------------------------------------------- #
def fit_final(estimator, grid: dict, X_train, y_train) -> GridSearchCV:
    """Tune on the full training set and refit the best configuration."""
    return make_search(estimator, grid).fit(X_train, y_train)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def save_artifacts(final_models: dict, cv_summary: pd.DataFrame, folds_long: pd.DataFrame, best_params: dict):
    """Write models, CV tables and best hyper-parameters to disk."""
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    for name, model in final_models.items():
        joblib.dump(model, MODELS_DIR / f"{_slug(name)}.joblib")

    real = cv_summary.drop(index="Majority-class baseline", errors="ignore")
    best_name = real["roc_auc_mean"].idxmax()
    joblib.dump(final_models[best_name], MODELS_DIR / "best_model.joblib")

    cv_summary.to_csv(RESULTS_DIR / "cv_results.csv")
    folds_long.to_csv(RESULTS_DIR / "cv_fold_scores.csv", index=False)
    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump({"selected_model": best_name, "params": best_params}, f, indent=2, default=str)
    return best_name


def run_training(n_repeats: int = 3, verbose: bool = True) -> str:
    """Full training stage. Returns the name of the selected model."""
    X_train, X_test, y_train, y_test = get_split()
    fold_frames, summary, final_models, best_params = [], {}, {}, {}

    base = baseline_cv(X_train, y_train, n_repeats=n_repeats)
    summary["Majority-class baseline"] = summarise(base)

    for name, (est, grid) in get_models().items():
        t0 = time.time()
        folds, fold_params = nested_cv(est, grid, X_train, y_train, n_repeats=n_repeats)
        summary[name] = summarise(folds)
        fold_frames.append(folds.assign(model=name))

        final = fit_final(est, grid, X_train, y_train)
        final_models[name] = final.best_estimator_
        best_params[name] = {k.replace("clf__", ""): v for k, v in final.best_params_.items()}
        if verbose:
            print(f"{name:<22} nested-CV ROC-AUC = {summary[name]['roc_auc_mean']:.3f} "
                  f"+/- {summary[name]['roc_auc_std']:.3f}   ({time.time() - t0:.0f}s)")

    cv_summary = pd.DataFrame(summary).T
    folds_long = pd.concat(fold_frames, ignore_index=True)
    best = save_artifacts(final_models, cv_summary, folds_long, best_params)
    if verbose:
        print(f"\nSelected model (highest nested-CV ROC-AUC): {best}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare MedPredict models.")
    parser.add_argument("--repeats", type=int, default=3, help="outer-CV repeats (default 3)")
    args = parser.parse_args()
    run_training(n_repeats=args.repeats)
