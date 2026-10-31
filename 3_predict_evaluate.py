"""STEP 3 - Predict and evaluate.

Run:  python 3_predict_evaluate.py      (after python 2_train.py)

What it does
    1. Loads the trained models and predicts the 61 held-out test patients
       (the test set was never used for training, tuning or model selection).
    2. Metrics: accuracy, precision, recall, specificity, F1, ROC-AUC + bootstrap 95 % CIs.
    3. Figures: confusion matrices, ROC, precision-recall, calibration, model comparison.
    4. Decision-threshold analysis (threshold chosen on TRAINING data only).
    5. Feature importance (permutation importance).
    6. Example: predict a single new patient.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd

from src import evaluate as ev
from src.preprocessing import FEATURES, MODELS_DIR, RESULTS_DIR, get_split

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


# --------------------------------------------------------------------------- #
# Load data and trained models
# --------------------------------------------------------------------------- #
X_train, X_test, y_train, y_test = get_split()
models = {name: joblib.load(MODELS_DIR / f"{slug(name)}.joblib") for name in ev.PALETTE}
cv_summary = pd.read_csv(RESULTS_DIR / "cv_results.csv", index_col=0)
selected = json.load(open(RESULTS_DIR / "best_params.json"))["selected_model"]
print(f"Test set: {len(y_test)} patients, {int(y_test.sum())} with disease ({y_test.mean():.1%})")
print(f"Selected model (chosen in step 2, before seeing the test set): {selected}")

# --------------------------------------------------------------------------- #
# 1. Predictions
# --------------------------------------------------------------------------- #
header("1. PREDICTIONS ON THE TEST SET")
predictions = pd.DataFrame({"actual": y_test.values}, index=X_test.index)
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    predictions[f"{name} - probability"] = proba.round(3)
    predictions[f"{name} - predicted"] = (proba >= 0.5).astype(int)
predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index_label="patient_row")
print(f"First 10 test patients (0 = no disease, 1 = disease), {selected}:")
cols = ["actual", f"{selected} - probability", f"{selected} - predicted"]
print(predictions[cols].head(10).to_string())
print(f"\nAll predictions saved -> {RESULTS_DIR / 'test_predictions.csv'}")

# --------------------------------------------------------------------------- #
# 2. Metrics
# --------------------------------------------------------------------------- #
header("2. TEST-SET PERFORMANCE (threshold = 0.5)")
test_metrics = ev.evaluate_models(models, X_test, y_test)
print(test_metrics[["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier"]]
      .round(3).to_string())
test_metrics.to_csv(RESULTS_DIR / "test_metrics.csv")

print("\nBootstrap 95 % confidence intervals (2000 resamples of the 61 test patients):")
ci_df = ev.ci_table(models, X_test, y_test)[["accuracy", "precision", "recall", "f1", "roc_auc"]]
print(ci_df.to_string())
ci_df.to_csv(RESULTS_DIR / "test_metrics_ci.csv")

print("\nErrors per model:")
print(test_metrics[["fn", "fp"]].astype(int)
      .rename(columns={"fn": "missed disease (FN)", "fp": "false alarms (FP)"}).to_string())

print("\nNested-CV (training) vs test ROC-AUC:")
print(pd.DataFrame({
    "CV ROC-AUC": [f"{cv_summary.loc[n, 'roc_auc_mean']:.3f} +/- {cv_summary.loc[n, 'roc_auc_std']:.3f}"
                   for n in models],
    "test ROC-AUC": [f"{test_metrics.loc[n, 'roc_auc']:.3f}" for n in models],
}, index=list(models)).to_string())
print("-> with only 61 test patients the CV estimate is the more reliable one; "
      "differences between the models are within noise.")

# --------------------------------------------------------------------------- #
# 3. Figures
# --------------------------------------------------------------------------- #
header("3. FIGURES")
figures = {
    "confusion_matrix.png": ev.plot_confusion_matrices(models, X_test, y_test),
    "roc_curve.png": ev.plot_roc_curves(models, X_test, y_test),
    "pr_curve.png": ev.plot_pr_curves(models, X_test, y_test),
    "calibration_curve.png": ev.plot_calibration(models, X_test, y_test),
    "model_comparison.png": ev.plot_model_comparison(cv_summary, test_metrics),
}
for filename, fig in figures.items():
    plt.close(fig)
    print(f"  [figure saved] {RESULTS_DIR / filename}")

# --------------------------------------------------------------------------- #
# 4. Decision threshold
# --------------------------------------------------------------------------- #
header("4. DECISION THRESHOLD: PRECISION vs RECALL")
oof = ev.oof_probabilities(models[selected], X_train, y_train)  # 5-fold CV predictions on the training set
fig, oof_table = ev.plot_threshold_analysis(y_train, oof, f"{selected} (out-of-fold, training set)")
plt.close(fig)
print("Chosen on out-of-fold TRAINING predictions (not on the test set):")
print(oof_table.loc[[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]].round(3).to_string())
threshold = float(oof_table.index[oof_table["recall"] >= 0.90].max())
print(f"\nScreening threshold = {threshold:.2f} (highest threshold with recall >= 90 % on the training data)")

proba_test = models[selected].predict_proba(X_test)[:, 1]
operating_points = pd.DataFrame({
    "default 0.50": ev.compute_metrics(y_test, proba_test, 0.5),
    f"screening {threshold:.2f}": ev.compute_metrics(y_test, proba_test, threshold),
}).T[["precision", "recall", "specificity", "f1", "fn", "fp"]]
operating_points[["fn", "fp"]] = operating_points[["fn", "fp"]].astype(int)
print(f"\n{selected} on the test set:")
print(operating_points.round(3).to_string())
operating_points.to_csv(RESULTS_DIR / "operating_points.csv")
print(f"  [figure saved] {RESULTS_DIR / 'threshold_analysis.png'}")

# --------------------------------------------------------------------------- #
# 5. Feature importance
# --------------------------------------------------------------------------- #
header("5. WHICH FEATURES DOES THE MODEL USE? (permutation importance)")
fig, importance = ev.plot_feature_importance(models[selected], selected, X_test, y_test)
plt.close(fig)
importance.to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)
print("Drop in test ROC-AUC when a feature is shuffled:")
print(importance.round(3).to_string(index=False))
print(f"  [figure saved] {RESULTS_DIR / 'feature_importance.png'}")

# --------------------------------------------------------------------------- #
# 6. Predict a single new patient
# --------------------------------------------------------------------------- #
header("6. EXAMPLE: PREDICT A NEW PATIENT")
new_patient = pd.DataFrame([{
    "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 240, "fbs": 0, "restecg": 2,
    "thalach": 120, "exang": 1, "oldpeak": 2.0, "slope": 2, "ca": 2, "thal": 7,
}])[FEATURES]  # illustrative values, not a real patient
print(new_patient.T.rename(columns={0: "value"}).to_string())

prob = models[selected].predict_proba(new_patient)[0, 1]
print(f"\nModel score ({selected}): {prob:.2f}")
print(f"  at threshold 0.50 -> {'higher predicted risk' if prob >= 0.5 else 'lower predicted risk'}")
print(f"  at threshold {threshold:.2f} -> {'higher predicted risk' if prob >= threshold else 'lower predicted risk'}")
print("\nThis is an educational prototype, NOT a medical diagnostic tool. "
      "The score is a model output, not a calibrated probability.")

print("\nDone. Tables and figures are in:", RESULTS_DIR)
