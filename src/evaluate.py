"""Evaluation metrics and plotting helpers for MedPredict."""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocessing import (
    FEATURE_DESCRIPTIONS, MODELS_DIR, RANDOM_STATE, RESULTS_DIR, get_split,
)

PALETTE = {
    "Logistic Regression": "#1f77b4",
    "Random Forest": "#2ca02c",
    "Gradient Boosting": "#ff7f0e",
    "XGBoost": "#d62728",
}
CLASS_NAMES = ["No disease", "Disease"]

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, filename: str | None):
    if filename:
        RESULTS_DIR.mkdir(exist_ok=True)
        fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Classification metrics at a probability threshold (positive class = disease)."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def evaluate_models(models: dict, X_test, y_test, threshold: float = 0.5) -> pd.DataFrame:
    """One row of test-set metrics per fitted model."""
    rows = {}
    for name, model in models.items():
        rows[name] = compute_metrics(y_test, model.predict_proba(X_test)[:, 1], threshold)
    return pd.DataFrame(rows).T


def bootstrap_ci(y_true, y_proba, threshold: float = 0.5, n_boot: int = 2000, seed: int = RANDOM_STATE):
    """Percentile 95 % bootstrap CIs on the test set for the main metrics."""
    rng = np.random.default_rng(seed)
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    n, samples = len(y_true), {m: [] for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], y_proba[idx], threshold)
        for k in samples:
            samples[k].append(m[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in samples.items()}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_roc_curves(models: dict, X_test, y_test, filename: str | None = "roc_curve.png"):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, model in models.items():
        p = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, lw=2, color=PALETTE.get(name), label=f"{name} (AUC = {roc_auc_score(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (recall)",
           title="ROC curves - held-out test set")
    ax.legend(loc="lower right")
    _save(fig, filename)
    return fig


def plot_pr_curves(models: dict, X_test, y_test, filename: str | None = "pr_curve.png"):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, model in models.items():
        p = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, lw=2, color=PALETTE.get(name),
                label=f"{name} (AP = {average_precision_score(y_test, p):.3f})")
    ax.axhline(np.mean(y_test), color="k", ls="--", lw=1, label="Prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves - held-out test set",
           ylim=(0, 1.02))
    ax.legend(loc="lower left")
    _save(fig, filename)
    return fig


def plot_confusion_matrices(models: dict, X_test, y_test, threshold: float = 0.5,
                            filename: str | None = "confusion_matrix.png"):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, (name, model) in zip(axes, models.items()):
        pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, vmin=0, vmax=max(np.bincount(y_test)),
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 16})
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    fig.suptitle(f"Confusion matrices - held-out test set (threshold = {threshold})", y=1.03)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_model_comparison(cv_summary: pd.DataFrame, test_metrics: pd.DataFrame,
                          filename: str | None = "model_comparison.png"):
    """Left: nested-CV ROC-AUC (mean +/- sd). Right: test-set metrics per model."""
    names = [n for n in test_metrics.index]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.8]})

    ax = axes[0]
    means = cv_summary.loc[names, "roc_auc_mean"]
    stds = cv_summary.loc[names, "roc_auc_std"]
    ax.barh(names, means, xerr=stds, color=[PALETTE[n] for n in names], capsize=4, alpha=0.9)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(0.505, i, f"{m:.3f}", va="center", color="white", fontweight="bold")
    ax.set(xlim=(0.5, 1.0), xlabel="ROC-AUC", title="Nested CV on training set (mean +/- SD)")
    ax.invert_yaxis()

    ax = axes[1]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    width = 0.8 / len(names)
    x = np.arange(len(metrics))
    for i, n in enumerate(names):
        vals = [test_metrics.loc[n, m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=n, color=PALETTE[n])
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set(ylim=(0.5, 1.0), title="Held-out test set (threshold = 0.5)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_cv_distribution(folds_long: pd.DataFrame, metric: str = "roc_auc",
                         filename: str | None = "cv_distribution.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = list(PALETTE)
    sns.boxplot(data=folds_long, x=metric, y="model", hue="model", order=order, hue_order=order,
                palette=PALETTE, legend=False, ax=ax, width=0.55)
    sns.stripplot(data=folds_long, x=metric, y="model", order=order, color="k", alpha=0.35, size=3, ax=ax)
    ax.set(title=f"Nested CV: {metric} across the 15 outer folds", xlabel=metric, ylabel="")
    _save(fig, filename)
    return fig


def plot_calibration(models: dict, X_test, y_test, n_bins: int = 5,
                     filename: str | None = "calibration_curve.png"):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, model in models.items():
        p = model.predict_proba(X_test)[:, 1]
        frac, mean_pred = calibration_curve(y_test, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", color=PALETTE.get(name), label=f"{name} (Brier = {brier_score_loss(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction with disease",
           title="Calibration - held-out test set (quantile bins)")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, filename)
    return fig


def threshold_table(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2) if thresholds is None else thresholds
    rows = [{"threshold": t, **{k: v for k, v in compute_metrics(y_true, y_proba, t).items()
                                if k in ("precision", "recall", "specificity", "f1", "fp", "fn")}}
            for t in thresholds]
    return pd.DataFrame(rows).set_index("threshold")


def plot_threshold_analysis(y_true, y_proba, model_name: str, filename: str | None = "threshold_analysis.png"):
    tbl = threshold_table(y_true, y_proba)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for col, style in [("recall", "-o"), ("specificity", "-s"), ("precision", "-^"), ("f1", "--")]:
        ax.plot(tbl.index, tbl[col], style, label=col.capitalize() if col != "f1" else "F1", ms=4)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set(xlabel="Decision threshold", ylabel="Score", ylim=(0, 1.02), title=f"{model_name}: metrics vs threshold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=9)
    ax = axes[1]
    ax.bar(tbl.index - 0.008, tbl["fn"], width=0.015, color="#d62728", label="False negatives (missed disease)")
    ax.bar(tbl.index + 0.008, tbl["fp"], width=0.015, color="#1f77b4", label="False positives (false alarms)")
    ax.set(xlabel="Decision threshold", ylabel="Number of errors", title="Error counts vs threshold")
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)
    return fig, tbl


def permutation_importance_table(model, X_test, y_test, n_repeats: int = 50) -> pd.DataFrame:
    """Permutation importance on the original 13 input features (drop in ROC-AUC)."""
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"feature": X_test.columns, "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def plot_feature_importance(model, model_name: str, X_test, y_test,
                            filename: str | None = "feature_importance.png"):
    tbl = permutation_importance_table(model, X_test, y_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    t = tbl.iloc[::-1]
    ax.barh([f"{FEATURE_DESCRIPTIONS[f].split(' (')[0]} [{f}]" for f in t["feature"]],
            t["importance_mean"], xerr=t["importance_std"], color="#4c72b0", capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="Drop in test ROC-AUC when the feature is shuffled",
           title=f"Permutation importance - {model_name}")
    fig.tight_layout()
    _save(fig, filename)
    return fig, tbl


# --------------------------------------------------------------------------- #
# Uncertainty, threshold selection and the one-shot evaluation script
# --------------------------------------------------------------------------- #
def ci_table(models: dict, X_test, y_test, threshold: float = 0.5, n_boot: int = 2000) -> pd.DataFrame:
    """'estimate [lo, hi]' strings for every model and main metric (percentile bootstrap)."""
    rows = {}
    for name, model in models.items():
        p = model.predict_proba(X_test)[:, 1]
        point, ci = compute_metrics(y_test, p, threshold), bootstrap_ci(y_test, p, threshold, n_boot)
        rows[name] = {m: f"{point[m]:.3f} [{lo:.2f}, {hi:.2f}]" for m, (lo, hi) in ci.items()}
    return pd.DataFrame(rows).T


def paired_bootstrap_auc_diff(y_true, p_a, p_b, n_boot: int = 2000, seed: int = RANDOM_STATE):
    """ROC-AUC(a) - ROC-AUC(b) on the same resampled patients. Returns (difference, lo, hi, share > 0)."""
    rng = np.random.default_rng(seed)
    y_true, p_a, p_b = np.asarray(y_true), np.asarray(p_a), np.asarray(p_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], p_a[idx]) - roc_auc_score(y_true[idx], p_b[idx]))
    diffs = np.asarray(diffs)
    point = roc_auc_score(y_true, p_a) - roc_auc_score(y_true, p_b)
    return point, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), float((diffs > 0).mean())


def oof_probabilities(model, X_train, y_train) -> np.ndarray:
    """Out-of-fold predicted probabilities on the training set (5-fold CV)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]


def select_threshold(model, X_train, y_train, min_recall: float = 0.90) -> tuple[float, pd.DataFrame]:
    """Choose a screening threshold using ONLY the training data.

    Out-of-fold probabilities are obtained with 5-fold CV on the training set; the returned
    threshold is the highest one whose out-of-fold recall is at least ``min_recall``
    (i.e. the most specific operating point that still catches ``min_recall`` of the disease cases).
    """
    tbl = threshold_table(y_train, oof_probabilities(model, X_train, y_train))
    ok = tbl[tbl["recall"] >= min_recall]
    return float(ok.index.max()), tbl


def run_evaluation(verbose: bool = True) -> pd.DataFrame:
    """Reproduce every table and figure of notebook 04 from the saved models.

        python -m src.evaluate
    """
    X_train, X_test, y_train, y_test = get_split()
    cv_summary = pd.read_csv(RESULTS_DIR / "cv_results.csv", index_col=0)
    selected = json.load(open(RESULTS_DIR / "best_params.json"))["selected_model"]
    models = {n: joblib.load(MODELS_DIR / f"{n.lower().replace(' ', '_')}.joblib") for n in PALETTE}

    test_metrics = evaluate_models(models, X_test, y_test)
    test_metrics.to_csv(RESULTS_DIR / "test_metrics.csv")
    ci_table(models, X_test, y_test).to_csv(RESULTS_DIR / "test_metrics_ci.csv")

    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_pr_curves(models, X_test, y_test)
    plot_model_comparison(cv_summary, test_metrics)
    plot_calibration(models, X_test, y_test)
    plot_feature_importance(models[selected], selected, X_test, y_test)[1].to_csv(
        RESULTS_DIR / "permutation_importance.csv", index=False)

    thr, _ = select_threshold(models[selected], X_train, y_train)
    p = models[selected].predict_proba(X_test)[:, 1]
    op = pd.DataFrame({"threshold 0.5": compute_metrics(y_test, p, 0.5),
                       f"threshold {thr:.2f} (recall >= 0.90 in training CV)": compute_metrics(y_test, p, thr)}).T
    op.to_csv(RESULTS_DIR / "operating_points.csv")
    plt.close("all")
    if verbose:
        print(f"Selected model: {selected}\n")
        print(test_metrics[["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]].round(3))
        print(f"\nFigures and tables written to {RESULTS_DIR}")
    return test_metrics


if __name__ == "__main__":
    run_evaluation()
