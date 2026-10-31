"""STEP 2 - Train the models.

Run:  python 2_train.py              (5-fold x 3-repeat nested CV, a few minutes)
      python 2_train.py --repeats 1  (faster)

What it does
    1. Splits the data once: 80 % train (242 patients) / 20 % test (61 patients, NOT used here).
    2. Trains 4 models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
       (every model is a Pipeline: imputation + scaling + encoding + classifier).
    3. Compares them with NESTED cross-validation on the training set
       (inner loop tunes hyper-parameters, outer loop measures performance).
    4. Selects the model with the highest CV ROC-AUC, re-tunes every model on the full
       training set and saves them to models/  (tables -> results/).
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from src import train as tr
from src.preprocessing import MODELS_DIR, RESULTS_DIR, get_split

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def main(n_repeats: int = 3) -> None:
    X_train, X_test, y_train, y_test = get_split()
    print(f"Training set: {X_train.shape[0]} patients ({y_train.mean():.1%} with disease)")
    print(f"Test set:     {X_test.shape[0]} patients - kept aside for step 3\n")

    summary, fold_frames, final_models, best_params = {}, [], {}, {}

    # Reference floor: always predict the majority class
    base = tr.baseline_cv(X_train, y_train, n_repeats=n_repeats)
    summary["Majority-class baseline"] = tr.summarise(base)
    print(f"Majority-class baseline: accuracy {base['accuracy'].mean():.3f}, ROC-AUC {base['roc_auc'].mean():.3f}\n")

    for name, (estimator, grid) in tr.get_models().items():
        t0 = time.time()
        print(f"Training {name} ...")
        print(f"   hyper-parameter grid: { {k.replace('clf__', ''): v for k, v in grid.items()} }")

        # 1) honest performance estimate (tuning happens inside each outer fold)
        folds, _ = tr.nested_cv(estimator, grid, X_train, y_train, n_repeats=n_repeats)
        summary[name] = tr.summarise(folds)
        fold_frames.append(folds.assign(model=name))

        # 2) final model: tune once on the whole training set
        final = tr.fit_final(estimator, grid, X_train, y_train)
        final_models[name] = final.best_estimator_
        best_params[name] = {k.replace("clf__", ""): v for k, v in final.best_params_.items()}

        print(f"   nested-CV ROC-AUC = {summary[name]['roc_auc_mean']:.3f} +/- {summary[name]['roc_auc_std']:.3f}"
              f"   best params: {best_params[name]}   ({time.time() - t0:.0f} s)\n")

    cv_summary = pd.DataFrame(summary).T
    folds_long = pd.concat(fold_frames, ignore_index=True)
    selected = tr.save_artifacts(final_models, cv_summary, folds_long, best_params)

    # Readable results table
    labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}
    table = pd.DataFrame({
        labels[m]: [f"{cv_summary.loc[n, m + '_mean']:.3f} +/- {cv_summary.loc[n, m + '_std']:.3f}"
                    for n in cv_summary.index]
        for m in tr.METRICS
    }, index=cv_summary.index)
    print("=" * 70)
    print(f"NESTED CROSS-VALIDATION RESULTS (training set, {len(folds_long) // len(final_models)} outer folds)")
    print("=" * 70)
    print(table.to_string())
    print(f"\nSelected model (highest CV ROC-AUC): {selected}")
    print(f"Saved models -> {MODELS_DIR}")
    print(f"Saved tables -> {RESULTS_DIR} (cv_results.csv, cv_fold_scores.csv, best_params.json)")
    print("\nNext:  python 3_predict_evaluate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare the MedPredict models.")
    parser.add_argument("--repeats", type=int, default=3, help="outer-CV repeats (default 3)")
    main(parser.parse_args().repeats)
