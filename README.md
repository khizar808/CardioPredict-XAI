# MedPredict

**Explainable Machine Learning for Cardiovascular Disease Risk Prediction**

Four classical machine-learning models are trained and compared on the UCI Heart Disease (Cleveland) data using a leakage-free pipeline, nested cross-validation and a one-time held-out test evaluation. SHAP explainability and a Streamlit app are the next stages (see [Status](#status)).

> **Disclaimer.** This is an educational / research prototype. It predicts a label from 13 structured attributes of a historical dataset. It is **not** a medical device and must not be used for diagnosis or treatment decisions.

## Status

| Stage | Notebook | State |
|---|---|---|
| 1. Data understanding and EDA | `01_data_exploration.ipynb` | Done |
| 2. Preprocessing pipeline | `02_preprocessing.ipynb` | Done |
| 3. Model training, tuning, selection | `03_model_training.ipynb` | Done |
| 4. Evaluation and performance | `04_model_evaluation.ipynb` | Done |
| 5. Explainability (SHAP) | `05_explainability.ipynb` | Next |
| 6. Streamlit app and deployment | `app/app.py` | Next |

## Problem statement and research questions

Can routine clinical attributes predict whether angiography will show heart disease, and how much do model choice and evaluation design change the answer on a very small dataset?

| # | Question | Where answered |
|---|---|---|
| RQ1 | How well can classical ML models predict heart disease from structured attributes? | Notebooks 03, 04 |
| RQ2 | How do the algorithms compare on accuracy, precision, recall, F1 and ROC-AUC? | Notebooks 03, 04 |
| RQ3 | Which features contribute most to the predictions? | Notebook 04 (permutation importance), SHAP next |
| RQ4 | Can SHAP explanations improve interpretability of tree-based models? | Stage 5 |

## Dataset

UCI Machine Learning Repository, *Heart Disease*, Cleveland Clinic Foundation subset: 303 patients, 13 input features, angiographic disease status `num` (0-4). The target is binarised: `0` = no disease (164 patients, 54.1 %), `1` = disease present (139 patients, 45.9 %). Six values are missing (`ca`: 4, `thal`: 2). Details, coding and credits: [`data/README.md`](data/README.md).

Source: https://archive.ics.uci.edu/dataset/45/heart+disease

## Methodology

```
303 patients
   |  stratified 80/20 split, seed 42
   |-- 61 test patients   (untouched until notebook 04)
   `-- 242 training patients
          |  Pipeline = imputation + scaling + one-hot encoding + classifier   (fitted inside every fold)
          |  Nested CV: 5 outer folds x 3 repeats, inner GridSearchCV (5-fold, ROC-AUC)
          |  Model selection rule: highest nested-CV ROC-AUC (fixed before seeing the test set)
          `-- final tuning on all 242 -> saved to models/
   |
   `-- one-time evaluation on the 61 test patients: metrics + bootstrap 95 % CIs,
       ROC / PR / calibration curves, confusion matrices, threshold analysis, permutation importance
```

Models: Logistic Regression (baseline), Random Forest, Gradient Boosting, XGBoost, plus a majority-class dummy as the reference floor.

## Exploratory data analysis (highlights)

Full analysis with tests and plots: `notebooks/01_data_exploration.ipynb`.

* Classes are nearly balanced; no duplicates; all values physiologically plausible; 6 missing values in 6 rows (2 %).
* Strongest signals: `thal` (22 % disease with a normal result vs 67-76 % with a defect), `ca` (26 % with 0 vessels rising to 85 % with 3), `exang`, `cp` (asymptomatic patients have the highest rate, 73 %, a known property of this referred-for-angiography cohort), `oldpeak` and `thalach`.
* `fbs` is essentially unrelated to the target; `chol` and `trestbps` are weak.
* 68 % of patients are men, and some categories are tiny (`restecg = 1`: 4 patients).

Figures: `results/eda_*.png`.

## Results

### Nested cross-validation (242 training patients, 15 outer folds, mean +/- SD)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.541 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.829 +/- 0.048 | 0.857 +/- 0.089 | 0.768 +/- 0.093 | 0.804 +/- 0.058 | 0.888 +/- 0.050 |
| **Random Forest** (selected) | 0.809 +/- 0.054 | 0.812 +/- 0.065 | 0.762 +/- 0.095 | 0.783 +/- 0.065 | **0.892 +/- 0.039** |
| Gradient Boosting | 0.804 +/- 0.055 | 0.802 +/- 0.080 | 0.771 +/- 0.094 | 0.782 +/- 0.064 | 0.867 +/- 0.049 |
| XGBoost | 0.796 +/- 0.056 | 0.787 +/- 0.067 | 0.768 +/- 0.103 | 0.773 +/- 0.067 | 0.876 +/- 0.043 |

### Held-out test set (61 patients, threshold 0.5)

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.869 | 0.812 | 0.929 | 0.818 | 0.867 | 0.958 |
| **Random Forest** (selected) | 0.902 | 0.893 | 0.893 | 0.909 | 0.893 | 0.953 |
| Gradient Boosting | 0.885 | 0.862 | 0.893 | 0.879 | 0.877 | 0.953 |
| XGBoost | 0.869 | 0.833 | 0.893 | 0.848 | 0.862 | 0.951 |

Bootstrap 95 % confidence intervals are in `results/test_metrics_ci.csv` (for example Random Forest accuracy 0.82-0.97, recall 0.76-1.00, ROC-AUC 0.90-0.99).

### What the results say

1. **All four models clearly beat the baseline** (cross-validated ROC-AUC 0.87-0.89 vs 0.50).
2. **The models are practically tied.** Differences between them are smaller than the fold-to-fold spread and than the test-set confidence intervals; a regularised logistic regression matches the tree ensembles. Random Forest was selected by the pre-defined rule, not because it is demonstrably better.
3. **Test scores are higher than the CV scores** (ROC-AUC about 0.95 vs 0.89). With only 61 test patients this is most plausibly an easy split, so the nested-CV figures are the more conservative estimate to quote.
4. **Missed cases matter.** At the default threshold roughly one in four (CV) to one in ten (test) patients with disease is missed. A screening threshold of 0.30, chosen on training out-of-fold predictions only, reaches 100 % recall on the test set at the cost of 12 instead of 3 false alarms; the out-of-fold estimate (about 92 % recall, 57 % specificity) is the more realistic expectation.
5. **Scores are not calibrated probabilities** (Random Forest is the least calibrated; Brier 0.113 vs 0.085 for Logistic Regression).
6. **Most used features** (permutation importance): `cp`, `thal`, `ca`, `oldpeak`, `exang`, `thalach`. This describes model behaviour, not causes.

Figures: `results/confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, `model_comparison.png`, `calibration_curve.png`, `threshold_analysis.png`, `feature_importance.png`, `cv_distribution.png`.

## Limitations

* Small dataset (303 patients, 61 in the test set): every estimate is uncertain.
* Single centre, historical data, 68 % men; no external validation.
* Patients were referred for angiography, so this is not a screening population.
* Scores are uncalibrated; feature importance is not causal.

## Ethical considerations

Model outputs are presented as model scores on a research dataset, never as a diagnosis. The choice of decision threshold trades missed cases against false alarms and is a clinical judgement, not a statistical one.

## Installation and usage

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the project as three Python scripts, in order:

```bash
python 1_eda.py                  # 1. understand the data   -> results/eda_*.png
python 2_train.py                # 2. train + compare models -> models/*.joblib (a few minutes)
python 3_predict_evaluate.py     # 3. predict + evaluate     -> results/ tables and figures
```

or everything at once: `python run_all.py` (`--repeats 1` for faster training, `--skip-train` to reuse the saved models).

The notebooks in `notebooks/` contain the same analysis with written observations (`jupyter lab`). The data file is downloaded automatically from UCI on first use if `data/raw/processed.cleveland.data` is missing. Tested with Python 3.12, scikit-learn 1.8, XGBoost 3.4, pandas 3.0.

## Project structure

```
medpredict/
|-- 1_eda.py, 2_train.py, 3_predict_evaluate.py, run_all.py   (run in this order)
|-- README.md, requirements.txt, LICENSE, .gitignore
|-- data/           README.md (provenance), raw/ (git-ignored)
|-- notebooks/      01_data_exploration, 02_preprocessing, 03_model_training, 04_model_evaluation
|-- src/            preprocessing.py, train.py, evaluate.py
|-- models/         saved pipelines (*.joblib) + README
|-- results/        figures (*.png) and tables (*.csv, *.json)
|-- app/            Streamlit app (next stage)
`-- screenshots/    app screenshots (next stage)
```

## Future work

SHAP global and per-patient explanations, Streamlit app with a selectable operating point, probability calibration, subgroup (sex, age) analysis, external validation on a larger dataset.

## Credits

Data: Janosi, Steinbrunn, Pfisterer, Detrano; donor David W. Aha (UCI Machine Learning Repository). Detrano, R., et al. (1989). *American Journal of Cardiology*, 64, 304-310.
