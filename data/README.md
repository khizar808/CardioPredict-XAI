# Data

**Dataset:** UCI Machine Learning Repository - *Heart Disease* (Cleveland Clinic Foundation subset)
**Page:** https://archive.ics.uci.edu/dataset/45/heart+disease
**File used:** `processed.cleveland.data` (303 patients, 13 features + diagnosis `num`)

## How to obtain it

Nothing to do: `src/preprocessing.py::load_data()` looks for `data/raw/processed.cleveland.data` and
**downloads it from UCI automatically** if it is missing. To download it manually:

```
https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
```

and save it as `data/raw/processed.cleveland.data`. The raw data is git-ignored on purpose (data provenance:
the UCI repository is the single source of truth).

## Format

Comma-separated, no header, `?` marks a missing value. Column order:
`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num`

| Column | Meaning | Coding |
|---|---|---|
| age | Age in years | numeric |
| sex | Sex | 0 = female, 1 = male |
| cp | Chest pain type | 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| trestbps | Resting blood pressure on admission (mm Hg) | numeric |
| chol | Serum cholesterol (mg/dl) | numeric |
| fbs | Fasting blood sugar > 120 mg/dl | 0 = no, 1 = yes |
| restecg | Resting ECG result | 0 normal, 1 ST-T abnormality, 2 LV hypertrophy |
| thalach | Maximum heart rate achieved | numeric |
| exang | Exercise-induced angina | 0 = no, 1 = yes |
| oldpeak | ST depression induced by exercise relative to rest | numeric |
| slope | Slope of the peak exercise ST segment | 1 upsloping, 2 flat, 3 downsloping |
| ca | Major vessels (0-3) coloured by fluoroscopy | 0-3 (4 missing) |
| thal | Thallium stress-test result | 3 normal, 6 fixed defect, 7 reversible defect (2 missing) |
| num | Angiographic disease status | 0 = none, 1-4 = disease present |

The modelling target is **binary**: `target = (num > 0)`.

## Citation / credit

The database creators asked that publications using the data credit the principal investigators:
Andras Janosi, M.D. (Hungarian Institute of Cardiology, Budapest); William Steinbrunn, M.D. (University Hospital,
Zurich); Matthias Pfisterer, M.D. (University Hospital, Basel); Robert Detrano, M.D., Ph.D. (V.A. Medical Center,
Long Beach and Cleveland Clinic Foundation). Donor: David W. Aha.

Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of
coronary artery disease. *American Journal of Cardiology*, 64, 304-310.
