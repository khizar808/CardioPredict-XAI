# Models

Created by `python -m src.train` (or notebook `03_model_training.ipynb`):

| File | Content |
|---|---|
| `logistic_regression.joblib`, `random_forest.joblib`, `gradient_boosting.joblib`, `xgboost.joblib` | Tuned scikit-learn `Pipeline` (preprocessing + classifier), fitted on the training split |
| `best_model.joblib` | Copy of the model with the highest nested-CV ROC-AUC |

Load one with:

```python
import joblib
model = joblib.load("models/best_model.joblib")
model.predict_proba(X)[:, 1]       # X = DataFrame with the 13 raw feature columns
```

`joblib` files depend on the scikit-learn / XGBoost versions they were created with. If you get a version
warning or error, simply re-run `python -m src.train` in your environment.
