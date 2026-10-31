"""STEP 1 - Understand the data (Exploratory Data Analysis).

Run:  python 1_eda.py

Prints the findings to the console and saves figures to results/eda_*.png
"""
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.preprocessing import (
    BINARY, FEATURE_DESCRIPTIONS, FEATURES, NOMINAL, NUMERIC, RESULTS_DIR, VALUE_LABELS, load_data,
)

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
RESULTS_DIR.mkdir(exist_ok=True)
PAL = {0: "#4c72b0", 1: "#c44e52"}  # 0 = no disease, 1 = disease


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save(fig, name: str) -> None:
    path = RESULTS_DIR / f"eda_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure saved] {path}")


# --------------------------------------------------------------------------- #
# 1. Load and overview
# --------------------------------------------------------------------------- #
header("1. DATASET OVERVIEW")
df = load_data()
print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}  (13 features + original label 'num' + binary 'target')\n")
print(df.head())
print("\nData types and non-null counts:")
df.info()
print("\nSummary statistics of numeric features:")
print(df[NUMERIC].describe().T.round(2))

# --------------------------------------------------------------------------- #
# 2. Data quality
# --------------------------------------------------------------------------- #
header("2. DATA QUALITY")
missing = df[FEATURES].isna().sum()
print("Missing values per column (only columns with gaps):")
print(missing[missing > 0].to_string())
rows_missing = df[FEATURES].isna().any(axis=1)
print(f"Rows with at least one missing value: {rows_missing.sum()} of {len(df)} ({rows_missing.mean():.1%})")
print(df.loc[rows_missing, ["age", "sex", "ca", "thal", "num", "target"]])

print(f"\nDuplicate rows: {df.duplicated().sum()}")
print("\nRange of numeric variables:")
print(df[NUMERIC].agg(["min", "max"]).T.to_string())
print("\nCategory counts:")
for c in NOMINAL + BINARY:
    print(f"  {c:<8}", df[c].value_counts(dropna=False).sort_index().to_dict())

# --------------------------------------------------------------------------- #
# 3. Target variable
# --------------------------------------------------------------------------- #
header("3. TARGET VARIABLE")
orig = df["num"].value_counts().sort_index()
binary = df["target"].value_counts().sort_index()
print("Original label 'num' (0 = none ... 4 = most severe):", orig.to_dict())
print(f"Binary target: no disease = {binary[0]} ({binary[0] / len(df):.1%}), "
      f"disease = {binary[1]} ({binary[1] / len(df):.1%})")
print("-> classes are nearly balanced; report precision/recall/F1/ROC-AUC, not only accuracy.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(orig.index.astype(str), orig.values, color=["#4c72b0"] + ["#c44e52"] * 4)
for i, v in enumerate(orig.values):
    axes[0].text(i, v + 2, str(v), ha="center")
axes[0].set(title="Original label `num` (0 = none ... 4 = most severe)", xlabel="num", ylabel="Patients")
axes[1].bar(["No disease (0)", "Disease (1)"], binary.values, color=[PAL[0], PAL[1]])
for i, v in enumerate(binary.values):
    axes[1].text(i, v + 2, f"{v}  ({v / len(df):.1%})", ha="center")
axes[1].set(title="Binary target used for modelling", ylabel="Patients")
fig.tight_layout()
save(fig, "target_distribution")

# --------------------------------------------------------------------------- #
# 4. Numeric features vs target
# --------------------------------------------------------------------------- #
header("4. NUMERIC FEATURES vs TARGET")
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
for j, c in enumerate(num_cols):
    sns.histplot(data=df, x=c, hue="target", palette=PAL, kde=True, stat="density",
                 common_norm=False, alpha=0.4, ax=axes[0, j], legend=(j == 0))
    axes[0, j].set_title(FEATURE_DESCRIPTIONS[c], fontsize=11)
    sns.boxplot(data=df, x="target", y=c, hue="target", palette=PAL, legend=False, ax=axes[1, j], width=0.5)
    axes[1, j].set_xticks([0, 1])
    axes[1, j].set_xticklabels(["No disease", "Disease"])
    axes[1, j].set_xlabel("")
fig.suptitle("Distribution of numeric features by disease status", y=1.01, fontsize=14)
fig.tight_layout()
save(fig, "numeric_distributions")

rows = []
for c in num_cols:
    a, b = df.loc[df.target == 0, c], df.loc[df.target == 1, c]
    rows.append({
        "feature": c,
        "mean (no disease)": a.mean(), "mean (disease)": b.mean(),
        "median (no disease)": a.median(), "median (disease)": b.median(),
        "skewness": df[c].skew(),
        "Mann-Whitney p": stats.mannwhitneyu(a, b).pvalue,
    })
num_tests = pd.DataFrame(rows).set_index("feature")
print(num_tests.to_string(float_format=lambda v: f"{v:.3f}", formatters={"Mann-Whitney p": "{:.2e}".format}))

q1, q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
iqr = q3 - q1
outliers = ((df[num_cols] < q1 - 1.5 * iqr) | (df[num_cols] > q3 + 1.5 * iqr)).sum()
print("\nIQR outliers per feature (kept: they are plausible measurements):")
print(pd.DataFrame({"IQR outliers": outliers, "max value": df[num_cols].max()}).to_string())

# --------------------------------------------------------------------------- #
# 5. Categorical features vs target
# --------------------------------------------------------------------------- #
header("5. CATEGORICAL FEATURES vs TARGET")
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "ca"]
overall = df["target"].mean()

fig, axes = plt.subplots(2, 4, figsize=(22, 9))
for ax, c in zip(axes.ravel(), cat_cols):
    g = df[[c, "target"]].dropna().groupby(c)["target"].agg(["mean", "count"])
    labels = [VALUE_LABELS[c].get(int(k), str(int(k))) if c in VALUE_LABELS else str(int(k)) for k in g.index]
    ax.bar(range(len(g)), g["mean"], color="#c44e52", alpha=0.85)
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(i, m + 0.02, f"{m:.0%}\n(n={n})", ha="center", fontsize=10)
    ax.axhline(overall, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set(ylim=(0, 1.05), title=FEATURE_DESCRIPTIONS[c], ylabel="Share with disease")
fig.suptitle(f"Disease rate by category (dashed line = overall rate, {overall:.1%})", y=1.0, fontsize=14)
fig.tight_layout()
save(fig, "categorical_vs_target")

rows = []
for c in cat_cols:
    ct = pd.crosstab(df[c], df["target"])
    chi2, p, _, expected = stats.chi2_contingency(ct)
    rows.append({"feature": c, "chi2": chi2, "p-value": p,
                 "Cramer's V": np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))),
                 "cells with expected count < 5": int((expected < 5).sum())})
cat_tests = pd.DataFrame(rows).set_index("feature").sort_values("Cramer's V", ascending=False)
print("Chi-square tests (sorted by effect size):")
print(cat_tests.to_string(formatters={"chi2": "{:.1f}".format, "p-value": "{:.1e}".format,
                                      "Cramer's V": "{:.2f}".format}))

print("\nDisease rate by category:")
for c in cat_cols:
    g = df.groupby(c)["target"].agg(["mean", "count"])
    print(f"  {c:<8}", {VALUE_LABELS.get(c, {}).get(int(k), int(k)): f"{m:.0%} (n={int(n)})"
                        for k, (m, n) in g.iterrows()})

# --------------------------------------------------------------------------- #
# 6. Correlations
# --------------------------------------------------------------------------- #
header("6. CORRELATIONS (Spearman)")
corr = df[FEATURES + ["target"]].corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Spearman correlation matrix (features and target)")
fig.tight_layout()
save(fig, "correlation_heatmap")

print("Correlation with target (sorted):")
print(corr["target"].drop("target").sort_values(ascending=False).round(2).to_string())
pairs = corr.where(~mask & (np.abs(corr) < 1)).drop(index="target", columns="target").stack()
print("\nStrongest feature-feature correlations (no serious multicollinearity expected):")
print(pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(5).round(2).to_string())

# --------------------------------------------------------------------------- #
# 7. Age and sex
# --------------------------------------------------------------------------- #
header("7. AGE AND SEX")
df["age_group"] = pd.cut(df["age"], bins=[28, 40, 50, 60, 80], labels=["<=40", "41-50", "51-60", ">60"])
by = df.groupby(["age_group", "sex"], observed=True)["target"].agg(["mean", "count"]).reset_index()
by["sex"] = by["sex"].map(VALUE_LABELS["sex"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(data=df, x="age_group", y="target", color="#c44e52", errorbar=("ci", 95), ax=axes[0])
axes[0].set(title="Disease rate by age group (95 % CI)", xlabel="Age group", ylabel="Share with disease", ylim=(0, 1))
sns.barplot(data=by, x="age_group", y="mean", hue="sex", ax=axes[1], palette=["#dd8452", "#4c72b0"])
axes[1].set(title="Disease rate by age group and sex", xlabel="Age group", ylabel="Share with disease", ylim=(0, 1))
fig.tight_layout()
save(fig, "age_sex")
print(df.groupby("age_group", observed=True)["target"].agg(["mean", "count"]).round(2).T.to_string())
print()
print(by.round(2).to_string(index=False))

# --------------------------------------------------------------------------- #
# 8. Summary -> decisions for the next steps
# --------------------------------------------------------------------------- #
header("8. SUMMARY AND DECISIONS FOR STEP 2 (TRAINING)")
print("""\
* 6 missing values (ca: 4, thal: 2), 2 % of rows  -> impute inside the pipeline (no dropna)
* Classes nearly balanced (54 % / 46 %)            -> stratified split; report precision, recall, F1, ROC-AUC
* Numeric features on different scales             -> standardise (needed by logistic regression)
* cp, restecg, slope, thal are unordered codes     -> one-hot encode
* ca is an ordered 0-3 count                       -> keep as numeric
* Small dataset (303 rows), rare categories        -> nested cross-validation, bootstrap CIs on the test set
* 68 % male, single hospital                       -> document as a limitation

Next:  python 2_train.py""")
