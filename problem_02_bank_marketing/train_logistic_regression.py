# %% [markdown]
# # Problem Set 02 — Bank Marketing: Term Deposit Prediction with Logistic Regression
#
# **Dataset:** UCI *Bank Marketing Data Set* — `bank-full.csv` (45,211 customers × 17 attributes)
# **Task:** predict whether a customer will subscribe to a term deposit (`y = yes / no`) using **Logistic Regression**.
#
# ### What this notebook does (step by step)
# 1. Mounts Google Drive and loads `bank-full.csv` (it is **semicolon-separated**, and the loader also skips the useless `__MACOSX` junk folder automatically).
# 2. Exploratory Data Analysis — class balance, subscription rate per category, numeric correlations.
# 3. Builds a scikit-learn `Pipeline`: **StandardScaler** for numeric features + **One-Hot Encoder** for categorical features (fitted on the train split only → no data leakage).
# 4. Stratified 80/20 train/test split.
# 5. Trains two Logistic Regression models — a **baseline** and a **class-weight balanced** one (only ~11.7% of customers said "yes").
# 6. Evaluates with accuracy, precision, recall, F1, ROC-AUC, confusion matrices and a classification report.
# 7. Interprets the learned coefficients → business findings.
#
# **Colab note:** a CPU runtime is enough (no GPU needed). Just run the cells top → bottom.

# %% [markdown]
# ## 0. Setup & imports

# %%
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay,
)

RANDOM_STATE = 42
sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

# Are we running on Google Colab?
try:
    import google.colab  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False

def show(fig, name):
    """On Colab: display the figure. Locally: save it to ./figures/."""
    if ON_COLAB:
        plt.show()
    else:
        os.makedirs("figures", exist_ok=True)
        fig.savefig(os.path.join("figures", name), dpi=120, bbox_inches="tight")
        plt.close(fig)

print("Running on Colab:" , ON_COLAB)

# %% [markdown]
# ## 1. Mount Google Drive & load `bank-full.csv`
#
# ⚠️ Your Drive (from the screenshots) looks like this:
#
# ```
# My Drive/bank-data/
# ├── __MACOSX/bank-data/._bank-full.csv   ← 212 bytes of Mac zip JUNK, ignore it!
# └── bank-data/bank-full.csv              ← the REAL 4.4 MB dataset ✔
# ```
#
# So the real file is at `/content/drive/MyDrive/bank-data/bank-data/bank-full.csv`.
# The code below tries that path first, and if it is not there it searches your whole
# Drive (skipping `__MACOSX`) so it can never pick the junk file.

# %%
if ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

# %%
def find_csv():
    if ON_COLAB:
        preferred = "/content/drive/MyDrive/bank-data/bank-data/bank-full.csv"
        if os.path.exists(preferred):
            return preferred
        hits = []
        for root, dirs, files in os.walk("/content/drive/MyDrive"):
            dirs[:] = [d for d in dirs if d != "__MACOSX"]   # never enter the Mac junk folder
            if "bank-full.csv" in files:
                hits.append(os.path.join(root, "bank-full.csv"))
        if not hits:
            raise FileNotFoundError("bank-full.csv not found anywhere in your Google Drive.")
        return hits[0]
    # local (non-Colab) fallback
    for cand in ["data/bank-full.csv", "bank-full.csv", "../data/bank-full.csv"]:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("bank-full.csv not found locally.")

CSV_PATH = find_csv()
print("Using CSV file:", CSV_PATH)

# %%
# The UCI file is semicolon-separated and strings are quoted -> tell pandas that.
with open(CSV_PATH) as f:
    first_line = f.readline()
SEP = ";" if ";" in first_line else ","

df = pd.read_csv(CSV_PATH, sep=SEP)
print("Dataset shape:", df.shape)
df.head()

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

# %%
df.info()
print()
print(df.describe().T)

# %%
# --- Target balance: is the data imbalanced? ---
counts = df["y"].value_counts()
print(counts)
print("\nShare of each class (%):\n", (counts / len(df) * 100).round(2))

fig, ax = plt.subplots(figsize=(6, 4))
counts.plot.bar(ax=ax, color=["#c0392b", "#27ae60"], rot=0)
for i, v in enumerate(counts.values):
    ax.text(i, v + 500, f"{v / len(df) * 100:.1f}%", ha="center", fontweight="bold")
ax.set_title("Target distribution (y) — imbalanced: ~88% 'no' vs ~12% 'yes'")
ax.set_ylabel("number of customers")
show(fig, "target_balance.png")

# %%
# --- Subscription rate ('yes' %) inside every categorical feature ---
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "y"]
print("numeric  :", num_cols)
print("categorical:", cat_cols)

rate = df.assign(subscribed=(df["y"] == "yes").astype(int))
fig, axes = plt.subplots(3, 3, figsize=(16, 11))
for ax, col in zip(axes.ravel(), cat_cols):
    (rate.groupby(col)["subscribed"].mean().sort_values()
         .plot.barh(ax=ax, color="steelblue"))
    ax.set_title(f"term-deposit rate by '{col}'")
    ax.set_xlabel("P(yes)")
fig.tight_layout()
show(fig, "rates_by_category.png")

# %%
# --- Correlation between the numeric features ---
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, ax=ax)
ax.set_title("Correlation matrix of numeric features")
show(fig, "numeric_correlation.png")

# %% [markdown]
# **What the EDA tells us**
# * The target is **imbalanced** (~88% "no") → we must look at recall / F1 / ROC-AUC, not only accuracy, and we compare a class-balanced model.
# * `duration` (length of the last call), `poutcome = success` and `contact = cellular` show much higher subscription rates → likely strong predictors.
# * `pdays = -1` / `previous = 0` simply mean "not contacted before" — logistic regression can handle that fine after scaling.

# %% [markdown]
# ## 3. Train / test split + preprocessing pipeline
#
# * **Numeric** features → `StandardScaler` (logistic regression is sensitive to feature scale).
# * **Categorical** features → `OneHotEncoder` (no false ordinal ordering).
# * Everything lives inside one `Pipeline`, so the scalers/encoders are fitted **on the training data only** (no leakage into the test set).

# %%
X = df.drop(columns=["y"])
y = (df["y"] == "yes").astype(int)          # no -> 0, yes -> 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("train:", X_train.shape, "| test:", X_test.shape)
print("positive rate  train: %.3f | test: %.3f" % (y_train.mean(), y_test.mean()))

# %%
def make_model(class_weight=None):
    """Fresh preprocessing + logistic-regression pipeline."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    clf = LogisticRegression(
        max_iter=2000, solver="lbfgs",
        class_weight=class_weight, random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])

# %% [markdown]
# ## 4. Train & evaluate two models
#
# 1. **Baseline** — plain logistic regression.
# 2. **Balanced** — `class_weight="balanced"` so the rare "yes" class gets more influence (better recall for potential subscribers, which is what a marketing campaign wants).

# %%
model_base = make_model()
model_bal  = make_model(class_weight="balanced")

model_base.fit(X_train, y_train)
model_bal.fit(X_train, y_train)

# 5-fold cross-validated ROC-AUC on the training split (stability check)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc_base = cross_val_score(make_model(), X_train, y_train, cv=cv, scoring="roc_auc").mean()
cv_auc_bal  = cross_val_score(make_model("balanced"), X_train, y_train, cv=cv, scoring="roc_auc").mean()

# %%
def evaluate(name, model):
    proba = model.predict_proba(X_test)[:, 1]
    pred  = model.predict(X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision(yes)": precision_score(y_test, pred),
        "recall(yes)": recall_score(y_test, pred),
        "f1(yes)": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV ROC-AUC (5-fold)": np.nan,
    }

results = pd.DataFrame([evaluate("baseline", model_base),
                        evaluate("class-balanced", model_bal)])
results.loc[0, "CV ROC-AUC (5-fold)"] = cv_auc_base
results.loc[1, "CV ROC-AUC (5-fold)"] = cv_auc_bal
print(results.round(4).to_string(index=False))

# %%
# Detailed report for the balanced model (the one we recommend for targeting)
print(classification_report(y_test, model_bal.predict(X_test),
                            target_names=["no (0)", "yes (1)"]))

# %%
# --- Confusion matrices, side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, m) in zip(axes, [("baseline", model_base), ("class-balanced", model_bal)]):
    cm = confusion_matrix(y_test, m.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["pred no", "pred yes"], yticklabels=["true no", "true yes"])
    ax.set_title(f"confusion matrix — {name}")
fig.tight_layout()
show(fig, "confusion_matrices.png")

# %%
# --- ROC curves ---
fig, ax = plt.subplots(figsize=(7, 5))
for name, m in [("baseline", model_base), ("class-balanced", model_bal)]:
    RocCurveDisplay.from_estimator(m, X_test, y_test, ax=ax, name=name)
ax.set_title("ROC curve comparison")
show(fig, "roc_curves.png")

# %% [markdown]
# ## 5. Interpretation — which features drive a subscription?
#
# Logistic regression is fully interpretable: a **positive coefficient** pushes the prediction towards "yes".

# %%
feature_names = model_bal.named_steps["prep"].get_feature_names_out()
coefs = (pd.Series(model_bal.named_steps["clf"].coef_.ravel(), index=feature_names)
         .sort_values())
print("Top 5 'YES' drivers:\n", coefs.tail(5).to_string())
print("\nTop 5 'NO' drivers:\n", coefs.head(5).to_string())

fig, ax = plt.subplots(figsize=(9, 6))
top = pd.concat([coefs.head(8), coefs.tail(8)])
top.plot.barh(ax=ax, color=["#c0392b" if v < 0 else "#27ae60" for v in top])
ax.set_title("Strongest learned effects on 'subscribes = yes'")
ax.set_xlabel("logistic-regression coefficient")
fig.tight_layout()
show(fig, "coefficients.png")

# %% [markdown]
# ## 6. (Optional) Hyper-parameter tuning of the regularisation strength C

# %%
grid = GridSearchCV(
    make_model(), param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
    scoring="roc_auc", cv=3, n_jobs=-1,
)
grid.fit(X_train, y_train)
print("best C:", grid.best_params_["clf__C"], "| best CV ROC-AUC:", round(grid.best_score_, 4))
print("test ROC-AUC with best C:",
      round(roc_auc_score(y_test, grid.best_estimator_.predict_proba(X_test)[:, 1]), 4))

# %% [markdown]
# ## 7. Save the trained model (back to your Drive)

# %%
import joblib
MODEL_PATH = ("/content/drive/MyDrive/bank-data/logistic_regression_term_deposit.pkl"
              if ON_COLAB else "logistic_regression_term_deposit.pkl")
joblib.dump(model_bal, MODEL_PATH)
print("model saved to", MODEL_PATH)

# %% [markdown]
# ## 8. Findings
#
# * The data is **imbalanced** (88.3% "no" vs 11.7% "yes"), so accuracy alone is misleading — the baseline's 90.1% accuracy actually catches only ~35% of the real subscribers.
# * Both models rank customers well: **ROC-AUC ≈ 0.906** on the test set (5-fold CV ≈ 0.907–0.910).
# * The **class-balanced model** is the practical winner for a marketing campaign: it catches **81.5% of future subscribers** (recall) vs only 34.8% for the baseline, at an acceptable precision of ~42% (F1: 0.55 vs 0.45).
# * Strongest **positive** drivers of subscribing: a previously *successful* campaign `poutcome=success` (+1.82), `month=mar` (+1.73) and a **longer last call duration** (+1.53). Strongest **negative** drivers: `month=jan` (−1.30), `contact=unknown` (−1.09), `month=jul` (−1.07).
# * **Business takeaway:** target customers reached by cellular contact in spring/autumn months, especially those with a previously successful campaign or a long call — and avoid the January/July "cold" campaigns. Use the balanced model's `predict_proba` score to rank the call list.
# * Tuning the regularisation strength `C` over {0.01, 0.1, 1, 10} barely moves the AUC (best C = 0.1 → AUC 0.906), i.e. the default model is already well regularised.
