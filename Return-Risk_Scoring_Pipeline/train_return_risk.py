"""
Part 1 -- Return-Risk Scoring Pipeline
Trains, evaluates, and saves the return-risk model used by Part 3's
check_return_risk tool.
"""
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, precision_recall_curve)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
report = {}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("orders_dataset.csv")

numeric_features = ["price_inr", "discount_pct", "customer_tenure_days",
                     "num_previous_orders", "num_previous_returns",
                     "delivery_distance_km", "delivery_days",
                     "is_weekend_order", "rating_given"]
categorical_features = ["product_category", "payment_method"]

X = df[numeric_features + categorical_features]
y = df["returned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# Task 3: Preprocessing pipeline (no leakage -- fit on train only)
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ---------------------------------------------------------------------------
# Task 4: Baseline -- DummyClassifier
# ---------------------------------------------------------------------------
dummy_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
])
dummy_pipe.fit(X_train, y_train)
dummy_pred = dummy_pipe.predict(X_test)
report["baseline"] = {
    "accuracy": round(accuracy_score(y_test, dummy_pred), 4),
    "f1_class1": round(f1_score(y_test, dummy_pred, pos_label=1, zero_division=0), 4),
}
print("Baseline DummyClassifier:", report["baseline"])

# ---------------------------------------------------------------------------
# Task 5: Logistic Regression + threshold sweep
# ---------------------------------------------------------------------------
logreg_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                random_state=RANDOM_STATE)),
])
logreg_pipe.fit(X_train, y_train)
logreg_proba = logreg_pipe.predict_proba(X_test)[:, 1]
logreg_pred_default = (logreg_proba >= 0.5).astype(int)

report["logreg_default"] = {
    "accuracy": round(accuracy_score(y_test, logreg_pred_default), 4),
    "f1_class1": round(f1_score(y_test, logreg_pred_default, pos_label=1), 4),
    "recall_class1": round(recall_score(y_test, logreg_pred_default, pos_label=1), 4),
    "precision_class1": round(precision_score(y_test, logreg_pred_default, pos_label=1, zero_division=0), 4),
    "roc_auc": round(roc_auc_score(y_test, logreg_proba), 4),
}
print("LogReg @0.5:", report["logreg_default"])

def sweep_threshold(y_true, proba, lo=0.10, hi=0.90, step=0.02):
    rows = []
    t = lo
    while t <= hi + 1e-9:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "f1": f1_score(y_true, pred, pos_label=1, zero_division=0),
            "recall": recall_score(y_true, pred, pos_label=1, zero_division=0),
            "precision": precision_score(y_true, pred, pos_label=1, zero_division=0),
        })
        t += step
    return pd.DataFrame(rows)

logreg_sweep = sweep_threshold(y_test, logreg_proba)
best_logreg_row = logreg_sweep.loc[logreg_sweep["f1"].idxmax()]
report["logreg_best_threshold"] = best_logreg_row.to_dict()
print("LogReg best-F1 threshold:", report["logreg_best_threshold"])
print("Recall gain vs default:",
      round(best_logreg_row["recall"] - report["logreg_default"]["recall_class1"], 4))

logreg_sweep.to_csv("docs/logreg_threshold_sweep.csv", index=False)

# ---------------------------------------------------------------------------
# Task 6: Random Forest + GridSearchCV
# ---------------------------------------------------------------------------
rf_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
])

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [6, 10, None],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(rf_pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
rf_proba = best_rf.predict_proba(X_test)[:, 1]
rf_test_auc = roc_auc_score(y_test, rf_proba)

report["rf_grid"] = {
    "best_params": grid.best_params_,
    "best_cv_roc_auc": round(grid.best_score_, 4),
    "test_roc_auc": round(rf_test_auc, 4),
}
print("RF GridSearch:", report["rf_grid"])

# ---------------------------------------------------------------------------
# Task 7: Feature importance (impurity-based + permutation)
# ---------------------------------------------------------------------------
ohe = best_rf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_names

importances = best_rf.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
top5 = imp_df.head(5)
report["rf_top5_impurity"] = top5.to_dict(orient="records")
print("\nTop 5 impurity-based importances:\n", top5)

perm = permutation_importance(best_rf, X_test, y_test, n_repeats=20,
                                random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
# map permutation importance back onto ORIGINAL (pre-onehot) columns
perm_df = pd.DataFrame({
    "feature": numeric_features + categorical_features,
})
# permutation_importance operates on raw X columns (pipeline handles preprocessing),
# so indices align to X.columns order = numeric_features + categorical_features
perm_df["perm_importance_mean"] = perm.importances_mean
perm_df["perm_importance_std"] = perm.importances_std
perm_df = perm_df.sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
report["rf_permutation_importance"] = perm_df.to_dict(orient="records")
print("\nPermutation importances (raw columns):\n", perm_df)

perm_df.to_csv("docs/permutation_importance.csv", index=False)
imp_df.to_csv("docs/impurity_importance.csv", index=False)

# ---------------------------------------------------------------------------
# Task 8: Subgroup analysis
# ---------------------------------------------------------------------------
X_test_copy = X_test.copy()
X_test_copy["y_true"] = y_test.values
X_test_copy["y_pred"] = (rf_proba >= 0.5).astype(int)

def subgroup_metrics(group_col):
    rows = []
    for g, sub in X_test_copy.groupby(group_col):
        rows.append({
            group_col: g,
            "n": len(sub),
            "recall": round(recall_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0), 4),
            "precision": round(precision_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0), 4),
        })
    return pd.DataFrame(rows)

subgroup_cat = subgroup_metrics("product_category")
subgroup_pay = subgroup_metrics("payment_method")
print("\nSubgroup by product_category:\n", subgroup_cat)
print("\nSubgroup by payment_method:\n", subgroup_pay)

subgroup_cat.to_csv("docs/subgroup_by_category.csv", index=False)
subgroup_pay.to_csv("docs/subgroup_by_payment.csv", index=False)

report["subgroup_category"] = subgroup_cat.to_dict(orient="records")
report["subgroup_payment"] = subgroup_pay.to_dict(orient="records")

# ---------------------------------------------------------------------------
# Task 9: Save artifact -- tuned Random Forest pipeline (final model)
# ---------------------------------------------------------------------------
rf_sweep = sweep_threshold(y_test, rf_proba)
best_rf_row = rf_sweep.loc[rf_sweep["f1"].idxmax()]
t_star_rf = float(best_rf_row["threshold"])
report["rf_threshold_sweep_best"] = best_rf_row.to_dict()
print("\nRandom Forest F1-maximising threshold t*_rf:", t_star_rf, "->", best_rf_row.to_dict())

rf_sweep.to_csv("docs/rf_threshold_sweep.csv", index=False)

import os
os.makedirs("models", exist_ok=True)
joblib.dump(best_rf, "models/return_risk_model.pkl")

meta = {
    "t_star_rf": t_star_rf,
    "feature_columns": numeric_features + categorical_features,
    "model_type": "RandomForestClassifier",
    "best_params": grid.best_params_,
    "test_roc_auc": round(rf_test_auc, 4),
}
with open("models/return_risk_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

with open("docs/part1_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\nSaved models/return_risk_model.pkl and models/return_risk_model_meta.json")
print("t*_rf =", t_star_rf)
