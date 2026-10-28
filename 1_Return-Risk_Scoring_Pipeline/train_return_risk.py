"""
Part 1 -- Return-Risk Scoring Pipeline.

Run order:
    python3 generate_orders.py
    python3 train_return_risk.py

Verifies the generated data, builds a leak-free preprocessing pipeline,
trains a DummyClassifier baseline, a Logistic Regression (with a threshold
sweep) and a GridSearchCV-tuned Random Forest, explains the forest with
impurity and permutation importance, breaks performance out by subgroup,
and saves the tuned Random Forest pipeline plus its own F1-maximising
threshold (t*_rf) for the support agent in Part 3.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
os.makedirs("docs", exist_ok=True)
os.makedirs("models", exist_ok=True)
report = {}

# ---------------------------------------------------------------- data check
df = pd.read_csv("orders_dataset.csv")
assert df.shape == (6000, 13), df.shape
miss_by_pay = df.groupby("payment_method")["rating_given"].apply(lambda s: s.isna().mean())
cod_rate = float(miss_by_pay["COD"])
non_cod_rate = float(df.loc[df.payment_method != "COD", "rating_given"].isna().mean())
report["data_verification"] = {
    "rows": int(len(df)), "columns": int(df.shape[1]),
    "overall_return_rate": round(float(df["returned"].mean()), 4),
    "rating_missing_pct": round(float(df["rating_given"].isna().mean() * 100), 2),
    "rating_missing_rate_COD": round(cod_rate, 4),
    "rating_missing_rate_non_COD": round(non_cod_rate, 4),
    "rating_missing_gap_pts": round((cod_rate - non_cod_rate) * 100, 2),
    "return_rate_by_category": df.groupby("product_category")["returned"].mean().round(4).to_dict(),
    "return_rate_by_payment": df.groupby("payment_method")["returned"].mean().round(4).to_dict(),
    "rating_missing_rate_by_payment": miss_by_pay.round(4).to_dict(),
}
print("DATA VERIFICATION:", json.dumps(report["data_verification"], indent=2))

numeric_features = ["price_inr", "discount_pct", "customer_tenure_days",
                    "num_previous_orders", "num_previous_returns",
                    "delivery_distance_km", "delivery_days",
                    "is_weekend_order", "rating_given"]
categorical_features = ["product_category", "payment_method"]

X = df[numeric_features + categorical_features]
y = df["returned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain rows: {len(X_train)} | Test rows: {len(X_test)}")

# ------------------------------------------------- preprocessing (no leakage)
# Every imputer / scaler / encoder lives inside the Pipeline, so .fit() only
# ever sees the training split; the test split is only transformed.
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

# ------------------------------------------------------------------ baseline
dummy = Pipeline([("pre", preprocessor),
                  ("clf", DummyClassifier(strategy="most_frequent"))]).fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
report["baseline"] = {
    "accuracy": round(accuracy_score(y_test, dummy_pred), 4),
    "f1_class1": round(f1_score(y_test, dummy_pred, zero_division=0), 4),
    "recall_class1": round(recall_score(y_test, dummy_pred, zero_division=0), 4),
}
print("\nBASELINE:", report["baseline"])


def sweep_threshold(y_true, proba, lo=0.10, hi=0.90, step=0.02):
    rows = []
    for t in np.arange(lo, hi + 1e-9, step):
        pred = (proba >= t).astype(int)
        rows.append({"threshold": round(float(t), 2),
                     "f1": f1_score(y_true, pred, zero_division=0),
                     "recall": recall_score(y_true, pred, zero_division=0),
                     "precision": precision_score(y_true, pred, zero_division=0)})
    return pd.DataFrame(rows)


# ------------------------------------------------------ logistic regression
logreg = Pipeline([("pre", preprocessor),
                   ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                              random_state=RANDOM_STATE))]).fit(X_train, y_train)
lr_proba = logreg.predict_proba(X_test)[:, 1]
lr_pred = (lr_proba >= 0.5).astype(int)
report["logreg_default"] = {
    "accuracy": round(accuracy_score(y_test, lr_pred), 4),
    "f1_class1": round(f1_score(y_test, lr_pred), 4),
    "recall_class1": round(recall_score(y_test, lr_pred), 4),
    "precision_class1": round(precision_score(y_test, lr_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, lr_proba), 4),
}
print("LOGREG @0.5:", report["logreg_default"])

lr_sweep = sweep_threshold(y_test, lr_proba)
lr_best = lr_sweep.loc[lr_sweep["f1"].idxmax()]
report["logreg_best_threshold"] = {k: round(float(v), 4) for k, v in lr_best.items()}
report["logreg_recall_gain_pts"] = round(
    (lr_best["recall"] - report["logreg_default"]["recall_class1"]) * 100, 2)
report["logreg_precision_drop_pts"] = round(
    (report["logreg_default"]["precision_class1"] - lr_best["precision"]) * 100, 2)
print("LOGREG best-F1 threshold:", report["logreg_best_threshold"],
      "| recall gain (pts):", report["logreg_recall_gain_pts"],
      "| precision drop (pts):", report["logreg_precision_drop_pts"])
lr_sweep.to_csv("docs/logreg_threshold_sweep.csv", index=False)

# ------------------------------------------------ random forest + GridSearch
rf_pipe = Pipeline([("pre", preprocessor),
                    ("clf", RandomForestClassifier(class_weight="balanced",
                                                   random_state=RANDOM_STATE))])
grid = GridSearchCV(
    rf_pipe,
    {"clf__n_estimators": [100, 200], "clf__max_depth": [6, 10, None]},
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1,
).fit(X_train, y_train)
best_rf = grid.best_estimator_
rf_proba = best_rf.predict_proba(X_test)[:, 1]
report["rf_grid"] = {
    "best_params": {k: (None if v is None else int(v)) for k, v in grid.best_params_.items()},
    "best_cv_roc_auc": round(float(grid.best_score_), 4),
    "test_roc_auc": round(float(roc_auc_score(y_test, rf_proba)), 4),
}
print("\nRF GRIDSEARCH:", report["rf_grid"])

# ----------------------------------------------------- feature importances
ohe = best_rf.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
feature_names = numeric_features + list(ohe.get_feature_names_out(categorical_features))
imp_df = (pd.DataFrame({"feature": feature_names,
                        "impurity_importance": best_rf.named_steps["clf"].feature_importances_})
          .sort_values("impurity_importance", ascending=False).reset_index(drop=True))
report["rf_top5_impurity"] = imp_df.head(5).round(4).to_dict(orient="records")
print("\nTOP-5 IMPURITY IMPORTANCES:\n", imp_df.head(5))

perm = permutation_importance(best_rf, X_test, y_test, n_repeats=20,
                              random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
perm_df = (pd.DataFrame({"feature": numeric_features + categorical_features,
                         "perm_importance_mean": perm.importances_mean,
                         "perm_importance_std": perm.importances_std})
           .sort_values("perm_importance_mean", ascending=False).reset_index(drop=True))
report["rf_permutation_importance"] = perm_df.round(4).to_dict(orient="records")
print("\nPERMUTATION IMPORTANCE (raw columns, test ROC-AUC drop):\n", perm_df)
imp_df.to_csv("docs/impurity_importance.csv", index=False)
perm_df.to_csv("docs/permutation_importance.csv", index=False)

# --------------------------------------------------------- subgroup analysis
rf_pred_default = (rf_proba >= 0.5).astype(int)
report["rf_overall_at_0.5"] = {
    "recall": round(recall_score(y_test, rf_pred_default), 4),
    "precision": round(precision_score(y_test, rf_pred_default), 4),
}
sub = X_test.copy()
sub["y_true"], sub["y_pred"] = y_test.values, rf_pred_default


def subgroup_table(col):
    rows = []
    for g, s in sub.groupby(col):
        rows.append({col: g, "n": len(s),
                     "recall": round(recall_score(s.y_true, s.y_pred, zero_division=0), 4),
                     "precision": round(precision_score(s.y_true, s.y_pred, zero_division=0), 4)})
    return pd.DataFrame(rows)


sub_cat, sub_pay = subgroup_table("product_category"), subgroup_table("payment_method")
print("\nRF overall @0.5:", report["rf_overall_at_0.5"])
print("\nSUBGROUP by product_category:\n", sub_cat, "\n\nSUBGROUP by payment_method:\n", sub_pay)
sub_cat.to_csv("docs/subgroup_by_category.csv", index=False)
sub_pay.to_csv("docs/subgroup_by_payment.csv", index=False)
report["subgroup_category"] = sub_cat.to_dict(orient="records")
report["subgroup_payment"] = sub_pay.to_dict(orient="records")

# --------------------------------------------- t*_rf + save the final artifact
rf_sweep = sweep_threshold(y_test, rf_proba)
rf_best = rf_sweep.loc[rf_sweep["f1"].idxmax()]
t_star_rf = round(float(rf_best["threshold"]), 2)
report["rf_t_star"] = {"t_star_rf": t_star_rf,
                       "recall": round(float(rf_best["recall"]), 4),
                       "precision": round(float(rf_best["precision"]), 4),
                       "f1": round(float(rf_best["f1"]), 4)}
print("\nt*_rf (F1-maximising threshold on the RF's own predict_proba):", report["rf_t_star"])
rf_sweep.to_csv("docs/rf_threshold_sweep.csv", index=False)

joblib.dump(best_rf, "models/return_risk_model.pkl")
with open("models/return_risk_model_meta.json", "w") as f:
    json.dump({"t_star_rf": t_star_rf,
               "feature_columns": numeric_features + categorical_features,
               "model_type": "RandomForestClassifier (inside sklearn Pipeline)",
               "best_params": report["rf_grid"]["best_params"],
               "test_roc_auc": report["rf_grid"]["test_roc_auc"]}, f, indent=2)
with open("docs/part1_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print("\nSaved models/return_risk_model.pkl, models/return_risk_model_meta.json, docs/part1_report.json")
