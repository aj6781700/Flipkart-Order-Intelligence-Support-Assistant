"""
Part 3, Task 3 -- check_return_risk tool.

Loads Part 1's ACTUAL saved artifact (models/return_risk_model.pkl, the
tuned Random Forest pipeline) and t*_rf from return_risk_model_meta.json.
Nothing here is a hardcoded stand-in -- calling this function reproduces
exactly what Part 1's saved model outputs for the same input features.

Risk buckets are anchored to t*_rf (NOT a fixed 0.3/0.6 split), because a
fixed split isn't self-calibrating: two equally valid Random Forest models
can produce probability distributions concentrated in very different
ranges, so a fixed cut point can silently collapse almost every order into
one bucket for a model whose probabilities happen to cluster elsewhere.
Anchoring to this specific model's own F1-maximising threshold keeps the
buckets meaningful for whatever probability range THIS trained model
actually produces.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

PART1_DIR = Path(__file__).resolve().parent.parent.parent / "1_Return-Risk_Scoring_Pipeline"
MODEL_PATH = PART1_DIR / "models" / "return_risk_model.pkl"
META_PATH = PART1_DIR / "models" / "return_risk_model_meta.json"

_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Part 1 model not found at {MODEL_PATH}. Run "
                f"1_Return-Risk_Scoring_Pipeline/train_return_risk.py first."
            )
        _model = joblib.load(MODEL_PATH)
        _meta = json.load(open(META_PATH))
    return _model, _meta


def check_return_risk(order_features: dict) -> dict:
    """
    Args:
        order_features: dict with the exact feature columns Part 1's
            pipeline expects, e.g.:
            {
              "price_inr": 1800, "discount_pct": 35, "customer_tenure_days": 40,
              "num_previous_orders": 1, "num_previous_returns": 0,
              "delivery_distance_km": 220, "delivery_days": 6,
              "is_weekend_order": 1, "rating_given": None,
              "product_category": "Apparel", "payment_method": "COD",
            }

    Returns:
        {
          "return_probability": float,
          "risk_bucket": "Low" | "Medium" | "High",
          "t_star_rf": float,          # the anchor threshold used
          "cut_points": {"low_below": float, "high_at_or_above": float},
        }
    """
    model, meta = _load()
    t_star = meta["t_star_rf"]

    # cut points anchored to t*_rf, per Task 3's spec:
    #   Low    if probability <  t*_rf
    #   High   if probability >= t*_rf + 0.15
    #   Medium otherwise
    low_below = t_star
    high_at_or_above = t_star + 0.15

    row = pd.DataFrame([order_features])[meta["feature_columns"]]
    proba = float(model.predict_proba(row)[0, 1])

    if proba < low_below:
        bucket = "Low"
    elif proba >= high_at_or_above:
        bucket = "High"
    else:
        bucket = "Medium"

    return {
        "return_probability": round(proba, 4),
        "risk_bucket": bucket,
        "t_star_rf": t_star,
        "cut_points": {"low_below": round(low_below, 4), "high_at_or_above": round(high_at_or_above, 4)},
    }


if __name__ == "__main__":
    example_order = {
        "price_inr": 1800, "discount_pct": 35, "customer_tenure_days": 40,
        "num_previous_orders": 1, "num_previous_returns": 0,
        "delivery_distance_km": 220, "delivery_days": 6,
        "is_weekend_order": 1, "rating_given": None,
        "product_category": "Apparel", "payment_method": "COD",
    }
    result = check_return_risk(example_order)
    print(result)
    print(
        f"\nBucket cut points are anchored to this model's own t*_rf = "
        f"{result['t_star_rf']}: Low if probability < {result['cut_points']['low_below']}, "
        f"High if probability >= {result['cut_points']['high_at_or_above']}, else Medium."
    )
