"""
Task 3 -- check_return_risk(order_features) -> dict

Loads Part 1's REAL saved artifact (1_Return-Risk_Scoring_Pipeline/models/
return_risk_model.pkl, the tuned Random Forest pipeline) and returns its
predict_proba output plus a risk bucket. Nothing here is a stand-in.

Bucket cut points are anchored to Part 1's t*_rf (the F1-maximising threshold
computed on THIS Random Forest's own predict_proba), not to fixed 0.3 / 0.6
values -- a fixed split is not self-calibrating, so it can collapse almost
every order into one bucket for a model whose probabilities cluster elsewhere:

    Low     if probability <  t*_rf
    Medium  if t*_rf <= probability < t*_rf + 0.15
    High    if probability >= t*_rf + 0.15
"""
import json
from pathlib import Path

import joblib
import pandas as pd

PART1_DIR = Path(__file__).resolve().parents[2] / "1_Return-Risk_Scoring_Pipeline"
MODEL_PATH = PART1_DIR / "models" / "return_risk_model.pkl"
META_PATH = PART1_DIR / "models" / "return_risk_model_meta.json"
HIGH_MARGIN = 0.15

_cache = {}


def _load():
    if "model" not in _cache:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Part 1 model not found at {MODEL_PATH}")
        _cache["model"] = joblib.load(MODEL_PATH)
        _cache["meta"] = json.loads(META_PATH.read_text())
    return _cache["model"], _cache["meta"]


def cut_points() -> dict:
    _, meta = _load()
    t = meta["t_star_rf"]
    return {"t_star_rf": t, "low_below": round(t, 4), "high_at_or_above": round(t + HIGH_MARGIN, 4)}


def check_return_risk(order_features: dict) -> dict:
    """
    order_features keys (Part 1's columns): price_inr, discount_pct,
    customer_tenure_days, num_previous_orders, num_previous_returns,
    delivery_distance_km, delivery_days, is_weekend_order, rating_given,
    product_category, payment_method.
    """
    model, meta = _load()
    row = pd.DataFrame([{c: order_features.get(c) for c in meta["feature_columns"]}])
    proba = float(model.predict_proba(row)[0, 1])      # the real model call
    cp = cut_points()
    if proba < cp["low_below"]:
        bucket = "Low"
    elif proba >= cp["high_at_or_above"]:
        bucket = "High"
    else:
        bucket = "Medium"
    return {"return_probability": round(proba, 4), "risk_bucket": bucket,
            "t_star_rf": cp["t_star_rf"], "cut_points": cp}


if __name__ == "__main__":
    example = {"price_inr": 32000, "discount_pct": 35, "customer_tenure_days": 60,
               "num_previous_orders": 2, "num_previous_returns": 1,
               "delivery_distance_km": 400, "delivery_days": 6, "is_weekend_order": 1,
               "rating_given": None, "product_category": "Electronics", "payment_method": "COD"}
    r = check_return_risk(example)
    cp = r["cut_points"]
    print(r)
    print(f"t*_rf = {cp['t_star_rf']}: Low if p < {cp['low_below']}, "
          f"Medium if {cp['low_below']} <= p < {cp['high_at_or_above']}, High if p >= {cp['high_at_or_above']}.")
