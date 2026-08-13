# Flipkart Order Intelligence & Support Assistant

One connected system: a return-risk model (Part 1), a product-image
categoriser (Part 2), and a LangGraph support agent (Part 3) that loads both
trained artifacts as real tools on top of a retrieval-augmented policy
knowledge base.

**Status:** Part 1 is fully built, tested, and passing every stated
acceptance criterion (see `docs/PART1_ANALYSIS.md` for the full write-up).
Parts 2 and 3 are scaffolded but need their full task briefs (dataset choice,
exact tool signatures, RAG corpus) before they can be built to the same
standard — see the note at the bottom of this README.

## Repo layout

```
generate_orders.py          # Task 1: exact seeded dataset generator
orders_dataset.csv          # generated output (6,000 rows, seed=42)
train_return_risk.py        # Tasks 3-9: preprocessing, baseline, LR, RF, save
models/
  return_risk_model.pkl     # final artifact: tuned RF pipeline
  return_risk_model_meta.json   # t*_rf threshold + feature column order
docs/
  PART1_ANALYSIS.md         # all required written analysis, with numbers
  part1_report.json         # machine-readable metrics dump
  *_sweep.csv, *_importance.csv, subgroup_by_*.csv   # supporting tables
```

## Part 1 — Return-Risk Scoring Pipeline

### Regenerate the dataset

```bash
python3 generate_orders.py
```

This is deterministic (`np.random.default_rng(42)`) — it will always produce
the same 6,000-row `orders_dataset.csv` with a ~22.75% return rate and
~13.05% missingness on `rating_given`.

### Train and evaluate the model

```bash
pip install scikit-learn pandas numpy joblib
python3 train_return_risk.py
```

This runs, in order: preprocessing pipeline build → DummyClassifier baseline
→ Logistic Regression + threshold sweep → Random Forest `GridSearchCV` →
impurity + permutation feature importance → subgroup analysis by category and
payment method → saves `models/return_risk_model.pkl` and
`models/return_risk_model_meta.json`.

### Key results (full detail in `docs/PART1_ANALYSIS.md`)

| Metric | Value |
|---|---|
| Baseline accuracy / F1(class 1) | 77.25% / 0.0 |
| Logistic Regression ROC-AUC / F1 @0.5 | 0.6253 / 0.3921 |
| LR best-F1 threshold / recall gain | 0.44 / +17.94 pts |
| RF best CV ROC-AUC | 0.6178 |
| RF test ROC-AUC | 0.6143 |
| RF t\*_rf (F1-max threshold) | 0.46 |

The saved `models/return_risk_model.pkl` is the exact fitted
`sklearn.Pipeline` (preprocessing + tuned `RandomForestClassifier`) that
Part 3's `check_return_risk` tool loads via `joblib.load(...)` and calls
`.predict_proba(...)` on directly — nothing is hardcoded.

```python
import joblib, json
model = joblib.load("models/return_risk_model.pkl")
meta = json.load(open("models/return_risk_model_meta.json"))

# example: score a single order (must match meta["feature_columns"] order)
import pandas as pd
order = pd.DataFrame([{
    "price_inr": 1800, "discount_pct": 35, "customer_tenure_days": 40,
    "num_previous_orders": 1, "num_previous_returns": 0,
    "delivery_distance_km": 220, "delivery_days": 6, "is_weekend_order": 1,
    "rating_given": None, "product_category": "Apparel",
    "payment_method": "COD",
}])
proba = model.predict_proba(order)[0, 1]
risk_bucket = "HIGH" if proba >= meta["t_star_rf"] else "LOW"
print(proba, risk_bucket)
```

## Part 2 — Product Image Categoriser (code complete, not yet executed)

`train_product_classifier.py`, `export_sample_images.py`, and
`src/classifier_inference.py` are written to the full spec — but **not yet
run**, because this was built in a sandbox with no internet access and no
PyTorch installed. Fashion-MNIST can't be downloaded, and fabricating numbers
would violate the brief's own "never a fabricated number" rule.

**→ See `docs/PART2_HOWTO.md` for exact run instructions (Google Colab free
GPU tier recommended, ~10-15 min).** Once you run it and send back the
output, I'll fill in the real results here and in a `PART2_ANALYSIS.md`, the
same way Part 1 was documented.

```
train_product_classifier.py     # full pipeline: load, split, preprocess,
                                 # transfer-learn (cached features), conditional
                                 # fine-tune, evaluate, save, report
export_sample_images.py         # exports 10 real test-split images as .png
src/classifier_inference.py     # classify_product_image() -- what Part 3's tool calls
requirements-part2.txt
docs/PART2_HOWTO.md             # how to run it + what's already spec-verified
```

## Part 3 — not yet built

Your brief's roadmap diagram names Part 3 (Agent + RAG + Safety, 40 marks),
but only Parts 1 and 2's tasks and acceptance criteria were spelled out in
full in what you shared with me. Building it to the same standard needs the
same level of detail — specifically the exact tool signatures for
`check_return_risk` and `classify_product_image` (now written, see above),
the RAG corpus content (or where it comes from), the LangGraph graph
structure expected, and the `MOCK_LLM` behavior contract.

Paste me the same full task-by-task text for Part 3 (the way you did for
Parts 1 and 2) and I'll build, run, and verify it the same way — same rigor,
same acceptance-criteria checking, saved artifacts, and README updates.

## Git workflow

This repo's history includes a feature branch (`part1-return-risk`) created,
committed to at least twice, and merged into `main` — visible via
`git log --graph --all`.
