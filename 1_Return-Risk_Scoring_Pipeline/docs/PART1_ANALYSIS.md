# Part 1 — Return-risk scoring: analysis

All numbers below are from `python3 train_return_risk.py` (full log: `docs/part1_run.log`, machine-readable: `docs/part1_report.json`). Data: 6,000 synthetic orders from `generate_orders.py` (seed 42), 80/20 stratified split (4,800 train / 1,200 test), overall return rate 22.75%.

## Data check
- Return rate by payment: COD 30.75%, Prepaid_Card 16.82%, Prepaid_UPI 16.92%, Wallet 17.85%.
- Return rate by category: Apparel 26.4%, Footwear 26.0%, Beauty 20.0%, Home 19.2%, Electronics 18.7%.
- `rating_given` is missing for 13.05% of rows, and the gap is structural: 22.8% for COD vs 6.1% for non-COD (16.8 points). Missingness is informative, so it is imputed inside the pipeline (median, fitted on train only) rather than dropped.

## Pipeline
`ColumnTransformer` (median-impute + scale numerics, mode-impute + one-hot categoricals) inside a `Pipeline`, fitted on the training split only, so no test statistics leak into preprocessing.

## Baseline vs models (test set)
| Model | Accuracy | Recall (returned) | Precision | ROC-AUC |
|---|---|---|---|---|
| Dummy (majority class) | 0.7725 | 0.000 | – | – |
| Logistic Regression, balanced, t=0.5 | 0.5917 | 0.579 | 0.296 | 0.625 |
| Logistic Regression, t*=0.44 | – | 0.758 | 0.280 | – |
| Random Forest (GridSearch: depth 6, 200 trees), t=0.5 | – | 0.550 | 0.324 | 0.620 |

The dummy model has the *highest accuracy and catches no returns at all* — accuracy is the wrong metric for a 22.75%-positive problem, so the work is judged on recall/precision/F1 of the returned class.

## Threshold tuning
Sweeping 0.10–0.90 (step 0.02) on Logistic Regression, F1 peaks at t*=0.44: recall rises from 0.579 to 0.758 (+17.9 points) at a cost of 1.6 points of precision. For ops, a missed return is costlier than a flagged-but-kept order, so the lower threshold is the better operating point.

## Random Forest threshold (t\*_rf)
The same sweep on the Random Forest's *own* `predict_proba` gives **t\*_rf = 0.50** (F1 0.408, recall 0.550, precision 0.324). This is the value saved in `models/return_risk_model_meta.json` and used by Part 3 for its risk buckets. (The Logistic Regression's 0.44 is not reused: a threshold belongs to the model whose probabilities produced it.)

## Feature importance
Impurity importance (biased towards high-cardinality numerics) ranks `payment_method_COD`, `price_inr`, `delivery_distance_km` highest. Permutation importance on the test set (ROC-AUC drop) is the more trustworthy view: `payment_method` (0.096) dominates by an order of magnitude, then `price_inr` (0.012), `num_previous_returns` (0.009), `product_category` (0.008). `customer_tenure_days` and `num_previous_orders` have ~zero/negative permutation importance — impurity importance over-credited them.

## Subgroup performance at t=0.5 (Random Forest)
- By payment: COD recall 0.936 / precision 0.327, but Prepaid_Card recall 0.020, Prepaid_UPI 0.042, Wallet 0.095. The model has largely learned "COD ⇒ returned" and is nearly blind to prepaid returns.
- By category: Electronics recall is lowest (0.442), Home highest (0.677, but precision only 0.235).

## Limitations
ROC-AUC ≈ 0.62 means modest signal. The data is synthetic, so the payment effect reflects how it was generated; on real data these scores would need re-validation, and the prepaid blind spot would need a dedicated look before using scores to act on customers.
