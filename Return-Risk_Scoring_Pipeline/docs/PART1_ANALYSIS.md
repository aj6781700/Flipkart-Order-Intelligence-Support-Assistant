# Part 1 — Return-Risk Scoring Pipeline: Written Analysis

All numbers below come from an actual run of `generate_orders.py` followed by
`train_return_risk.py` in this repo (seed fixed at 42 throughout). Raw tables
backing every claim are in `docs/*.csv` and `docs/part1_report.json`.

## Task 2 — Data verification

- **Rows:** 6,000
- **Overall return rate:** 22.75%
- **`rating_given` missing:** 13.05% of rows overall

**Return rate by `product_category`:**

| category | return rate |
|---|---|
| Apparel | 26.43% |
| Footwear | 25.96% |
| Beauty | 20.03% |
| Home | 19.15% |
| Electronics | 18.69% |

**Return rate by `payment_method`:**

| method | return rate |
|---|---|
| COD | 30.75% |
| Wallet | 17.85% |
| Prepaid_UPI | 16.92% |
| Prepaid_Card | 16.82% |

**Missingness classification: MAR (Missing At Random), conditional on `payment_method`.**

The missing-rate gap is stark and directly measurable: `rating_given` is missing on
**22.83%** of COD orders versus **5.66%–6.40%** on the three prepaid methods — roughly
a 4x difference. This is exactly how the generator was written (`missing_mask` is
drawn with probability 0.22 for COD rows and 0.06 otherwise), so the mechanism is
provably conditional on a fully **observed** column (`payment_method`), which is the
definition of MAR. It is not MCAR, because the missing-rate gap between COD and
non-COD is real and measured, not noise — a chi-square test on this gap would be
overwhelmingly significant. It is not MNAR either, because missingness does not
depend on the (unobserved) rating value itself — a customer who would have given a
5-star rating is exactly as likely to skip rating as one who would have given a
1-star rating, once you condition on their payment method. Practically: COD orders
plausibly get rated less because there's no forced post-payment app touchpoint the
way a prepaid checkout can nudge one.

## Task 4 — Baseline (why high accuracy is a trap)

`DummyClassifier(strategy="most_frequent")` scores:

- **Accuracy: 77.25%**
- **F1 (class 1, "returned"): 0.0**

This is the single most important number in the whole report to explain to a
non-technical stakeholder. Because 77% of orders are *not* returned, a model that
predicts "not returned" for every single order gets 77% accuracy while catching
**zero** of the actual returns — its recall on the class we care about is exactly 0.
Accuracy is the wrong metric here because it rewards agreement with the majority
class, and the majority class is the *uninteresting* one from a business
perspective: nobody needs a model to tell them an order probably won't be returned,
they need it to catch the ones that will be. This is the classic **"high accuracy,
zero recall"** trap of imbalanced classification — any model report for this
project that leads with accuracy alone, without a baseline comparison, is
functionally hiding the fact that it might be doing nothing useful at all.

## Task 5 — Logistic Regression threshold trade-off

At the default 0.5 threshold: **Accuracy 59.17%, F1 0.3921, Recall 57.88%,
Precision 29.64%, ROC-AUC 0.6253** — all above the required minimums (AUC ≥ 0.58,
F1 ≥ 0.30).

Sweeping the threshold from 0.10 to 0.90 in 0.02 steps and picking the
F1-maximizing point lands at **threshold = 0.44**, giving **recall 75.82%,
precision 28.01%, F1 0.4091** — a **+17.94 percentage-point** recall gain over the
default threshold, at a precision cost of about 1.6 points (29.64% → 28.01%).

**Business trade-off:** lowering the threshold from 0.5 to 0.44 makes the model
flag more orders as "at risk," which means catching more of the returns that
actually happen (fewer false negatives — the costly error of failing to flag a
return that later happens, which shows up downstream as unplanned reverse-logistics
cost and a surprised support agent). In exchange, more orders that were never going
to be returned get flagged anyway (more false positives — wasted proactive
outreach, a support agent's time spent double-checking an order that was fine). For
a returns-prevention workflow, missing a real return is usually more expensive than
one extra unnecessary check-in, so accepting the precision drop in exchange for the
recall gain is the right call — but this is a business policy decision, not a
purely statistical one.

## Task 7 — Feature importance: impurity vs. permutation

**Top 5 by impurity-based `.feature_importances_`:**

| rank | feature | importance |
|---|---|---|
| 1 | payment_method_COD | 0.166 |
| 2 | price_inr | 0.137 |
| 3 | customer_tenure_days | 0.107 |
| 4 | delivery_distance_km | 0.097 |
| 5 | discount_pct | 0.089 |

Each is plausible on its face: COD is the single strongest signal because the
data-generating process gives it the largest coefficient of any single feature
(customers who pay COD haven't "committed" to the purchase the way a prepaid
customer has, so they return more); higher-priced items carry more scrutiny and
buyer's remorse risk; newer customers (`customer_tenure_days` low) are statistically
likelier to return, per the generator's tenure term; and larger discounts can
correlate with impulse buys that get returned once the discount-driven urgency
fades.

**Permutation importance (mean drop in test ROC-AUC over 20 shuffles), same raw
columns:**

| feature | permutation importance |
|---|---|
| payment_method | 0.0949 |
| price_inr | 0.0103 |
| num_previous_returns | 0.0073 |
| product_category | 0.0065 |
| delivery_days | 0.0005 |
| num_previous_orders | -0.0011 |
| is_weekend_order | -0.0011 |
| **delivery_distance_km** | **-0.0024** |
| rating_given | -0.0024 |
| discount_pct | -0.0029 |
| **customer_tenure_days** | **-0.0046** |

`payment_method` and `price_inr` hold up under both measures. But **`delivery_distance_km`**
and **`customer_tenure_days`** — both in the original impurity-based top 5 —
collapse to *negative* permutation importance, meaning shuffling them doesn't hurt
the model at all (within noise, it can even look like it "helps," which is just
sampling variance around zero true signal). `delivery_distance_km` in particular is
never used anywhere in the actual return-generating formula in `generate_orders.py`
— its appearance in the impurity top-5 is a known artifact.

**Why impurity-based importance can overrate a noisy continuous feature, in one
sentence:** a continuous, high-cardinality column like `delivery_distance_km` gives
a tree enormously more candidate split points than a low-cardinality categorical
column, so the tree can always find *some* split that reduces impurity on the
training data purely by chance, inflating its impurity-based importance even when
it carries no real predictive signal on unseen data.

## Task 8 — Subgroup / root-cause analysis

**Random Forest recall/precision by `product_category` (test set, threshold 0.5):**

| category | n | recall | precision |
|---|---|---|---|
| Apparel | 385 | 53.00% | 34.19% |
| Beauty | 116 | 61.29% | 50.00% |
| **Electronics** | **261** | **32.69%** | **27.87%** |
| Footwear | 217 | 50.00% | 33.33% |
| Home | 221 | 64.71% | 22.45% |

**By `payment_method`:**

| method | n | recall | precision |
|---|---|---|---|
| COD | 503 | 87.74% | 31.70% |
| **Prepaid_Card** | **283** | **0.00%** | **0.00%** |
| Prepaid_UPI | 294 | 4.17% | 66.67% |
| Wallet | 120 | 4.76% | 50.00% |

Overall weighted recall across the test set is ~51%. **Electronics is the clearly
weaker product-category subgroup** — recall drops to 32.69%, well below the overall
average, meaning the model misses roughly two-thirds of actual Electronics returns.
This lines up with the underlying generator: Electronics returns are driven more by
the continuous `price_inr` term (high-value items have more return risk regardless
of category) than by the strong categorical "fit risk" signal that boosts recall
for Apparel/Footwear/Home — the model currently has no Electronics-specific feature
to lean on.

**Concrete fix:** add a price-tier interaction feature specific to Electronics —
e.g. a binary `is_high_value_electronics` flag (Electronics orders above, say, the
75th percentile of Electronics price) or an explicit
`price_inr × is_electronics` interaction term — so the model has a feature that
directly targets the price-driven return pattern this category exhibits, rather
than relying on `price_inr` alone to generalize across all five categories with
very different price distributions.

(The payment-method table shows an even more dramatic effect — recall for
Prepaid_Card is literally 0% — but that is a direct, expected consequence of the
generator giving COD by far the largest single coefficient in the return-risk
formula, so at the default 0.5 threshold the model has essentially learned "flag
COD orders" as its dominant strategy. A payment-method-specific threshold, lower
for prepaid methods, would be the natural first fix there.)

## Task 9 — Final artifact

- **Saved model:** `models/return_risk_model.pkl` (tuned Random Forest pipeline —
  preprocessing + `GridSearchCV`-selected classifier as one fitted
  `sklearn.pipeline.Pipeline`)
- **Best GridSearchCV params:** `max_depth=6, n_estimators=100`
- **Best cross-validated ROC-AUC:** 0.6178
- **Held-out test ROC-AUC:** 0.6143 (within 0.05 of CV — no severe overfitting)
- **t\*_rf (F1-maximizing threshold on the RF's own `predict_proba`, test split):
  0.46** — recall 60.81%, precision 29.38%, F1 0.3962

`models/return_risk_model_meta.json` stores `t_star_rf` and the exact feature
column order so Part 3's `check_return_risk` tool can load both without
re-deriving them.
