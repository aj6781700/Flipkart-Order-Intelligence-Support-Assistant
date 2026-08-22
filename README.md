# Flipkart Order Intelligence & Support Assistant

A support-agent system for e-commerce order operations, combining a
tabular return-risk model, a computer-vision product categoriser, and a
retrieval-augmented conversational agent that calls both models as tools.

The three components aren't separate demos — the agent in
`3_Flipkart_Support_Agent` loads the actual trained artifacts produced by
`1_Return-Risk_Scoring_Pipeline` and
`2_Product_Image_Categoriser_via_Transfer_Learning` and calls their real
inference methods. Nothing is mocked or hardcoded downstream of training.

## What it does

- **Predicts return risk** for an incoming order using a Random Forest
  trained on order history (price, category, payment method, customer
  tenure, delivery details, etc.), with risk buckets calibrated to the
  model's own probability distribution rather than arbitrary fixed
  thresholds.
- **Classifies product images** into 10 apparel/footwear/accessory
  categories using a ResNet-18 backbone fine-tuned via transfer learning.
- **Answers policy questions** (return windows, refund timelines, delivery
  SLAs, reverse pickup) through retrieval-augmented generation over a
  hand-authored knowledge base, with groundedness checks that refuse to
  answer rather than fabricate a policy when nothing relevant is retrieved.
- **Routes and remembers**: a LangGraph agent classifies user intent,
  branches to the right tool or retrieval path, and carries short-term
  conversational context across turns (e.g. a follow-up question about an
  order mentioned earlier in the same conversation).
- **Runs entirely offline** in its default mode — no API keys, no network
  calls at inference time, fully deterministic and reproducible.

## Repository structure

```
.
├── 1_Return-Risk_Scoring_Pipeline/
│   ├── generate_orders.py            # deterministic synthetic order-data generator
│   ├── orders_dataset.csv
│   ├── train_return_risk.py          # preprocessing, baseline, tuned models, save
│   ├── models/
│   │   ├── return_risk_model.pkl
│   │   └── return_risk_model_meta.json
│   └── docs/
│       ├── PART1_ANALYSIS.md
│       └── (supporting metrics/CSVs)
│
├── 2_Product_Image_Categoriser_via_Transfer_Learning/
│   ├── train_product_classifier.py   # ResNet-18 transfer learning
│   ├── export_sample_images.py
│   ├── requirements-part2.txt
│   ├── models/product_classifier.pt
│   ├── data/sample_images/
│   ├── src/classifier_inference.py
│   └── docs/
│
├── 3_Flipkart_Support_Agent/
│   ├── policy_kb/documents.py        # knowledge base, sentence-level chunking
│   ├── build_index.py                # embeddings + Faiss vector index
│   ├── retrieval.py
│   ├── tools/
│   │   ├── check_return_risk.py      # calls Part 1's saved model
│   │   └── classify_product_image.py # calls Part 2's saved model
│   ├── graph.py                      # LangGraph agent definition
│   ├── prompts.py
│   ├── mock_llm.py                   # deterministic offline response generator
│   ├── guardrails.py                 # prompt-injection + groundedness checks
│   ├── run_transcripts.py
│   ├── evaluate_retrieval.py
│   ├── transcripts/
│   ├── requirements-part3.txt
│   └── docs/
│
└── README.md
```

## 1. Return-risk model

```bash
cd 1_Return-Risk_Scoring_Pipeline
pip install scikit-learn pandas numpy joblib
python3 generate_orders.py
python3 train_return_risk.py
```

Trains and compares a baseline, a Logistic Regression, and a
GridSearchCV-tuned Random Forest; saves the final pipeline plus its
F1-optimal decision threshold. Full methodology and results — including
missing-data diagnosis, threshold trade-offs, permutation vs. impurity
feature importance, and subgroup performance breakdowns — are in
[`1_Return-Risk_Scoring_Pipeline/docs/PART1_ANALYSIS.md`](1_Return-Risk_Scoring_Pipeline/docs/PART1_ANALYSIS.md).

`models/return_risk_model.pkl` is what the agent loads directly.

## 2. Product image classifier

```bash
cd 2_Product_Image_Categoriser_via_Transfer_Learning
pip install -r requirements-part2.txt
python3 train_product_classifier.py
python3 export_sample_images.py
```

Fine-tunes an ImageNet-pretrained ResNet-18 on Fashion-MNIST — frozen
backbone with a trained classifier head, with conditional fine-tuning of
the final block if the initial validation accuracy is insufficient.
Preprocessing, training configuration, and evaluation results are in
[`2_Product_Image_Categoriser_via_Transfer_Learning/docs/`](2_Product_Image_Categoriser_via_Transfer_Learning/docs/).

`models/product_classifier.pt` and the exported `data/sample_images/*.png`
are what the agent loads and classifies against.

## 3. Support agent

```bash
cd 3_Flipkart_Support_Agent
pip install -r requirements-part3.txt
python3 build_index.py
python3 run_transcripts.py
python3 evaluate_retrieval.py
```

A LangGraph agent with conditional routing: an intent classifier decides
whether a message is a policy question, a return-risk query, or a
product-category query, and branches accordingly to retrieval or the
relevant tool. Runs against a deterministic, fully offline response
generator by default — no API key required.

**Return-risk tool correctness is independently verifiable**: calling
`check_return_risk` through the agent and calling the saved model directly
with the same input return identical probabilities, confirming the tool
calls the real artifact rather than standing in for it.

Design details — chunking strategy, the groundedness threshold, and the
full test-conversation transcripts (including a blocked prompt-injection
attempt and a correctly-refused ungrounded question) — are in
[`3_Flipkart_Support_Agent/docs/`](3_Flipkart_Support_Agent/docs/) and
[`3_Flipkart_Support_Agent/transcripts/`](3_Flipkart_Support_Agent/transcripts/).

## How it fits together

```
Return-risk model (Random Forest) ──┐
                                      ├──► Agent tools ──► LangGraph agent ──► Response
Product classifier (ResNet-18)    ──┘
                                          ▲
Policy knowledge base ──► Faiss index ────┘   (retrieval, for policy-intent queries)
```

## Notes on environment

- All components run fully offline once dependencies and one-time model
  downloads (Fashion-MNIST, the pretrained ResNet-18 weights, and the
  sentence-embedding model) are complete.
- If installing the support-agent's dependencies raises a NumPy-related
  import error, pin NumPy below version 2 (`pip install "numpy<2"
  --force-reinstall`) — some vector-index libraries aren't yet built
  against NumPy 2.x. `requirements-part3.txt` already pins this.
