# Flipkart Order Intelligence & Support Assistant


One connected system: a return-risk model trained on Flipkart-style order
history (Part 1), a product-image categoriser trained with transfer
learning (Part 2), and a LangGraph support agent (Part 3) that loads both
trained artifacts as real, callable tools on top of its own
retrieval-augmented policy knowledge base. Nothing in Part 3 is a
hardcoded stand-in for Parts 1 or 2 — both tools load the actual saved
model files and call their real inference methods.

| Part | Description | Marks |
|---|---|---|
| [Part 1](#part-1--return-risk-scoring-pipeline-35-marks) | Return-Risk Scoring Pipeline | 35 |
| [Part 2](#part-2--product-image-categoriser-25-marks) | Product Image Categoriser (Transfer Learning) | 25 |
| [Part 3](#part-3--flipkart-support-agent-40-marks) | Flipkart Support Agent (LangGraph + RAG) | 40 |

---

## Repository Structure

```
Flipkart-Order-Intelligence-Support-Assistant/
├── 1_Return-Risk_Scoring_Pipeline/
│   ├── generate_orders.py            # exact seeded dataset generator
│   ├── orders_dataset.csv            # 6,000-row generated dataset
│   ├── train_return_risk.py          # preprocessing, baseline, LR, RF, save
│   ├── models/
│   │   ├── return_risk_model.pkl     # final artifact: tuned RF pipeline
│   │   └── return_risk_model_meta.json  # t*_rf threshold + feature order
│   └── docs/
│       ├── PART1_ANALYSIS.md         # full written analysis, real numbers
│       └── *.csv, part1_report.json  # supporting tables
│
├── 2_Product_Image_Categoriser_via_Transfer_Learning/
│   ├── train_product_classifier.py   # ResNet-18 transfer learning pipeline
│   ├── export_sample_images.py       # exports real test-split images as .png
│   ├── requirements-part2.txt
│   ├── models/product_classifier.pt  # trained model artifact
│   ├── data/sample_images/           # real exported test-split .png files
│   ├── src/classifier_inference.py   # classify_product_image() -- called by Part 3
│   └── docs/                         # confusion matrix, report, how-to
│
├── 3_Flipkart_Support_Agent/
│   ├── policy_kb/documents.py        # 14 policy docs, sentence-wise chunked
│   ├── build_index.py                # embeds chunks, builds Faiss index
│   ├── retrieval.py                  # top-k similarity search
│   ├── tools/
│   │   ├── check_return_risk.py      # real Part 1 model call
│   │   └── classify_product_image.py # real Part 2 model call
│   ├── graph.py                      # 5-node LangGraph graph
│   ├── prompts.py                    # 4S-annotated prompt + few-shot intent classifier
│   ├── mock_llm.py                   # deterministic structured-JSON generator
│   ├── guardrails.py                 # prompt-injection filter + groundedness check
│   ├── run_transcripts.py            # generates the 8 required transcripts
│   ├── evaluate_retrieval.py         # Precision@3 / Recall@3
│   ├── transcripts/                  # all 8+ test-conversation transcripts
│   ├── requirements-part3.txt
│   └── docs/PART3_HOWTO.md
│
└── README.md                         # this file
```

---

## Part 1 — Return-Risk Scoring Pipeline (35 marks)

Predicts whether an order will be returned, trained on a deterministic
6,000-row synthetic Flipkart order dataset.

### How to regenerate the dataset and model

```bash
cd 1_Return-Risk_Scoring_Pipeline
pip install scikit-learn pandas numpy joblib
python3 generate_orders.py      # regenerates orders_dataset.csv (seed=42, deterministic)
python3 train_return_risk.py    # trains baseline, Logistic Regression, Random Forest; saves the final model
```

### Key results

| Metric | Value |
|---|---|
| Dataset | 6,000 rows, return rate 22.75% |
| Baseline (DummyClassifier) accuracy / F1(class 1) | 77.25% / 0.0 |
| Logistic Regression ROC-AUC / F1 @0.5 | 0.6253 / 0.3921 |
| Random Forest best CV ROC-AUC | 0.6178 |
| Random Forest test ROC-AUC | 0.6143 |
| **t\*_rf** (F1-maximising threshold on RF's own predict_proba) | **0.46** |
| Weakest subgroup | Electronics (recall 32.69% vs. ~51% overall) |

Full analysis — missingness classification (MAR), the baseline "high
accuracy, zero recall" trap, threshold trade-offs, impurity vs. permutation
feature importance, and subgroup root-cause analysis — is in
[`1_Return-Risk_Scoring_Pipeline/docs/PART1_ANALYSIS.md`](1_Return-Risk_Scoring_Pipeline/docs/PART1_ANALYSIS.md).

`models/return_risk_model.pkl` is the exact fitted `sklearn.Pipeline`
(preprocessing + tuned `RandomForestClassifier`) that Part 3's
`check_return_risk` tool loads directly.

---

## Part 2 — Product Image Categoriser (25 marks)

Transfer-learns a ResNet-18 (frozen backbone + trained head, with
conditional fine-tuning) on Fashion-MNIST to classify 10
apparel/footwear/accessory categories.

### How to run training and evaluation

```bash
cd 2_Product_Image_Categoriser_via_Transfer_Learning
pip install -r requirements-part2.txt
python3 train_product_classifier.py   # downloads Fashion-MNIST, trains, evaluates, saves model + confusion matrix
python3 export_sample_images.py       # exports 10 real test-split images as .png
```

Splits: 50,000 train / 10,000 validation / 10,000 test (stratified,
test split untouched until final evaluation). Preprocessing: grayscale→3
channels, resized to 224×224, ImageNet-normalized. Full settings, the
confusion-matrix output, and per-class precision/recall are documented in
[`2_.../docs/`](2_Product_Image_Categoriser_via_Transfer_Learning/docs/).

`models/product_classifier.pt` and `data/sample_images/*.png` are the
artifacts Part 3's `classify_product_image` tool loads directly.

---

## Part 3 — Flipkart Support Agent (40 marks)

A LangGraph agent that answers policy questions via RAG over a
14-document knowledge base (30 sentence-wise chunks, embedded with
`all-MiniLM-L6-v2`, indexed with Faiss), and calls Part 1's and Part 2's
real saved models as tools for order-risk and product-category questions.

### How to run the agent in default MOCK_LLM mode

```bash
cd 3_Flipkart_Support_Agent
pip install -r requirements-part3.txt
python3 build_index.py          # embeds the KB, builds the Faiss vector index
python3 run_transcripts.py      # runs all 8+ required test conversations, saves to transcripts/
python3 evaluate_retrieval.py   # Precision@3 / Recall@3, document-level, per-query
```

No API keys or network calls are needed at run time (only once, up front,
to download the embedding model) — `MOCK_LLM` is the default mode and every
transcript in `transcripts/` is run against it.

### check_return_risk — bucket cut points

Anchored to Part 1's own **t\*_rf = 0.46** (not a fixed 0.3/0.6 split):
**Low** if probability < 0.46, **High** if probability ≥ 0.61 (0.46 + 0.15),
**Medium** otherwise. This keeps the buckets meaningful for whatever
probability range this specific trained Random Forest actually produces,
rather than assuming a fixed split works for any model.

### Verified tool correctness

Spot-check required by the brief: calling `check_return_risk` through the
agent and calling `1_Return-Risk_Scoring_Pipeline/models/return_risk_model.pkl`
directly, with the same input, both return **0.6212** — identical, confirming
the tool is a real function call, not a hardcoded stand-in.

### Example transcript

See `3_Flipkart_Support_Agent/transcripts/` for all 8+ transcripts,
covering: two policy questions answered via RAG, one return-risk question,
one product-category question, a multi-turn exchange showing state carried
across turns plus the matching fresh-conversation transcript showing state
correctly reset, a prompt-injection attempt visibly blocked by the
input-side guardrail, and an ungrounded policy question correctly refused
by the output-side groundedness check (with the retrieved chunk's
similarity score printed against the threshold).

Retrieval evaluation (Precision@3 / Recall@3, computed at the document
level across 7 test queries, with per-query arithmetic) is produced by
`evaluate_retrieval.py` and detailed in
[`3_Flipkart_Support_Agent/docs/PART3_HOWTO.md`](3_Flipkart_Support_Agent/docs/PART3_HOWTO.md).

---

## How the Three Parts Connect

```
Part 1 (Random Forest)  ──┐
                           ├──► Part 3's tools/  ──► LangGraph agent ──► Support-agent answer
Part 2 (ResNet-18)      ──┘
                                      ▲
Policy KB (14 docs) ──► Faiss index ──┘  (RAG retrieval, when intent = policy question)
```

---

## Git Workflow

This repository's commit history includes feature branches
(`part1-return-risk`, `part2-image-classifier`, `part3-support-agent`),
each created and committed to multiple times, then merged into `main`:

```bash
git log --graph --all --oneline
```

---

## Notes on Environment

- All three Parts run fully offline once dependencies are installed and,
  for Part 2/3, the one-time downloads (Fashion-MNIST, ImageNet-pretrained
  ResNet-18, `all-MiniLM-L6-v2`) are complete.
- If you hit a NumPy-related import error when installing Part 3's
  dependencies (`faiss-cpu` and similar libraries are not yet built
  against NumPy 2.x), pin NumPy below version 2:
  ```bash
  pip install "numpy<2" --force-reinstall
  ```
  `requirements-part3.txt` already pins this.
