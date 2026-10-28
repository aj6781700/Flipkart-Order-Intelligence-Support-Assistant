# Flipkart Order Intelligence & Support Assistant

An end-to-end e-commerce support stack built in three parts: a return-risk model, a product-image classifier, and a conversational support agent that combines both with a retrieval-backed policy knowledge base.

| Part | What it does | Headline result |
|---|---|---|
| [`1_Return-Risk_Scoring_Pipeline`](1_Return-Risk_Scoring_Pipeline) | Predicts whether an order will be returned (scikit-learn Pipeline, Random Forest, tuned threshold) | ROC-AUC 0.62; t\*_rf = 0.50 (recall 0.55, precision 0.32) |
| [`2_Product_Image_Categoriser_via_Transfer_Learning`](2_Product_Image_Categoriser_via_Transfer_Learning) | Classifies product photos into 10 categories (EfficientNet-B0 transfer learning on Fashion-MNIST) | 91.38% test accuracy |
| [`3_Flipkart_Support_Agent`](3_Flipkart_Support_Agent) | LangGraph agent: policy RAG (MiniLM + Faiss), the two models as tools, guardrails, memory | Retrieval Recall@3 0.93 over 7 queries |

## How the parts connect
```
user message ─► guardrail ─► intent ─┬─► policy RAG (Faiss over 14 policy docs)
                                     └─► tools ─┬─ check_return_risk  → Part 1 model
                                                └─ classify_product_image → Part 2 model
```

## Setup
```bash
git clone https://github.com/aj6781700/Flipkart-Order-Intelligence-Support-Assistant.git
cd Flipkart-Order-Intelligence-Support-Assistant
pip install -r 1_Return-Risk_Scoring_Pipeline/requirements.txt \
            -r 2_Product_Image_Categoriser_via_Transfer_Learning/requirements.txt \
            -r 3_Flipkart_Support_Agent/requirements-part3.txt
```
Python 3.10+. Trained artifacts are committed, so Parts 3's agent runs without retraining.

## Part 1 — return-risk pipeline
```bash
cd 1_Return-Risk_Scoring_Pipeline
python3 generate_orders.py        # writes orders_dataset.csv (6,000 rows, seed 42)
python3 train_return_risk.py      # trains, tunes, writes models/ and docs/
```
Write-up with all numbers: [`docs/PART1_ANALYSIS.md`](1_Return-Risk_Scoring_Pipeline/docs/PART1_ANALYSIS.md).

## Part 2 — image classifier
```bash
cd 2_Product_Image_Categoriser_via_Transfer_Learning
python3 export_sample_images.py       # 10 sample PNGs -> data/sample_images/
python3 train_product_classifier.py   # downloads data/weights, trains; ~1.5 h on 2 CPU cores
python3 src/classifier_inference.py   # classify the sample PNGs
```
Split 50,000 / 10,000 / 10,000; frozen backbone + linear head reached 92.2% validation, so no fine-tuning was run; test accuracy 91.38%. Confusion matrix and error analysis: [`docs/PART2_ANALYSIS.md`](2_Product_Image_Categoriser_via_Transfer_Learning/docs/PART2_ANALYSIS.md).

## Part 3 — support agent
```bash
cd 3_Flipkart_Support_Agent
python3 build_index.py          # optional: index files are committed
python3 run_transcripts.py      # regenerate every transcript
python3 evaluate_retrieval.py   # Precision@3 / Recall@3
python3 -c "from graph import build_graph, ask; print(ask(build_graph(), 't', 'How many days do I have to return a shirt?')['response'])"
```
`MOCK_LLM=1` is the default (deterministic, no API key, no network).

Design notes and results: [`docs/PART3_ANALYSIS.md`](3_Flipkart_Support_Agent/docs/PART3_ANALYSIS.md). Example conversations:
- Policy RAG: [return window](3_Flipkart_Support_Agent/transcripts/01_policy_rag_return_window.txt), [COD refund](3_Flipkart_Support_Agent/transcripts/02_policy_rag_cod_refund.txt), [pickup vs self-ship](3_Flipkart_Support_Agent/transcripts/09_policy_rag_pickup_vs_selfship.txt)
- Tools: [return risk](3_Flipkart_Support_Agent/transcripts/03_return_risk_tool.txt), [image classification](3_Flipkart_Support_Agent/transcripts/04_image_classification_tool.txt)
- Memory: [multi-turn](3_Flipkart_Support_Agent/transcripts/05_multi_turn_memory.txt), [fresh conversation](3_Flipkart_Support_Agent/transcripts/06_fresh_conversation_reset.txt)
- Guardrails: [injection blocked](3_Flipkart_Support_Agent/transcripts/07_prompt_injection_blocked.txt), [ungrounded refusal](3_Flipkart_Support_Agent/transcripts/08_ungrounded_refusal.txt)

## Notes and limitations
- The order data is synthetic; the return-risk model has modest signal (AUC ≈ 0.62) and is nearly blind to prepaid returns — see the Part 1 analysis.
- Part 2 uses EfficientNet-B0 (ImageNet weights from the EfficientNet-PyTorch GitHub release) because the torchvision weights host was unreachable when it was built.
- The sentence-embedding model `all-MiniLM-L6-v2` is loaded from the `all-minilm-l6-v2-model` PyPI package (or the Hugging Face hub if you have access).
- The groundedness threshold (0.45) is calibrated on a small query set.
