# Flipkart Order Intelligence Support Assistant

An end-to-end AI/ML capstone project combining Machine Learning, Deep Learning, and an AI-powered support agent.

This repository contains all three required parts in one public GitHub repository:

- Part 1: Return-Risk Scoring Pipeline
- Part 2: Product Image Categorization via Transfer Learning
- Part 3: Flipkart Support Agent using LangGraph, RAG, and Tools

---

## Repository Structure

```text
Flipkart-Order-Intelligence-Support-Assistant/
│
├── 1_Return-Risk_Scoring_Pipeline/
│   ├── docs/
│   ├── models/
│   │   └── return_risk_model.pkl
│   ├── .gitignore
│   ├── README.md
│   ├── generate_orders.py
│   ├── orders_dataset.csv
│   ├── train_return_risk.py
│   └── requirements-part1.txt
│
├── 2_Product_Image_Categoriser_via_Transfer_Learning/
│   ├── data/
│   ├── docs/
│   ├── models/
│   │   └── product_classifier.pt
│   ├── notebooks/
│   ├── src/
│   ├── .gitignore
│   ├── README.md
│   ├── export_sample_images.py
│   ├── train_product_classifier.py
│   └── requirements-part2.txt
│
├── 3_Flipkart_Support_Agent/
│   ├── docs/
│   ├── policy_kb/
│   ├── tools/
│   ├── transcripts/
│   ├── .gitignore
│   ├── build_index.py
│   ├── evaluate_retrieval.py
│   ├── graph.py
│   ├── guardrails.py
│   ├── mock_llm.py
│   ├── prompts.py
│   └── requirements-part3.txt
│
└── README.md
```

---

# Part 1: Return-Risk Scoring Pipeline

## Objective

Part 1 predicts the probability that a customer order will be returned.

The pipeline includes:

- Synthetic order data generation
- Dataset creation
- Data preprocessing
- Feature engineering
- Return-risk model training
- Model evaluation
- Saved trained model

## Part 1 Files

```text
1_Return-Risk_Scoring_Pipeline/
│
├── generate_orders.py
├── orders_dataset.csv
├── train_return_risk.py
├── models/
│   └── return_risk_model.pkl
└── requirements-part1.txt
```

## How to Run Part 1

Open the terminal in the root repository folder and run:

```bash
cd 1_Return-Risk_Scoring_Pipeline
```

Install dependencies:

```bash
pip install -r requirements-part1.txt
```

Generate the order dataset:

```bash
python generate_orders.py
```

Run the training and evaluation pipeline:

```bash
python train_return_risk.py
```

## Part 1 Output

The generated dataset is:

```text
orders_dataset.csv
```

The trained model is:

```text
models/return_risk_model.pkl
```

---

# Part 2: Product Image Categorization via Transfer Learning

## Objective

Part 2 classifies product images using Transfer Learning.

The pipeline includes:

- Product image data preparation
- Transfer learning
- Model training
- Model evaluation
- Confusion-matrix output
- Saved PyTorch model

## Part 2 Files

```text
2_Product_Image_Categoriser_via_Transfer_Learning/
│
├── data/
├── docs/
├── models/
│   └── product_classifier.pt
├── notebooks/
├── src/
├── export_sample_images.py
├── train_product_classifier.py
└── requirements-part2.txt
```

## How to Run Part 2

From the root repository folder:

```bash
cd 2_Product_Image_Categoriser_via_Transfer_Learning
```

Install dependencies:

```bash
pip install -r requirements-part2.txt
```

Run the product image classification training pipeline:

```bash
python train_product_classifier.py
```

## Part 2 Output

The trained product classification model is:

```text
models/product_classifier.pt
```

The project also includes model evaluation and confusion-matrix output.

---

# Part 3: Flipkart Support Agent

## Objective

Part 3 implements an AI-powered Flipkart support agent using:

- LangGraph
- Retrieval-Augmented Generation (RAG)
- Knowledge-base retrieval
- Vector indexing
- Tool calling
- Guardrails
- Prompt engineering
- Retrieval evaluation
- Test conversations

## Part 3 Files

```text
3_Flipkart_Support_Agent/
│
├── docs/
├── policy_kb/
├── tools/
├── transcripts/
├── build_index.py
├── evaluate_retrieval.py
├── graph.py
├── guardrails.py
├── mock_llm.py
├── prompts.py
└── requirements-part3.txt
```

## Knowledge Base

The support and policy knowledge-base files are stored in:

```text
policy_kb/
```

These files are used by the retrieval system to answer relevant support questions.

## Vector Index

The vector index is created using:

```text
build_index.py
```

## Tool Implementations

The support-agent tool implementations are stored in:

```text
tools/
```

## LangGraph Agent

The main AI agent workflow is implemented in:

```text
graph.py
```

The workflow handles:

```text
User Query
    ↓
Guardrails
    ↓
Knowledge-Base Retrieval
    ↓
Tool Selection
    ↓
Tool Execution
    ↓
AI Support Response
```

## Retrieval Evaluation

Retrieval performance is evaluated using:

```text
evaluate_retrieval.py
```

The evaluation produces retrieval performance numbers for the knowledge-base system.

## Test Conversations

The repository contains 8 or more test support conversations inside:

```text
transcripts/
```

## How to Run Part 3

From the root repository folder:

```bash
cd 3_Flipkart_Support_Agent
```

Install dependencies:

```bash
pip install -r requirements-part3.txt
```

Build the vector index:

```bash
python build_index.py
```

Run retrieval evaluation:

```bash
python evaluate_retrieval.py
```

Run the LangGraph support agent:

```bash
python graph.py
```

---

# Complete Project Workflow

```text
PART 1: RETURN-RISK SCORING

Synthetic Order Data
        ↓
Data Generation
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
models/return_risk_model.pkl
```

```text
PART 2: PRODUCT IMAGE CATEGORIZATION

Product Images
        ↓
Data Preparation
        ↓
Transfer Learning
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Confusion Matrix
        ↓
models/product_classifier.pt
```

```text
PART 3: FLIPKART SUPPORT AGENT

User Support Query
        ↓
Guardrails
        ↓
Knowledge-Base Retrieval
        ↓
LangGraph Agent
        ↓
Tool Selection
        ↓
Tool Execution
        ↓
AI Support Response
```

---

# Complete Installation and Run Instructions

## Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

## Step 2: Open the Repository

```bash
cd Flipkart-Order-Intelligence-Support-Assistant
```

---

## Run Part 1

```bash
cd 1_Return-Risk_Scoring_Pipeline
pip install -r requirements-part1.txt
python generate_orders.py
python train_return_risk.py
```

Return to the root folder:

```bash
cd ..
```

---

## Run Part 2

```bash
cd 2_Product_Image_Categoriser_via_Transfer_Learning
pip install -r requirements-part2.txt
python train_product_classifier.py
```

Return to the root folder:

```bash
cd ..
```

---

## Run Part 3

```bash
cd 3_Flipkart_Support_Agent
pip install -r requirements-part3.txt
python build_index.py
python evaluate_retrieval.py
python graph.py
```

---

# Technologies Used

```text
PART 1: MACHINE LEARNING

Python
Pandas
NumPy
Scikit-learn


PART 2: DEEP LEARNING

Python
PyTorch
Torchvision
Transfer Learning


PART 3: AI AGENT SYSTEM

Python
LangGraph
Retrieval-Augmented Generation
Vector Indexing
Tool Calling
Prompt Engineering
Guardrails
```

---

# Submission Contents

This single public GitHub repository contains all required project artifacts.

## Part 1

```text
✓ generate_orders.py
✓ orders_dataset.csv
✓ Training code
✓ Evaluation code
✓ models/return_risk_model.pkl
```

## Part 2

```text
✓ Product image training code
✓ Product image evaluation code
✓ Confusion-matrix output
✓ models/product_classifier.pt
```

## Part 3

```text
✓ Knowledge-base files
✓ Vector-index build code
✓ Retrieval evaluation code and numbers
✓ Tool implementations
✓ LangGraph agent code
✓ Guardrails
✓ Prompt definitions
✓ transcripts/ with 8+ test conversations
```

---

# Conclusion

This capstone project demonstrates a progression across three major areas of Artificial Intelligence.

```text
PART 1
Machine Learning
Return-Risk Prediction
        ↓
PART 2
Deep Learning
Product Image Classification
        ↓
PART 3
AI Agent System
LangGraph + RAG + Tools
```

All source code, datasets, trained model artifacts, evaluations, knowledge-base files, transcripts, and documentation are included in this single GitHub repository.

---

# Author

Adarsh Kumar Jha

Capstone Project: Flipkart Order Intelligence Support Assistant
`git log --graph --all`.
