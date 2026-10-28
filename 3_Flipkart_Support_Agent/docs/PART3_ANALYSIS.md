# Part 3 — Support agent: design notes and results

## Knowledge base and retrieval
- 14 policy documents (`policy_kb/documents.py`, ~13.7 KB of text) split **sentence-wise** into 31 chunks; every chunk carries `chunk_id` (e.g. `D01-00`) and its parent `doc_id`.
- Embeddings: `all-MiniLM-L6-v2` (384-d, L2-normalised); Faiss `IndexFlatIP`, so inner product = cosine. `policy_kb/faiss_index.bin` and `policy_kb/chunk_metadata.pkl` are committed, so the agent runs without rebuilding (`python3 build_index.py` regenerates them).

## Retrieval evaluation (document level, k = 3)
Chunks are mapped to parent documents and de-duplicated before scoring. Per-query arithmetic is in `docs/retrieval_eval.txt`.

| Query (short) | Relevant docs | Precision@3 | Recall@3 |
|---|---|---|---|
| days to return a shirt | D01 | 1/2 = 0.500 | 1/1 = 1.000 |
| COD refund timing | D05 | 1/2 = 0.500 | 1/1 = 1.000 |
| laptop after 20 days | D02 | 1/2 = 0.500 | 1/1 = 1.000 |
| village delivery | D08 | 1/2 = 0.500 | 1/1 = 1.000 |
| pickup vs self-ship | D09, D10 | 1/3 = 0.333 | 1/2 = 0.500 |
| shoe size exchange | D11 | 1/2 = 0.500 | 1/1 = 1.000 |
| broken order | D12 | 1/1 = 1.000 | 1/1 = 1.000 |
| **Mean** | | **0.548** | **0.929** |

Precision is low by construction: each query has one or two relevant documents but top-3 returns up to three distinct ones. The weak spot is pickup vs self-ship, where D10 (self-ship) was not retrieved.

## Tools
- `check_return_risk` loads Part 1's real `return_risk_model.pkl`. Buckets are anchored to the model's own threshold, **t\*_rf = 0.50**: Low if p < 0.50, Medium if 0.50 ≤ p < 0.65, High if p ≥ 0.65. Spot-check (transcript 03): tool output 0.5627 equals a direct `predict_proba` on the same row.
- `classify_product_image` loads Part 2's real `product_classifier.pt` and runs it on a committed PNG (transcript 04: `07_sneaker.png` → Sneaker, 0.993).

## Graph
`guardrail → intent → (rag_retrieval | tool_calling) → response`, with conditional edges after `guardrail` (blocked → response) and after `intent`. Compiled with LangGraph's `MemorySaver`; `order_context` persists per `thread_id`, all other fields are reset every turn, and a new `thread_id` starts with empty state.

## Prompting
`prompts.py` holds the system prompt (role prompting + 4S: Specific, Short, Surround, Single, each annotated) and the few-shot intent exemplars. Routing embeds the message and picks the intent of the nearest exemplar; the matched exemplar and cosine are printed in each transcript. A `.png` path in the message routes to the image tool by rule. Weak matches (cosine < 0.35) or risk questions mentioning no order fall back to policy RAG, where the groundedness check can still refuse.

## Structured output
Every reply is `{"answer", "source", "confidence"}` with `source ∈ {policy_kb, return_risk_tool, image_classifier_tool}`. Confidence: top retrieval cosine for policy, max(p, 1−p) for risk, softmax probability for images. Replies that ask for missing details or refuse report confidence 0 (or the failing score).

## LLM mode
`MOCK_LLM=1` (default) is a deterministic template generator grounded in the retrieved chunks/tool output: no key, no network. A live-model path exists but the committed transcripts were all produced with the mock.

## Guardrails
- Prompt-injection filter (regex list) blocks before any retrieval or tool call (transcript 07).
- Groundedness: the best chunk must reach cosine **0.45**. Calibration on this KB: in-scope queries' best chunk scored 0.47–0.65; out-of-scope ones scored 0.07–0.36 (car dealership 0.364, cricket 0.185, pizza 0.294). Refusals print score vs threshold (transcript 08). The margin is narrow on the low side, so borderline questions can be refused or answered wrongly; the threshold is calibrated on a handful of queries, not a large set.

## Transcripts (`transcripts/`)
| File | Shows |
|---|---|
| 01, 02, 09 | Policy RAG with sources |
| 03 | Return-risk tool + spot-check |
| 04 | Image tool on a real PNG |
| 05 | Multi-turn memory: turn 3 recalls order #48213 from state |
| 06 | Same question in a fresh thread: no state, agent asks for details |
| 07 | Prompt injection blocked |
| 08 | Ungrounded questions refused, score vs threshold |
