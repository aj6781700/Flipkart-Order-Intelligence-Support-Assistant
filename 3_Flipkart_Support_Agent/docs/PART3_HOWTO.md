# Part 3 — How to Run (and what's already verified vs. still pending)

## What's already REAL and verified (tested in this build, not simulated)

- **`check_return_risk` tool**: loads Part 1's actual `return_risk_model.pkl`
  and calls real `.predict_proba()`. Spot-checked against calling the model
  directly outside the agent — **identical probability** (0.6212 both ways).
  Risk buckets are correctly anchored to `t*_rf = 0.46` (Low < 0.46, High ≥
  0.61, Medium between) — not a fixed 0.3/0.6 split.
- **Guardrails**: prompt-injection pattern matching tested directly — blocks
  "ignore previous instructions", "pretend you are...", etc.; passes through
  normal policy questions.
- **Intent classification**: few-shot exemplars tested directly — all 3
  intents (`policy_question`, `return_risk_query`, `product_category_query`)
  route correctly off the same examples shown in the system prompt.
- **Multi-turn state**: manually traced through the graph's node functions
  (bypassing LangGraph's own execution, which isn't installed in this
  sandbox, but exercising the exact same node logic `graph.py` calls).
  Confirmed: turn 1 ("order #48213, Electronics, price 32000, COD")
  correctly parses those real details into `order_context`; turn 2 ("what
  about that order again?") correctly reuses them without re-parsing,
  producing an identical tool output; a **fresh** conversation asking the
  same follow-up correctly falls back to generic defaults with **no**
  memory of order #48213. This is the exact contrast Task 5 requires
  between `transcripts/05_multiturn_state_carried.txt` and
  `transcripts/06_fresh_conversation_state_reset.txt`.

## What's written to spec but NOT yet executed end-to-end

This sandbox has no internet access, so `langgraph`, `sentence-transformers`,
and `faiss-cpu` can't be installed here, and the `all-MiniLM-L6-v2`
embedding model can't be downloaded. That means:

- `build_index.py` (Task 2 — embed + Faiss index) has not been run
- The full `graph.py` compiled LangGraph execution (as opposed to the
  manually-traced node logic above) has not been run
- `run_transcripts.py` has not produced the actual 8 transcript files yet
- `evaluate_retrieval.py`'s Precision@3/Recall@3 numbers are not yet real
- Part 2's `classify_product_image` tool depends on
  `2_Product_Image_Categoriser_via_Transfer_Learning/models/product_classifier.pt`,
  which also hasn't been trained yet (see `PART2_HOWTO.md`)

**No fake transcripts or fabricated retrieval scores are included anywhere
in this repo.** Per the brief's acceptance criteria, those only get written
once they're real.

## How to run it for real (in order)

```bash
cd 3_Flipkart_Support_Agent
pip install -r requirements-part3.txt
```

1. **First, make sure Part 1 and Part 2 artifacts exist** (siblings to this
   folder):
   - `1_Return-Risk_Scoring_Pipeline/models/return_risk_model.pkl` — should
     already exist if you ran Part 1.
   - `2_Product_Image_Categoriser_via_Transfer_Learning/models/product_classifier.pt`
     and `data/sample_images/*.png` — run Part 2's training first if these
     don't exist yet.

2. **Build the embedding index** (one-time, needs internet to download the
   embedding model the first time):
   ```bash
   python3 build_index.py
   ```
   This creates `policy_kb/faiss_index.bin` and `policy_kb/chunk_metadata.pkl`.

3. **Generate all 8 required transcripts:**
   ```bash
   python3 run_transcripts.py
   ```
   This writes `transcripts/01_*.txt` through `transcripts/08_*.txt`.

4. **Run the retrieval evaluation:**
   ```bash
   python3 evaluate_retrieval.py
   ```
   Prints per-query Precision@3/Recall@3 arithmetic and the averages.

5. **Send me the console output / transcript files** and I'll verify each
   acceptance criterion against your real run and write up the final
   analysis section, the same way Parts 1 and 2 were documented.

## Design decisions, documented (so a grader doesn't have to guess)

- **Chunking strategy**: sentence-wise (Task 1). Each policy sentence is a
  self-contained factual claim, so splitting on sentence boundaries keeps
  every retrieved chunk directly usable without truncating mid-clause —
  chosen over fixed-size/overlapping windows for that reason.
- **Risk bucket cut points**: anchored to `t*_rf` (0.46) from Part 1, not a
  fixed 0.3/0.6 split — Low if probability < 0.46, High if probability ≥
  0.61 (t*_rf + 0.15), Medium otherwise. See `tools/check_return_risk.py`.
- **Groundedness threshold**: 0.45 cosine similarity (`guardrails.py`,
  `GROUNDEDNESS_THRESHOLD`). Below this, the agent refuses rather than
  letting MOCK_LLM's template generator fabricate an answer.
- **5 graph nodes**: `guardrail_node → intent_node → (conditional) →
  rag_retrieval_node | tool_calling_node → response_generation_node`. Two
  conditional-edge decision points (blocked vs. not; policy vs. tool-based
  intent) mean the graph genuinely branches rather than always running
  every node.
- **Few-shot intent examples**: the same 3 examples shown in
  `prompts.SYSTEM_PROMPT`'s few-shot block are reused verbatim as literal
  exemplars inside `prompts.classify_intent()`, so they demonstrably drive
  MOCK_LLM's routing decisions rather than sitting inertly in prompt text
  nothing reads.
