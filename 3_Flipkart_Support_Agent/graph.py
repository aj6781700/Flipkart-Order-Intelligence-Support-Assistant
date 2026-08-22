"""
Part 3, Task 5 -- LangGraph graph.

5 nodes: guardrail_node -> intent_node -> (conditional) -> rag_retrieval_node
                                                          -> tool_calling_node
         both -> response_generation_node -> END

Conditional edges branch by (a) whether the input was blocked as a prompt
injection, and (b) the classified intent (policy / return-risk /
product-category). The graph does NOT always run every node -- a
return-risk question skips rag_retrieval_node entirely, and a blocked
input skips straight to response generation.

Short-term conversational state: `order_context` in the graph state
persists across turns WITHIN one conversation (the caller re-passes the
same state dict on each `run_turn` call). A follow-up like "what about
that order's delivery time?" resolves order features from the previous
turn's order_context. A freshly-started conversation uses an empty state
dict, so order_context starts absent/reset -- see
transcripts/06_fresh_conversation_state_reset.txt for this contrast with
transcripts/05_multiturn_state_carried.txt.
"""
import pickle
import re
from pathlib import Path
from typing import TypedDict, Optional

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = "END"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from guardrails import check_prompt_injection, check_groundedness
from prompts import classify_intent
from mock_llm import (generate_policy_answer, generate_return_risk_answer,
                       generate_image_classification_answer, generate_refusal,
                       generate_injection_deflection)
from tools.check_return_risk import check_return_risk
from tools.classify_product_image import classify_product_image

INDEX_PATH = Path(__file__).resolve().parent / "policy_kb" / "faiss_index.bin"
METADATA_PATH = Path(__file__).resolve().parent / "policy_kb" / "chunk_metadata.pkl"

_embed_model = None
_faiss_index = None
_chunk_metadata = None


def _load_retrieval_assets():
    global _embed_model, _faiss_index, _chunk_metadata
    if _embed_model is None:
        import faiss
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _faiss_index = faiss.read_index(str(INDEX_PATH))
        _chunk_metadata = pickle.load(open(METADATA_PATH, "rb"))
    return _embed_model, _faiss_index, _chunk_metadata


def retrieve_chunks(query: str, k: int = 3) -> list:
    model, index, chunks = _load_retrieval_assets()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        c = dict(chunks[idx])
        c["score"] = float(score)
        results.append(c)
    return results


# ---------------------------------------------------------------------------
# Graph state schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    user_input: str
    blocked: bool
    injection_pattern: Optional[str]
    intent: str
    intent_method: str
    retrieved_chunks: list
    tool_output: dict
    final_answer: dict
    order_context: dict   # persists across turns -- the multi-turn "memory"
    history: list


ORDER_ID_RE = re.compile(r"order\s*#?\s*(\w+)", re.IGNORECASE)

DEFAULT_ORDER_FEATURES = {
    "price_inr": 1500, "discount_pct": 20, "customer_tenure_days": 100,
    "num_previous_orders": 3, "num_previous_returns": 0,
    "delivery_distance_km": 150, "delivery_days": 4,
    "is_weekend_order": 0, "rating_given": None,
    "product_category": "Apparel", "payment_method": "COD",
}
KNOWN_CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
KNOWN_PAYMENTS = ["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"]


def extract_order_features_from_text(text: str) -> dict:
    """Best-effort parse of order details mentioned in free text, layered
    on top of DEFAULT_ORDER_FEATURES for anything not mentioned. This is
    what makes turn 1 of a return-risk conversation actually reflect what
    the user said, rather than always falling back to generic defaults --
    which in turn is what makes carrying it into turn 2 meaningful."""
    features = dict(DEFAULT_ORDER_FEATURES)

    price_m = re.search(r"(?:price|₹|rs\.?)\s*(\d{3,7})", text, re.IGNORECASE)
    if price_m:
        features["price_inr"] = float(price_m.group(1))

    discount_m = re.search(r"(\d{1,2})\s*%\s*discount", text, re.IGNORECASE)
    if discount_m:
        features["discount_pct"] = float(discount_m.group(1))

    for cat in KNOWN_CATEGORIES:
        if cat.lower() in text.lower():
            features["product_category"] = cat
            break

    for pay in KNOWN_PAYMENTS:
        if pay.lower().replace("_", " ") in text.lower() or pay.lower() in text.lower():
            features["payment_method"] = pay
            break
    if "cod" in text.lower() or "cash on delivery" in text.lower():
        features["payment_method"] = "COD"

    return features


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def guardrail_node(state: AgentState) -> AgentState:
    result = check_prompt_injection(state["user_input"])
    state["blocked"] = result["blocked"]
    state["injection_pattern"] = result["matched_pattern"]
    return state


def intent_node(state: AgentState) -> AgentState:
    result = classify_intent(state["user_input"])
    state["intent"] = result["intent"]
    state["intent_method"] = result["method"]
    return state


def rag_retrieval_node(state: AgentState) -> AgentState:
    state["retrieved_chunks"] = retrieve_chunks(state["user_input"], k=3)
    return state


def tool_calling_node(state: AgentState) -> AgentState:
    text = state["user_input"]
    order_context = state.get("order_context", {}) or {}

    if state["intent"] == "return_risk_query":
        # Resolve order features: if THIS turn mentions concrete details
        # (price/category/payment), parse and use those. Otherwise -- e.g.
        # a bare follow-up like "what about that order again?" -- fall back
        # to whatever was carried in order_context from a previous turn
        # (the multi-turn memory mechanism). Only if neither is available
        # does it fall back to generic defaults.
        mentions_details = bool(re.search(r"(?:price|₹|rs\.?)\s*\d{3,7}", text, re.IGNORECASE)) or \
                            any(c.lower() in text.lower() for c in KNOWN_CATEGORIES)
        if mentions_details or not order_context.get("features"):
            features = extract_order_features_from_text(text)
        else:
            features = order_context["features"]
        state["tool_output"] = check_return_risk(features)
        state["order_context"] = {**order_context, "features": features, "last_intent": "return_risk_query"}

    elif state["intent"] == "product_category_query":
        m = re.search(r"([\w./-]+\.png)", text)
        image_path = m.group(1) if m else order_context.get("last_image_path")
        state["tool_output"] = classify_product_image(image_path)
        state["order_context"] = {**order_context, "last_image_path": image_path, "last_intent": "product_category_query"}

    return state


def response_generation_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        state["final_answer"] = generate_injection_deflection()
        return state

    if state["intent"] == "policy_question":
        grounded = check_groundedness(state.get("retrieved_chunks", []))
        if not grounded["grounded"]:
            state["final_answer"] = generate_refusal(
                f"no retrieved policy chunk cleared the similarity threshold "
                f"(top score {grounded['top_score']} < threshold {grounded['threshold']})."
            )
        else:
            state["final_answer"] = generate_policy_answer(state["retrieved_chunks"])

    elif state["intent"] == "return_risk_query":
        state["final_answer"] = generate_return_risk_answer(state["tool_output"])

    elif state["intent"] == "product_category_query":
        state["final_answer"] = generate_image_classification_answer(state["tool_output"])

    else:
        state["final_answer"] = generate_refusal("intent could not be determined.")

    return state


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------
def route_after_guardrail(state: AgentState) -> str:
    return "response_generation_node" if state.get("blocked") else "intent_node"


def route_after_intent(state: AgentState) -> str:
    if state["intent"] == "policy_question":
        return "rag_retrieval_node"
    return "tool_calling_node"  # return_risk_query or product_category_query


def build_graph():
    if StateGraph is None:
        raise ImportError("langgraph is not installed. pip install langgraph")

    graph = StateGraph(AgentState)
    graph.add_node("guardrail_node", guardrail_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("rag_retrieval_node", rag_retrieval_node)
    graph.add_node("tool_calling_node", tool_calling_node)
    graph.add_node("response_generation_node", response_generation_node)

    graph.set_entry_point("guardrail_node")
    graph.add_conditional_edges("guardrail_node", route_after_guardrail,
                                 {"intent_node": "intent_node", "response_generation_node": "response_generation_node"})
    graph.add_conditional_edges("intent_node", route_after_intent,
                                 {"rag_retrieval_node": "rag_retrieval_node", "tool_calling_node": "tool_calling_node"})
    graph.add_edge("rag_retrieval_node", "response_generation_node")
    graph.add_edge("tool_calling_node", "response_generation_node")
    graph.add_edge("response_generation_node", END)

    return graph.compile()


def run_turn(compiled_graph, state: dict, user_input: str) -> dict:
    """Run one conversational turn. Pass the SAME `state` dict back in on
    the next call to carry order_context across turns (multi-turn memory).
    Pass a fresh empty dict {} to start a new conversation with state
    correctly absent."""
    state = dict(state)  # shallow copy
    state["user_input"] = user_input
    result = compiled_graph.invoke(state)
    return result


if __name__ == "__main__":
    g = build_graph()
    state = {}
    r1 = run_turn(g, state, "How many days do I have to return a shirt?")
    print(r1["final_answer"])
