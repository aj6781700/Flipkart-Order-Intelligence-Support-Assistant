"""
Task 5 -- LangGraph conversation graph.

    START -> guardrail --(blocked)--------------------------> response -> END
                 |
                 +--(ok)--> intent --(policy_question)-----> rag_retrieval --> response
                                  |
                                  +--(return_risk | image)--> tool_calling --> response

Nodes: guardrail, intent, rag_retrieval, tool_calling, response   (5)
Conditional edges: after `guardrail` (blocked?) and after `intent` (which route?)

Multi-turn memory: the graph is compiled with a MemorySaver checkpointer, so
state persists per `thread_id`.  `order_context` (orders the customer has
described, keyed by order id) lives in that persistent state.  Everything else
in the state is per-turn and is reset on every call (see `ask`).  A new
thread_id is a fresh conversation with empty state.
"""
import re
from typing import Optional, TypedDict

import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from embedder import embed
from guardrails import GROUNDEDNESS_THRESHOLD, check_groundedness, check_prompt_injection
from mock_llm import generate
from prompts import INTENT_EXEMPLARS, OUTPUT_FEWSHOT, SYSTEM_PROMPT
from retrieval import retrieve
from tools.check_return_risk import check_return_risk
from tools.classify_product_image import classify_product_image


class AgentState(TypedDict, total=False):
    # ---- persistent across turns of one thread
    order_context: dict            # {"orders": {order_id: features}, "last_order_id": str|None}
    # ---- per-turn (reset on every call)
    user_input: str
    blocked: bool
    injection: dict
    intent: str
    intent_info: dict
    retrieved: list
    grounded: dict
    tool_name: Optional[str]
    tool_result: Optional[dict]
    need_info: Optional[str]
    prompt: str
    response: dict
    trace: list


# ----------------------------------------------------------------- helpers
_exemplar_cache = {}
MIN_ROUTE_SIM = 0.35


def _exemplars():
    if not _exemplar_cache:
        texts, labels = [], []
        for label, items in INTENT_EXEMPLARS.items():
            texts += items
            labels += [label] * len(items)
        _exemplar_cache.update(texts=texts, labels=labels, vecs=embed(texts))
    return _exemplar_cache


def classify_intent(text: str) -> dict:
    """Few-shot routing: nearest exemplar (cosine) decides the intent."""
    if re.search(r"\.png\b", text, re.IGNORECASE):
        return {"intent": "image_classification", "method": "rule: .png file path in message",
                "matched_exemplar": None, "similarity": None}
    ex = _exemplars()
    sims = ex["vecs"] @ embed([text])[0]
    i = int(np.argmax(sims))
    info = {"intent": ex["labels"][i], "method": "few-shot nearest exemplar",
            "matched_exemplar": ex["texts"][i], "similarity": round(float(sims[i]), 4)}
    # Safety net: a tool route needs real evidence. Weak match, or a risk question that
    # mentions no order/customer, is handled as a policy question (the RAG groundedness
    # check then refuses it if the KB cannot answer).
    weak = info["similarity"] < MIN_ROUTE_SIM
    no_order_ref = info["intent"] == "return_risk" and not re.search(r"order|#\s*\d|customer", text, re.I)
    if info["intent"] != "policy_question" and (weak or no_order_ref):
        info.update(intent="policy_question",
                    method="few-shot nearest exemplar (weak/unsupported tool match -> default to policy RAG)")
    return info


CATEGORIES = ["Apparel", "Electronics", "Footwear", "Home", "Beauty"]


def parse_order_details(text: str) -> dict:
    """Pull whatever order fields the customer stated; absent fields are simply not returned."""
    t, f = text, {}
    m = re.search(r"(?:₹|rs\.?\s*|inr\s*)\s*([\d,]+)", t, re.I)
    if m:
        f["price_inr"] = float(m.group(1).replace(",", ""))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:discount|off)", t, re.I)
    if m:
        f["discount_pct"] = float(m.group(1))
    m = re.search(r"(\d+)\s*km", t, re.I)
    if m:
        f["delivery_distance_km"] = float(m.group(1))
    m = re.search(r"deliver\w*\s+(?:in|took|takes)\s+(\d+)\s+days", t, re.I)
    if m:
        f["delivery_days"] = float(m.group(1))
    m = re.search(r"customer\s+(?:for|since)\s+(\d+)\s+days", t, re.I)
    if m:
        f["customer_tenure_days"] = float(m.group(1))
    m = re.search(r"(no|\d+)\s+previous\s+orders?", t, re.I)
    if m:
        f["num_previous_orders"] = 0.0 if m.group(1).lower() == "no" else float(m.group(1))
    m = re.search(r"(no|\d+)\s+previous\s+returns?", t, re.I)
    if m:
        f["num_previous_returns"] = 0.0 if m.group(1).lower() == "no" else float(m.group(1))
    if re.search(r"weekend", t, re.I):
        f["is_weekend_order"] = 1
    elif re.search(r"weekday", t, re.I):
        f["is_weekend_order"] = 0
    m = re.search(r"rated\s+(\d)", t, re.I)
    f["rating_given"] = float(m.group(1)) if m else None
    for c in CATEGORIES:
        if re.search(rf"\b{c}\b", t, re.I):
            f["product_category"] = c
    if re.search(r"\bCOD\b|cash on delivery", t, re.I):
        f["payment_method"] = "COD"
    elif re.search(r"\bUPI\b", t, re.I):
        f["payment_method"] = "Prepaid_UPI"
    elif re.search(r"\bcard\b", t, re.I):
        f["payment_method"] = "Prepaid_Card"
    elif re.search(r"\bwallet\b", t, re.I):
        f["payment_method"] = "Wallet"
    return f


REQUIRED = ["price_inr", "discount_pct", "customer_tenure_days", "num_previous_orders",
            "num_previous_returns", "delivery_distance_km", "delivery_days",
            "product_category", "payment_method"]


# ------------------------------------------------------------------- nodes
def guardrail_node(state):
    g = check_prompt_injection(state["user_input"])
    return {"blocked": g["blocked"], "injection": g,
            "trace": state["trace"] + [f"guardrail: injection_blocked={g['blocked']}"]}


def intent_node(state):
    info = classify_intent(state["user_input"])
    return {"intent": info["intent"], "intent_info": info,
            "trace": state["trace"] + [f"intent: {info['intent']} via {info['method']}"]}


def rag_node(state):
    chunks = retrieve(state["user_input"], k=3)
    g = check_groundedness(chunks)
    return {"retrieved": chunks, "grounded": g,
            "trace": state["trace"] + [f"rag_retrieval: top score {g['top_score']} vs threshold {g['threshold']}"]}


def tool_node(state):
    text, intent = state["user_input"], state["intent"]
    ctx = state.get("order_context") or {"orders": {}, "last_order_id": None}
    trace = list(state["trace"])

    if intent == "image_classification":
        m = re.search(r"[\w./\\-]+\.png", text, re.I)
        if not m:
            return {"tool_name": "classify_product_image", "need_info": "Please share the path of the .png image to classify.",
                    "trace": trace + ["tool_calling: classify_product_image -> no image path given"]}
        res = classify_product_image(m.group(0))
        return {"tool_name": "classify_product_image", "tool_result": res,
                "trace": trace + [f"tool_calling: classify_product_image({m.group(0)})"]}

    # ---- return_risk
    m = re.search(r"#\s*(\d+)", text)
    explicit_id = m.group(1) if m else None
    order_id = explicit_id or ctx.get("last_order_id")
    stated = parse_order_details(text)
    orders = dict(ctx["orders"])
    have = {**orders.get(order_id, {}), **{k: v for k, v in stated.items() if v is not None}} if order_id \
        else {k: v for k, v in stated.items() if v is not None}
    if order_id is None and have:
        order_id = "current"
    missing = [k for k in REQUIRED if k not in have]
    if order_id is None or missing:
        which = f"order #{explicit_id}" if explicit_id else "that order"
        what = ("any details" if len(missing) == len(REQUIRED)
                else "only partial details (missing: " + ", ".join(missing) + ")")
        return {"tool_name": "check_return_risk",
                "need_info": (f"I don't have {what} for {which} in this conversation. Please share the price, "
                              "discount, category, payment method, delivery distance and days, and the customer's "
                              "tenure and previous orders/returns so I can score it."),
                "trace": trace + [f"tool_calling: check_return_risk -> missing details for {which}"]}
    have.setdefault("is_weekend_order", 0)
    have.setdefault("rating_given", None)
    orders[order_id] = have
    res = check_return_risk(have)
    res["order_id"] = f"#{order_id}" if order_id != "current" else ""
    return {"tool_name": "check_return_risk", "tool_result": res,
            "order_context": {"orders": orders, "last_order_id": order_id},
            "trace": trace + [f"tool_calling: check_return_risk(order #{order_id}) "
                              f"[details {'from this message' if any(v is not None for v in stated.values()) else 'recalled from conversation state'}]"]}


def response_node(state):
    trace = list(state["trace"])
    if state.get("blocked"):
        resp = {"answer": "I can't act on instructions that try to change how I work. "
                          "I can help with Flipkart return/refund policy, order return risk, or product images.",
                "source": "policy_kb", "confidence": 0.0}
        return {"response": resp, "prompt": "", "trace": trace + ["response: blocked by prompt-injection guardrail"]}
    if state.get("need_info"):
        src = "image_classifier_tool" if state["intent"] == "image_classification" else "return_risk_tool"
        return {"response": {"answer": state["need_info"], "source": src, "confidence": 0.0},
                "prompt": "", "trace": trace + ["response: asked the customer for missing details"]}

    if state["intent"] == "policy_question":
        g = state["grounded"]
        if not g["grounded"]:
            resp = {"answer": "I couldn't find this in Flipkart's policy documents, so I won't guess. "
                              "Please contact Flipkart customer support for this one.",
                    "source": "policy_kb", "confidence": g["top_score"]}
            return {"response": resp, "prompt": "",
                    "trace": trace + [f"response: REFUSED - groundedness {g['top_score']} < threshold {g['threshold']}"]}
        ctx_text = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in state["retrieved"])
        kind, payload = "policy", state["retrieved"]
    else:
        ctx_text = str(state["tool_result"])
        kind = "risk" if state["tool_name"] == "check_return_risk" else "image"
        payload = state["tool_result"]
    prompt = SYSTEM_PROMPT.format(context=ctx_text)
    resp = generate(prompt, kind, payload)
    return {"response": resp, "prompt": prompt, "trace": trace + [f"response: generated ({kind})"]}


def route_after_guardrail(state):
    return "response" if state["blocked"] else "intent"


def route_after_intent(state):
    return "rag_retrieval" if state["intent"] == "policy_question" else "tool_calling"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("guardrail", guardrail_node)
    g.add_node("intent", intent_node)
    g.add_node("rag_retrieval", rag_node)
    g.add_node("tool_calling", tool_node)
    g.add_node("response", response_node)
    g.add_edge(START, "guardrail")
    g.add_conditional_edges("guardrail", route_after_guardrail, {"response": "response", "intent": "intent"})
    g.add_conditional_edges("intent", route_after_intent, {"rag_retrieval": "rag_retrieval", "tool_calling": "tool_calling"})
    g.add_edge("rag_retrieval", "response")
    g.add_edge("tool_calling", "response")
    g.add_edge("response", END)
    return g.compile(checkpointer=MemorySaver())


def ask(graph, thread_id: str, text: str) -> dict:
    """One conversational turn. Per-turn fields are reset; order_context persists via the checkpointer."""
    fresh = {"user_input": text, "blocked": False, "injection": {}, "intent": "", "intent_info": {},
             "retrieved": [], "grounded": {}, "tool_name": None, "tool_result": None,
             "need_info": None, "prompt": "", "response": {}, "trace": []}
    return graph.invoke(fresh, config={"configurable": {"thread_id": thread_id}})


if __name__ == "__main__":
    import json
    app = build_graph()
    out = ask(app, "demo", "How many days do I have to return a shirt?")
    print(json.dumps(out["response"], indent=2))
