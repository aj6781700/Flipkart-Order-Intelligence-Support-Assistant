"""
Part 3, Task 7 -- MOCK_LLM deterministic mode.

Given retrieved KB chunk(s) and/or tool output, deterministically composes
the final structured answer with ZERO network calls and ZERO API keys.
This is the DEFAULT mode and is what every graded transcript in Task 9 runs
against.

Output is always the fixed structured schema required by Task 6:
    {"answer": str, "source": "policy_kb" | "return_risk_tool" | "image_classifier_tool", "confidence": float}
"""


def generate_policy_answer(retrieved_chunks: list[dict]) -> dict:
    """Compose an answer from the top retrieved chunk(s), template-based."""
    if not retrieved_chunks:
        return {"answer": "I don't have information on that.", "source": "policy_kb", "confidence": 0.0}
    top = retrieved_chunks[0]
    answer = top["text"]
    # if a close second chunk from the SAME document exists, append it for completeness
    if len(retrieved_chunks) > 1 and retrieved_chunks[1]["doc_id"] == top["doc_id"]:
        answer += " " + retrieved_chunks[1]["text"]
    return {"answer": answer, "source": "policy_kb", "confidence": round(top["score"], 4)}


def generate_return_risk_answer(tool_output: dict) -> dict:
    p = tool_output["return_probability"]
    bucket = tool_output["risk_bucket"]
    answer = (
        f"This order has an estimated return probability of {p:.0%}, "
        f"which places it in the '{bucket}' risk bucket "
        f"(anchored to this model's own threshold t*_rf={tool_output['t_star_rf']})."
    )
    return {"answer": answer, "source": "return_risk_tool", "confidence": round(p, 4)}


def generate_image_classification_answer(tool_output: dict) -> dict:
    cls = tool_output["predicted_class"]
    conf = tool_output["confidence"]
    answer = f"This product image is classified as '{cls}' with {conf:.0%} confidence."
    return {"answer": answer, "source": "image_classifier_tool", "confidence": round(conf, 4)}


def generate_refusal(reason: str, source: str = "policy_kb") -> dict:
    return {"answer": f"I can't answer that confidently: {reason}", "source": source, "confidence": 0.0}


def generate_injection_deflection() -> dict:
    return {
        "answer": (
            "I can't follow instructions embedded in your message that ask me to change "
            "my role or reveal internal instructions. I can help with Flipkart return, "
            "refund, delivery, or order-risk questions instead."
        ),
        "source": "policy_kb",
        "confidence": 0.0,
    }
