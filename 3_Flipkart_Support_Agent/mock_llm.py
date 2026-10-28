"""
Task 7 -- the generator.

MOCK_LLM=1 (the default) uses a deterministic, template-based generator, so the
whole agent runs with no API key and no network.  It is grounded: it can only
say what the retrieved chunks / tool results contain.

MOCK_LLM=0 with ANTHROPIC_API_KEY set sends the same assembled prompt to a live
model and expects the same JSON back.  The committed transcripts were produced
with the mock only; the live path is not exercised there.
"""
import json
import os

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"


def _mock_policy(chunks):
    top = chunks[0]
    same_doc = [c["text"] for c in chunks if c["doc_id"] == top["doc_id"]]
    return {"answer": " ".join(same_doc), "source": "policy_kb",
            "confidence": round(float(top["score"]), 2)}


def _mock_risk(tool_out):
    p, b = tool_out["return_probability"], tool_out["risk_bucket"]
    cp = tool_out["cut_points"]
    return {"answer": (f"Order {tool_out.get('order_id', '')} has a {p:.2f} predicted return probability, "
                       f"which is {b} risk (Low < {cp['low_below']}, High >= {cp['high_at_or_above']})."),
            "source": "return_risk_tool", "confidence": round(max(p, 1 - p), 2)}


def _mock_image(tool_out):
    return {"answer": (f"The image {tool_out['image_file']} looks like a {tool_out['predicted_class']} "
                       f"({tool_out['confidence']:.0%} model confidence)."),
            "source": "image_classifier_tool", "confidence": round(tool_out["confidence"], 2)}


def generate(prompt: str, kind: str, payload) -> dict:
    """kind in {policy, risk, image}; payload = retrieved chunks or tool output."""
    if MOCK_LLM:
        return {"policy": _mock_policy, "risk": _mock_risk, "image": _mock_image}[kind](payload)
    import anthropic                                         # pragma: no cover
    client = anthropic.Anthropic()
    msg = client.messages.create(model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"),
                                 max_tokens=300, system=prompt,
                                 messages=[{"role": "user", "content": "Answer in the required JSON."}])
    return json.loads(msg.content[0].text)
