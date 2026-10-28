"""
Task 6 -- prompting.

SYSTEM_PROMPT is annotated with the 4S framework (Specific, Short, Surround,
Single) plus role prompting.  INTENT_EXEMPLARS are the few-shot examples the
agent uses to route a message: the incoming message is embedded and compared
to every exemplar, and the intent of the nearest one wins (the matched
exemplar and its similarity are printed in the transcripts, so you can see
the few-shot examples actually driving the routing).
"""

# --------------------------------------------------------------------------
# System prompt  (role prompting + 4S)
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """\
[ROLE]      You are "Kira", a senior customer-support specialist at Flipkart with
            five years of experience resolving order, return and refund queries.
[SPECIFIC]  Answer only questions about (a) Flipkart return/refund/delivery policy,
            (b) the return-risk of a given order, (c) what product category an
            uploaded product image shows. Politely decline anything else.
[SHORT]     Reply in at most three sentences, plain language, no preamble.
[SURROUND]  Policy facts come ONLY from the <context> block below; order-risk and
            image facts come ONLY from the tool result. Never invent a number of
            days, an amount, or a policy that is not in the context.
            <context>{context}</context>
[SINGLE]    Do one job per reply: give exactly one answer, and return it as JSON:
            {{"answer": str, "source": "policy_kb" | "return_risk_tool" | "image_classifier_tool",
              "confidence": float between 0 and 1}}
"""

# JSON output examples (few-shot for the generator's output format)
OUTPUT_FEWSHOT = [
    {"user": "How many days do I have to return a shirt?",
     "assistant": {"answer": "Apparel can be returned within 14 days of delivery.",
                   "source": "policy_kb", "confidence": 0.6}},
    {"user": "Is order #1 likely to be returned?",
     "assistant": {"answer": "Order #1 has a 0.71 return probability, which is High risk.",
                   "source": "return_risk_tool", "confidence": 0.71}},
]

# --------------------------------------------------------------------------
# Few-shot intent exemplars used for routing
# --------------------------------------------------------------------------
INTENT_EXEMPLARS = {
    "policy_question": [
        "How many days do I have to return a product?",
        "When will I get my refund after I return an item?",
        "Does Flipkart offer exchange for a different size?",
        "How long does delivery take to my area?",
        "What happens if my order arrives damaged?",
        "Can I cancel an order after it is placed?",
    ],
    "return_risk": [
        "How likely is this order to be returned?",
        "Check the return risk for this order.",
        "What is the risk bucket for order number 123?",
        "Going back to that order, what was its risk level again?",
        "Is this customer going to return the purchase?",
    ],
    "image_classification": [
        "What category is the product in this image?",
        "Classify the product photo I uploaded.",
        "Can you tell me what this picture shows? data/sample_images/photo.png",
        "Identify the item in this image file.",
    ],
}

# Human-readable routing table, printed in docs
ROUTING = {
    "policy_question": "rag_retrieval  -> policy_kb",
    "return_risk": "tool_calling -> check_return_risk",
    "image_classification": "tool_calling -> classify_product_image",
}
