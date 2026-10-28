"""
Task 8 -- guardrails.

1) Input side: prompt-injection filter (blocks messages that try to override
   the agent's instructions or change its persona).
2) Output side: groundedness check -- a policy question is only answered if
   the best retrieved chunk clears a minimum cosine similarity; otherwise the
   agent refuses instead of letting the generator invent a policy.
"""
import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(of )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"ignore (all|every) (the )?(rules|instructions)",
    r"disregard (your|the|all) (instructions|rules|prompt|guidelines)",
    r"pretend (you are|to be|you're)",
    r"act as (if you (are|were)|an? )",
    r"you are now (a|an|the|in)\b",
    r"forget (everything|all) (you were told|above|your instructions)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"what (is|are) your (system )?(prompt|instructions)",
    r"developer mode|jailbreak",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# Minimum cosine similarity (all-MiniLM-L6-v2) the top retrieved chunk must reach.
# Calibrated in docs/PART3_ANALYSIS.md: the 7 in-scope answer-key queries score 0.47-0.65 at best,
# off-topic queries score 0.07-0.36; 0.45 sits in that gap.
GROUNDEDNESS_THRESHOLD = 0.45


def check_prompt_injection(user_input: str) -> dict:
    for pat in _COMPILED:
        m = pat.search(user_input)
        if m:
            return {"blocked": True, "matched_pattern": pat.pattern, "matched_text": m.group(0)}
    return {"blocked": False, "matched_pattern": None, "matched_text": None}


def check_groundedness(retrieved_chunks: list) -> dict:
    top = max((c["score"] for c in retrieved_chunks), default=0.0)
    return {"grounded": bool(top >= GROUNDEDNESS_THRESHOLD),
            "top_score": round(float(top), 4), "threshold": GROUNDEDNESS_THRESHOLD}
