"""Grounding / faithfulness

Faithfulness = of the claims the answer makes, what fraction are supported by the
retrieved context? 

"""

from __future__ import annotations

from ..config import ModelSpec
from .judge import generate_answer, _parse_json

_DECOMPOSE = """Break the following ANSWER into a list of atomic factual claims \
(each independently checkable). Return ONLY JSON: {{"claims": ["...", "..."]}}.

ANSWER:
{answer}"""

_VERIFY = """Given the CONTEXT, is the CLAIM supported? Answer strictly.
Return ONLY JSON: {{"supported": true | false}}.

CONTEXT:
{context}

CLAIM:
{claim}"""


def faithfulness(model: ModelSpec, answer: str, contexts: list[str]) -> float:
    """Fraction of atomic claims in `answer` supported by `contexts`.

    Uses an LLM to decompose then verify. Deterministic-ish at temperature 0.
    Returns NaN if the answer contains no checkable claims.
    """
    context_block = "\n\n".join(contexts) or "(none)"

    claims = _parse_json(generate_answer(_DECOMPOSE.format(answer=answer), model)).get("claims", [])
    if not claims:
        return float("nan")

    supported = 0
    for claim in claims:
        v = _parse_json(generate_answer(_VERIFY.format(context=context_block, claim=claim), model))
        supported += int(bool(v.get("supported")))
    return supported / len(claims)
