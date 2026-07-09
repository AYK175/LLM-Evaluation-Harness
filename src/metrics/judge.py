""" LLM-as-judge (pointwise G-Eval style + pairwise).

Two judging modes:
  - pointwise: score a single answer 1-5 on a rubric, with chain-of-thought
    reasoning first (the G-Eval move). Returns a structured verdict.
  - pairwise: given answers A and B, pick a winner. This is what feeds the
    position-bias analysis.

Verdicts are forced into JSON so parsing is deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..config import ModelSpec
from ..generate import generate_answer  # reuse the provider dispatch


POINTWISE_RUBRIC = """You are an expert evaluator of long-form answers. Judge the \
ANSWER to the QUESTION on these dimensions, each 1-5:
  - correctness: are the claims factually accurate?
  - completeness: does it address the full question?
  - grounding: are claims supported by the CONTEXT (if provided)?
  - clarity: is it well-organized and readable?

First reason briefly about each dimension, then output your verdict.
Return ONLY a JSON object, no prose outside it, of the form:
{{"reasoning": "<2-3 sentences>", "correctness": int, "completeness": int, \
"grounding": int, "clarity": int, "overall": int}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}"""


PAIRWISE_RUBRIC = """You are an expert evaluator. Two answers (A and B) respond to \
the same QUESTION. Decide which is better overall, considering correctness, \
completeness, grounding, and clarity.

Return ONLY a JSON object: {{"reasoning": "<1-2 sentences>", "winner": "A" | "B" | "tie"}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}"""


@dataclass
class PointwiseVerdict:
    judge: str
    correctness: int
    completeness: int
    grounding: int
    clarity: int
    overall: int
    reasoning: str


@dataclass
class PairwiseVerdict:
    judge: str
    winner: str  # "A" | "B" | "tie"
    reasoning: str


def _parse_json(raw: str) -> dict:
    """Judges sometimes wrap JSON in ```json fences or add stray text."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start : end + 1])


def judge_pointwise(
    judge: ModelSpec, question: str, answer: str, context: str = ""
) -> PointwiseVerdict:
    prompt = POINTWISE_RUBRIC.format(context=context or "(none)", question=question, answer=answer)
    raw = generate_answer(prompt, judge)
    d = _parse_json(raw)
    return PointwiseVerdict(
        judge=judge.name,
        correctness=int(d["correctness"]),
        completeness=int(d["completeness"]),
        grounding=int(d["grounding"]),
        clarity=int(d["clarity"]),
        overall=int(d["overall"]),
        reasoning=d.get("reasoning", ""),
    )


def judge_pairwise(
    judge: ModelSpec, question: str, answer_a: str, answer_b: str, context: str = ""
) -> PairwiseVerdict:
    prompt = PAIRWISE_RUBRIC.format(
        context=context or "(none)", question=question, answer_a=answer_a, answer_b=answer_b
    )
    raw = generate_answer(prompt, judge)
    d = _parse_json(raw)
    return PairwiseVerdict(judge=judge.name, winner=d["winner"], reasoning=d.get("reasoning", ""))
