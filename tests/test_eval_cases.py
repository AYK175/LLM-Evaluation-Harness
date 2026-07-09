"""pytest for LLMs


 deepeval test run tests/test_eval_cases.py
or
 pytest tests/test_eval_cases.py -k "not llm"


"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GEN_FILE = Path("data/raw/asqa_meta_eval_v1__generations.jsonl")


def _load_generations() -> list[dict]:
    if not GEN_FILE.exists():
        pytest.skip(f"No generations at {GEN_FILE}; run `python -m src.generate` first.")
    return [json.loads(line) for line in GEN_FILE.read_text().splitlines()]


# Plain assertions

def test_generations_are_nonempty():
    for rec in _load_generations():
        assert rec["answer"].strip(), f"empty answer from {rec['model_name']} on {rec['example_id']}"


def test_every_example_has_all_models():
    recs = _load_generations()
    by_example: dict[str, set[str]] = {}
    for r in recs:
        by_example.setdefault(r["example_id"], set()).add(r["model_name"])
    counts = {len(v) for v in by_example.values()}
    assert len(counts) == 1, f"uneven model coverage across examples: {counts}"



# LLM-backed gates (DeepEval)

@pytest.mark.llm
def test_faithfulness_gate():
    """Example DeepEval gate: answers must clear a faithfulness threshold.

    Fill in with real DeepEval metrics once generations exist. Sketch:

        from deepeval import assert_test
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        for rec in _load_generations():
            case = LLMTestCase(
                input=rec["question"],
                actual_output=rec["answer"],
                retrieval_context=rec["contexts"],
            )
            assert_test(case, [FaithfulnessMetric(threshold=0.7)])
    """
    pytest.skip("Enable once DeepEval metrics are configured with an API key.")
