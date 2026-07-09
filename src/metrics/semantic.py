"""Semantic (embedding-based) metrics: BERTScore.

A step up from lexical overlap because it rewards meaning, not exact wording.
"""

from __future__ import annotations


def bertscore(predictions: list[str], references: list[str]) -> list[float]:
    """Batched BERTScore F1. Batched because model load is the expensive part."""
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(predictions, references, lang="en", rescale_with_baseline=True)
    return f1.tolist()


def score_batch(
    predictions: list[str], references: list[str], which: list[str]
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    if "bertscore" in which:
        out["bertscore"] = bertscore(predictions, references)
    # BLEURT left as a stretch metric — it needs a heavier checkpoint download.
    if "bleurt" in which:
        raise NotImplementedError("Wire up BLEURT-20 checkpoint if you want it.")
    return out
