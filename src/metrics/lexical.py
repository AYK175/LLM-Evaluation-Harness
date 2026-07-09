"""Lexical (surface-overlap) metrics: ROUGE, BLEU, METEOR.

These are the reference-based baselines. The whole rhetorical arc of the project
starts here: we want to see if any of these metrics correlate with human judgment.
If they don't, we need to find better metrics. If they do, we can use them to evaluate models
All functions take (prediction, reference) strings and return a float in [0, 1].
"""

from __future__ import annotations


def rouge_l(prediction: str, reference: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def bleu(prediction: str, reference: str) -> float:
    import sacrebleu

    # sentence_bleu expects a hypothesis string and a list of references
    return sacrebleu.sentence_bleu(prediction, [reference]).score / 100.0


def meteor(prediction: str, reference: str) -> float:
    import nltk
    from nltk.translate.meteor_score import meteor_score

    # METEOR needs wordnet + tokenizer data; download once, quietly.
    for pkg in ("wordnet", "omw-1.4", "punkt"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    return meteor_score([reference.split()], prediction.split())


REGISTRY = {"rouge": rouge_l, "bleu": bleu, "meteor": meteor}


def score(prediction: str, reference: str, which: list[str]) -> dict[str, float]:
    """Score one prediction against one reference across the requested metrics."""
    if not reference:
        return {name: float("nan") for name in which}
    return {name: REGISTRY[name](prediction, reference) for name in which}
