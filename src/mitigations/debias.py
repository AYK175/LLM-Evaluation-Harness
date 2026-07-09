"""Mitigations. Measure a bias, apply a fix, re-measure, report the delta.

Two mitigations, both cheap and both grounded in the literature:

  balanced_position: for every pairwise comparison, always run both orders and
    resolve to a single verdict. Ties/disagreements between orders become "tie".
    This directly attacks the position bias measured earlier.

  jury: poll multiple judges and aggregate (majority vote for pairwise, mean for
    pointwise). A panel-of-judges reduces any single judge's idiosyncratic bias,
    including self-preference.
"""

from __future__ import annotations

from collections import Counter


def balanced_position_verdict(winner_ab: str, winner_ba: str) -> str:
    """Resolve a pair judged in both orders into one order-invariant verdict.

    winner_ab: verdict with original order (A,B)
    winner_ba: verdict with swapped order (B,A) — letters refer to slots
    Returns the winning ANSWER as "A"/"B", or "tie" if the judge was inconsistent.
    """
    # Map the swapped-order letter back to the original answer.
    ba_mapped = {"A": "B", "B": "A", "tie": "tie"}[winner_ba]
    if winner_ab == ba_mapped and winner_ab != "tie":
        return winner_ab
    return "tie"  # judge disagreed with itself across orders -> no reliable winner


def jury_pairwise(verdicts: list[str]) -> str:
    """Majority vote across judges; ties broken to 'tie'."""
    counts = Counter(v for v in verdicts if v in ("A", "B"))
    if not counts:
        return "tie"
    (top, top_n), *rest = counts.most_common()
    if rest and rest[0][1] == top_n:
        return "tie"
    return top


def jury_pointwise(scores: list[float]) -> float:
    """Mean pointwise score across judges (drops NaN)."""
    import numpy as np

    arr = np.asarray(scores, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")
