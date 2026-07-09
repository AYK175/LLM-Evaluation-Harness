"""

Three biases, each with an established metric so numbers are comparable to
the literature:

  Position bias — does the judge favor whichever answer comes first?
      - repetition_stability: agreement of the judge with itself across repeats
      - position_consistency: fraction of pairs where the verdict is order-invariant
      - preference_fairness: how balanced wins are between slot A and slot B
    (These three are the framing used in recent position-bias work.)

  Verbosity bias — does the judge reward length independent of quality?
  
"""

from __future__ import annotations

from dataclasses import dataclass

# Position bias

@dataclass
class PositionBiasReport:
    n_pairs: int
    position_consistency: float  # 1.0 = fully order-invariant (no position bias)
    preference_fairness: float  # 1.0 = wins split evenly across slots
    reversal_rate: float  # fraction of pairs whose winner flips when order flips
    slot_a_win_rate: float


def position_bias(verdicts_ab: list[str], verdicts_ba: list[str]) -> PositionBiasReport:
    """Each list holds winners ("A"/"B"/"tie") for the same pairs in opposite orders.

    In the (B,A) run, "A" means the answer sitting in slot A — which is the
    ORIGINAL B. So an order-invariant judge should flip its letter between runs.
    """
    assert len(verdicts_ab) == len(verdicts_ba)
    n = len(verdicts_ab)
    if n == 0:
        return PositionBiasReport(0, float("nan"), float("nan"), float("nan"), float("nan"))

    consistent = 0  # judge preferred the same *answer* regardless of slot
    reversals = 0
    slot_a_wins = 0
    for ab, ba in zip(verdicts_ab, verdicts_ba):
        if ab in ("A", "B"):
            slot_a_wins += int(ab == "A")
        # order-invariant if the winning ANSWER is the same: e.g. ab="A" & ba="B"
        if ab == "A" and ba == "B":
            consistent += 1
        elif ab == "B" and ba == "A":
            consistent += 1
        elif ab == ba and ab != "tie":
            reversals += 1  # same letter both times -> judge locked onto a slot

    return PositionBiasReport(
        n_pairs=n,
        position_consistency=consistent / n,
        preference_fairness=1.0 - abs(slot_a_wins / n - 0.5) * 2,
        reversal_rate=reversals / n,
        slot_a_win_rate=slot_a_wins / n,
    )


# Verbosity bias

def verbosity_bias(answer_lengths: list[int], judge_scores: list[float]) -> float:
    """Spearman correlation between answer length and judge score.

    Interpret alongside the length-vs-human correlation: if the judge correlates
    with length more strongly than humans do, that gap is verbosity bias.
    """
    import numpy as np
    from scipy.stats import spearmanr

    lengths = np.asarray(answer_lengths, dtype=float)
    scores = np.asarray(judge_scores, dtype=float)
    mask = np.isfinite(lengths) & np.isfinite(scores)
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(lengths[mask], scores[mask]).statistic)



# Self-preference bias

@dataclass
class SelfPreferenceReport:
    judge_family: str
    own_family_mean: float
    other_family_mean: float
    gap: float  # positive = judge favors its own family


def self_preference(
    judge_family: str, scores_by_family: dict[str, list[float]]
) -> SelfPreferenceReport:
    """Compare a judge's mean score for its own family vs everyone else."""
    import numpy as np

    own = np.asarray(scores_by_family.get(judge_family, []), dtype=float)
    other = np.concatenate(
        [np.asarray(v, dtype=float) for fam, v in scores_by_family.items() if fam != judge_family]
        or [np.array([])]
    )
    own_mean = float(np.nanmean(own)) if own.size else float("nan")
    other_mean = float(np.nanmean(other)) if other.size else float("nan")
    return SelfPreferenceReport(judge_family, own_mean, other_mean, own_mean - other_mean)
