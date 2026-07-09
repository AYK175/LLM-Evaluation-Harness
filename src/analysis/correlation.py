""" Meta-evaluation — how well does each automated metric track humans?

This is the heart of the project. I have, per answer, a bunch of automated
scores (rouge, bertscore, judge_overall, faithfulness, ...) and one human score
from the gold set. For each metric, correlate its scores against the human
scores across all answers. The metric with the highest correlation is the one
I'd actually trust and usually it is NOT the lexical ones.

Report three coefficients because they answer different questions:
  - Spearman: monotonic relationship (rank agreement) — the headline number.
  - Kendall's tau: rank agreement robust to ties/outliers.
  - Cohen's kappa: agreement on discretized buckets (chance-corrected).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricAgreement:
    metric: str
    spearman: float
    kendall_tau: float
    cohen_kappa: float
    n: int


def _bucketize(values: list[float], n_bins: int = 3) -> list[int]:
    """Discretize continuous scores into ordinal buckets for kappa."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0] * len(values)
    edges = np.quantile(finite, [i / n_bins for i in range(1, n_bins)])
    return [int(np.digitize(v, edges)) if np.isfinite(v) else 0 for v in values]


def agreement(metric_scores: list[float], human_scores: list[float], metric: str) -> MetricAgreement:
    """Correlate one metric's scores against human scores (drops NaN pairs)."""
    import numpy as np
    from scipy.stats import kendalltau, spearmanr
    from sklearn.metrics import cohen_kappa_score

    m = np.asarray(metric_scores, dtype=float)
    h = np.asarray(human_scores, dtype=float)
    mask = np.isfinite(m) & np.isfinite(h)
    m, h = m[mask], h[mask]

    if m.size < 3:
        nan = float("nan")
        return MetricAgreement(metric, nan, nan, nan, int(m.size))

    rho = spearmanr(m, h).statistic
    tau = kendalltau(m, h).statistic
    kappa = cohen_kappa_score(_bucketize(m.tolist()), _bucketize(h.tolist()), weights="quadratic")
    return MetricAgreement(metric, float(rho), float(tau), float(kappa), int(m.size))


def agreement_table(
    scores_by_metric: dict[str, list[float]], human_scores: list[float]
) -> list[MetricAgreement]:
    """Build the headline results table, sorted best-correlating metric first."""
    rows = [agreement(v, human_scores, name) for name, v in scores_by_metric.items()]
    return sorted(rows, key=lambda r: (r.spearman if r.spearman == r.spearman else -1), reverse=True)
