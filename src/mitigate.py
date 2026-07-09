"""Mitigations. Measure a bias, apply a fix, re-measure, report the delta.

Two mitigations, both from the literature:

  balanced-position calibration (swap-augmented evaluation, Wang et al. 2023):
      run every pairwise comparison in BOTH orders and keep the verdict only if it
      survives the swap; inconsistent pairs collapse to "tie". The surviving
      verdicts are position-invariant *by construction*
      A judge with heavy position bias pays a high tie tax.

  panel-of-judges / jury:
      aggregate multiple judges (mean for pointwise, majority vote for pairwise).
      This attacks self-preference: each judge inflates its own family, but a
      GPT judge's pro-OpenAI tilt and a Claude judge's pro-Anthropic tilt partially
      cancel in the average, shrinking the worst-case family deviation.

Outputs:
  data/raw/<exp>__answer_scores_mitigated.jsonl   (adds a judge_jury_overall column)
  data/raw/mitigation_report.json                 (before/after, rendered by dashboard)

"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .analyze import build_position_bias, load_answer_scores, load_pairwise, _slot_letter
from .config import PipelineConfig, load_config
from .mitigations.debias import balanced_position_verdict, jury_pairwise, jury_pointwise



# Balanced-position calibration

def balance_pairwise(pairwise: list[dict]) -> tuple[dict[tuple, str], dict[str, float]]:
    """Collapse each comparison's two orders into one order-invariant verdict.

    Returns:
      balanced[(example, a, b, judge)] = winner_model | "tie"
      tie_rate_by_judge[judge]         = fraction of complete pairs that became tie
                                         (the cost of buying position-invariance)
    """
    grouped: dict[tuple, dict[str, tuple[str, str]]] = defaultdict(dict)
    for r in pairwise:
        gkey = (r["example_id"], r["model_a"], r["model_b"], r["judge"])
        grouped[gkey][r["order"]] = (r["winner_model"], r["slot_a_model"])

    balanced: dict[tuple, str] = {}
    total = defaultdict(int)
    ties = defaultdict(int)
    for (ex, a, b, judge), orders in grouped.items():
        if "ab" not in orders or "ba" not in orders:
            continue
        ab_letter = _slot_letter(*orders["ab"])
        ba_letter = _slot_letter(*orders["ba"])
        verdict_letter = balanced_position_verdict(ab_letter, ba_letter)  # "A"/"B"/"tie"
        # In the canonical (ab) framing, slot A = model_a.
        winner = {"A": a, "B": b, "tie": "tie"}[verdict_letter]
        balanced[(ex, a, b, judge)] = winner
        total[judge] += 1
        ties[judge] += int(winner == "tie")

    tie_rate = {j: (ties[j] / total[j] if total[j] else float("nan")) for j in total}
    return balanced, tie_rate



# Jury (panel of judges)

def jury_pointwise_scores(answer_scores: list[dict], cfg: PipelineConfig) -> list[dict]:
    """Add a judge_jury_overall column = mean of individual judges' pointwise scores."""
    judge_cols = [f"judge_{j.name}_overall" for j in cfg.judges]
    out = []
    for r in answer_scores:
        vals = [r[c] for c in judge_cols if isinstance(r.get(c), (int, float))]
        row = dict(r)
        row["judge_jury_overall"] = jury_pointwise(vals) if vals else float("nan")
        out.append(row)
    return out


def jury_pairwise_verdicts(
    balanced: dict[tuple, str], cfg: PipelineConfig
) -> dict[tuple, str]:
    """Majority vote across judges over the balanced (position-invariant) verdicts."""
    # regroup balanced verdicts by (example, a, b), collecting one vote per judge
    by_pair: dict[tuple, list[str]] = defaultdict(list)
    for (ex, a, b, _judge), winner in balanced.items():
        by_pair[(ex, a, b)].append(winner)

    jury: dict[tuple, str] = {}
    for (ex, a, b), winners in by_pair.items():
        # map model winners -> slot letters for jury_pairwise, then back
        letters = ["A" if w == a else "B" if w == b else "tie" for w in winners]
        v = jury_pairwise(letters)
        jury[(ex, a, b)] = {"A": a, "B": b, "tie": "tie"}[v]
    return jury



# Self-preference before/after

def _family_means(scores_by_family: dict[str, list[float]]) -> tuple[dict[str, float], float]:
    import numpy as np

    means = {
        fam: float(np.nanmean(v)) for fam, v in scores_by_family.items() if len(v)
    }
    grand = float(np.nanmean([s for v in scores_by_family.values() for s in v])) if means else float("nan")
    return means, grand


def self_preference_delta(mitigated: list[dict], cfg: PipelineConfig) -> dict:
    """Individual judges' own-family gaps vs the jury's worst-case family deviation."""
    import numpy as np

    # Individual judges: own-family mean minus other-family mean.
    individual = {}
    for j in cfg.judges:
        col = f"judge_{j.name}_overall"
        by_fam: dict[str, list[float]] = defaultdict(list)
        for r in mitigated:
            if isinstance(r.get(col), (int, float)):
                by_fam[r["model_family"]].append(float(r[col]))
        own = np.asarray(by_fam.get(j.family, []), dtype=float)
        other = np.asarray([s for f, v in by_fam.items() if f != j.family for s in v], dtype=float)
        if own.size and other.size:
            individual[j.name] = float(np.nanmean(own) - np.nanmean(other))

    worst_individual = max((abs(g) for g in individual.values()), default=float("nan"))

    # Jury: how far does each family sit from the grand mean? Worst-case deviation.
    jury_by_fam: dict[str, list[float]] = defaultdict(list)
    for r in mitigated:
        v = r.get("judge_jury_overall")
        if isinstance(v, (int, float)):
            jury_by_fam[r["model_family"]].append(float(v))
    fam_means, grand = _family_means(jury_by_fam)
    jury_deviations = {f: m - grand for f, m in fam_means.items()}
    worst_jury = max((abs(d) for d in jury_deviations.values()), default=float("nan"))

    reduction = (
        worst_individual - worst_jury
        if worst_individual == worst_individual and worst_jury == worst_jury
        else float("nan")
    )
    return {
        "method": "panel-of-judges (mean pointwise)",
        "individual_own_family_gap": individual,
        "worst_individual_gap": worst_individual,
        "jury_family_deviation": jury_deviations,
        "worst_jury_deviation": worst_jury,
        "worst_case_reduction": reduction,  # positive = jury is fairer across families
    }



# Orchestration

def run(cfg: PipelineConfig) -> None:
    answer_scores = load_answer_scores(cfg)
    pairwise = load_pairwise(cfg)
    raw = Path(cfg.output_dir)

    # --- position bias: before (raw) and the cost of the fix ---
    raw_position = build_position_bias(pairwise)  # reversal rate / consistency per judge
    balanced, tie_rate = balance_pairwise(pairwise)
    position_report = {
        "method": "balanced-position calibration (swap-augmented, Wang et al. 2023)",
        "before": raw_position,  # measured position bias per judge
        "after_tie_rate": tie_rate,  # cost: comparisons discarded to reach invariance
        "note": (
            "After calibration, surviving verdicts are position-invariant by "
            "construction; after_tie_rate is the fraction discarded as inconsistent."
        ),
    }

    # --- self-preference: individual judges vs jury ---
    mitigated = jury_pointwise_scores(answer_scores, cfg)
    jury_pw = jury_pairwise_verdicts(balanced, cfg)
    self_pref_report = self_preference_delta(mitigated, cfg)

    # persist mitigated per-answer scores (adds the jury column) for re-analysis
    mit_path = raw / f"{cfg.experiment_name}__answer_scores_mitigated.jsonl"
    with mit_path.open("w") as f:
        for r in mitigated:
            f.write(json.dumps(r) + "\n")

    report = {
        "position_bias_mitigation": position_report,
        "self_preference_mitigation": self_pref_report,
        "jury_pairwise_n": len(jury_pw),
        "notes": [
            "Verbosity bias is not addressed by these two mitigations; the honest "
            "framing is that it needs a length-controlled rubric or length-matched "
            "pairs, which is a good 'future work' line.",
            "Report the tie-rate cost alongside the invariance gain, not just the gain.",
        ],
    }
    out = raw / "mitigation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"mitigated scores -> {mit_path}")
    print(f"mitigation report -> {out}")
    print("\nWeek 4 mitigations done. Next: run the reproduction harness "
          "(experiments/reproduce_position_bias.py) for the strongest expert signal.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
