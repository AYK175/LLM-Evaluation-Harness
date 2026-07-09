"""
TARGET FINDING
    Zheng et al., 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
    (arXiv:2306.05685) introduced LLM-as-a-judge and documented position bias:
    a judge's pairwise verdict can flip when the two answers are swapped. Follow-up
    reporting quantifies the effect at roughly 10-15 points of win-rate swing by
    slot order, and notes GPT-4 favored the first response in a large share of the
    comparisons where its verdict changed on swap. Wang et al. (2023) proposed
    swap-augmented calibration as the fix (implemented in src/mitigations/debias.py).

METRIC
    position_consistency = fraction of comparisons whose winning *answer* is
    unchanged when the presentation order is swapped. A perfectly unbiased judge
    scores 1.0; the paper's finding is that real judges score meaningfully below 1.0.

The schema each pairwise JSONL row must have:
    {"example_id","model_a","model_b","judge","order","slot_a_model","winner_model"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analyze import build_position_bias


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing pairwise file: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def mean_consistency(pairwise: list[dict]) -> tuple[float, int]:
    """Average position_consistency across judges, weighted by pair count."""
    report = build_position_bias(pairwise)
    total_pairs = sum(v["n_pairs"] for v in report.values())
    if total_pairs == 0:
        return float("nan"), 0
    weighted = sum(v["position_consistency"] * v["n_pairs"] for v in report.values())
    return weighted / total_pairs, total_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, help="MT-Bench-derived pairwise JSONL")
    ap.add_argument("--domain", help="your own domain pairwise JSONL (the 'extend' step)")
    ap.add_argument(
        "--target",
        type=float,
        default=None,
        help="position_consistency reported by the paper you are reproducing",
    )
    args = ap.parse_args()

    bench_c, bench_n = mean_consistency(_load(Path(args.benchmark)))
    print("=" * 60)
    print("REPRODUCE — position consistency on the benchmark")
    print(f"  measured: {bench_c:.3f}  (over {bench_n} pairs)")
    if args.target is not None:
        print(f"  paper:    {args.target:.3f}")
        print(f"  gap:      {bench_c - args.target:+.3f}")
        verdict = "reproduced" if abs(bench_c - args.target) <= 0.05 else "DIVERGES — investigate"
        print(f"  -> {verdict} (within 0.05 tolerance)")
    else:
        print("  (pass --target <paper_value> to score the reproduction)")

    if args.domain:
        dom_c, dom_n = mean_consistency(_load(Path(args.domain)))
        print("\n" + "=" * 60)
        print("EXTEND — does the effect transfer to your domain?")
        print(f"  benchmark consistency: {bench_c:.3f}")
        print(f"  domain consistency:    {dom_c:.3f}  (over {dom_n} pairs)")
        print(f"  shift:                 {dom_c - bench_c:+.3f}")
        print(
            "  -> Lower domain consistency means position bias is WORSE on long-form "
            "answers than on MT-Bench; higher means your task is more robust."
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
