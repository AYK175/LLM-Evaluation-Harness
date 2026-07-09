"""Meta-evaluation

Reads the three artifacts from earlier stages and answers the two questions:

  1. Which automated metrics actually track human judgment?
     -> joins every per-answer metric in __answer_scores.jsonl to the human score
        for the same (example, model) and correlates them. Writes agreement_table.csv.

  2. How biased are the LLM judges, as measurement instruments?
     -> position bias  (from __pairwise.jsonl: same pair judged in both orders)
        verbosity bias  (does the judge reward length beyond what humans do?)
        self-preference (does a judge over-rate its own model family?)
        Writes bias_report.json.

"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .analysis import bias as biasmod
from .analysis.correlation import agreement_table
from .config import PipelineConfig, load_config

# Columns in __answer_scores.jsonl that identify a row rather than score it.
_IDENTITY = {"example_id", "model_name", "model_family"}



# Loading

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the earlier stage first.")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_answer_scores(cfg: PipelineConfig) -> list[dict]:
    return _load_jsonl(Path(cfg.output_dir) / f"{cfg.experiment_name}__answer_scores.jsonl")


def load_pairwise(cfg: PipelineConfig) -> list[dict]:
    return _load_jsonl(Path(cfg.output_dir) / f"{cfg.experiment_name}__pairwise.jsonl")


def load_gold(gold_csv: Path, gold_key: Path) -> dict[tuple[str, str], float]:
    """Join the human labels (by row_id) back to (example_id, model_name).

    The key file de-anonymizes the row_id to the original (example_id, model_name) pair.
    Returns {(example_id, model_name): human_score}.
    """
    if not gold_csv.exists():
        raise FileNotFoundError(
            f"No gold labels at {gold_csv}. Build one with `python -m scripts.gold template`, "
            "fill in human_score_1to5, then re-run."
        )
    key = {json.loads(l)["row_id"]: json.loads(l) for l in gold_key.read_text().splitlines() if l.strip()}

    out: dict[tuple[str, str], float] = {}
    with gold_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            val = (row.get("human_score_1to5") or "").strip()
            if not val:
                continue
            rid = int(row["row_id"])
            meta = key[rid]
            out[(meta["example_id"], meta["model_name"])] = float(val)
    return out


# Q1: correlation table

def _numeric_metric_columns(answer_scores: list[dict]) -> list[str]:
    cols: set[str] = set()
    for r in answer_scores:
        for k, v in r.items():
            if k in _IDENTITY:
                continue
            if isinstance(v, (int, float)):  # includes answer_len_tokens as a length baseline
                cols.add(k)
    return sorted(cols)


def build_correlation_table(
    answer_scores: list[dict], gold: dict[tuple[str, str], float], out_path: Path
) -> list[dict]:
    """For every metric column, align to human scores and correlate."""
    metric_cols = _numeric_metric_columns(answer_scores)

    # Build aligned parallel lists over the (example, model) keys that have a gold label.
    scores_by_metric: dict[str, list[float]] = {c: [] for c in metric_cols}
    human: list[float] = []
    matched = 0
    for r in answer_scores:
        k = (r["example_id"], r["model_name"])
        if k not in gold:
            continue
        matched += 1
        human.append(gold[k])
        for c in metric_cols:
            v = r.get(c, float("nan"))
            scores_by_metric[c].append(float(v) if isinstance(v, (int, float)) else float("nan"))

    if matched == 0:
        raise ValueError(
            "No overlap between gold labels and answer_scores. Check that the gold key "
            "file matches this experiment's generations."
        )

    rows = agreement_table(scores_by_metric, human)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "spearman", "kendall_tau", "cohen_kappa", "n"])
        for r in rows:
            w.writerow([r.metric, f"{r.spearman:.4f}", f"{r.kendall_tau:.4f}",
                        f"{r.cohen_kappa:.4f}", r.n])
    print(f"correlation: {matched} labeled answers, {len(metric_cols)} metrics -> {out_path}")
    return [asdict(r) for r in rows]


# Q2: bias report

def _slot_letter(winner_model: str, slot_a_model: str) -> str:
    """Map a normalized winner back to the SLOT it occupied (for position analysis)."""
    if winner_model in ("tie", "error"):
        return "tie"
    return "A" if winner_model == slot_a_model else "B"


def build_position_bias(pairwise: list[dict]) -> dict:
    """Per judge: pair up each comparison's two orders and run the position metrics."""
    # group[(judge, example, a, b)] = {"ab": slot_letter, "ba": slot_letter}
    grouped: dict[tuple, dict[str, str]] = defaultdict(dict)
    for r in pairwise:
        gkey = (r["judge"], r["example_id"], r["model_a"], r["model_b"])
        grouped[gkey][r["order"]] = _slot_letter(r["winner_model"], r["slot_a_model"])

    per_judge_ab: dict[str, list[str]] = defaultdict(list)
    per_judge_ba: dict[str, list[str]] = defaultdict(list)
    for (judge, _ex, _a, _b), orders in grouped.items():
        if "ab" in orders and "ba" in orders:  # only complete pairs
            per_judge_ab[judge].append(orders["ab"])
            per_judge_ba[judge].append(orders["ba"])

    report = {}
    for judge in sorted(per_judge_ab):
        r = biasmod.position_bias(per_judge_ab[judge], per_judge_ba[judge])
        report[judge] = asdict(r)
    return report


def build_verbosity_bias(answer_scores: list[dict], cfg: PipelineConfig,
                         gold: dict[tuple[str, str], float]) -> dict:
    """Per judge: judge-score vs length, compared to human-score vs length.

    The gap between the two is the real verbosity-bias signal: if the judge tracks
    length more strongly than humans do, it's rewarding verbosity beyond merit.
    """
    lengths = [r.get("answer_len_tokens", float("nan")) for r in answer_scores]

    report = {}
    for j in cfg.judges:
        col = f"judge_{j.name}_overall"
        if not any(col in r for r in answer_scores):
            continue
        judge_scores = [r.get(col, float("nan")) for r in answer_scores]
        judge_len_corr = biasmod.verbosity_bias(lengths, judge_scores)

        # Human-length correlation over the labeled subset, for reference.
        h_len, h_score = [], []
        for r in answer_scores:
            k = (r["example_id"], r["model_name"])
            if k in gold:
                h_len.append(r.get("answer_len_tokens", float("nan")))
                h_score.append(gold[k])
        human_len_corr = biasmod.verbosity_bias(h_len, h_score) if h_len else float("nan")

        gap = (judge_len_corr - human_len_corr
               if judge_len_corr == judge_len_corr and human_len_corr == human_len_corr
               else float("nan"))
        report[j.name] = {
            "judge_length_corr": judge_len_corr,
            "human_length_corr": human_len_corr,
            "verbosity_gap": gap,  # >0 => judge rewards length more than humans do
        }
    return report


def build_self_preference(answer_scores: list[dict], cfg: PipelineConfig) -> dict:
    """Per judge: mean pointwise score for own-family vs other-family generators."""
    report = {}
    for j in cfg.judges:
        col = f"judge_{j.name}_overall"
        by_family: dict[str, list[float]] = defaultdict(list)
        for r in answer_scores:
            if col in r and isinstance(r[col], (int, float)):
                by_family[r["model_family"]].append(float(r[col]))
        if not by_family:
            continue
        report[j.name] = asdict(biasmod.self_preference(j.family, by_family))
    return report


def build_bias_report(answer_scores: list[dict], pairwise: list[dict],
                      cfg: PipelineConfig, gold: dict, out_path: Path) -> dict:
    report = {
        "position_bias": build_position_bias(pairwise),
        "verbosity_bias": build_verbosity_bias(answer_scores, cfg, gold),
        "self_preference": build_self_preference(answer_scores, cfg),
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"bias report -> {out_path}")
    return report



# Orchestration

def run(cfg: PipelineConfig, gold_csv: Path, gold_key: Path) -> None:
    answer_scores = load_answer_scores(cfg)
    pairwise = load_pairwise(cfg)
    gold = load_gold(gold_csv, gold_key)
    print(f"Loaded {len(answer_scores)} answer rows, {len(pairwise)} pairwise rows, "
          f"{len(gold)} gold labels")

    raw = Path(cfg.output_dir)
    build_correlation_table(answer_scores, gold, raw / "agreement_table.csv")
    build_bias_report(answer_scores, pairwise, cfg, gold, raw / "bias_report.json")
    print("\nWeek 3 done. `make dashboard` to view; then week 4 applies mitigations.")


def _default_gold_paths(cfg: PipelineConfig) -> tuple[Path, Path]:
    csv_path = Path(cfg.gold_path).with_suffix(".template.csv")
    key_path = Path(str(csv_path).replace(".csv", ".key.jsonl"))
    return csv_path, key_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--gold-csv", default=None, help="completed labeling CSV")
    ap.add_argument("--gold-key", default=None, help="de-anonymization key JSONL")
    args = ap.parse_args()

    cfg = load_config(args.config)
    default_csv, default_key = _default_gold_paths(cfg)
    gold_csv = Path(args.gold_csv) if args.gold_csv else default_csv
    gold_key = Path(args.gold_key) if args.gold_key else (
        Path(str(gold_csv).replace(".csv", ".key.jsonl"))
    )
    run(cfg, gold_csv, gold_key)


if __name__ == "__main__":
    main()
