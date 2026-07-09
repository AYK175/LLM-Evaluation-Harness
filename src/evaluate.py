"""Evaluation stage.

Reads the generations produced by `src.generate` and scores every answer with the
full stack, writing two tidy artifacts that analysis consumes directly:

  data/raw/<exp>__answer_scores.jsonl
      one row per (example, model): reference metrics (rouge/meteor/bertscore),
      grounding (faithfulness), and each judge's pointwise overall + dimensions.

  data/raw/<exp>__pairwise.jsonl
      one row per (example, model_a, model_b, judge, order): the winner. Every
      comparison is run in BOTH orders on purpose — that is the raw material for
      the position-bias analysis.

Design notes:
  - LLM calls are the expensive part, so both files are written incrementally and
    the runner is resumable (--resume skips keys already on disk). A crash mid-run
    costs nothing.
  - Reference metrics are local and cheap, so they're computed in one batched pass
    up front (BERTScore especially wants batching) and merged into each row.
  - Pairwise is O(models^2) per example, so it's capped by --pairwise-examples to
    keep the token bill sane; scoring is independent of that cap.

"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

from .config import PipelineConfig, load_config
from .metrics import grounding, lexical, semantic
from .metrics.judge import judge_pairwise, judge_pointwise


# IO helpers

def _load_generations(cfg: PipelineConfig) -> list[dict]:
    path = Path(cfg.output_dir) / f"{cfg.experiment_name}__generations.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No generations at {path}. Run `python -m src.generate` first."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _existing_keys(path: Path, key_fields: tuple[str, ...]) -> set[tuple]:
    """Read a JSONL output file and return the set of already-completed keys."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add(tuple(row[f] for f in key_fields))
    return done


def _append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


# Reference metrics (cheap, local, batched)

def compute_reference_metrics(
    records: list[dict], cfg: PipelineConfig
) -> dict[tuple[str, str], dict[str, float]]:
    """Return {(example_id, model_name): {metric: score}} for lexical+semantic.

    Skips any record without a gold reference (metrics are reference-based).
    """
    out: dict[tuple[str, str], dict[str, float]] = {}

    # Lexical: per-record, keyed by (example_id, model_name)
    for r in records:
        key = (r["example_id"], r["model_name"])
        gold = r.get("gold_answer") or ""
        out[key] = lexical.score(r["answer"], gold, cfg.metrics.lexical)

    # Semantic: batched across all records that have a gold reference
    if cfg.metrics.semantic:
        idx = [i for i, r in enumerate(records) if r.get("gold_answer")]
        preds = [records[i]["answer"] for i in idx]
        refs = [records[i]["gold_answer"] for i in idx]
        if preds:
            sem = semantic.score_batch(preds, refs, cfg.metrics.semantic)
            for name, values in sem.items():
                for pos, i in enumerate(idx):
                    key = (records[i]["example_id"], records[i]["model_name"])
                    out[key][name] = values[pos]
    return out



# Stage A: per-answer scores (grounding + pointwise judging)

def run_answer_scores(cfg: PipelineConfig, records: list[dict], resume: bool) -> Path:
    out_path = Path(cfg.output_dir) / f"{cfg.experiment_name}__answer_scores.jsonl"
    key_fields = ("example_id", "model_name")
    done = _existing_keys(out_path, key_fields) if resume else set()
    if not resume and out_path.exists():
        out_path.unlink()

    ref = compute_reference_metrics(records, cfg)
    judges = {j.name: j for j in cfg.judges}

    # Grounding uses the first judge as its scorer (documented convention).
    grounding_model = cfg.judges[0] if (cfg.metrics.grounding and cfg.judges) else None

    n_written = 0
    for r in records:
        key = (r["example_id"], r["model_name"])
        if key in done:
            continue

        row: dict = {
            "example_id": r["example_id"],
            "model_name": r["model_name"],
            "model_family": r["model_family"],
            "answer_len_tokens": len(r["answer"].split()),
        }
        row.update(ref.get(key, {}))  # rouge / meteor / bertscore

        # Grounding / faithfulness
        if grounding_model is not None:
            try:
                row["faithfulness"] = grounding.faithfulness(
                    grounding_model, r["answer"], r.get("contexts", [])
                )
            except Exception as e:
                print(f"  [warn] faithfulness failed {key}: {e}")
                row["faithfulness"] = float("nan")

        # Pointwise judging: one overall + dims per judge
        for jname in cfg.metrics.judges:
            judge = judges[jname]
            try:
                v = judge_pointwise(
                    judge, r["question"], r["answer"], "\n\n".join(r.get("contexts", []))
                )
                row[f"judge_{jname}_overall"] = v.overall
                row[f"judge_{jname}_correctness"] = v.correctness
                row[f"judge_{jname}_grounding"] = v.grounding
            except Exception as e:
                print(f"  [warn] pointwise {jname} failed {key}: {e}")
                row[f"judge_{jname}_overall"] = float("nan")

        _append(out_path, row)
        n_written += 1

    print(f"answer_scores: wrote {n_written} rows ({len(done)} skipped) -> {out_path}")
    return out_path



# Stage B: pairwise judging (both orders) for position-bias analysis

def run_pairwise(
    cfg: PipelineConfig, records: list[dict], resume: bool, pairwise_examples: int
) -> Path:
    out_path = Path(cfg.output_dir) / f"{cfg.experiment_name}__pairwise.jsonl"
    key_fields = ("example_id", "model_a", "model_b", "judge", "order")
    done = _existing_keys(out_path, key_fields) if resume else set()
    if not resume and out_path.exists():
        out_path.unlink()

    judges = {j.name: j for j in cfg.judges}

    # Group answers by example so we can pair models within an example.
    by_example: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_example[r["example_id"]][r["model_name"]] = r

    example_ids = sorted(by_example)[:pairwise_examples]
    model_names = [m.name for m in cfg.generators]

    n_written = 0
    for ex_id in example_ids:
        answers = by_example[ex_id]
        context = "\n\n".join(next(iter(answers.values())).get("contexts", []))
        question = next(iter(answers.values()))["question"]

        for a, b in itertools.combinations(model_names, 2):
            if a not in answers or b not in answers:
                continue
            for jname in cfg.metrics.judges:
                judge = judges[jname]
                # Two orders: (a,b) and (b,a). "order" records which model sat in slot A.
                for order, (slot_a, slot_b) in (("ab", (a, b)), ("ba", (b, a))):
                    key = (ex_id, a, b, jname, order)
                    if key in done:
                        continue
                    try:
                        v = judge_pairwise(
                            judge,
                            question,
                            answers[slot_a]["answer"],
                            answers[slot_b]["answer"],
                            context,
                        )
                        # Normalize winner to a MODEL name (not a slot letter).
                        if v.winner == "A":
                            winner_model = slot_a
                        elif v.winner == "B":
                            winner_model = slot_b
                        else:
                            winner_model = "tie"
                    except Exception as e:
                        print(f"  [warn] pairwise {jname} failed {key}: {e}")
                        winner_model = "error"

                    _append(
                        out_path,
                        {
                            "example_id": ex_id,
                            "model_a": a,  # canonical pair ordering (a<b by list order)
                            "model_b": b,
                            "judge": jname,
                            "order": order,  # "ab" = a in slot A; "ba" = b in slot A
                            "slot_a_model": slot_a,
                            "winner_model": winner_model,
                        },
                    )
                    n_written += 1

    print(f"pairwise: wrote {n_written} rows ({len(done)} skipped) -> {out_path}")
    return out_path



# Orchestration

def run(cfg: PipelineConfig, resume: bool, pairwise_examples: int) -> None:
    records = _load_generations(cfg)
    print(f"Loaded {len(records)} generation records")
    run_answer_scores(cfg, records, resume)
    run_pairwise(cfg, records, resume, pairwise_examples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--resume", action="store_true", help="skip keys already on disk")
    ap.add_argument(
        "--pairwise-examples",
        type=int,
        default=25,
        help="cap pairwise judging to the first N examples (token budget control)",
    )
    args = ap.parse_args()
    run(load_config(args.config), args.resume, args.pairwise_examples)


if __name__ == "__main__":
    main()
