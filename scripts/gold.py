"""
The human gold set is the ground truth every automated metric is measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def _load_generations(exp: str, raw_dir: str) -> list[dict]:
    path = Path(raw_dir) / f"{exp}__generations.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_template(exp: str, raw_dir: str, gold_path: str, n: int, seed: int) -> Path:
    """Emit a blind labeling CSV: one row per (example, answer), model hidden."""
    records = _load_generations(exp, raw_dir)
    rng = random.Random(seed)
    rng.shuffle(records)
    rows = records[:n]

    out = Path(gold_path).with_suffix(".template.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    # A hidden key maps the anonymized row back to (example_id, model) after labeling.
    key_path = out.with_suffix(".key.jsonl")
    with out.open("w", newline="") as f, key_path.open("w") as kf:
        w = csv.writer(f)
        w.writerow(["row_id", "question", "answer", "human_score_1to5", "notes"])
        for i, r in enumerate(rows):
            w.writerow([i, r["question"], r["answer"], "", ""])
            kf.write(
                json.dumps(
                    {"row_id": i, "example_id": r["example_id"], "model_name": r["model_name"]}
                )
                + "\n"
            )

    print(f"Wrote blind template: {out}  ({len(rows)} rows)")
    print(f"Wrote de-anonymization key: {key_path}")
    print("Fill in human_score_1to5 (1=poor, 5=excellent), then join on row_id via the key.")
    return out


def _read_scores(path: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            val = row.get("human_score_1to5", "").strip()
            if val:
                scores[int(row["row_id"])] = float(val)
    return scores


def report_agreement(path_a: str, path_b: str) -> None:
    """Inter-annotator agreement on the rows both annotators scored."""
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    a, b = _read_scores(path_a), _read_scores(path_b)
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        print(f"Only {len(shared)} shared rows — label more overlap before trusting this.")
        return

    xa = [a[i] for i in shared]
    xb = [b[i] for i in shared]
    rho = spearmanr(xa, xb).statistic
    kappa = cohen_kappa_score(
        [int(round(x)) for x in xa], [int(round(x)) for x in xb], weights="quadratic"
    )
    print(f"Shared items: {len(shared)}")
    print(f"Spearman:            {rho:.3f}")
    print(f"Quadratic-weighted kappa: {kappa:.3f}")
    print(f"Mean abs difference: {np.mean(np.abs(np.array(xa) - np.array(xb))):.2f} points")
    print("\nThis is the ceiling: no automated metric should be expected to beat it.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="build a blind labeling CSV")
    t.add_argument("--config", default="config/default.yaml")
    t.add_argument("--n", type=int, default=60)

    g = sub.add_parser("agreement", help="inter-annotator agreement between two label files")
    g.add_argument("file_a")
    g.add_argument("file_b")

    args = ap.parse_args()
    if args.cmd == "template":
        from src.config import load_config

        cfg = load_config(args.config)
        build_template(cfg.experiment_name, cfg.output_dir, cfg.gold_path, args.n, cfg.seed)
    elif args.cmd == "agreement":
        report_agreement(args.file_a, args.file_b)


if __name__ == "__main__":
    main()
