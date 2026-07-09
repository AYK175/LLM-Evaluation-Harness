# LLM Meta-Evaluation Harness

**Question:** when an LLM grades another LLM's answer, how much should I trust the grade?

I built a pipeline that scores the same long-form QA answers with every evaluation
method I could wire up — ROUGE/METEOR, BERTScore, RAGAS-style faithfulness, and
LLM-as-judge (pointwise + pairwise) — then checked which of those methods actually
agree with human judgment, and how biased the LLM judges are as instruments
(position, verbosity, self-preference). Where a bias showed up, I applied a known
fix and measured whether it actually worked, cost included.

Domain: long-form factual QA over **ASQA** (gold references + a retrieval corpus,
so every metric type applies). Everything is one YAML config
(`config/*.yaml`) — swapping domains doesn't touch the pipeline.

## Results (hybrid run: 3 local models generating, 2 Claude models judging, ~300 human labels)

| What I measured | Result |
|---|---|
| Cheap metrics vs. human judgment | ROUGE 0.28, METEOR 0.27, BERTScore 0.32 Spearman — barely informative |
| LLM judges vs. human judgment | Claude Sonnet 0.86, Claude Haiku 0.76 Spearman — the gap that motivates using judges at all |
| Standalone faithfulness score vs. human judgment | **-0.11** (no relationship) — grounding ≠ overall quality here |
| Verbosity bias | Small (0.03–0.05 gap) and in the *opposite* direction expected — longer answers scored slightly worse, by both judges and me |
| Position bias | Sonnet: 88% order-consistent but skews 69% to slot A. Haiku: 84% consistent, 62% to slot A. Consistency and fairness are different axes |
| Cost of fixing position bias | Swap-and-discard calibration ties out 12–16% of verdicts to buy invariance |
| Self-preference | Unmeasurable in this run — judges (Anthropic) share no family with any generator, by design. Needs `config/default.yaml` |
| Panel-of-judges | Surfaced a real capability gap (Gemma 3 12B +0.54, Qwen-Coder 7B -0.80 vs. mean), not a bias fix, given the above |

Full write-up of *why* each of these numbers looks the way it does is below the fold.
Raw artifacts: `data/raw/agreement_table.csv`, `bias_report.json`, `mitigation_report.json`.

## How it works

```
generate  → retrieve top-k passages, prompt, generate from every configured model
gold      → blind-label a sample by hand (model name hidden) — this is ground truth
evaluate  → score every answer: lexical + semantic + faithfulness + judge (pointwise & pairwise, both slot orders)
analyze   → correlate every metric against human labels; compute position/verbosity/self-preference bias
mitigate  → apply balanced-position calibration + panel-of-judges; report before/after, cost included
reproduce → replay Zheng et al.'s position-consistency metric against an external MT-Bench file, then my own domain
```

Config-driven throughout (`src/config.py`, Pydantic-validated). Judging prompts
use the G-Eval "reason before scoring" pattern; pairwise comparisons always run
in both slot orders so position bias is measurable; faithfulness decomposes
answers into atomic claims and verifies each against retrieved context
(RAGAS-style). A Streamlit dashboard (`dashboard/app.py`) renders all of it.

## Literature this leans on

- **Zheng et al., 2023** (*MT-Bench / Chatbot Arena*, arXiv:2306.05685) — the
  position/verbosity/self-preference bias framing and the reproduction target.
- **Wang et al., 2023** — swap-augmented position-bias calibration (the mitigation).
- **G-Eval** (Liu et al., 2023) — chain-of-thought-then-score judging prompts.
- **RAGAS** (Es et al., 2023) — claim-decomposition faithfulness scoring.
- **BERTScore** (Zhang et al., 2020) — the semantic baseline.

## Honest gaps

- Only one human labeler — no inter-annotator ceiling computed (the tooling for
  it exists in `scripts/gold.py agreement`, just not exercised here).
- Self-preference needs a generator that shares a family with a judge; the
  hybrid config intentionally has none.
- MT-Bench reproduction script is wired but needs an external MT-Bench pairwise
  file I didn't bundle; only the "extend to my domain" half ran here.
- No mitigation for verbosity bias — would need a length-controlled rubric.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in keys only for providers you use

make check       # validate config, free
make generate    # -> data/raw/<exp>__generations.jsonl
make gold        # -> blind labeling template
make evaluate    # -> __answer_scores.jsonl + __pairwise.jsonl
make analyze     # -> agreement_table.csv + bias_report.json
make mitigate    # -> mitigation_report.json
make dashboard   # streamlit UI
make test        # pytest
```

`make` targets use `config/default.yaml` (paid APIs). To reproduce the numbers
above, pass `--config config/hybrid.yaml` to each `python -m src.*` command
directly (needs `ollama serve` + `mistral:latest`, `gemma3:12b`,
`qwen2.5-coder:7b` pulled). `config/local.yaml` runs fully local/free.
`generate`/`evaluate` support `--resume` to skip completed work after a crash.

## Layout

```
config/          default.yaml (paid APIs) · hybrid.yaml (local gen + Claude judges, run above) · local.yaml
src/
  config.py       schema + loader
  generate.py     retrieve → prompt → generate
  evaluate.py     reference metrics + grounding + judging
  analyze.py      correlation table + bias report
  mitigate.py     balanced-position + jury, before/after
  metrics/        lexical.py, semantic.py, judge.py, grounding.py
  analysis/       correlation.py, bias.py
  mitigations/    debias.py
scripts/gold.py                              blind labeling + inter-annotator agreement
experiments/reproduce_position_bias.py       reproduces Zheng et al. (2023)
tests/test_eval_cases.py                     pytest-for-LLMs suite
dashboard/app.py                             Streamlit results dashboard
data/{raw,gold}/                             generations, scores, reports, gold labels
```
