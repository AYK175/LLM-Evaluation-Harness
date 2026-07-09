# LLM Meta-Evaluation Harness

I built this to answer a question I kept running into and never trusted an easy
answer for: **when I use an LLM to grade another LLM's output, how much should I
actually believe the grade?** Everyone quotes ROUGE scores they don't trust and
"LLM-as-judge" numbers they've never audited. I wanted a project that didn't just
*use* evaluation metrics but *evaluated the evaluators* — scored the same set of
answers with every method I could reasonably wire up, checked which of those
methods track what a human actually thinks, and then measured how badly the
LLM judges are biased as measurement instruments before trusting them.

It ended up being three things at once: a small research project (a correlation
study plus a bias audit, with one published finding reproduced), a config-driven
pipeline I can point at a new domain in one YAML edit, and a "pytest for LLMs"
style eval suite a product team would recognize.

## The question, and why I set it up this way

Surface-overlap metrics like ROUGE, BLEU, and METEOR are cheap and everywhere,
but for long-form answers they mostly measure word overlap, not correctness. LLM
judges are the fix everyone reaches for now, but a judge is just another model
with its own failure modes: it can favor whichever answer happens to sit first
in the prompt, it can reward length instead of quality, and it can rate its own
model family more generously than others. None of that is hypothetical — it's
documented in the literature I leaned on (below), and I wanted to see it, and
measure it, on my own data rather than take it on faith.

That meant the project had to do three things, in order: show *why* the cheap
metrics fail (not just report that they do), quantify how much I can trust the
expensive metrics by checking them against real human labels, and reproduce at
least one known result so the numbers are anchored to something outside my own
run.

I picked **long-form factual QA over ASQA** (Ambiguous Sig-QA) as the domain
because it's a good stress test for exactly this: it ships gold reference
answers, so lexical and semantic metrics have something to compare against and
I can show precisely where they break down; it comes with a supporting-knowledge
corpus, so retrieval-grounding and faithfulness are first-class, checkable
dimensions and not an afterthought; and it's small enough that I could hand-label
a few hundred answers myself instead of hand-waving the "ground truth." Nothing
about the pipeline is ASQA-specific — the dataset is one block in
`config/default.yaml`, so swapping domains doesn't touch the harness.

## How the pipeline works

Everything is driven by a single YAML config, validated against a Pydantic
schema in `src/config.py` (`ModelSpec`, `DatasetSpec`, `MetricsSpec`, `BiasSpec`,
...), so an experiment — which models generate, which models judge, which
metrics run, how big the sample is — is fully described by its config file and
nothing is hardcoded in the pipeline itself. I run three variants of it:
`config/default.yaml` (paid APIs: GPT-4o-mini, Claude Haiku, Qwen and Llama via
Together, judged by GPT-4o and Claude Sonnet), `config/local.yaml` (everything
local through Ollama, for a free/offline run), and `config/hybrid.yaml`, which
is the one I actually ran end to end and whose numbers are below: three local
Ollama models as the generators being evaluated (Mistral 7B, Gemma 3 12B,
Qwen2.5-Coder 7B — deliberately no family overlap with the judges) and two
Claude models (Haiku, Sonnet) as judges.

From there the pipeline runs as one linear flow:

**Generate** (`src/generate.py`). For each question, it embeds the query with
`sentence-transformers` and retrieves the top-k passages from ASQA's own
supporting-knowledge corpus by cosine similarity, builds a grounded prompt, and
calls every configured generator model. Provider calls are dispatched through a
thin wrapper (`_generate_once`) with exponential backoff on rate limits, so a
free-tier 429 doesn't kill a run. Output is one JSONL file — the single source
of truth every later stage reads from.

**Build the gold set** (`scripts/gold.py`). This is the part that makes
everything downstream meaningful. It samples generated answers, shuffles them,
and writes a *blind* labeling CSV — the model name is stripped out and held in
a separate de-anonymization key, so I couldn't unconsciously grade "the GPT
answer" more kindly. I scored roughly 300 (question, answer) pairs myself,
1–5, before looking at any automated metric.

**Evaluate** (`src/evaluate.py`). Every generated answer gets scored by the
full stack: ROUGE-L and METEOR against the gold reference (`src/metrics/lexical.py`),
BERTScore F1 for a semantic/embedding-level comparison (`src/metrics/semantic.py`),
a RAGAS-style faithfulness score that decomposes the answer into atomic claims
and checks each one against the retrieved context with an LLM
(`src/metrics/grounding.py`), and pointwise G-Eval-style judging where a judge
model reasons through a rubric (correctness, completeness, grounding, clarity)
before emitting a structured 1–5 verdict (`src/metrics/judge.py`). Separately,
every model-pair comparison is run as a **pairwise** judgment, and — this is the
part that makes the bias analysis possible later — every pair is judged in
*both* orders, (A,B) and (B,A). Both stages write incrementally and are
resumable, because the LLM calls are the expensive part and a crash mid-run
shouldn't cost anything.

**Analyze** (`src/analyze.py`). This is the actual meta-evaluation. For every
automated metric column, it joins that metric's scores to my human labels on
the same (example, model) key and correlates them — Spearman for the headline
rank agreement, Kendall's tau because it's more robust to ties, and a
quadratic-weighted Cohen's kappa on binned scores, because a chance-corrected
agreement number tells a different story than a raw correlation. That produces
`agreement_table.csv`. Separately, it uses the both-orders pairwise data to
compute **position bias** (does the winning *answer* change when I swap which
slot it sits in?), **verbosity bias** (does the judge's score correlate with
answer length more than a human's does?), and **self-preference** (does a judge
score its own model family higher than others?) — all with metric definitions
chosen to be comparable to the papers that introduced them, not invented from
scratch. That produces `bias_report.json`.

**Mitigate** (`src/mitigate.py`). Once a bias is measured, I apply a fix and
re-measure, honestly, including the cost. **Balanced-position calibration**
keeps a pairwise verdict only if it survives being run in both orders;
anything that flips becomes a discarded tie, so the surviving verdicts are
position-invariant *by construction* — the open question is how much gets
thrown away to buy that. A **panel-of-judges / jury** averages the pointwise
scores (or majority-votes the pairwise verdicts) across both judges, on the
theory that if a GPT judge tilts pro-OpenAI and a Claude judge tilts
pro-Anthropic, averaging cancels some of that out. Output is
`mitigation_report.json`, always framed as before/after with the cost stated
next to the gain.

**Reproduce** (`experiments/reproduce_position_bias.py`). The strongest signal
in a project like this is showing a result isn't just an artifact of my own
setup, so I built a small harness around the same `position_consistency` metric
that reproduces Zheng et al.'s MT-Bench finding on an external pairwise-judgment
file and then runs the identical calculation on my own domain data, reporting
the shift between the two. I didn't have a redistribution-clean copy of MT-Bench
judgments to bundle into this repo, so the "reproduce the paper's number" half
needs you to point `--benchmark` at your own MT-Bench-derived pairwise JSONL;
the "extend to my domain" half is real and is the same position-bias numbers
reported below.

There's also a CI-style eval suite (`tests/test_eval_cases.py`) — plain
assertions (no empty generations, every example has answers from every model)
plus a `@pytest.mark.llm`-gated slot for wiring in DeepEval metric thresholds —
and a Streamlit dashboard (`dashboard/app.py`) that renders all four artifacts
(correlation table, bias report, mitigation report, and a per-answer inspector)
as charts instead of raw JSON.

## What the literature told me to build

- **Zheng et al., 2023**, *"Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena"* (arXiv:2306.05685) is the paper that put LLM-as-judge on the map and
  is also the one that documented its failure modes — position bias, verbosity
  bias, and self-enhancement bias — with concrete numbers. My bias metrics
  (`position_consistency`, `preference_fairness`, `reversal_rate`) are framed to
  be directly comparable to that paper's, and `experiments/reproduce_position_bias.py`
  targets its specific reversal-rate finding.
- **Wang et al., 2023** proposed swap-augmented calibration — judge every pair
  in both orders and only trust verdicts that agree — as the fix for position
  bias. That's exactly what `balance_pairwise` in `src/mitigations/debias.py`
  implements.
- **G-Eval** (Liu et al., 2023) is where the pointwise judging prompt design
  comes from: chain-of-thought reasoning over an explicit rubric before the
  model commits to a score, rather than asking for a bare number.
- **RAGAS** (Es et al., 2023) is the source for the faithfulness metric's
  design: decompose an answer into atomic, independently-checkable claims, then
  verify each one against the retrieved context, rather than asking a model to
  eyeball "is this grounded?" as one holistic judgment.
- **BERTScore** (Zhang et al., 2020) is the semantic baseline — embedding-level
  similarity to the reference, a step up from lexical overlap but still
  reference-bound, which is exactly the property I wanted to test against human
  judgment alongside the cruder metrics.

## What I found

All the numbers below are from the hybrid run — three local 7–12B models
generating, two Claude models judging, ~300 human-labeled (question, answer)
pairs as ground truth. The raw artifacts are in `data/raw/agreement_table.csv`,
`bias_report.json`, and `mitigation_report.json`.

**The cheap metrics really are close to uninformative.** Against my human
labels, ROUGE-L reached a Spearman correlation of 0.28, METEOR 0.27, and
BERTScore — supposedly the "semantic" upgrade — only 0.32. Compare that to the
LLM judges' overall-score correlation: Claude Sonnet at 0.86 and Claude Haiku
at 0.76. That's not a small gap; it's the difference between a metric I'd
actually use to rank models and one I wouldn't. This is the empirical version
of the claim the whole project opens with, on my own data rather than someone
else's benchmark.

**The standalone faithfulness score didn't track human judgment at all** — its
correlation with my labels came out slightly *negative* (-0.11). The judges'
own internal "grounding" rubric dimension, by contrast, correlated at 0.57–0.74.
My read is that the two are measuring different things: the RAGAS-style
decompose-and-verify faithfulness score is answering "are these specific claims
supported by the retrieved passages," while my 1–5 holistic score was answering
"is this a good answer" — and on 7–12B local models, the bottleneck on answer
quality was more often completeness or coherence than hallucination. That's a
genuinely useful negative result: faithfulness is a real, checkable property,
but it's not a proxy for overall answer quality, and treating it as one would
have been a mistake.

**Neither judges nor I rewarded length in this run** — if anything, the
opposite. Human-score-vs-length correlation was -0.13, and both judges tracked
length almost identically (-0.08 for Haiku, -0.11 for Sonnet), giving small
verbosity gaps of 0.05 and 0.03. The classic "judges love long answers" failure
mode from the literature didn't show up here, most likely because these
particular local models' longer answers tended to ramble rather than add
substance, and both the judges and I penalized that. I'd treat this as a
domain- and model-scale-dependent finding, not a universal one — the literature
that reports strong verbosity bias mostly does so on stronger, chattier
frontier models.

**Position bias is real, and consistency and fairness are not the same axis.**
Claude Sonnet was the more internally consistent judge — 88% of its pairwise
verdicts survived an order swap unchanged, versus 84% for Haiku, and Sonnet's
raw reversal rate (7.7%) was about half of Haiku's (15.3%). But Sonnet's
consistency was also more *skewed*: when it had a stable opinion, it favored
whatever sat in slot A 69% of the time, against 62% for Haiku, which is why
Sonnet's `preference_fairness` score (0.63) is actually worse than Haiku's
(0.76). A judge can be more self-consistent and more order-biased at the same
time — those are two different failure modes and I'd have missed that by only
reporting one number.

**Fixing position bias has a real, quantifiable cost.** Balanced-position
calibration makes the surviving verdicts position-invariant by construction,
but "surviving" is the operative word: 16% of Haiku's pairwise judgments and
12% of Sonnet's had to be discarded as ties because they didn't agree with
themselves across the swap. Any writeup of "we fixed position bias" that
doesn't mention the tie-rate tax is hiding half the result.

**Self-preference came back unmeasurable, for an honest and slightly
embarrassing reason.** I set the hybrid config up so the judges (both
Anthropic-family) share no model family with any generator (Mistral, Gemma,
Qwen) — which was intentional, to keep the *generation* comparison clean of
judge-family favoritism, but it means there's no "own family" data point for
the self-preference metric to compare against, so `own_family_mean` comes back
as `NaN` in both `bias_report.json` and `mitigation_report.json`. Measuring
self-preference properly needs at least one generator sharing a family with a
judge — the `default.yaml` config (GPT-4o judging GPT-4o-mini's answers among
others) is built for exactly that and I'd run that config next to close this
gap.

**The jury (panel-of-judges) surfaced a real capability spread instead of a
bias signal, given the setup above.** Averaging the two Claude judges' scores
per generator family, Qwen2.5-Coder-7B sat 0.80 points below the grand mean (on
the 1–5 scale), Gemma 3 12B sat 0.54 above it, and Mistral 7B was close to
neutral (+0.26). That reads less like "the jury cancelled out bias" and more
like "a 12B general model outperforms two 7B models, and a code-tuned 7B
checkpoint underperforms on long-form prose QA" — which is a believable,
mundane, and probably correct result. Because there was no valid
individual-judge self-preference baseline to compare against (the point above),
`worst_case_reduction` also comes back `NaN` — I report that as a gap rather
than papering over it with a made-up number.

## Honest limitations

- **Single annotator.** `scripts/gold.py` has an `agreement` subcommand built
  specifically to compute inter-annotator Spearman/kappa between two labelers
  as a reliability ceiling for everything else, but I only had one labeler
  (myself) label the hybrid gold set, so that ceiling is unmeasured here. Every
  correlation number above should be read as "against one person's judgment,"
  not "against ground truth."
- **Self-preference needs a same-family generator/judge pair**, which the
  hybrid run deliberately didn't have (see above).
- **The MT-Bench reproduction is wired but not executed end-to-end in this
  repo** — the domain-side position-consistency numbers are real and reported
  above; the paper-comparison half needs an MT-Bench pairwise file supplied at
  run time.
- **Verbosity bias has no mitigation implemented.** Position bias and
  self-preference each have one; verbosity doesn't, and the honest fix would be
  a length-controlled rubric or length-matched pairs, which I've left as future
  work rather than a half-implemented mitigation.

## How to run it

```bash
git clone <this-repo> && cd llm-eval-harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in only the providers you're actually using
```

`.env` needs API keys only for the providers a given config actually calls —
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`. Nothing is required
for `provider: local` models, which run through Ollama's OpenAI-compatible
local endpoint instead — start it with `ollama serve` and pull whatever models
a config references (for `config/hybrid.yaml`: `mistral:latest`, `gemma3:12b`,
`qwen2.5-coder:7b`).

Validate a config before spending a token, then run the pipeline stages in
order:

```bash
make check                  # validates the YAML against the Pydantic schema, free
make generate                # -> data/raw/<experiment>__generations.jsonl
make gold                    # -> a blind labeling template (you fill in scores by hand)
make evaluate                 # -> __answer_scores.jsonl + __pairwise.jsonl
make analyze                  # -> agreement_table.csv + bias_report.json
make mitigate                 # -> mitigation_report.json (before/after, with costs)
make dashboard                # streamlit run dashboard/app.py — view all of it
```

Each `make` target hardcodes `config/default.yaml`; to reproduce the hybrid run
whose numbers are quoted above, pass `config/hybrid.yaml` (or `config/local.yaml`
for a fully local/free run) to each stage directly, e.g.:

```bash
python -m src.generate  --config config/hybrid.yaml
python -m scripts.gold   template --config config/hybrid.yaml --n 60
python -m src.evaluate  --config config/hybrid.yaml --pairwise-examples 25
python -m src.analyze   --config config/hybrid.yaml
python -m src.mitigate  --config config/hybrid.yaml
```

`evaluate` and `generate` are resumable — pass `--resume` to skip
already-completed keys if a run gets interrupted; the LLM calls are what cost
money and time, so nothing gets re-done unnecessarily.

To run the MT-Bench reproduction, point it at an MT-Bench-derived pairwise
JSONL of your own (schema: `example_id, model_a, model_b, judge, order,
slot_a_model, winner_model`) and, optionally, your domain's pairwise file to
run the extend step in the same pass:

```bash
python -m experiments.reproduce_position_bias \
  --benchmark <your_mtbench_pairwise>.jsonl \
  --domain data/raw/asqa_hybrid_v1__pairwise.jsonl \
  --target 0.65
```

Finally, run the eval suite:

```bash
make test                    # pytest tests/ -v
```

## Repository layout

```
llm-eval-harness/
├── config/                    # default.yaml (paid APIs), hybrid.yaml (local gen + Claude judges,
│                               #   the run reported above), local.yaml (fully local via Ollama)
├── src/
│   ├── config.py               # Pydantic schema + loader
│   ├── generate.py             # retrieve -> prompt -> generate -> JSONL
│   ├── evaluate.py             # reference metrics + grounding + pointwise/pairwise judging
│   ├── analyze.py              # correlation table + bias report (the meta-evaluation)
│   ├── mitigate.py             # balanced-position + jury, with before/after deltas
│   ├── metrics/                # lexical.py, semantic.py, judge.py, grounding.py
│   ├── analysis/                # correlation.py, bias.py
│   └── mitigations/             # debias.py
├── scripts/gold.py              # blind labeling template + inter-annotator agreement
├── experiments/reproduce_position_bias.py   # reproduces Zheng et al. (2023), extends to my domain
├── tests/test_eval_cases.py     # pytest-for-LLMs eval suite (plain assertions + DeepEval gate)
├── dashboard/app.py              # Streamlit results dashboard
└── data/{raw,gold}/              # generations, per-answer scores, bias/mitigation reports, gold labels
```
