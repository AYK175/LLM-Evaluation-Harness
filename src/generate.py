"""Generation pipeline.

For each example in the dataset:
  1. (optional) retrieve top-k passages from the corpus
  2. build a grounded prompt
  3. generate an answer from every configured generator model
  4. write one JSONL record per (example, model) to output_dir

The output of this stage is the single source of truth that every later stage
(metrics, judging, bias analysis) reads from. 

"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ModelSpec, PipelineConfig, load_config


@dataclass
class GenerationRecord:
    """One generated answer, plus everything needed to evaluate it later."""

    example_id: str
    question: str
    gold_answer: str | None
    contexts: list[str]  # retrieved passages used for grounding
    model_name: str
    model_family: str
    answer: str


# Dataset loading

def load_examples(cfg: PipelineConfig) -> list[dict]:
    """Load n examples as dicts with keys: id, question, gold_answer, corpus.

    Implemented for ASQA via HuggingFace `datasets`. Add elif branches for other
    corpora; downstream code only depends on this normalized shape.
    """
    if cfg.dataset.name == "asqa":
        from datasets import load_dataset

        ds = load_dataset(cfg.dataset.hf_path, split=cfg.dataset.split)
        ds = ds.select(range(min(cfg.dataset.n_examples, len(ds))))
        examples = []
        for i, row in enumerate(ds):
            # ASQA: 'ambiguous_question' + 'annotations' (long-form answers)
            gold = None
            if row.get("annotations"):
                gold = row["annotations"][0].get("long_answer")
            examples.append(
                {
                    "id": f"asqa-{i}",
                    "question": row["ambiguous_question"],
                    "gold_answer": gold,
                    # ASQA ships supporting knowledge; treat as the retrieval corpus
                    "corpus": [
                        k["content"] for k in row.get("knowledge", []) if k.get("content")
                    ],
                }
            )
        return examples

    raise NotImplementedError(
        f"Dataset '{cfg.dataset.name}' not wired up yet. "
        "Add a branch in load_examples() returning id/question/gold_answer/corpus."
    )



# Retrieval

def retrieve(query: str, corpus: list[str], cfg: PipelineConfig) -> list[str]:
    """Return top-k passages by embedding cosine similarity.

    Uses sentence-transformers locally so it costs nothing and is deterministic.
    If retrieval is disabled or the corpus is empty, returns [].
    """
    if not cfg.retrieval.enabled or not corpus:
        return []

    from sentence_transformers import SentenceTransformer, util

    # Cache the model on the function to avoid reloading per call.
    model = getattr(retrieve, "_model", None)
    if model is None:
        model = SentenceTransformer(cfg.retrieval.embedding_model)
        retrieve._model = model  # type: ignore[attr-defined]

    q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    c_emb = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, c_emb)[0]
    top = scores.topk(min(cfg.retrieval.top_k, len(corpus)))
    return [corpus[idx] for idx in top.indices.tolist()]



# Prompting + generation

GEN_TEMPLATE = """Answer the question using the provided context. Write a \
complete, well-organized long-form answer. If the context is insufficient, \
answer from your own knowledge but do not fabricate specifics.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) or "(none)"
    return GEN_TEMPLATE.format(context=context_block, question=question)


def _is_rate_limit(err: Exception) -> bool:
    """Heuristic: does this exception look like a 429 / rate-limit / quota error?"""
    s = str(err).lower()
    return any(k in s for k in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted", "too many requests"))


def generate_answer(prompt: str, model: ModelSpec, max_retries: int = 6) -> str:
    """Provider dispatch with automatic backoff on rate-limit errors.

    Free tiers (e.g. Gemini's ~15 req/min) will 429 during a run; instead of
    failing, we wait and retry with exponential backoff + jitter. Non-rate-limit
    errors are raised immediately so real bugs surface fast.
    """
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return _generate_once(prompt, model)
        except Exception as e:
            if _is_rate_limit(e) and attempt < max_retries - 1:
                wait = delay + random.uniform(0, 1)
                print(f"    [rate-limit] {model.name}: waiting {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)
                delay = min(delay * 2, 60)  # cap the backoff
                continue
            raise
    raise RuntimeError(f"{model.name}: exhausted retries")


def _generate_once(prompt: str, model: ModelSpec) -> str:
    """Single provider call, no retry. Thin wrappers keep providers swappable.

    Reads API keys from the environment:
        OPENAI_API_KEY, ANTHROPIC_API_KEY
    """
    if model.provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=model.temperature,
            max_tokens=model.max_tokens,
        )
        return resp.choices[0].message.content or ""

    if model.provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model.model_id,
            max_tokens=model.max_tokens,
            temperature=model.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


    if model.provider == "local":
        # Ollama exposes an OpenAI-compatible endpoint
    
        from openai import OpenAI

        client = OpenAI(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # required by the SDK, ignored by Ollama
        )
        resp = client.chat.completions.create(
            model=model.model_id,  # an Ollama tag, e.g. "mistral:latest"
            messages=[{"role": "user", "content": prompt}],
            temperature=model.temperature,
            max_tokens=model.max_tokens,
        )
        return resp.choices[0].message.content or ""

    raise NotImplementedError(f"Provider '{model.provider}' not implemented.")



# Orchestration

def run(cfg: PipelineConfig) -> Path:
    random.seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.experiment_name}__generations.jsonl"

    examples = load_examples(cfg)
    print(f"Loaded {len(examples)} examples from {cfg.dataset.name}")

    n_written = 0
    with out_path.open("w") as f:
        for ex in examples:
            contexts = retrieve(ex["question"], ex["corpus"], cfg)
            prompt = build_prompt(ex["question"], contexts)
            for model in cfg.generators:
                try:
                    answer = generate_answer(prompt, model)
                except Exception as e:  # keep going; log the failure
                    print(f"  [warn] {model.name} failed on {ex['id']}: {e}")
                    continue
                rec = GenerationRecord(
                    example_id=ex["id"],
                    question=ex["question"],
                    gold_answer=ex["gold_answer"],
                    contexts=contexts,
                    model_name=model.name,
                    model_family=model.family,
                    answer=answer,
                )
                f.write(json.dumps(asdict(rec)) + "\n")
                n_written += 1
            print(f"  {ex['id']}: {len(cfg.generators)} answers")

    print(f"\nWrote {n_written} generation records -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
