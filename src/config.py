"""Config schema for the evaluation harness.

Everything the pipeline does is driven by a single YAML file validated against
these Pydantic models. This means an experiment is fully described by its config
and swapping the domain/corpus is a one-line change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelSpec(BaseModel):
    """A single model under evaluation (a generator) or acting as a judge."""

    name: str  # human-readable label used in results tables
    provider: Literal["openai", "anthropic", "together", "gemini", "local"] = "openai"
    model_id: str  # provider-specific id, e.g. "gpt-4o-mini"
    family: str  # "openai" / "anthropic" / "qwen" ... used for self-preference analysis
    temperature: float = 0.0
    max_tokens: int = 1024


class DatasetSpec(BaseModel):
    """The corpus. Swap this block to change domains; the harness is unchanged."""

    name: str  # "asqa" | "eli5" | "factscore_bios" | custom
    split: str = "validation"
    n_examples: int = 100
    hf_path: str | None = None  # HuggingFace dataset path, if applicable
    has_gold_reference: bool = True
    has_retrieval_corpus: bool = True


class RetrievalSpec(BaseModel):
    enabled: bool = True
    top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class MetricsSpec(BaseModel):
    lexical: list[Literal["bleu", "rouge", "meteor"]] = Field(
        default_factory=lambda: ["rouge", "meteor"]
    )
    semantic: list[Literal["bertscore", "bleurt"]] = Field(
        default_factory=lambda: ["bertscore"]
    )
    judges: list[str] = Field(default_factory=list)  # names must exist in `judges`
    grounding: list[Literal["ragas_faithfulness", "ragas_relevance"]] = Field(
        default_factory=lambda: ["ragas_faithfulness"]
    )


class BiasSpec(BaseModel):
    position: bool = True  # run each pairwise comparison in both orders
    verbosity: bool = True
    self_preference: bool = True


class PipelineConfig(BaseModel):
    experiment_name: str
    seed: int = 13
    dataset: DatasetSpec
    retrieval: RetrievalSpec = RetrievalSpec()
    generators: list[ModelSpec]
    judges: list[ModelSpec] = Field(default_factory=list)
    metrics: MetricsSpec = MetricsSpec()
    bias: BiasSpec = BiasSpec()
    output_dir: str = "data/raw"
    gold_path: str = "data/gold/gold.jsonl"

    @field_validator("generators")
    @classmethod
    def _need_at_least_one_generator(cls, v: list[ModelSpec]) -> list[ModelSpec]:
        if not v:
            raise ValueError("Provide at least one generator model.")
        return v

    def judge_names(self) -> set[str]:
        return {j.name for j in self.judges}


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML config into a typed PipelineConfig.

    Also loads a local .env file (if present) so API keys are
    available to every stage without manual `export`. Safe if python-dotenv isn't
    installed or no .env exists.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()  # reads .env from the current working directory, if present
    except ImportError:
        pass

    raw = yaml.safe_load(Path(path).read_text())
    cfg = PipelineConfig(**raw)

    # cross-field check: every judge referenced in metrics must be defined
    missing = set(cfg.metrics.judges) - cfg.judge_names()
    if missing:
        raise ValueError(f"metrics.judges references undefined judges: {missing}")
    return cfg


if __name__ == "__main__":
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml")
    print(f"Loaded '{cfg.experiment_name}' OK")
    print(f"  dataset:    {cfg.dataset.name} (n={cfg.dataset.n_examples})")
    print(f"  generators: {[m.name for m in cfg.generators]}")
    print(f"  judges:     {[m.name for m in cfg.judges]}")
