# RADAR Bench

## Overview

RADAR Bench is an installable Python library for reproducing experiments from
[RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs](https://openreview.net/forum?id=9k1oXUtrhO).

This work is being completed as part of
[vLLM Semantic Router issue #1166](https://github.com/vllm-project/semantic-router/issues/1166).

The current scope is limited to reproducing the paper's formulation and
reported results. GPQA-Diamond is the initial reproduction target because it
shows the largest reported improvement in the paper.

The library will provide reusable components for loading benchmark data,
estimating query difficulty and model ability, training the RADAR formulation,
and evaluating routing results. Repository-specific runtime integration is
outside the current scope.

## Project structure

```text
radar-bench/
├── configs/              # Versioned experiment configurations
├── src/
│   └── radar_bench/      # Installable Python package
├── tests/                # Unit and integration tests
├── README.md
├── pyproject.toml
└── uv.lock
```

## Installation

The project requires Python 3.12 and uses
[`uv`](https://docs.astral.sh/uv/) for dependency and environment management.

Clone the repository and install the package with its development dependencies:

```bash
uv sync
```

Verify that the package can be imported:

```bash
uv run python -c "import radar_bench; print(radar_bench.__file__)"
```

Because the project uses a `src` layout, run development commands through
`uv run` so they use the managed environment and installed package.

## Reproducing the benchmark

GPQA-Diamond is the initial reproduction target. The loader uses the pinned
dataset revision and creates a deterministic train/test split from the 198
Diamond questions.

Before running an experiment:

1. Accept the GPQA dataset terms on Hugging Face.
2. Install and start Ollama.
3. Pull the required Qwen models and embedding model.

For example:

```bash
ollama pull qwen3:4b
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

Run a small resumable generation experiment:

```bash
caffeinate -i uv run python scripts/run_gpqa_experiment.py \
  --models qwen3-4b qwen3-8b \
  --budgets 0 256 \
  --train-count 1 \
  --test-count 1 \
  --num-ctx 4096 \
  --output-dir outputs/gpqa-smoke
```

Evaluate saved records using Chebyshev scalarization and latency cost:

```bash
uv run python scripts/evaluate_gpqa_experiment.py \
  --train-records outputs/gpqa-smoke/diamond-split_revision-633f5ee89ab8_seed-42/train_n-1.jsonl \
  --test-records outputs/gpqa-smoke/diamond-split_revision-633f5ee89ab8_seed-42/test_n-1.jsonl \
  --scalarization chebyshev \
  --cost-metric latency
```

The evaluator supports three cost metrics:

| Metric | Meaning |
|---|---|
| `latency` | Average observed generation latency |
| `output-tokens` | Average reasoning and completion token count |
| `token-price` | Average output-token price, matching the paper's cost formulation |

The `token-price` metric requires a JSON pricing file. For example:

```json
[
  {
    "model_id": "qwen3-4b",
    "input_price_per_million_tokens": 0.0,
    "output_price_per_million_tokens": 1.0,
    "currency": "USD",
    "source": "Example only; replace with documented provider pricing",
    "effective_date": "2026-09-05"
  }
]
```

Run token-price evaluation with:

```bash
uv run python scripts/evaluate_gpqa_experiment.py \
  --train-records path/to/train-records.jsonl \
  --test-records path/to/test-records.jsonl \
  --scalarization chebyshev \
  --cost-metric token-price \
  --pricing-file path/to/pricing.json
```

The paper does not provide its exact historical per-model pricing table.
Therefore, pricing sources and effective dates must be recorded explicitly.
Synthetic or proxy prices must not be presented as reproduced paper results.

Generated datasets, model responses, checkpoints, and reports should not be
committed to Git.

## Development

Run the test and lint checks with:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

## Current status

- [x] Create the installable `radar_bench` package
- [x] Initialize dependency management with `uv`
- [x] Review and document the paper's experimental setup
- [x] Implement the RADAR formulation
- [x] Implement a reproducible GPQA-Diamond local pilot
- [ ] Reproduce the full GPQA-Diamond experiment
- [ ] Compare the reproduced results with the paper
- [ ] Document deviations and findings