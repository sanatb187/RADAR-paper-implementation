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
â”œâ”€â”€ configs/              # Versioned experiment configurations
â”œâ”€â”€ src/
â”‚   â””â”€â”€ radar_bench/      # Installable Python package
â”œâ”€â”€ tests/                # Unit and integration tests
â”œâ”€â”€ README.md
â”œâ”€â”€ pyproject.toml
â””â”€â”€ uv.lock
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

GPQA-Diamond is the initial reproduction target. Two dataset layouts are
supported:

- A deterministic split of the 198 Diamond questions for small local pilots.
- A paper-aligned split using GPQA Main excluding Diamond for training
  (250 questions) and full GPQA-Diamond for testing (198 questions).

Both loaders use the pinned dataset revision and deterministic choice ordering.

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

### Experiment provenance

vLLM experiments require a runtime metadata file so that model, hardware,
software, dataset, configuration, and Git provenance are recorded alongside
the generated responses.

Copy the example metadata before running an experiment:

```bash
cp \
  configs/vllm-runtime-kaggle-t4.example.json \
  configs/vllm-runtime-kaggle-t4.json
```
Populate the copied file with the exact model revision, vLLM and PyTorch
versions, CUDA version, GPU model, quantization, and server arguments used by
the active server.

Pass the metadata and current Git commit to the experiment runner:

```bash
uv run python scripts/run_gpqa_vllm_experiment.py \
  --model qwen3-4b-awq \
  --budgets 0 256 1024 \
  --train-count 16 \
  --test-count 16 \
  --runtime-metadata configs/vllm-runtime-kaggle-t4.json \
  --git-commit "$(git rev-parse HEAD)" \
  --output-dir outputs/gpqa-vllm
```
Use `--git-dirty` only when intentionally running with uncommitted changes.
The runner creates a resumable manifest before generation and marks it
completed after all expected records have been written.
Then run:
```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
git diff --check
git status --short --untracked-files=all
git diff --stat
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

## Reproduction results

Two evaluation setups were completed.

| Setup | Cost metric | Fixed frontier | RADAR frontier | Difference |
|---|---:|---:|---:|---:|
| Local Ollama pilot (64 train / 32 test) | Latency | 0.466408 | 0.448391 | -0.018017 |
| Local Ollama pilot (64 train / 32 test) | Output tokens | 0.468787 | 0.469762 | +0.000975 |
| Paper-aligned GPQA split (250 train / 198 test) | Output tokens | 0.347359 | 0.330996 | -0.016363 |

The paper-aligned experiment used GPQA Main excluding Diamond for training and
the full GPQA-Diamond dataset for testing. It evaluated Qwen3 4B/8B AWQ models
with reasoning budgets of 0, 256, and 1024 tokens. Generations ran through vLLM
on NVIDIA T4 GPUs, while routing used Qwen3-Embedding-8B and Chebyshev
scalarization.

Results from the paper-aligned experiment:

| Metric | Result |
|---|---:|
| Best fixed accuracy | 0.419 |
| Best RADAR accuracy | 0.338 |
| Oracle accuracy | 0.682 |
| Fixed frontier | 0.347359 |
| RADAR frontier | 0.330996 |
| RADAR difference from fixed | -0.016363 |
| Calibrated-classifier frontier | 0.352483 |
| Calibrated-classifier difference from fixed | +0.005124 |
| IRT training loss | 0.394328 |
| IRT test loss | 0.699372 |

The oracle upper bound shows substantial complementarity among the
configurations, but the learned IRT probabilities did not generalize. RADAR
did not outperform the fixed frontier, while the simpler calibrated classifier
slightly did.

This is a scoped reproduction rather than the paper's complete experimental
suite. It uses one benchmark, six self-hosted AWQ configurations, and
output-token cost instead of provider pricing. These deviations prevent a
direct comparison with the paper's reported absolute improvement and do not
support Semantic Router runtime integration at this stage.

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
- [x] Run one paper-aligned GPQA experiment
- [x] Compare the reproduced results with the paper
- [x] Document deviations and findings