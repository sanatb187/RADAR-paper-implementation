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

The first supported reproduction will be GPQA-Diamond. The exact command,
versioned configuration, required model outputs, and expected artifacts will be
documented here alongside the implementation so that a run can be repeated
without relying on undocumented local settings.

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
- [ ] Review and document the paper's experimental setup
- [ ] Implement the RADAR formulation
- [ ] Reproduce the GPQA-Diamond experiment
- [ ] Compare the reproduced results with the paper
- [ ] Document deviations and findings
