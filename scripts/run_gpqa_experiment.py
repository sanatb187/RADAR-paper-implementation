import argparse
from pathlib import Path
from typing import Any

from radar_bench.configurations import (
    QWEN3_OLLAMA_MODELS,
    QWEN3_REASONING_BUDGETS,
    build_qwen3_ollama_configurations,
)
from radar_bench.datasets.gpqa import (
    GPQA_REVISION,
    load_gpqa_splits,
)
from radar_bench.experiment import (
    run_gpqa_experiment,
    select_query_subset,
)
from radar_bench.ollama_generation import run_ollama_generation
from radar_bench.schemas import (
    GenerationResult,
    ModelConfiguration,
    Query,
    TokenBudget,
)

MODEL_IDS = tuple(model.model_id for model in QWEN3_OLLAMA_MODELS)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable local GPQA experiment."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_IDS,
        required=True,
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        choices=QWEN3_REASONING_BUDGETS,
        required=True,
    )
    parser.add_argument(
        "--train-count",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--test-count",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--revision",
        default=GPQA_REVISION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gpqa"),
    )

    return parser.parse_args()


def select_configurations(
    model_ids: list[str],
    budgets: list[int],
) -> list[ModelConfiguration]:
    requested_models = set(model_ids)
    requested_budgets = set(budgets)

    return [
        configuration
        for configuration in build_qwen3_ollama_configurations()
        if configuration.model_spec.model_id in requested_models
        and isinstance(
            configuration.reasoning_budget,
            TokenBudget,
        )
        and configuration.reasoning_budget.value in requested_budgets
    ]


def run_with_progress(
    **kwargs: Any,
) -> GenerationResult:
    query: Query = kwargs["query"]
    configuration: ModelConfiguration = kwargs["configuration"]

    print(f"Running {configuration.configuration_id} on {query.query_id}")

    generation = run_ollama_generation(**kwargs)

    print(
        "Completed "
        f"{generation.generation_id}: "
        f"{generation.token_usage.total_tokens} tokens, "
        f"{generation.latency_seconds:.2f}s"
    )

    return generation


def main() -> None:
    arguments = parse_arguments()

    if arguments.num_ctx < max(arguments.budgets) + 2048:
        raise ValueError(
            "num-ctx must be at least 2048 tokens larger "
            "than the largest reasoning budget"
        )

    configurations = select_configurations(
        arguments.models,
        arguments.budgets,
    )

    splits = load_gpqa_splits(
        seed=arguments.seed,
        revision=arguments.revision,
    )

    train_queries = select_query_subset(
        splits.train,
        count=arguments.train_count,
        seed=arguments.seed,
    )
    test_queries = select_query_subset(
        splits.test,
        count=arguments.test_count,
        seed=arguments.seed,
    )

    experiment_directory = arguments.output_dir / (
        f"revision-{arguments.revision[:12]}_seed-{arguments.seed}"
    )

    train_output = experiment_directory / f"train_n-{arguments.train_count}.jsonl"
    test_output = experiment_directory / f"test_n-{arguments.test_count}.jsonl"

    generation_count = (len(train_queries) + len(test_queries)) * len(configurations)

    print(f"Configurations: {len(configurations)}")
    print(f"Train queries: {len(train_queries)}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Planned generations: {generation_count}")

    request_options = {
        "temperature": 0.0,
        "seed": arguments.seed,
        "num_ctx": arguments.num_ctx,
    }

    train_records = run_gpqa_experiment(
        train_queries,
        configurations,
        output_path=train_output,
        request_options=request_options,
        generation_function=run_with_progress,
    )

    test_records = run_gpqa_experiment(
        test_queries,
        configurations,
        output_path=test_output,
        request_options=request_options,
        generation_function=run_with_progress,
    )

    print("GPQA experiment completed")
    print(f"Train records: {len(train_records)}")
    print(f"Test records: {len(test_records)}")
    print(f"Train output: {train_output}")
    print(f"Test output: {test_output}")


if __name__ == "__main__":
    main()
