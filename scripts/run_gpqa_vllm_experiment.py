import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from radar_bench.configurations import (
    QWEN3_REASONING_BUDGETS,
    QWEN3_VLLM_MODELS,
    build_qwen3_vllm_configurations,
)
from radar_bench.datasets.gpqa import (
    GPQA_REVISION,
    load_gpqa_diamond_splits,
)
from radar_bench.experiment import (
    run_gpqa_experiment,
    select_query_subset,
)
from radar_bench.schemas import (
    GenerationResult,
    ModelConfiguration,
    Query,
    TokenBudget,
)
from radar_bench.vllm_generation import run_vllm_generation

MODEL_IDS = tuple(model.model_id for model in QWEN3_VLLM_MODELS)

ProgressGenerationFunction = Callable[..., GenerationResult]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run a resumable GPQA experiment through a vLLM server.")
    )

    parser.add_argument(
        "--model",
        choices=MODEL_IDS,
        required=True,
        help="Model name exposed by the active vLLM server.",
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
        "--base-url",
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--revision",
        default=GPQA_REVISION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gpqa-vllm"),
    )

    return parser.parse_args()


def select_configurations(
    model_id: str,
    budgets: list[int],
) -> list[ModelConfiguration]:
    requested_budgets = set(budgets)

    return [
        configuration
        for configuration in build_qwen3_vllm_configurations()
        if configuration.model_spec.model_id == model_id
        and isinstance(
            configuration.reasoning_budget,
            TokenBudget,
        )
        and configuration.reasoning_budget.value in requested_budgets
    ]


def build_progress_generation_function(
    *,
    base_url: str,
    request_timeout_seconds: float,
) -> ProgressGenerationFunction:
    def run_with_progress(
        **kwargs: Any,
    ) -> GenerationResult:
        query: Query = kwargs["query"]
        configuration: ModelConfiguration = kwargs["configuration"]

        print(f"Running {configuration.configuration_id} on {query.query_id}")

        generation = run_vllm_generation(
            **kwargs,
            base_url=base_url,
            request_timeout_seconds=request_timeout_seconds,
        )

        print(
            "Completed "
            f"{generation.generation_id}: "
            f"{generation.token_usage.total_tokens} tokens, "
            f"{generation.latency_seconds:.2f}s"
        )

        return generation

    return run_with_progress


def main() -> None:
    arguments = parse_arguments()

    if arguments.request_timeout <= 0:
        raise ValueError("request-timeout must be positive")

    configurations = select_configurations(
        arguments.model,
        arguments.budgets,
    )

    splits = load_gpqa_diamond_splits(
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
        f"diamond-split_revision-{arguments.revision[:12]}_seed-{arguments.seed}"
    )

    train_output = experiment_directory / f"train_n-{arguments.train_count}.jsonl"
    test_output = experiment_directory / f"test_n-{arguments.test_count}.jsonl"

    generation_count = (len(train_queries) + len(test_queries)) * len(configurations)

    print(f"Model: {arguments.model}")
    print(f"Configurations: {len(configurations)}")
    print(f"Train queries: {len(train_queries)}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Planned generations: {generation_count}")

    generation_function = build_progress_generation_function(
        base_url=arguments.base_url,
        request_timeout_seconds=arguments.request_timeout,
    )

    request_options = {
        "seed": arguments.seed,
    }

    train_records = run_gpqa_experiment(
        train_queries,
        configurations,
        output_path=train_output,
        request_options=request_options,
        generation_function=generation_function,
    )

    test_records = run_gpqa_experiment(
        test_queries,
        configurations,
        output_path=test_output,
        request_options=request_options,
        generation_function=generation_function,
    )

    print("vLLM GPQA experiment completed")
    print(f"Cumulative train records: {len(train_records)}")
    print(f"Cumulative test records: {len(test_records)}")
    print(f"Train output: {train_output}")
    print(f"Test output: {test_output}")


if __name__ == "__main__":
    main()
