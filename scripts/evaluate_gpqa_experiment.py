import argparse
from collections.abc import Sequence
from pathlib import Path

from radar_bench.datasets.gpqa import (
    GPQA_REVISION,
    load_gpqa_diamond_splits,
)
from radar_bench.experiment import (
    load_evaluation_records,
)
from radar_bench.radar_evaluation import (
    evaluate_radar_experiment,
)
from radar_bench.routing_evaluation import (
    PairedRoutingComparison,
    RoutingEvaluation,
    count_configuration_selections,
)
from radar_bench.schemas import Query


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RADAR against fixed routing."
    )

    parser.add_argument(
        "--train-records",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--test-records",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
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

    return parser.parse_args()


def select_record_queries(
    queries: Sequence[Query],
    query_ids: set[str],
) -> list[Query]:
    return [query for query in queries if query.query_id in query_ids]


def print_results(
    heading: str,
    results: Sequence[RoutingEvaluation],
) -> None:
    print()
    print(heading)
    print("strategy | accuracy | average latency")
    print("-" * 70)

    for result in results:
        print(
            f"{result.strategy} | "
            f"{result.accuracy:.3f} | "
            f"{result.average_latency_seconds:.3f}s"
        )


def _format_query_ids(
    query_ids: tuple[str, ...],
) -> str:
    if not query_ids:
        return "none"

    return ", ".join(query_ids)


def print_routing_diagnostics(
    results: Sequence[RoutingEvaluation],
    comparisons: Sequence[PairedRoutingComparison],
) -> None:
    print()
    print("Routing diagnostics")
    print("-" * 70)

    for result, comparison in zip(
        results,
        comparisons,
        strict=True,
    ):
        selection_counts = count_configuration_selections(result)
        selection_summary = ", ".join(
            f"{configuration_id}={count}"
            for configuration_id, count in selection_counts.items()
        )

        print()
        print(result.strategy)
        print(f"Compared with: {comparison.baseline_strategy}")
        print(f"Selections: {selection_summary}")
        print(
            f"Improved ({len(comparison.improved_query_ids)}): "
            f"{_format_query_ids(comparison.improved_query_ids)}"
        )
        print(
            f"Regressed ({len(comparison.regressed_query_ids)}): "
            f"{_format_query_ids(comparison.regressed_query_ids)}"
        )
        print(f"Both correct: {len(comparison.both_correct_query_ids)}")
        print(f"Both incorrect: {len(comparison.both_incorrect_query_ids)}")


def main() -> None:
    arguments = parse_arguments()

    train_records = load_evaluation_records(arguments.train_records)
    test_records = load_evaluation_records(arguments.test_records)

    splits = load_gpqa_diamond_splits(
        seed=arguments.seed,
        revision=arguments.revision,
    )

    train_query_ids = {record.generation.query_id for record in train_records}
    test_query_ids = {record.generation.query_id for record in test_records}

    train_queries = select_record_queries(
        splits.train,
        train_query_ids,
    )
    test_queries = select_record_queries(
        splits.test,
        test_query_ids,
    )

    report = evaluate_radar_experiment(
        train_queries,
        test_queries,
        train_records,
        test_records,
        performance_weights=arguments.weights,
        num_epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        random_seed=arguments.seed,
    )

    print("RADAR evaluation completed")
    print(f"Train records: {len(train_records)}")
    print(f"Test records: {len(test_records)}")
    print(f"Initial IRT loss: {report.training_loss_history[0]:.6f}")
    print(f"Final IRT loss: {report.training_loss_history[-1]:.6f}")

    print_results(
        "Training fixed-configuration results",
        report.train_fixed_results,
    )

    print_results(
        "Test fixed-configuration baselines",
        report.fixed_results,
    )

    print()
    print("Oracle upper bounds")
    print("-" * 70)
    print(f"Train oracle accuracy: {report.train_oracle_accuracy:.3f}")
    print(f"Test oracle accuracy: {report.test_oracle_accuracy:.3f}")
    print(f"Best fixed test baseline: {report.best_fixed_result.strategy}")

    print_results(
        "RADAR routing",
        report.radar_results,
    )

    print_routing_diagnostics(
        report.radar_results,
        report.radar_comparisons,
    )


if __name__ == "__main__":
    main()
