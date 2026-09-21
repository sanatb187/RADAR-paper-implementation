import argparse
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from radar_bench.configurations import (
    build_qwen3_ollama_configurations,
)
from radar_bench.cost import load_pricing_file
from radar_bench.datasets.gpqa import (
    GPQA_REVISION,
    load_gpqa_diamond_splits,
    load_gpqa_splits,
)
from radar_bench.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    embed_queries,
)
from radar_bench.experiment import (
    load_evaluation_records,
)
from radar_bench.radar_evaluation import (
    RadarEvaluationReport,
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
        default=100,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
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
        "--scalarization",
        choices=(
            "linear",
            "chebyshev",
        ),
        default="linear",
    )
    parser.add_argument(
        "--cost-metric",
        choices=("latency", "output-tokens", "token-price"),
        default="latency",
    )
    parser.add_argument(
        "--pricing-file",
        type=Path,
    )
    parser.add_argument(
        "--irt-seed",
        type=int,
        default=None,
        help="Random seed for IRT training; defaults to --seed.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Ollama embedding model used by the IRT and classifier models.",
    )
    parser.add_argument(
        "--paper-split",
        action="store_true",
        help="Use GPQA Main minus Diamond for training and full Diamond for test.",
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


def print_irt_diagnostics(
    report: RadarEvaluationReport,
) -> None:
    train_observed = {
        result.strategy.removeprefix("fixed:"): result.accuracy
        for result in report.train_fixed_results
    }
    test_observed = {
        result.strategy.removeprefix("fixed:"): result.accuracy
        for result in report.fixed_results
    }

    print()
    print("IRT diagnostics")
    print(
        "configuration | ability | train observed | "
        "train predicted | test observed | test predicted"
    )
    print("-" * 110)

    for configuration_id, ability in report.configuration_abilities.items():
        print(
            f"{configuration_id} | "
            f"{ability:.3f} | "
            f"{train_observed[configuration_id]:.3f} | "
            f"{report.train_mean_predicted_probabilities[configuration_id]:.3f} | "
            f"{test_observed[configuration_id]:.3f} | "
            f"{report.test_mean_predicted_probabilities[configuration_id]:.3f}"
        )

    print()
    print(
        "Negative discrimination fraction: "
        f"train={report.train_negative_discrimination_fraction:.3f}, "
        f"test={report.test_negative_discrimination_fraction:.3f}"
    )


def print_probability_variation(
    report: RadarEvaluationReport,
) -> None:
    print()
    print("Predicted probability variation")
    print("configuration | train range | train std | test range | test std")
    print("-" * 90)

    for configuration_id in report.configuration_abilities:
        print(
            f"{configuration_id} | "
            f"{report.train_probability_ranges[configuration_id]:.6f} | "
            f"{report.train_probability_standard_deviations[configuration_id]:.6f} | "
            f"{report.test_probability_ranges[configuration_id]:.6f} | "
            f"{report.test_probability_standard_deviations[configuration_id]:.6f}"
        )


def print_cost_diagnostics(
    report: RadarEvaluationReport,
) -> None:
    print()
    print(f"Normalized routing costs ({report.cost_metric})")
    print("configuration | normalized cost")
    print("-" * 70)

    for configuration_id, cost in report.normalized_costs.items():
        print(f"{configuration_id} | {cost:.6f}")


def print_hypervolume(
    report: RadarEvaluationReport,
) -> None:
    difference = report.radar_hypervolume - report.fixed_hypervolume
    classifier_difference = report.classifier_hypervolume - report.fixed_hypervolume
    random_pair_difference = report.random_pair_hypervolume - report.fixed_hypervolume

    print()
    print("Hypervolume")
    print("-" * 70)
    print(f"Fixed-configuration frontier: {report.fixed_hypervolume:.6f}")
    print(f"RADAR frontier: {report.radar_hypervolume:.6f}")
    print(f"RADAR difference: {difference:+.6f}")
    print(f"Calibrated-classifier frontier: {report.classifier_hypervolume:.6f}")
    print(f"Calibrated-classifier difference: {classifier_difference:+.6f}")
    print(f"Random-Pair frontier: {report.random_pair_hypervolume:.6f}")
    print(f"Random-Pair difference: {random_pair_difference:+.6f}")


def _format_cpt(
    cost_fraction: float | None,
) -> str:
    if cost_fraction is None:
        return "unreachable"

    return f"{cost_fraction:.4f} ({cost_fraction * 100:.2f}%)"


def print_cpt(
    report: RadarEvaluationReport,
) -> None:
    print()
    print("Proxy CPT (90%)")
    print("-" * 70)
    print(f"Reference configuration: {report.cpt_reference_configuration_id}")
    print(f"Reference accuracy: {report.cpt_reference_accuracy:.6f}")
    print(f"Reference raw cost ({report.cost_metric}): {report.cpt_reference_cost:.6f}")
    print(f"RADAR: {_format_cpt(report.radar_cpt_90)}")
    print(f"Calibrated classifier: {_format_cpt(report.classifier_cpt_90)}")
    print(f"Random-Pair: {_format_cpt(report.random_pair_cpt_90)}")


def print_aurc(
    report: RadarEvaluationReport,
) -> None:
    print()
    print("Area under the risk-coverage curve")
    print("-" * 70)

    for strategy, area in report.aurc_by_strategy.items():
        print(f"{strategy}: {area:.6f}")


def main() -> None:
    arguments = parse_arguments()

    configurations = None
    pricing_by_model_id = None

    if arguments.cost_metric == "token-price":
        if arguments.pricing_file is None:
            raise ValueError(
                "--pricing-file is required when --cost-metric=token-price"
            )

        configurations = build_qwen3_ollama_configurations()
        pricing_by_model_id = load_pricing_file(arguments.pricing_file)

    train_records = load_evaluation_records(arguments.train_records)
    test_records = load_evaluation_records(arguments.test_records)

    split_loader = (
        load_gpqa_splits if arguments.paper_split else load_gpqa_diamond_splits
    )

    splits = split_loader(
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
    irt_seed = arguments.seed if arguments.irt_seed is None else arguments.irt_seed
    embedding_function = partial(
        embed_queries,
        model=arguments.embedding_model,
    )
    report = evaluate_radar_experiment(
        train_queries,
        test_queries,
        train_records,
        test_records,
        performance_weights=arguments.weights,
        num_epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        batch_size=arguments.batch_size,
        max_gradient_norm=arguments.max_gradient_norm,
        scalarization=arguments.scalarization,
        embedding_function=embedding_function,
        calibration_bins=arguments.calibration_bins,
        cost_metric=arguments.cost_metric,
        configurations=configurations,
        pricing_by_model_id=pricing_by_model_id,
        random_seed=irt_seed,
    )
    classifier_difference = report.classifier_hypervolume - report.fixed_hypervolume

    print("RADAR evaluation completed")
    print(f"Train records: {len(train_records)}")
    print(f"Test records: {len(test_records)}")
    print(f"Routing cost metric: {report.cost_metric}")
    print(f"Embedding model: {arguments.embedding_model}")
    print(f"IRT training seed: {irt_seed}")
    print(f"Initial IRT loss: {report.training_loss_history[0]:.6f}")
    print(f"Final IRT loss: {report.training_loss_history[-1]:.6f}")
    print(f"Test IRT loss: {report.test_irt_loss:.6f}")
    print(f"Calibrated-classifier frontier: {report.classifier_hypervolume:.6f}")
    print(f"Calibrated-classifier difference: {classifier_difference:+.6f}")
    print(f"Test Brier score: {report.test_brier_score:.6f}")
    print(f"Classifier test loss: {report.classifier_test_loss:.6f}")
    print(f"Classifier test Brier score: {report.classifier_test_brier_score:.6f}")
    print(
        "Classifier test expected calibration error: "
        f"{report.classifier_test_expected_calibration_error:.6f}"
    )
    print(
        f"Test expected calibration error: {report.test_expected_calibration_error:.6f}"
    )
    print_irt_diagnostics(report)
    print_probability_variation(report)
    print_cost_diagnostics(report)

    print_results(
        "Training fixed-configuration results",
        report.train_fixed_results,
    )

    print_results(
        "Test fixed-configuration baselines",
        report.fixed_results,
    )

    print_results(
        "Routing-sampling baselines",
        report.routing_sampling_results,
    )

    print()
    print("Random-Pair endpoints")
    print("-" * 70)
    print(f"Lower-cost configuration: {report.random_pair_lower_configuration_id}")
    print(f"Upper-cost configuration: {report.random_pair_upper_configuration_id}")

    print_results(
        "Random-Pair runs",
        report.random_pair_results,
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

    print_hypervolume(report)

    print_cpt(report)

    print_aurc(report)

    print_routing_diagnostics(
        report.radar_results,
        report.radar_comparisons,
    )


if __name__ == "__main__":
    main()
