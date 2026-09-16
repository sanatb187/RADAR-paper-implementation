from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from radar_bench.calibrated_classifier import train_calibrated_classifier
from radar_bench.cost import (
    CostMetric,
    estimate_routing_costs,
    normalize_costs,
)
from radar_bench.embeddings import embed_queries
from radar_bench.irt import train_irt_model
from radar_bench.metrics import (
    PerformanceCostPoint,
    build_performance_cost_points,
    calculate_area_under_risk_coverage,
    calculate_brier_score,
    calculate_cost_at_performance_threshold,
    calculate_expected_calibration_error,
    calculate_hypervolume,
)
from radar_bench.response_matrix import ResponseMatrix
from radar_bench.routing_evaluation import (
    PairedRoutingComparison,
    RoutingEvaluation,
    ScalarizationMethod,
    calculate_oracle_accuracy,
    compare_routing_results,
    evaluate_fixed_configurations,
    evaluate_radar_routing,
    evaluate_random_pair,
    evaluate_routing_sampling,
    select_best_fixed_result,
)
from radar_bench.schemas import (
    EvaluationRecord,
    ModelConfiguration,
    Pricing,
    Query,
)

QueryEmbeddingFunction = Callable[
    [Sequence[Query]],
    torch.Tensor,
]


DEFAULT_ROUTING_SAMPLING_SEEDS: tuple[int, ...] = tuple(range(10))


@dataclass(frozen=True)
class RadarEvaluationReport:
    training_loss_history: tuple[float, ...]
    cost_metric: CostMetric
    normalized_costs: dict[str, float]
    train_fixed_results: tuple[RoutingEvaluation, ...]
    fixed_results: tuple[RoutingEvaluation, ...]
    radar_results: tuple[RoutingEvaluation, ...]
    classifier_results: tuple[RoutingEvaluation, ...]
    routing_sampling_results: tuple[RoutingEvaluation, ...]
    random_pair_results: tuple[RoutingEvaluation, ...]
    random_pair_lower_configuration_id: str
    random_pair_upper_configuration_id: str
    best_fixed_result: RoutingEvaluation
    train_oracle_accuracy: float
    test_oracle_accuracy: float
    radar_comparisons: tuple[PairedRoutingComparison, ...]
    cpt_reference_configuration_id: str
    cpt_reference_accuracy: float
    cpt_reference_cost: float
    radar_cpt_90: float | None
    classifier_cpt_90: float | None
    random_pair_cpt_90: float | None
    configuration_abilities: dict[str, float]
    classifier_test_loss: float
    classifier_test_brier_score: float
    classifier_test_expected_calibration_error: float
    train_mean_predicted_probabilities: dict[str, float]
    test_mean_predicted_probabilities: dict[str, float]
    train_negative_discrimination_fraction: float
    test_negative_discrimination_fraction: float
    test_irt_loss: float
    test_brier_score: float
    test_expected_calibration_error: float
    aurc_by_strategy: dict[str, float]
    fixed_hypervolume: float
    radar_hypervolume: float
    classifier_hypervolume: float
    random_pair_hypervolume: float
    train_probability_ranges: dict[str, float]
    test_probability_ranges: dict[str, float]
    train_probability_standard_deviations: dict[str, float]
    test_probability_standard_deviations: dict[str, float]


def _order_queries(
    queries: Sequence[Query],
    query_ids: Sequence[str],
) -> list[Query]:
    queries_by_id = {query.query_id: query for query in queries}

    if len(queries_by_id) != len(queries):
        raise ValueError("Query IDs must be unique")

    expected_ids = set(query_ids)
    actual_ids = set(queries_by_id)

    if expected_ids != actual_ids:
        missing_ids = sorted(expected_ids - actual_ids)
        unexpected_ids = sorted(actual_ids - expected_ids)

        raise ValueError(
            "Queries do not match response matrix. "
            f"Missing: {missing_ids}; "
            f"unexpected: {unexpected_ids}"
        )

    return [queries_by_id[query_id] for query_id in query_ids]


def _calculate_routing_aurc(
    results: Sequence[RoutingEvaluation],
    predicted_probabilities: torch.Tensor,
    response_matrix: ResponseMatrix,
) -> dict[str, float]:
    configuration_index = {
        configuration_id: row
        for row, configuration_id in enumerate(response_matrix.configuration_ids)
    }

    aurc_by_strategy: dict[str, float] = {}

    for result in results:
        if len(result.selected_configuration_ids) != len(response_matrix.query_ids):
            raise ValueError(
                f"Selections do not match queries for strategy: {result.strategy}"
            )

        selected_confidences: list[torch.Tensor] = []
        selected_correctness: list[float] = []

        for column, configuration_id in enumerate(result.selected_configuration_ids):
            if configuration_id not in configuration_index:
                raise ValueError(f"Unknown selected configuration: {configuration_id}")

            row = configuration_index[configuration_id]
            selected_confidences.append(predicted_probabilities[row, column])
            selected_correctness.append(float(response_matrix.values[row, column]))

        confidences = torch.stack(selected_confidences)
        correctness = torch.tensor(
            selected_correctness,
            dtype=confidences.dtype,
            device=confidences.device,
        )

        aurc_by_strategy[result.strategy] = calculate_area_under_risk_coverage(
            confidences,
            correctness,
        )

    return aurc_by_strategy


def evaluate_radar_experiment(
    train_queries: Sequence[Query],
    test_queries: Sequence[Query],
    train_records: Sequence[EvaluationRecord],
    test_records: Sequence[EvaluationRecord],
    *,
    performance_weights: Sequence[float] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ),
    num_epochs: int = 100,
    learning_rate: float = 5e-4,
    batch_size: int = 32,
    calibration_bins: int = 10,
    max_gradient_norm: float = 1.0,
    routing_sampling_seeds: Sequence[int] = DEFAULT_ROUTING_SAMPLING_SEEDS,
    random_seed: int = 42,
    embedding_function: QueryEmbeddingFunction = embed_queries,
    scalarization: ScalarizationMethod = "linear",
    cost_metric: CostMetric = "latency",
    configurations: Sequence[ModelConfiguration] | None = None,
    pricing_by_model_id: Mapping[str, Pricing] | None = None,
) -> RadarEvaluationReport:
    """Train IRT and compare RADAR with fixed routing."""

    if not performance_weights:
        raise ValueError("performance_weights cannot be empty")

    if not routing_sampling_seeds:
        raise ValueError("routing_sampling_seeds cannot be empty")

    if len(set(routing_sampling_seeds)) != len(routing_sampling_seeds):
        raise ValueError("routing_sampling_seeds must be unique")

    if any(weight < 0.0 or weight > 1.0 for weight in performance_weights):
        raise ValueError("performance_weights must be between 0 and 1")

    train_matrix = ResponseMatrix.from_records(train_records)
    test_matrix = ResponseMatrix.from_records(test_records)

    if train_matrix.configuration_ids != test_matrix.configuration_ids:
        raise ValueError("Train and test configuration IDs must match")

    ordered_train_queries = _order_queries(
        train_queries,
        train_matrix.query_ids,
    )
    ordered_test_queries = _order_queries(
        test_queries,
        test_matrix.query_ids,
    )

    train_embeddings = embedding_function(ordered_train_queries)
    test_embeddings = embedding_function(ordered_test_queries)

    if train_embeddings.shape[1] != test_embeddings.shape[1]:
        raise ValueError("Train and test embedding dimensions must match")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)

        model, loss_history = train_irt_model(
            response_matrix=train_matrix,
            query_embeddings=train_embeddings,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_gradient_norm=max_gradient_norm,
        )

        classifier = train_calibrated_classifier(
            response_matrix=train_matrix,
            query_embeddings=train_embeddings,
            num_epochs=num_epochs,
            calibration_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_gradient_norm=max_gradient_norm,
            random_seed=random_seed,
        )

        classifier.eval()

    model.eval()

    with torch.no_grad():
        train_predicted_probabilities = model.predict_probabilities(train_embeddings)
        predicted_probabilities = model.predict_probabilities(test_embeddings)
        test_logits = model(test_embeddings)

        test_targets = torch.as_tensor(
            test_matrix.values,
            dtype=torch.float32,
            device=test_logits.device,
        )
        classifier_predicted_probabilities = classifier.predict_probabilities(
            test_embeddings
        )

        classifier_test_loss = float(
            F.binary_cross_entropy(
                classifier_predicted_probabilities,
                test_targets,
            ).item()
        )
        classifier_test_brier_score = calculate_brier_score(
            classifier_predicted_probabilities,
            test_targets,
        )
        classifier_test_expected_calibration_error = (
            calculate_expected_calibration_error(
                classifier_predicted_probabilities,
                test_targets,
                num_bins=calibration_bins,
            )
        )

        test_irt_loss = float(
            F.binary_cross_entropy_with_logits(
                test_logits,
                test_targets,
            ).item()
        )

        test_brier_score = calculate_brier_score(
            predicted_probabilities,
            test_targets,
        )
        test_expected_calibration_error = calculate_expected_calibration_error(
            predicted_probabilities,
            test_targets,
            num_bins=calibration_bins,
        )

        train_probability_ranges = {
            configuration_id: float(value)
            for configuration_id, value in zip(
                train_matrix.configuration_ids,
                (
                    train_predicted_probabilities.max(dim=1).values
                    - train_predicted_probabilities.min(dim=1).values
                )
                .cpu()
                .tolist(),
                strict=True,
            )
        }

        test_probability_ranges = {
            configuration_id: float(value)
            for configuration_id, value in zip(
                test_matrix.configuration_ids,
                (
                    predicted_probabilities.max(dim=1).values
                    - predicted_probabilities.min(dim=1).values
                )
                .cpu()
                .tolist(),
                strict=True,
            )
        }

        train_probability_standard_deviations = {
            configuration_id: float(value)
            for configuration_id, value in zip(
                train_matrix.configuration_ids,
                train_predicted_probabilities.std(
                    dim=1,
                    unbiased=False,
                )
                .cpu()
                .tolist(),
                strict=True,
            )
        }

        test_probability_standard_deviations = {
            configuration_id: float(value)
            for configuration_id, value in zip(
                test_matrix.configuration_ids,
                predicted_probabilities.std(
                    dim=1,
                    unbiased=False,
                )
                .cpu()
                .tolist(),
                strict=True,
            )
        }

        train_discriminations = train_embeddings @ model.discrimination_weights
        test_discriminations = test_embeddings @ model.discrimination_weights

    configuration_abilities = {
        configuration_id: float(ability)
        for configuration_id, ability in zip(
            train_matrix.configuration_ids,
            model.abilities.detach().cpu().tolist(),
            strict=True,
        )
    }

    train_mean_predicted_probabilities = {
        configuration_id: float(probability)
        for configuration_id, probability in zip(
            train_matrix.configuration_ids,
            train_predicted_probabilities.mean(dim=1).cpu().tolist(),
            strict=True,
        )
    }

    test_mean_predicted_probabilities = {
        configuration_id: float(probability)
        for configuration_id, probability in zip(
            test_matrix.configuration_ids,
            predicted_probabilities.mean(dim=1).cpu().tolist(),
            strict=True,
        )
    }

    train_negative_discrimination_fraction = float(
        (train_discriminations < 0).float().mean().item()
    )
    test_negative_discrimination_fraction = float(
        (test_discriminations < 0).float().mean().item()
    )

    costs = estimate_routing_costs(
        train_records,
        train_matrix.configuration_ids,
        metric=cost_metric,
        configurations=configurations,
        pricing_by_model_id=pricing_by_model_id,
    )
    normalized_costs = normalize_costs(costs)

    if len(train_matrix.configuration_ids) < 2:
        raise ValueError("Random-Pair requires at least two configurations")

    random_pair_lower_configuration_id = min(
        train_matrix.configuration_ids,
        key=lambda configuration_id: (
            normalized_costs[configuration_id],
            configuration_id,
        ),
    )
    random_pair_upper_configuration_id = max(
        train_matrix.configuration_ids,
        key=lambda configuration_id: (
            normalized_costs[configuration_id],
            configuration_id,
        ),
    )

    train_fixed_results = evaluate_fixed_configurations(
        train_matrix,
        train_records,
    )

    fixed_results = evaluate_fixed_configurations(
        test_matrix,
        test_records,
    )

    routing_sampling_results = tuple(
        evaluate_routing_sampling(
            test_matrix,
            test_records,
            random_seed=sampling_seed,
        )
        for sampling_seed in routing_sampling_seeds
    )

    random_pair_results = tuple(
        evaluate_random_pair(
            test_matrix,
            test_records,
            lower_configuration_id=random_pair_lower_configuration_id,
            upper_configuration_id=random_pair_upper_configuration_id,
            upper_probability=performance_weight,
            random_seed=sampling_seed,
        )
        for performance_weight in performance_weights
        for sampling_seed in routing_sampling_seeds
    )

    best_fixed_result = select_best_fixed_result(fixed_results)

    fixed_results_by_configuration_id = {
        result.strategy.removeprefix("fixed:"): result for result in fixed_results
    }

    cpt_reference_configuration_id = random_pair_upper_configuration_id
    cpt_reference_result = fixed_results_by_configuration_id[
        cpt_reference_configuration_id
    ]
    cpt_reference_accuracy = cpt_reference_result.accuracy
    cpt_reference_cost = costs[cpt_reference_configuration_id]

    if cpt_reference_cost <= 0.0:
        raise ValueError("CPT reference cost must be greater than zero")

    cpt_cost_fractions = {
        configuration_id: cost / cpt_reference_cost
        for configuration_id, cost in costs.items()
    }

    radar_results = tuple(
        evaluate_radar_routing(
            predicted_probabilities,
            test_matrix,
            test_records,
            normalized_costs,
            performance_weight=performance_weight,
            scalarization=scalarization,
        )
        for performance_weight in performance_weights
    )

    classifier_strategy_prefix = "calibrated-classifier"

    if scalarization == "chebyshev":
        classifier_strategy_prefix += "-chebyshev"

    classifier_results = tuple(
        evaluate_radar_routing(
            classifier_predicted_probabilities,
            test_matrix,
            test_records,
            normalized_costs,
            performance_weight=performance_weight,
            scalarization=scalarization,
            strategy_prefix=classifier_strategy_prefix,
        )
        for performance_weight in performance_weights
    )

    aurc_by_strategy = _calculate_routing_aurc(
        (
            *fixed_results,
            *routing_sampling_results,
            *radar_results,
        ),
        predicted_probabilities,
        test_matrix,
    )

    aurc_by_strategy.update(
        _calculate_routing_aurc(
            classifier_results,
            classifier_predicted_probabilities,
            test_matrix,
        )
    )

    fixed_points = build_performance_cost_points(
        fixed_results,
        normalized_costs,
    )
    radar_points = build_performance_cost_points(
        radar_results,
        normalized_costs,
    )

    classifier_points = build_performance_cost_points(
        classifier_results,
        normalized_costs,
    )

    random_pair_points = build_performance_cost_points(
        random_pair_results,
        normalized_costs,
    )

    seed_count = len(routing_sampling_seeds)

    random_pair_mean_points = tuple(
        PerformanceCostPoint(
            strategy=f"random-pair:{performance_weight:g}",
            accuracy=(sum(point.accuracy for point in weight_points) / seed_count),
            normalized_cost=(
                sum(point.normalized_cost for point in weight_points) / seed_count
            ),
        )
        for weight_index, performance_weight in enumerate(performance_weights)
        for weight_points in (
            random_pair_points[
                weight_index * seed_count : (weight_index + 1) * seed_count
            ],
        )
    )

    radar_cpt_points = build_performance_cost_points(
        radar_results,
        cpt_cost_fractions,
    )
    classifier_cpt_points = build_performance_cost_points(
        classifier_results,
        cpt_cost_fractions,
    )
    random_pair_cpt_seed_points = build_performance_cost_points(
        random_pair_results,
        cpt_cost_fractions,
    )

    random_pair_cpt_mean_points = tuple(
        PerformanceCostPoint(
            strategy=f"random-pair:{performance_weight:g}",
            accuracy=(sum(point.accuracy for point in weight_points) / seed_count),
            normalized_cost=(
                sum(point.normalized_cost for point in weight_points) / seed_count
            ),
        )
        for weight_index, performance_weight in enumerate(performance_weights)
        for weight_points in (
            random_pair_cpt_seed_points[
                weight_index * seed_count : (weight_index + 1) * seed_count
            ],
        )
    )

    radar_cpt_90 = calculate_cost_at_performance_threshold(
        radar_cpt_points,
        reference_accuracy=cpt_reference_accuracy,
    )
    classifier_cpt_90 = calculate_cost_at_performance_threshold(
        classifier_cpt_points,
        reference_accuracy=cpt_reference_accuracy,
    )
    random_pair_cpt_90 = calculate_cost_at_performance_threshold(
        random_pair_cpt_mean_points,
        reference_accuracy=cpt_reference_accuracy,
    )

    fixed_hypervolume = calculate_hypervolume(fixed_points)
    radar_hypervolume = calculate_hypervolume(radar_points)
    classifier_hypervolume = calculate_hypervolume(classifier_points)
    random_pair_hypervolume = calculate_hypervolume(random_pair_mean_points)
    radar_comparisons = tuple(
        compare_routing_results(
            radar_result,
            best_fixed_result,
            test_matrix,
        )
        for radar_result in radar_results
    )

    return RadarEvaluationReport(
        training_loss_history=tuple(loss_history),
        cost_metric=cost_metric,
        normalized_costs=normalized_costs,
        train_fixed_results=train_fixed_results,
        fixed_results=fixed_results,
        routing_sampling_results=routing_sampling_results,
        radar_results=radar_results,
        best_fixed_result=best_fixed_result,
        train_oracle_accuracy=calculate_oracle_accuracy(train_matrix),
        test_oracle_accuracy=calculate_oracle_accuracy(test_matrix),
        radar_comparisons=radar_comparisons,
        configuration_abilities=configuration_abilities,
        train_mean_predicted_probabilities=train_mean_predicted_probabilities,
        test_mean_predicted_probabilities=test_mean_predicted_probabilities,
        train_probability_ranges=train_probability_ranges,
        test_probability_ranges=test_probability_ranges,
        train_probability_standard_deviations=train_probability_standard_deviations,
        test_probability_standard_deviations=test_probability_standard_deviations,
        train_negative_discrimination_fraction=train_negative_discrimination_fraction,
        test_negative_discrimination_fraction=test_negative_discrimination_fraction,
        test_irt_loss=test_irt_loss,
        test_brier_score=test_brier_score,
        test_expected_calibration_error=test_expected_calibration_error,
        classifier_results=classifier_results,
        classifier_test_loss=classifier_test_loss,
        classifier_test_brier_score=classifier_test_brier_score,
        classifier_test_expected_calibration_error=(
            classifier_test_expected_calibration_error
        ),
        classifier_hypervolume=classifier_hypervolume,
        aurc_by_strategy=aurc_by_strategy,
        fixed_hypervolume=fixed_hypervolume,
        radar_hypervolume=radar_hypervolume,
        random_pair_results=random_pair_results,
        random_pair_lower_configuration_id=(random_pair_lower_configuration_id),
        random_pair_upper_configuration_id=(random_pair_upper_configuration_id),
        random_pair_hypervolume=random_pair_hypervolume,
        cpt_reference_configuration_id=cpt_reference_configuration_id,
        cpt_reference_accuracy=cpt_reference_accuracy,
        cpt_reference_cost=cpt_reference_cost,
        radar_cpt_90=radar_cpt_90,
        classifier_cpt_90=classifier_cpt_90,
        random_pair_cpt_90=random_pair_cpt_90,
    )
