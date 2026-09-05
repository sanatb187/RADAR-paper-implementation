from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from radar_bench.cost import (
    CostMetric,
    estimate_routing_costs,
    normalize_costs,
)
from radar_bench.embeddings import embed_queries
from radar_bench.irt import train_irt_model
from radar_bench.response_matrix import ResponseMatrix
from radar_bench.routing_evaluation import (
    PairedRoutingComparison,
    RoutingEvaluation,
    ScalarizationMethod,
    calculate_oracle_accuracy,
    compare_routing_results,
    evaluate_fixed_configurations,
    evaluate_radar_routing,
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


@dataclass(frozen=True)
class RadarEvaluationReport:
    training_loss_history: tuple[float, ...]
    cost_metric: CostMetric
    normalized_costs: dict[str, float]
    train_fixed_results: tuple[RoutingEvaluation, ...]
    fixed_results: tuple[RoutingEvaluation, ...]
    radar_results: tuple[RoutingEvaluation, ...]
    best_fixed_result: RoutingEvaluation
    train_oracle_accuracy: float
    test_oracle_accuracy: float
    radar_comparisons: tuple[PairedRoutingComparison, ...]
    configuration_abilities: dict[str, float]
    train_mean_predicted_probabilities: dict[str, float]
    test_mean_predicted_probabilities: dict[str, float]
    train_negative_discrimination_fraction: float
    test_negative_discrimination_fraction: float


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
    max_gradient_norm: float = 1.0,
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

    model.eval()

    with torch.no_grad():
        train_predicted_probabilities = model.predict_probabilities(train_embeddings)
        predicted_probabilities = model.predict_probabilities(test_embeddings)

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

    train_fixed_results = evaluate_fixed_configurations(
        train_matrix,
        train_records,
    )

    fixed_results = evaluate_fixed_configurations(
        test_matrix,
        test_records,
    )

    best_fixed_result = select_best_fixed_result(fixed_results)

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
        radar_results=radar_results,
        best_fixed_result=best_fixed_result,
        train_oracle_accuracy=calculate_oracle_accuracy(train_matrix),
        test_oracle_accuracy=calculate_oracle_accuracy(test_matrix),
        radar_comparisons=radar_comparisons,
        configuration_abilities=configuration_abilities,
        train_mean_predicted_probabilities=train_mean_predicted_probabilities,
        test_mean_predicted_probabilities=test_mean_predicted_probabilities,
        train_negative_discrimination_fraction=train_negative_discrimination_fraction,
        test_negative_discrimination_fraction=test_negative_discrimination_fraction,
    )
