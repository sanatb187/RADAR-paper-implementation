from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from radar_bench.cost import (
    estimate_configuration_latency_costs,
    normalize_costs,
)
from radar_bench.embeddings import embed_queries
from radar_bench.irt import train_irt_model
from radar_bench.response_matrix import ResponseMatrix
from radar_bench.routing_evaluation import (
    RoutingEvaluation,
    evaluate_fixed_configurations,
    evaluate_radar_routing,
)
from radar_bench.schemas import EvaluationRecord, Query

QueryEmbeddingFunction = Callable[
    [Sequence[Query]],
    torch.Tensor,
]


@dataclass(frozen=True)
class RadarEvaluationReport:
    training_loss_history: tuple[float, ...]
    normalized_latency_costs: dict[str, float]
    fixed_results: tuple[RoutingEvaluation, ...]
    radar_results: tuple[RoutingEvaluation, ...]


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
    num_epochs: int = 500,
    learning_rate: float = 0.01,
    random_seed: int = 42,
    embedding_function: QueryEmbeddingFunction = (embed_queries),
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
        )

    model.eval()

    with torch.no_grad():
        predicted_probabilities = model.predict_probabilities(test_embeddings)

    latency_costs = estimate_configuration_latency_costs(
        train_records,
        train_matrix.configuration_ids,
    )
    normalized_latency_costs = normalize_costs(latency_costs)

    fixed_results = evaluate_fixed_configurations(
        test_matrix,
        test_records,
    )

    radar_results = tuple(
        evaluate_radar_routing(
            predicted_probabilities,
            test_matrix,
            test_records,
            normalized_latency_costs,
            performance_weight=performance_weight,
        )
        for performance_weight in performance_weights
    )

    return RadarEvaluationReport(
        training_loss_history=tuple(loss_history),
        normalized_latency_costs=(normalized_latency_costs),
        fixed_results=fixed_results,
        radar_results=radar_results,
    )
