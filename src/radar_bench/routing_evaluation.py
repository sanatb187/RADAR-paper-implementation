from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from radar_bench.optimization import (
    select_configuration,
)
from radar_bench.response_matrix import ResponseMatrix
from radar_bench.schemas import EvaluationRecord


@dataclass(frozen=True)
class RoutingEvaluation:
    strategy: str
    accuracy: float
    average_latency_seconds: float
    selected_configuration_ids: tuple[str, ...]


@dataclass(frozen=True)
class PairedRoutingComparison:
    candidate_strategy: str
    baseline_strategy: str
    improved_query_ids: tuple[str, ...]
    regressed_query_ids: tuple[str, ...]
    both_correct_query_ids: tuple[str, ...]
    both_incorrect_query_ids: tuple[str, ...]


def _build_record_lookup(
    response_matrix: ResponseMatrix,
    evaluation_records: Sequence[EvaluationRecord],
) -> dict[tuple[str, str], EvaluationRecord]:
    expected_pairs = {
        (configuration_id, query_id)
        for configuration_id in (response_matrix.configuration_ids)
        for query_id in response_matrix.query_ids
    }

    records_by_pair: dict[
        tuple[str, str],
        EvaluationRecord,
    ] = {}

    for record in evaluation_records:
        pair = (
            record.generation.configuration_id,
            record.generation.query_id,
        )

        if pair not in expected_pairs:
            raise ValueError(f"Unexpected evaluation record: {pair}")

        if pair in records_by_pair:
            raise ValueError(f"Duplicate evaluation record: {pair}")

        records_by_pair[pair] = record

    missing_pairs = expected_pairs - records_by_pair.keys()

    if missing_pairs:
        raise ValueError(f"Missing evaluation records: {sorted(missing_pairs)}")

    return records_by_pair


def evaluate_fixed_configurations(
    response_matrix: ResponseMatrix,
    evaluation_records: Sequence[EvaluationRecord],
) -> tuple[RoutingEvaluation, ...]:
    """Evaluate always selecting each configuration."""

    records_by_pair = _build_record_lookup(
        response_matrix,
        evaluation_records,
    )

    results: list[RoutingEvaluation] = []

    for row, configuration_id in enumerate(response_matrix.configuration_ids):
        correct_count = int(response_matrix.values[row].sum())

        latencies = [
            records_by_pair[(configuration_id, query_id)].generation.latency_seconds
            for query_id in response_matrix.query_ids
        ]

        query_count = len(response_matrix.query_ids)

        results.append(
            RoutingEvaluation(
                strategy=f"fixed:{configuration_id}",
                accuracy=correct_count / query_count,
                average_latency_seconds=(sum(latencies) / query_count),
                selected_configuration_ids=(configuration_id,) * query_count,
            )
        )

    return tuple(results)


def evaluate_radar_routing(
    predicted_probabilities: torch.Tensor,
    response_matrix: ResponseMatrix,
    evaluation_records: Sequence[EvaluationRecord],
    normalized_costs: Mapping[str, float],
    *,
    performance_weight: float,
) -> RoutingEvaluation:
    """Evaluate RADAR selections against observed results."""

    expected_shape = (
        len(response_matrix.configuration_ids),
        len(response_matrix.query_ids),
    )

    if tuple(predicted_probabilities.shape) != (expected_shape):
        raise ValueError(
            "predicted_probabilities shape must match "
            f"the response matrix shape {expected_shape}"
        )

    records_by_pair = _build_record_lookup(
        response_matrix,
        evaluation_records,
    )

    configuration_index = {
        configuration_id: index
        for index, configuration_id in enumerate(response_matrix.configuration_ids)
    }

    selected_configuration_ids: list[str] = []
    correct_count = 0
    total_latency = 0.0

    for column, query_id in enumerate(response_matrix.query_ids):
        selected_configuration_id = select_configuration(
            predicted_probabilities=(predicted_probabilities[:, column]),
            configuration_ids=(response_matrix.configuration_ids),
            normalized_costs=normalized_costs,
            performance_weight=performance_weight,
        )

        selected_configuration_ids.append(selected_configuration_id)

        row = configuration_index[selected_configuration_id]

        correct_count += int(response_matrix.values[row, column])

        total_latency += records_by_pair[
            (selected_configuration_id, query_id)
        ].generation.latency_seconds

    query_count = len(response_matrix.query_ids)

    return RoutingEvaluation(
        strategy=f"radar:{performance_weight:g}",
        accuracy=correct_count / query_count,
        average_latency_seconds=(total_latency / query_count),
        selected_configuration_ids=tuple(selected_configuration_ids),
    )


def count_configuration_selections(
    result: RoutingEvaluation,
) -> dict[str, int]:
    """Count how many queries were routed to each configuration."""

    return dict(
        sorted(
            Counter(result.selected_configuration_ids).items(),
        )
    )


def calculate_oracle_accuracy(
    response_matrix: ResponseMatrix,
) -> float:
    """Calculate accuracy if an oracle selected a correct configuration."""

    query_count = len(response_matrix.query_ids)

    if query_count == 0:
        raise ValueError("response matrix cannot contain zero queries")

    solved_count = sum(
        any(
            bool(response_matrix.values[row, column])
            for row in range(len(response_matrix.configuration_ids))
        )
        for column in range(query_count)
    )

    return solved_count / query_count


def select_best_fixed_result(
    fixed_results: Sequence[RoutingEvaluation],
) -> RoutingEvaluation:
    """Select the most accurate fixed configuration, breaking ties by latency."""

    if not fixed_results:
        raise ValueError("fixed_results cannot be empty")

    return min(
        fixed_results,
        key=lambda result: (
            -result.accuracy,
            result.average_latency_seconds,
            result.strategy,
        ),
    )


def compare_routing_results(
    candidate: RoutingEvaluation,
    baseline: RoutingEvaluation,
    response_matrix: ResponseMatrix,
) -> PairedRoutingComparison:
    """Compare two routing strategies query by query."""

    query_count = len(response_matrix.query_ids)

    if len(candidate.selected_configuration_ids) != query_count:
        raise ValueError("candidate selections must match response matrix queries")

    if len(baseline.selected_configuration_ids) != query_count:
        raise ValueError("baseline selections must match response matrix queries")

    configuration_index = {
        configuration_id: row
        for row, configuration_id in enumerate(response_matrix.configuration_ids)
    }

    improved_query_ids: list[str] = []
    regressed_query_ids: list[str] = []
    both_correct_query_ids: list[str] = []
    both_incorrect_query_ids: list[str] = []

    for column, query_id in enumerate(response_matrix.query_ids):
        candidate_configuration = candidate.selected_configuration_ids[column]
        baseline_configuration = baseline.selected_configuration_ids[column]

        if candidate_configuration not in configuration_index:
            raise ValueError(
                f"Unknown candidate configuration: {candidate_configuration}"
            )

        if baseline_configuration not in configuration_index:
            raise ValueError(
                f"Unknown baseline configuration: {baseline_configuration}"
            )

        candidate_correct = bool(
            response_matrix.values[
                configuration_index[candidate_configuration],
                column,
            ]
        )
        baseline_correct = bool(
            response_matrix.values[
                configuration_index[baseline_configuration],
                column,
            ]
        )

        if candidate_correct and not baseline_correct:
            improved_query_ids.append(query_id)
        elif baseline_correct and not candidate_correct:
            regressed_query_ids.append(query_id)
        elif candidate_correct:
            both_correct_query_ids.append(query_id)
        else:
            both_incorrect_query_ids.append(query_id)

    return PairedRoutingComparison(
        candidate_strategy=candidate.strategy,
        baseline_strategy=baseline.strategy,
        improved_query_ids=tuple(improved_query_ids),
        regressed_query_ids=tuple(regressed_query_ids),
        both_correct_query_ids=tuple(both_correct_query_ids),
        both_incorrect_query_ids=tuple(both_incorrect_query_ids),
    )
