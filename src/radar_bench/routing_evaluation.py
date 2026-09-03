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
