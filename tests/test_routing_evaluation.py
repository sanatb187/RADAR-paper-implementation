import pytest
import torch

from radar_bench.response_matrix import ResponseMatrix
from radar_bench.routing_evaluation import (
    RoutingEvaluation,
    calculate_oracle_accuracy,
    compare_routing_results,
    count_configuration_selections,
    evaluate_fixed_configurations,
    evaluate_radar_routing,
    select_best_fixed_result,
)
from radar_bench.schemas import (
    EvaluationRecord,
    GenerationResult,
    TokenUsage,
)


def make_record(
    configuration_id: str,
    query_id: str,
    *,
    correct: bool,
    latency_seconds: float,
) -> EvaluationRecord:
    return EvaluationRecord(
        generation=GenerationResult(
            generation_id=(f"{query_id}__{configuration_id}__run-0"),
            query_id=query_id,
            configuration_id=configuration_id,
            response_text=(r"\boxed{B}" if correct else r"\boxed{A}"),
            reasoning_text=None,
            token_usage=TokenUsage(
                prompt_tokens=10,
                reasoning_tokens=10,
                completion_tokens=5,
            ),
            latency_seconds=latency_seconds,
        ),
        parsed_answer="B" if correct else "A",
        correct=correct,
    )


def make_records() -> list[EvaluationRecord]:
    return [
        make_record(
            "config-a",
            "query-1",
            correct=True,
            latency_seconds=1.0,
        ),
        make_record(
            "config-a",
            "query-2",
            correct=False,
            latency_seconds=1.0,
        ),
        make_record(
            "config-b",
            "query-1",
            correct=False,
            latency_seconds=3.0,
        ),
        make_record(
            "config-b",
            "query-2",
            correct=True,
            latency_seconds=3.0,
        ),
    ]


def test_evaluates_fixed_configurations() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    results = evaluate_fixed_configurations(
        matrix,
        records,
    )

    assert len(results) == 2

    assert results[0].strategy == "fixed:config-a"
    assert results[0].accuracy == 0.5
    assert results[0].average_latency_seconds == 1.0

    assert results[1].strategy == "fixed:config-b"
    assert results[1].accuracy == 0.5
    assert results[1].average_latency_seconds == 3.0


def test_radar_selects_best_configuration_per_query() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    predicted_probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )

    result = evaluate_radar_routing(
        predicted_probabilities,
        matrix,
        records,
        {
            "config-a": 0.0,
            "config-b": 1.0,
        },
        performance_weight=1.0,
    )

    assert result.strategy == "radar:1"
    assert result.accuracy == 1.0
    assert result.average_latency_seconds == 2.0
    assert result.selected_configuration_ids == (
        "config-a",
        "config-b",
    )


def test_radar_can_prefer_lower_cost() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    predicted_probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )

    result = evaluate_radar_routing(
        predicted_probabilities,
        matrix,
        records,
        {
            "config-a": 0.0,
            "config-b": 1.0,
        },
        performance_weight=0.0,
    )

    assert result.accuracy == 0.5
    assert result.average_latency_seconds == 1.0
    assert result.selected_configuration_ids == (
        "config-a",
        "config-a",
    )


def test_rejects_wrong_probability_shape() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    with pytest.raises(
        ValueError,
        match="shape must match",
    ):
        evaluate_radar_routing(
            torch.tensor([0.5, 0.5]),
            matrix,
            records,
            {
                "config-a": 0.0,
                "config-b": 1.0,
            },
            performance_weight=0.5,
        )


def test_counts_configuration_selections() -> None:
    result = RoutingEvaluation(
        strategy="radar:0.5",
        accuracy=0.5,
        average_latency_seconds=2.0,
        selected_configuration_ids=(
            "config-b",
            "config-a",
            "config-b",
        ),
    )

    assert count_configuration_selections(result) == {
        "config-a": 1,
        "config-b": 2,
    }


def test_calculates_oracle_accuracy() -> None:
    matrix = ResponseMatrix.from_records(make_records())

    assert calculate_oracle_accuracy(matrix) == 1.0


def test_selects_best_fixed_result_by_accuracy_then_latency() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    results = evaluate_fixed_configurations(
        matrix,
        records,
    )

    best_result = select_best_fixed_result(results)

    assert best_result.strategy == "fixed:config-a"
    assert best_result.accuracy == 0.5
    assert best_result.average_latency_seconds == 1.0


def test_compares_routing_results_query_by_query() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    fixed_results = evaluate_fixed_configurations(
        matrix,
        records,
    )

    radar_result = evaluate_radar_routing(
        torch.tensor(
            [
                [0.9, 0.1],
                [0.1, 0.9],
            ]
        ),
        matrix,
        records,
        {
            "config-a": 0.0,
            "config-b": 1.0,
        },
        performance_weight=1.0,
    )

    comparison = compare_routing_results(
        radar_result,
        fixed_results[0],
        matrix,
    )

    assert comparison.candidate_strategy == "radar:1"
    assert comparison.baseline_strategy == "fixed:config-a"
    assert comparison.improved_query_ids == ("query-2",)
    assert comparison.regressed_query_ids == ()
    assert comparison.both_correct_query_ids == ("query-1",)
    assert comparison.both_incorrect_query_ids == ()


def test_radar_supports_chebyshev_scalarization() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    result = evaluate_radar_routing(
        torch.tensor(
            [
                [0.9, 0.1],
                [0.1, 0.9],
            ]
        ),
        matrix,
        records,
        {
            "config-a": 0.0,
            "config-b": 1.0,
        },
        performance_weight=1.0,
        scalarization="chebyshev",
    )

    assert result.strategy == "radar-chebyshev:1"
    assert result.accuracy == 1.0
    assert result.selected_configuration_ids == (
        "config-a",
        "config-b",
    )
