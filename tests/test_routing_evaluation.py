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
    evaluate_random_pair,
    evaluate_routing_sampling,
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


def test_routing_sampling_is_reproducible() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    first_result = evaluate_routing_sampling(
        matrix,
        records,
        random_seed=17,
    )
    second_result = evaluate_routing_sampling(
        matrix,
        records,
        random_seed=17,
    )

    assert first_result == second_result
    assert first_result.strategy == "routing-sampling:17"


def test_routing_sampling_uses_observed_configuration_results() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    result = evaluate_routing_sampling(
        matrix,
        records,
        random_seed=42,
    )

    assert len(result.selected_configuration_ids) == len(matrix.query_ids)
    assert set(result.selected_configuration_ids) <= set(matrix.configuration_ids)

    correct_by_pair = {
        (
            record.generation.configuration_id,
            record.generation.query_id,
        ): record.correct
        for record in records
    }
    latency_by_pair = {
        (
            record.generation.configuration_id,
            record.generation.query_id,
        ): record.generation.latency_seconds
        for record in records
    }

    expected_correct_count = sum(
        correct_by_pair[(configuration_id, query_id)]
        for configuration_id, query_id in zip(
            result.selected_configuration_ids,
            matrix.query_ids,
            strict=True,
        )
    )
    expected_latency = sum(
        latency_by_pair[(configuration_id, query_id)]
        for configuration_id, query_id in zip(
            result.selected_configuration_ids,
            matrix.query_ids,
            strict=True,
        )
    ) / len(matrix.query_ids)

    assert result.accuracy == expected_correct_count / len(matrix.query_ids)
    assert result.average_latency_seconds == expected_latency


def test_routing_supports_custom_strategy_prefix() -> None:
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
        strategy_prefix="classifier",
    )

    assert result.strategy == "classifier:1"
    assert result.selected_configuration_ids == (
        "config-a",
        "config-b",
    )


def test_random_pair_selects_lower_endpoint_at_zero_probability() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    result = evaluate_random_pair(
        matrix,
        records,
        lower_configuration_id="config-a",
        upper_configuration_id="config-b",
        upper_probability=0.0,
        random_seed=17,
    )

    assert result.strategy == "random-pair:0:seed-17"
    assert result.selected_configuration_ids == (
        "config-a",
        "config-a",
    )
    assert result.accuracy == 0.5
    assert result.average_latency_seconds == 1.0


def test_random_pair_selects_upper_endpoint_at_one_probability() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    result = evaluate_random_pair(
        matrix,
        records,
        lower_configuration_id="config-a",
        upper_configuration_id="config-b",
        upper_probability=1.0,
        random_seed=17,
    )

    assert result.strategy == "random-pair:1:seed-17"
    assert result.selected_configuration_ids == (
        "config-b",
        "config-b",
    )
    assert result.accuracy == 0.5
    assert result.average_latency_seconds == 3.0


def test_random_pair_is_reproducible() -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    first_result = evaluate_random_pair(
        matrix,
        records,
        lower_configuration_id="config-a",
        upper_configuration_id="config-b",
        upper_probability=0.5,
        random_seed=17,
    )
    second_result = evaluate_random_pair(
        matrix,
        records,
        lower_configuration_id="config-a",
        upper_configuration_id="config-b",
        upper_probability=0.5,
        random_seed=17,
    )

    assert first_result == second_result
    assert set(first_result.selected_configuration_ids) <= {
        "config-a",
        "config-b",
    }


@pytest.mark.parametrize(
    "upper_probability",
    [-0.1, 1.1],
)
def test_random_pair_rejects_invalid_probability(
    upper_probability: float,
) -> None:
    records = make_records()
    matrix = ResponseMatrix.from_records(records)

    with pytest.raises(
        ValueError,
        match="upper_probability must be between 0 and 1",
    ):
        evaluate_random_pair(
            matrix,
            records,
            lower_configuration_id="config-a",
            upper_configuration_id="config-b",
            upper_probability=upper_probability,
            random_seed=17,
        )
