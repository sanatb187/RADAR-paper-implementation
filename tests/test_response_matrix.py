import numpy as np
import pytest

from radar_bench.response_matrix import ResponseMatrix
from radar_bench.schemas import (
    EvaluationRecord,
    GenerationResult,
    TokenUsage,
)


def make_evaluation_record(
    configuration_id: str,
    query_id: str,
    correct: bool,
    *,
    generation_id: str | None = None,
) -> EvaluationRecord:
    generation = GenerationResult(
        generation_id=generation_id or f"{configuration_id}__{query_id}",
        query_id=query_id,
        configuration_id=configuration_id,
        response_text="Example response",
        token_usage=TokenUsage(
            prompt_tokens=10,
            reasoning_tokens=20,
            completion_tokens=5,
        ),
        latency_seconds=1.0,
    )

    return EvaluationRecord(
        generation=generation,
        parsed_answer="A",
        correct=correct,
    )


def test_builds_complete_response_matrix() -> None:
    records = [
        make_evaluation_record("config-b", "query-2", False),
        make_evaluation_record("config-a", "query-1", True),
        make_evaluation_record("config-b", "query-1", True),
        make_evaluation_record("config-a", "query-3", False),
        make_evaluation_record("config-a", "query-2", True),
        make_evaluation_record("config-b", "query-3", True),
    ]

    response_matrix = ResponseMatrix.from_records(records)

    assert response_matrix.configuration_ids == (
        "config-a",
        "config-b",
    )

    assert response_matrix.query_ids == (
        "query-1",
        "query-2",
        "query-3",
    )

    expected_values = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        response_matrix.values,
        expected_values,
    )


def test_matrix_uses_int8_values() -> None:
    records = [
        make_evaluation_record("config-a", "query-1", True),
    ]

    response_matrix = ResponseMatrix.from_records(records)

    assert response_matrix.values.dtype == np.int8


def test_rejects_empty_record_list() -> None:
    with pytest.raises(
        ValueError,
        match="evaluation_records cannot be empty",
    ):
        ResponseMatrix.from_records([])


def test_rejects_duplicate_pairs() -> None:
    records = [
        make_evaluation_record(
            "config-a",
            "query-1",
            True,
            generation_id="generation-1",
        ),
        make_evaluation_record(
            "config-a",
            "query-1",
            False,
            generation_id="generation-2",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate evaluation record",
    ):
        ResponseMatrix.from_records(records)


def test_rejects_missing_pairs() -> None:
    records = [
        make_evaluation_record("config-a", "query-1", True),
        make_evaluation_record("config-a", "query-2", False),
        make_evaluation_record("config-b", "query-1", True),
    ]

    with pytest.raises(
        ValueError,
        match="Missing evaluation records",
    ):
        ResponseMatrix.from_records(records)
