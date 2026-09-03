import pytest

from radar_bench.cost import (
    estimate_configuration_latency_costs,
    normalize_costs,
)
from radar_bench.schemas import (
    EvaluationRecord,
    GenerationResult,
    TokenUsage,
)


def make_record(
    configuration_id: str,
    query_id: str,
    latency_seconds: float,
) -> EvaluationRecord:
    return EvaluationRecord(
        generation=GenerationResult(
            generation_id=(f"{query_id}__{configuration_id}__run-0"),
            query_id=query_id,
            configuration_id=configuration_id,
            response_text=r"\boxed{B}",
            reasoning_text=None,
            token_usage=TokenUsage(
                prompt_tokens=10,
                reasoning_tokens=0,
                completion_tokens=5,
            ),
            latency_seconds=latency_seconds,
        ),
        parsed_answer="B",
        correct=True,
    )


def test_estimates_average_latency_costs() -> None:
    records = [
        make_record("config-a", "query-1", 1.0),
        make_record("config-a", "query-2", 3.0),
        make_record("config-b", "query-1", 4.0),
        make_record("config-b", "query-2", 4.0),
    ]

    costs = estimate_configuration_latency_costs(
        records,
        ("config-a", "config-b"),
    )

    assert costs == {
        "config-a": 2.0,
        "config-b": 4.0,
    }

    assert normalize_costs(costs) == {
        "config-a": 0.0,
        "config-b": 1.0,
    }


def test_rejects_missing_configuration_records() -> None:
    records = [
        make_record("config-a", "query-1", 1.0),
    ]

    with pytest.raises(
        ValueError,
        match="config-b",
    ):
        estimate_configuration_latency_costs(
            records,
            ("config-a", "config-b"),
        )


def test_rejects_unknown_configuration() -> None:
    records = [
        make_record("config-c", "query-1", 1.0),
    ]

    with pytest.raises(
        ValueError,
        match="Unknown configuration ID",
    ):
        estimate_configuration_latency_costs(
            records,
            ("config-a", "config-b"),
        )
