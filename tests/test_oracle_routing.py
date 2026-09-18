import torch

from radar_bench.response_matrix import ResponseMatrix
from radar_bench.routing_evaluation import (
    calculate_oracle_accuracy,
    evaluate_radar_routing,
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
) -> EvaluationRecord:
    return EvaluationRecord(
        generation=GenerationResult(
            generation_id=f"{query_id}__{configuration_id}__run-0",
            query_id=query_id,
            configuration_id=configuration_id,
            response_text=r"\boxed{B}" if correct else r"\boxed{A}",
            token_usage=TokenUsage(
                prompt_tokens=10,
                reasoning_tokens=10,
                completion_tokens=2,
            ),
            latency_seconds=1.0,
        ),
        parsed_answer="B" if correct else "A",
        correct=correct,
    )


def test_oracle_probabilities_reach_oracle_accuracy() -> None:
    records = [
        make_record("config-a", "query-1", correct=True),
        make_record("config-a", "query-2", correct=False),
        make_record("config-b", "query-1", correct=False),
        make_record("config-b", "query-2", correct=True),
    ]

    matrix = ResponseMatrix.from_records(records)

    result = evaluate_radar_routing(
        predicted_probabilities=torch.from_numpy(matrix.values).float(),
        response_matrix=matrix,
        evaluation_records=records,
        normalized_costs={
            "config-a": 0.0,
            "config-b": 1.0,
        },
        performance_weight=1.0,
        scalarization="chebyshev",
    )

    assert result.accuracy == calculate_oracle_accuracy(matrix)
    assert result.accuracy == 1.0
