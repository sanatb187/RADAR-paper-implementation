from collections.abc import Sequence

import torch

from radar_bench.radar_evaluation import (
    evaluate_radar_experiment,
)
from radar_bench.schemas import (
    EvaluationRecord,
    GenerationResult,
    Query,
    TokenUsage,
)


def make_query(
    query_id: str,
    split: str,
) -> Query:
    return Query(
        query_id=query_id,
        prompt=f"Question {query_id}?",
        choices=("One", "Two", "Three", "Four"),
        gold_answer="B",
        dataset="test",
        split=split,
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


def fake_embeddings(
    queries: Sequence[Query],
) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0] if query.query_id.endswith("1") else [0.0, 1.0]
            for query in queries
        ]
    )


def test_evaluates_radar_experiment() -> None:
    train_queries = [
        make_query("train-2", "train"),
        make_query("train-1", "train"),
    ]
    test_queries = [
        make_query("test-2", "test"),
        make_query("test-1", "test"),
    ]

    train_records = [
        make_record(
            "config-a",
            "train-1",
            correct=True,
            latency_seconds=1.0,
        ),
        make_record(
            "config-a",
            "train-2",
            correct=False,
            latency_seconds=1.0,
        ),
        make_record(
            "config-b",
            "train-1",
            correct=False,
            latency_seconds=3.0,
        ),
        make_record(
            "config-b",
            "train-2",
            correct=True,
            latency_seconds=3.0,
        ),
    ]

    test_records = [
        make_record(
            "config-a",
            "test-1",
            correct=True,
            latency_seconds=1.0,
        ),
        make_record(
            "config-a",
            "test-2",
            correct=False,
            latency_seconds=1.0,
        ),
        make_record(
            "config-b",
            "test-1",
            correct=False,
            latency_seconds=3.0,
        ),
        make_record(
            "config-b",
            "test-2",
            correct=True,
            latency_seconds=3.0,
        ),
    ]

    report = evaluate_radar_experiment(
        train_queries,
        test_queries,
        train_records,
        test_records,
        performance_weights=(0.0, 1.0),
        num_epochs=20,
        embedding_function=fake_embeddings,
    )

    assert len(report.training_loss_history) == 20

    assert report.normalized_costs == {
        "config-a": 0.0,
        "config-b": 1.0,
    }

    assert len(report.fixed_results) == 2
    assert len(report.radar_results) == 2

    lowest_cost_result = report.radar_results[0]

    assert lowest_cost_result.strategy == "radar:0"
    assert lowest_cost_result.selected_configuration_ids == ("config-a", "config-a")
    assert lowest_cost_result.accuracy == 0.5
    assert lowest_cost_result.average_latency_seconds == 1.0
