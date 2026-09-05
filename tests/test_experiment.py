from pathlib import Path
from typing import Any

import pytest

from radar_bench.experiment import (
    load_evaluation_records,
    run_gpqa_experiment,
    select_query_subset,
)
from radar_bench.schemas import (
    GenerationResult,
    ModelConfiguration,
    ModelSpec,
    Query,
    TokenBudget,
    TokenUsage,
)


def make_query(index: int) -> Query:
    return Query(
        query_id=f"query-{index:02d}",
        prompt=f"Question {index}?",
        choices=("One", "Two", "Three", "Four"),
        gold_answer="B",
        dataset="test",
        split="test",
    )


def make_configuration(
    budget: int,
) -> ModelConfiguration:
    return ModelConfiguration(
        configuration_id=f"qwen3-0.6b__tokens-{budget}",
        model_spec=ModelSpec(
            model_id="qwen3-0.6b",
            litellm_model="ollama/qwen3:0.6b",
        ),
        reasoning_budget=TokenBudget(value=budget),
    )


def make_generation(**kwargs: Any) -> GenerationResult:
    query: Query = kwargs["query"]
    configuration: ModelConfiguration = kwargs["configuration"]
    run_index: int = kwargs["run_index"]

    return GenerationResult(
        generation_id=(
            f"{query.query_id}__{configuration.configuration_id}__run-{run_index}"
        ),
        query_id=query.query_id,
        configuration_id=configuration.configuration_id,
        response_text=r"\boxed{B}",
        reasoning_text="Test reasoning",
        token_usage=TokenUsage(
            prompt_tokens=10,
            reasoning_tokens=20,
            completion_tokens=5,
        ),
        latency_seconds=0.1,
        run_index=run_index,
    )


def test_select_query_subset_is_deterministic() -> None:
    queries = [make_query(index) for index in range(10)]

    first = select_query_subset(
        queries,
        count=4,
        seed=42,
    )
    second = select_query_subset(
        list(reversed(queries)),
        count=4,
        seed=42,
    )

    assert first == second

    selected_ids = [query.query_id for query in first]

    assert selected_ids == sorted(selected_ids)
    assert len(selected_ids) == 4
    assert len(set(selected_ids)) == 4


def test_rejects_invalid_subset_count() -> None:
    queries = [make_query(0)]

    with pytest.raises(
        ValueError,
        match="count must be positive",
    ):
        select_query_subset(
            queries,
            count=0,
            seed=42,
        )

    with pytest.raises(
        ValueError,
        match="Cannot select",
    ):
        select_query_subset(
            queries,
            count=2,
            seed=42,
        )


def test_run_experiment_saves_and_resumes(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "records.jsonl"
    calls: list[str] = []

    def capture_generation(**kwargs: Any) -> GenerationResult:
        configuration: ModelConfiguration = kwargs["configuration"]
        calls.append(configuration.configuration_id)
        return make_generation(**kwargs)

    queries = [make_query(0)]
    configurations = [
        make_configuration(0),
        make_configuration(256),
    ]

    records = run_gpqa_experiment(
        queries,
        configurations,
        output_path=output_path,
        generation_function=capture_generation,
    )

    assert len(records) == 2
    assert all(record.correct for record in records)
    assert len(calls) == 2

    saved_records = load_evaluation_records(output_path)

    assert saved_records == records

    calls.clear()

    resumed_records = run_gpqa_experiment(
        queries,
        configurations,
        output_path=output_path,
        generation_function=capture_generation,
    )

    assert resumed_records == records
    assert calls == []


def test_resume_after_interrupted_run(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "records.jsonl"
    queries = [make_query(0)]
    configurations = [
        make_configuration(0),
        make_configuration(256),
    ]

    attempted_calls = 0

    def interrupted_generation(
        **kwargs: Any,
    ) -> GenerationResult:
        nonlocal attempted_calls
        attempted_calls += 1

        if attempted_calls == 2:
            raise RuntimeError("Simulated interruption")

        return make_generation(**kwargs)

    with pytest.raises(
        RuntimeError,
        match="Simulated interruption",
    ):
        run_gpqa_experiment(
            queries,
            configurations,
            output_path=output_path,
            generation_function=interrupted_generation,
        )

    saved_records = load_evaluation_records(output_path)

    assert len(saved_records) == 1

    resumed_calls: list[str] = []

    def resumed_generation(
        **kwargs: Any,
    ) -> GenerationResult:
        configuration: ModelConfiguration = kwargs["configuration"]
        resumed_calls.append(configuration.configuration_id)
        return make_generation(**kwargs)

    records = run_gpqa_experiment(
        queries,
        configurations,
        output_path=output_path,
        generation_function=resumed_generation,
    )

    assert len(records) == 2
    assert resumed_calls == ["qwen3-0.6b__tokens-256"]


def test_rejects_invalid_jsonl(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "records.jsonl"
    output_path.write_text(
        "not valid JSON\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid evaluation record on line 1",
    ):
        load_evaluation_records(output_path)


def test_sequential_configuration_runs_share_output(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "records.jsonl"
    queries = [make_query(0)]
    calls: list[str] = []

    def capture_generation(**kwargs: Any) -> GenerationResult:
        configuration: ModelConfiguration = kwargs["configuration"]
        calls.append(configuration.configuration_id)
        return make_generation(**kwargs)

    first_records = run_gpqa_experiment(
        queries,
        [make_configuration(0)],
        output_path=output_path,
        generation_function=capture_generation,
    )

    assert len(first_records) == 1
    assert calls == ["qwen3-0.6b__tokens-0"]

    calls.clear()

    combined_records = run_gpqa_experiment(
        queries,
        [make_configuration(256)],
        output_path=output_path,
        generation_function=capture_generation,
    )

    assert len(combined_records) == 2
    assert calls == ["qwen3-0.6b__tokens-256"]

    saved_configuration_ids = {
        record.generation.configuration_id
        for record in load_evaluation_records(output_path)
    }

    assert saved_configuration_ids == {
        "qwen3-0.6b__tokens-0",
        "qwen3-0.6b__tokens-256",
    }
