import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from radar_bench.evaluators.multiple_choice import (
    evaluate_multiple_choice_generation,
)
from radar_bench.ollama_generation import run_ollama_generation
from radar_bench.schemas import (
    EvaluationRecord,
    GenerationResult,
    ModelConfiguration,
    Query,
)

GenerationFunction = Callable[..., GenerationResult]


def select_query_subset(
    queries: Sequence[Query],
    *,
    count: int,
    seed: int,
) -> list[Query]:
    """Select a deterministic subset and return it in stable order."""

    if count <= 0:
        raise ValueError("count must be positive")

    if count > len(queries):
        raise ValueError(
            f"Cannot select {count} queries from {len(queries)} available queries"
        )

    candidates = sorted(
        queries,
        key=lambda query: query.query_id,
    )

    selected = random.Random(seed).sample(
        candidates,
        count,
    )

    return sorted(
        selected,
        key=lambda query: query.query_id,
    )


def load_evaluation_records(
    output_path: Path,
) -> list[EvaluationRecord]:
    """Load existing JSONL evaluation records."""

    if not output_path.exists():
        return []

    records: list[EvaluationRecord] = []
    generation_ids: set[str] = set()

    with output_path.open(encoding="utf-8") as output_file:
        for line_number, line in enumerate(
            output_file,
            start=1,
        ):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                record = EvaluationRecord.model_validate_json(stripped_line)
            except ValidationError as error:
                raise ValueError(
                    f"Invalid evaluation record on line {line_number}"
                ) from error

            generation_id = record.generation.generation_id

            if generation_id in generation_ids:
                raise ValueError(f"Duplicate generation ID: {generation_id}")

            generation_ids.add(generation_id)
            records.append(record)

    return records


def run_gpqa_experiment(
    queries: Sequence[Query],
    configurations: Sequence[ModelConfiguration],
    *,
    output_path: Path,
    run_index: int = 0,
    request_options: Mapping[str, Any] | None = None,
    generation_function: GenerationFunction = run_ollama_generation,
) -> list[EvaluationRecord]:
    """Run configurations over queries and checkpoint results as JSONL."""

    if run_index < 0:
        raise ValueError("run_index cannot be negative")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_evaluation_records(output_path)

    completed_generation_ids = {record.generation.generation_id for record in records}

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as output_file:
        for configuration in configurations:
            for query in queries:
                generation_id = (
                    f"{query.query_id}__"
                    f"{configuration.configuration_id}__"
                    f"run-{run_index}"
                )

                if generation_id in completed_generation_ids:
                    continue

                generation = generation_function(
                    query=query,
                    configuration=configuration,
                    run_index=run_index,
                    request_options=request_options,
                )

                evaluation = evaluate_multiple_choice_generation(
                    query=query,
                    generation=generation,
                )

                output_file.write(evaluation.model_dump_json())
                output_file.write("\n")
                output_file.flush()

                records.append(evaluation)
                completed_generation_ids.add(generation_id)

    return records
