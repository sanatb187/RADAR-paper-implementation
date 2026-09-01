import random
from collections.abc import Mapping
from typing import Any, Literal

from datasets import load_dataset  # type: ignore[import-untyped]

from radar_bench.schemas import Query

GPQA_DATASET_ID = "Idavidrein/gpqa"
GPQA_DIAMOND_CONFIG = "gpqa_diamond"
GPQA_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"

REQUIRED_FIELDS = {
    "Record ID",
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
}

AnswerLabel = Literal["A", "B", "C", "D"]

ANSWER_LABELS: tuple[AnswerLabel, ...] = (
    "A",
    "B",
    "C",
    "D",
)


def convert_gpqa_row(
    row: Mapping[str, Any],
    *,
    seed: int = 0,
) -> Query:
    """Convert one GPQA-Diamond row into a RADAR Query."""

    missing_fields = REQUIRED_FIELDS - row.keys()

    if missing_fields:
        raise ValueError(
            "GPQA row is missing required fields: " + ", ".join(sorted(missing_fields))
        )

    record_id = str(row["Record ID"])
    correct_answer = str(row["Correct Answer"])

    answers = [
        (correct_answer, True),
        (str(row["Incorrect Answer 1"]), False),
        (str(row["Incorrect Answer 2"]), False),
        (str(row["Incorrect Answer 3"]), False),
    ]

    # A separate deterministic shuffle is created for each question.
    random_generator = random.Random(f"{seed}:{record_id}")
    random_generator.shuffle(answers)

    choices: tuple[str, ...] = tuple(answer for answer, _ in answers)
    correct_index = next(
        index for index, (_, is_correct) in enumerate(answers) if is_correct
    )
    gold_answer: AnswerLabel = ANSWER_LABELS[correct_index]
    return Query(
        query_id=f"gpqa_diamond::{record_id}",
        prompt=str(row["Question"]),
        choices=choices,
        gold_answer=gold_answer,
        dataset="gpqa_diamond",
        split="train",
    )


def load_gpqa_diamond(
    *,
    seed: int = 0,
    revision: str = GPQA_REVISION,
) -> list[Query]:
    """Load GPQA-Diamond and convert it into RADAR queries."""

    dataset = load_dataset(
        GPQA_DATASET_ID,
        GPQA_DIAMOND_CONFIG,
        split="train",
        revision=revision,
    )

    return [convert_gpqa_row(row, seed=seed) for row in dataset]
