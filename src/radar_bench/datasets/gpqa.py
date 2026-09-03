import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from datasets import load_dataset  # type: ignore[import-untyped]

from radar_bench.schemas import Query

GPQA_DATASET_ID = "Idavidrein/gpqa"
GPQA_DIAMOND_CONFIG = "gpqa_diamond"
GPQA_MAIN_CONFIG = "gpqa_main"
GPQA_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
GPQA_DIAMOND_TRAIN_FRACTION = 0.8
GPQASplit = Literal["train", "test"]

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


@dataclass(frozen=True)
class GPQASplits:
    train: tuple[Query, ...]
    test: tuple[Query, ...]


def convert_gpqa_row(
    row: Mapping[str, Any],
    *,
    seed: int = 0,
    split: GPQASplit = "train",
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
        query_id=f"gpqa::{record_id}",
        prompt=str(row["Question"]),
        choices=choices,
        gold_answer=gold_answer,
        dataset="gpqa",
        split=split,
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

    return [convert_gpqa_row(row, seed=seed, split="test") for row in dataset]


def load_gpqa_diamond_splits(
    *,
    seed: int = 0,
    revision: str = GPQA_REVISION,
    train_fraction: float = GPQA_DIAMOND_TRAIN_FRACTION,
) -> GPQASplits:
    """Split GPQA-Diamond into deterministic train and test sets."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    diamond_rows = list(
        load_dataset(
            GPQA_DATASET_ID,
            GPQA_DIAMOND_CONFIG,
            split="train",
            revision=revision,
        )
    )

    if len(diamond_rows) < 2:
        raise ValueError("GPQA-Diamond must contain at least two records")

    record_ids = [str(row["Record ID"]) for row in diamond_rows]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError("GPQA-Diamond contains duplicate record IDs")

    # Sorting makes the split stable even if dataset iteration order changes.
    shuffled_rows = sorted(
        diamond_rows,
        key=lambda row: str(row["Record ID"]),
    )

    random_generator = random.Random(seed)
    random_generator.shuffle(shuffled_rows)

    train_size = int(len(shuffled_rows) * train_fraction)
    train_size = max(
        1,
        min(train_size, len(shuffled_rows) - 1),
    )

    train_rows = shuffled_rows[:train_size]
    test_rows = shuffled_rows[train_size:]

    train_queries = tuple(
        convert_gpqa_row(
            row,
            seed=seed,
            split="train",
        )
        for row in train_rows
    )

    test_queries = tuple(
        convert_gpqa_row(
            row,
            seed=seed,
            split="test",
        )
        for row in test_rows
    )

    return GPQASplits(
        train=train_queries,
        test=test_queries,
    )


def load_gpqa_splits(
    *,
    seed: int = 0,
    revision: str = GPQA_REVISION,
) -> GPQASplits:
    """Load non-Diamond GPQA as train and Diamond as test."""

    main_rows = list(
        load_dataset(
            GPQA_DATASET_ID, GPQA_MAIN_CONFIG, split="train", revision=revision
        )
    )

    diamond_rows = list(
        load_dataset(
            GPQA_DATASET_ID,
            GPQA_DIAMOND_CONFIG,
            split="train",
            revision=revision,
        )
    )

    main_ids = {str(row["Record ID"]) for row in main_rows}
    diamond_ids = {str(row["Record ID"]) for row in diamond_rows}

    missing_diamond_ids = diamond_ids - main_ids

    if missing_diamond_ids:
        raise ValueError(
            f"{len(missing_diamond_ids)} Diamond records were not found in GPQA Main"
        )

    training_rows = [
        row for row in main_rows if str(row["Record ID"]) not in diamond_ids
    ]

    train_queries = tuple(
        convert_gpqa_row(
            row,
            seed=seed,
            split="train",
        )
        for row in training_rows
    )

    test_queries = tuple(
        convert_gpqa_row(
            row,
            seed=seed,
            split="test",
        )
        for row in diamond_rows
    )

    return GPQASplits(
        train=train_queries,
        test=test_queries,
    )
