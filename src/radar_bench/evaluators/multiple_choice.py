import re
from typing import Literal, cast

from radar_bench.schemas import (
    EvaluationRecord,
    GenerationResult,
    Query,
)

AnswerLabel = Literal["A", "B", "C", "D"]

ANSWER_LABELS: tuple[AnswerLabel, ...] = (
    "A",
    "B",
    "C",
    "D",
)

BOXED_ANSWER_PATTERN = re.compile(
    r"\\boxed\s*\{\s*([A-D])\s*\}",
    re.IGNORECASE,
)


def format_multiple_choice_question(
    query: Query,
) -> str:
    """Format a question without answer instructions."""

    if len(query.choices) != 4:
        raise ValueError("The RADAR multiple-choice prompt requires exactly 4 choices")

    option_lines = [
        f"{label}) {choice}"
        for label, choice in zip(
            ANSWER_LABELS,
            query.choices,
            strict=True,
        )
    ]

    return "\n".join(
        [
            "Answer the following multiple choice question.",
            query.prompt,
            *option_lines,
        ]
    )


def format_multiple_choice_prompt(query: Query) -> str:
    """Format a four-choice question using the RADAR prompt."""

    return "\n".join(
        [
            format_multiple_choice_question(query),
            (
                "Please reason step by step, and put your final "
                r"answer option within \boxed{}."
            ),
            (
                r"Only put the letter in the box, e.g. \boxed{A}. "
                "There is only one correct answer."
            ),
        ]
    )


def parse_boxed_answer(response_text: str) -> AnswerLabel | None:
    """Extract the final boxed A-D answer from a response."""

    matches = BOXED_ANSWER_PATTERN.findall(response_text)

    if not matches:
        return None

    return cast(AnswerLabel, matches[-1].upper())


def evaluate_multiple_choice_generation(
    query: Query,
    generation: GenerationResult,
) -> EvaluationRecord:
    """Evaluate a generation against its query's gold answer."""

    if generation.query_id != query.query_id:
        raise ValueError("Generation query_id does not match Query query_id")

    parsed_answer = parse_boxed_answer(generation.response_text)

    return EvaluationRecord(
        generation=generation,
        parsed_answer=parsed_answer,
        correct=parsed_answer == query.gold_answer,
    )
