import pytest

from radar_bench.evaluators.multiple_choice import (
    evaluate_multiple_choice_generation,
    format_multiple_choice_prompt,
    parse_boxed_answer,
)
from radar_bench.schemas import (
    GenerationResult,
    Query,
    TokenUsage,
)


def make_query() -> Query:
    return Query(
        query_id="sample-query",
        prompt="Which option is correct?",
        choices=(
            "First",
            "Second",
            "Third",
            "Fourth",
        ),
        gold_answer="B",
        dataset="test",
        split="test",
    )


def make_generation(
    response_text: str,
    *,
    query_id: str = "sample-query",
) -> GenerationResult:
    return GenerationResult(
        generation_id="sample-generation",
        query_id=query_id,
        configuration_id="sample-configuration",
        response_text=response_text,
        reasoning_text=None,
        token_usage=TokenUsage(
            prompt_tokens=100,
            reasoning_tokens=20,
            completion_tokens=5,
        ),
        latency_seconds=1.0,
        run_index=0,
    )


def test_format_multiple_choice_prompt() -> None:
    prompt = format_multiple_choice_prompt(make_query())

    expected = (
        "Answer the following multiple choice question.\n"
        "Which option is correct?\n"
        "A) First\n"
        "B) Second\n"
        "C) Third\n"
        "D) Fourth\n"
        "Please reason step by step, and put your final "
        "answer option within \\boxed{}.\n"
        "Only put the letter in the box, e.g. \\boxed{A}. "
        "There is only one correct answer."
    )

    assert prompt == expected


@pytest.mark.parametrize(
    ("response_text", "expected"),
    [
        (r"The answer is \boxed{A}.", "A"),
        (r"My final answer is \boxed{b}.", "B"),
        (r"The answer is \boxed{ C }.", "C"),
        ("There is no boxed answer.", None),
        (r"The answer is \boxed{E}.", None),
    ],
)
def test_parse_boxed_answer(
    response_text: str,
    expected: str | None,
) -> None:
    assert parse_boxed_answer(response_text) == expected


def test_parser_uses_last_boxed_answer() -> None:
    response = (
        r"I first considered \boxed{A}, "
        r"but the final answer is \boxed{B}."
    )

    assert parse_boxed_answer(response) == "B"


def test_correct_generation() -> None:
    generation = make_generation(r"After reasoning, the answer is \boxed{B}.")

    evaluation = evaluate_multiple_choice_generation(
        make_query(),
        generation,
    )

    assert evaluation.parsed_answer == "B"
    assert evaluation.correct is True


def test_incorrect_generation() -> None:
    generation = make_generation(r"After reasoning, the answer is \boxed{C}.")

    evaluation = evaluate_multiple_choice_generation(
        make_query(),
        generation,
    )

    assert evaluation.parsed_answer == "C"
    assert evaluation.correct is False


def test_missing_answer_is_incorrect() -> None:
    generation = make_generation("I could not determine the answer.")

    evaluation = evaluate_multiple_choice_generation(
        make_query(),
        generation,
    )

    assert evaluation.parsed_answer is None
    assert evaluation.correct is False


def test_rejects_mismatched_query_id() -> None:
    generation = make_generation(
        r"\boxed{B}",
        query_id="different-query",
    )

    with pytest.raises(
        ValueError,
        match="query_id does not match",
    ):
        evaluate_multiple_choice_generation(
            make_query(),
            generation,
        )
