from typing import Any

import pytest

from radar_bench.generation import run_litellm_generation
from radar_bench.schemas import (
    EffortBudget,
    ModelConfiguration,
    ModelSpec,
    Query,
    TokenBudget,
)


def make_query() -> Query:
    return Query(
        query_id="query-1",
        prompt="Which option is correct?",
        choices=("One", "Two", "Three", "Four"),
        gold_answer="B",
        dataset="test",
        split="test",
    )


def make_effort_configuration() -> ModelConfiguration:
    return ModelConfiguration(
        configuration_id="model-a__effort-high",
        model_spec=ModelSpec(
            model_id="model-a",
            litellm_model="openai/model-a",
        ),
        reasoning_budget=EffortBudget(value="high"),
    )


def fake_completion(**kwargs: Any) -> dict[str, Any]:
    del kwargs

    return {
        "choices": [
            {
                "message": {
                    "content": r"\boxed{B}",
                    "reasoning_content": "The second option is correct.",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "completion_tokens_details": {
                "reasoning_tokens": 80,
            },
        },
    }


def test_run_litellm_generation() -> None:
    captured_kwargs: dict[str, Any] = {}

    def capture_completion(**kwargs: Any) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        return fake_completion()

    generation = run_litellm_generation(
        query=make_query(),
        configuration=make_effort_configuration(),
        run_index=2,
        completion_function=capture_completion,
    )

    assert captured_kwargs["model"] == "openai/model-a"
    assert captured_kwargs["reasoning_effort"] == "high"

    messages = captured_kwargs["messages"]

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Which option is correct?" in messages[0]["content"]

    assert generation.generation_id == ("query-1__model-a__effort-high__run-2")
    assert generation.query_id == "query-1"
    assert generation.configuration_id == ("model-a__effort-high")
    assert generation.response_text == r"\boxed{B}"
    assert generation.reasoning_text == ("The second option is correct.")

    assert generation.token_usage.prompt_tokens == 100
    assert generation.token_usage.reasoning_tokens == 80
    assert generation.token_usage.completion_tokens == 20
    assert generation.token_usage.output_tokens == 100
    assert generation.latency_seconds >= 0


def test_forwards_additional_request_kwargs() -> None:
    captured_kwargs: dict[str, Any] = {}

    def capture_completion(**kwargs: Any) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        return fake_completion()

    run_litellm_generation(
        query=make_query(),
        configuration=make_effort_configuration(),
        request_kwargs={
            "temperature": 0.0,
            "timeout": 60,
        },
        completion_function=capture_completion,
    )

    assert captured_kwargs["temperature"] == 0.0
    assert captured_kwargs["timeout"] == 60


def test_rejects_reserved_request_fields() -> None:
    with pytest.raises(
        ValueError,
        match="reserved fields",
    ):
        run_litellm_generation(
            query=make_query(),
            configuration=make_effort_configuration(),
            request_kwargs={
                "model": "different-model",
            },
            completion_function=fake_completion,
        )


def test_rejects_token_budget() -> None:
    configuration = ModelConfiguration(
        configuration_id="qwen__tokens-512",
        model_spec=ModelSpec(
            model_id="qwen",
            litellm_model="openai/qwen",
        ),
        reasoning_budget=TokenBudget(value=512),
    )

    with pytest.raises(
        TypeError,
        match="TokenBudget requires a provider-specific runner",
    ):
        run_litellm_generation(
            query=make_query(),
            configuration=configuration,
            completion_function=fake_completion,
        )


def test_rejects_response_without_choices() -> None:
    def empty_completion(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"choices": [], "usage": {}}

    with pytest.raises(
        ValueError,
        match="contains no choices",
    ):
        run_litellm_generation(
            query=make_query(),
            configuration=make_effort_configuration(),
            completion_function=empty_completion,
        )


def test_rejects_response_without_usage() -> None:
    def no_usage_completion(**kwargs: Any) -> dict[str, Any]:
        del kwargs

        return {
            "choices": [
                {
                    "message": {
                        "content": r"\boxed{B}",
                    }
                }
            ]
        }

    with pytest.raises(
        ValueError,
        match="contains no token usage",
    ):
        run_litellm_generation(
            query=make_query(),
            configuration=make_effort_configuration(),
            completion_function=no_usage_completion,
        )
