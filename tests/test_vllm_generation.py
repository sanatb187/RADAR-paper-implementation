from typing import Any

import pytest

from radar_bench.schemas import (
    EffortBudget,
    ModelConfiguration,
    ModelSpec,
    Query,
    TokenBudget,
)
from radar_bench.vllm_generation import run_vllm_generation


def make_query() -> Query:
    return Query(
        query_id="query-1",
        prompt="Which option is correct?",
        choices=("One", "Two", "Three", "Four"),
        gold_answer="B",
        dataset="test",
        split="test",
    )


def make_configuration(budget: int) -> ModelConfiguration:
    return ModelConfiguration(
        configuration_id=f"qwen3-4b-awq__tokens-{budget}",
        model_spec=ModelSpec(
            model_id="qwen3-4b-awq",
            litellm_model="openai/qwen3-4b-awq",
        ),
        reasoning_budget=TokenBudget(value=budget),
    )


def test_zero_budget_uses_one_request() -> None:
    captured_requests: list[dict[str, Any]] = []

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        captured_requests.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )

        return {
            "choices": [
                {
                    "message": {
                        "content": "B",
                        "reasoning": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 47,
                "completion_tokens": 2,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                },
            },
        }

    generation = run_vllm_generation(
        query=make_query(),
        configuration=make_configuration(0),
        base_url="http://127.0.0.1:8000/v1",
        final_answer_max_tokens=16,
        request_options={"seed": 42},
        post_function=fake_post,
    )

    assert len(captured_requests) == 1

    request = captured_requests[0]

    assert request["url"] == ("http://127.0.0.1:8000/v1/chat/completions")

    payload = request["json"]

    assert payload["model"] == "qwen3-4b-awq"
    assert payload["max_tokens"] == 16
    assert payload["stream"] is False
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
    }
    assert payload["structured_outputs"] == {
        "choice": ["A", "B", "C", "D"],
    }

    assert generation.response_text == r"\boxed{B}"
    assert generation.reasoning_text is None
    assert generation.token_usage.prompt_tokens == 47
    assert generation.token_usage.reasoning_tokens == 0
    assert generation.token_usage.completion_tokens == 2


def test_positive_budget_uses_reasoning_then_answer() -> None:
    captured_requests: list[dict[str, Any]] = []

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        captured_requests.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )

        if len(captured_requests) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "\n\nB",
                            "reasoning": "The second option is correct.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 43,
                    "completion_tokens": 244,
                    "completion_tokens_details": {
                        "reasoning_tokens": 239,
                    },
                },
            }

        return {
            "choices": [
                {
                    "message": {
                        "content": "B",
                        "reasoning": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 302,
                "completion_tokens": 2,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                },
            },
        }

    generation = run_vllm_generation(
        query=make_query(),
        configuration=make_configuration(256),
        base_url="http://127.0.0.1:8000/v1",
        final_answer_max_tokens=16,
        request_options={"seed": 42},
        post_function=fake_post,
    )

    assert len(captured_requests) == 2

    reasoning_payload = captured_requests[0]["json"]
    answer_payload = captured_requests[1]["json"]

    assert reasoning_payload["max_tokens"] == 256
    assert reasoning_payload["chat_template_kwargs"] == {
        "enable_thinking": True,
    }
    assert "structured_outputs" not in reasoning_payload

    assert answer_payload["max_tokens"] == 16
    assert answer_payload["chat_template_kwargs"] == {
        "enable_thinking": False,
    }
    assert answer_payload["structured_outputs"] == {
        "choice": ["A", "B", "C", "D"],
    }
    assert "The second option is correct." in (answer_payload["messages"][0]["content"])

    assert generation.response_text == r"\boxed{B}"
    assert generation.reasoning_text == ("The second option is correct.")

    assert generation.token_usage.prompt_tokens == 345
    assert generation.token_usage.reasoning_tokens == 239

    # Five non-reasoning tokens from the first request plus two
    # final-answer tokens from the second request.
    assert generation.token_usage.completion_tokens == 7
    assert generation.token_usage.total_tokens == 591


def test_rejects_effort_budget() -> None:
    configuration = ModelConfiguration(
        configuration_id="qwen3-4b-awq__effort-high",
        model_spec=ModelSpec(
            model_id="qwen3-4b-awq",
            litellm_model="openai/qwen3-4b-awq",
        ),
        reasoning_budget=EffortBudget(value="high"),
    )

    with pytest.raises(
        TypeError,
        match="supports TokenBudget",
    ):
        run_vllm_generation(
            query=make_query(),
            configuration=configuration,
            base_url="http://127.0.0.1:8000/v1",
        )


def test_rejects_controlled_request_override() -> None:
    with pytest.raises(
        ValueError,
        match="cannot override controlled fields",
    ):
        run_vllm_generation(
            query=make_query(),
            configuration=make_configuration(0),
            base_url="http://127.0.0.1:8000/v1",
            request_options={
                "max_tokens": 100,
            },
        )


def test_rejects_missing_reasoning_text() -> None:
    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        del url, json, timeout

        return {
            "choices": [
                {
                    "message": {
                        "content": "B",
                        "reasoning": None,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "completion_tokens_details": {
                    "reasoning_tokens": 10,
                },
            },
        }

    with pytest.raises(
        ValueError,
        match="contains no reasoning text",
    ):
        run_vllm_generation(
            query=make_query(),
            configuration=make_configuration(256),
            base_url="http://127.0.0.1:8000/v1",
            post_function=fake_post,
        )


def test_rejects_nonexact_structured_answer() -> None:
    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        del url, json, timeout

        return {
            "choices": [
                {
                    "message": {
                        "content": "B) Two",
                        "reasoning": None,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                },
            },
        }

    with pytest.raises(
        ValueError,
        match="must be one of A, B, C, or D",
    ):
        run_vllm_generation(
            query=make_query(),
            configuration=make_configuration(0),
            base_url="http://127.0.0.1:8000/v1",
            post_function=fake_post,
        )
