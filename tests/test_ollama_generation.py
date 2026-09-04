from typing import Any

import pytest

from radar_bench.ollama_generation import run_ollama_generation
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


def make_token_configuration(
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


def test_zero_budget_uses_one_request() -> None:
    captured_calls: list[dict[str, Any]] = []

    def fake_chat(**kwargs: Any) -> dict[str, Any]:
        captured_calls.append(kwargs)

        return {
            "message": {
                "content": '{"answer":"B"}',
                "thinking": "",
            },
            "prompt_eval_count": 100,
            "eval_count": 12,
        }

    generation = run_ollama_generation(
        query=make_query(),
        configuration=make_token_configuration(0),
        request_options={
            "seed": 42,
        },
        chat_function=fake_chat,
    )

    assert len(captured_calls) == 1

    call = captured_calls[0]

    assert call["format"]["properties"]["answer"]["enum"] == [
        "A",
        "B",
        "C",
        "D",
    ]

    assert call["model"] == "qwen3:0.6b"
    assert call["think"] is False
    assert call["stream"] is False
    assert call["options"] == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "seed": 42,
        "num_predict": 256,
    }

    answer_prompt = call["messages"][0]["content"]

    assert "/no_think" in answer_prompt
    assert "Please reason step by step" not in answer_prompt
    assert "Return a JSON object" in answer_prompt

    assert generation.response_text == r"\boxed{B}"
    assert generation.reasoning_text is None
    assert generation.token_usage.prompt_tokens == 100
    assert generation.token_usage.reasoning_tokens == 0
    assert generation.token_usage.completion_tokens == 12


def test_positive_budget_uses_two_requests() -> None:
    captured_calls: list[dict[str, Any]] = []

    def fake_chat(**kwargs: Any) -> dict[str, Any]:
        captured_calls.append(kwargs)

        if len(captured_calls) == 1:
            return {
                "message": {
                    "content": "",
                    "thinking": "The second option is correct.",
                },
                "prompt_eval_count": 100,
                "eval_count": 256,
            }

        return {
            "message": {
                "content": '{"answer":"B"}',
                "thinking": "",
            },
            "prompt_eval_count": 380,
            "eval_count": 12,
        }

    generation = run_ollama_generation(
        query=make_query(),
        configuration=make_token_configuration(256),
        run_index=2,
        request_options={
            "seed": 42,
        },
        chat_function=fake_chat,
    )

    assert len(captured_calls) == 2

    reasoning_call = captured_calls[0]
    answer_call = captured_calls[1]

    assert reasoning_call["think"] is True
    assert reasoning_call["options"]["num_predict"] == 256

    assert answer_call["think"] is False
    assert answer_call["options"]["num_predict"] == 256

    assert reasoning_call["options"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 42,
        "num_predict": 256,
    }

    assert answer_call["options"] == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "seed": 42,
        "num_predict": 256,
    }

    assert answer_call["format"]["properties"]["answer"]["enum"] == [
        "A",
        "B",
        "C",
        "D",
    ]

    assert "format" not in reasoning_call

    final_prompt = answer_call["messages"][0]["content"]

    reasoning_prompt = reasoning_call["messages"][0]["content"]

    assert "/think" in reasoning_prompt
    assert "/no_think" in final_prompt
    assert "Please reason step by step" not in final_prompt
    assert "Return a JSON object" in final_prompt

    assert "The second option is correct." in final_prompt
    assert r"\boxed{" not in final_prompt

    assert generation.generation_id == ("query-1__qwen3-0.6b__tokens-256__run-2")
    assert generation.response_text == r"\boxed{B}"
    assert generation.reasoning_text == "The second option is correct."

    assert generation.token_usage.prompt_tokens == 480
    assert generation.token_usage.reasoning_tokens == 256
    assert generation.token_usage.completion_tokens == 12
    assert generation.token_usage.total_tokens == 748
    assert generation.latency_seconds >= 0


def test_rejects_effort_budget() -> None:
    configuration = ModelConfiguration(
        configuration_id="qwen3-0.6b__effort-high",
        model_spec=ModelSpec(
            model_id="qwen3-0.6b",
            litellm_model="ollama/qwen3:0.6b",
        ),
        reasoning_budget=EffortBudget(value="high"),
    )

    with pytest.raises(
        TypeError,
        match="supports TokenBudget",
    ):
        run_ollama_generation(
            query=make_query(),
            configuration=configuration,
        )


def test_rejects_num_predict_override() -> None:
    with pytest.raises(
        ValueError,
        match="cannot override",
    ):
        run_ollama_generation(
            query=make_query(),
            configuration=make_token_configuration(256),
            request_options={
                "num_predict": 100,
            },
        )


def test_rejects_negative_run_index() -> None:
    with pytest.raises(
        ValueError,
        match="run_index cannot be negative",
    ):
        run_ollama_generation(
            query=make_query(),
            configuration=make_token_configuration(0),
            run_index=-1,
        )


def test_rejects_missing_message() -> None:
    def fake_chat(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {}

    with pytest.raises(
        ValueError,
        match="contains no message",
    ):
        run_ollama_generation(
            query=make_query(),
            configuration=make_token_configuration(0),
            chat_function=fake_chat,
        )
