from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

import httpx

from radar_bench.evaluators.multiple_choice import (
    format_multiple_choice_prompt,
    format_multiple_choice_question,
)
from radar_bench.schemas import (
    GenerationResult,
    ModelConfiguration,
    Query,
    TokenBudget,
    TokenUsage,
)

VLLMPostFunction = Callable[..., Mapping[str, Any]]

VLLM_THINKING_OPTIONS: dict[str, float | int] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}

VLLM_ANSWER_OPTIONS: dict[str, float | int] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
}

ANSWER_CHOICES = [
    "A",
    "B",
    "C",
    "D",
]

FINAL_ANSWER_MAX_TOKENS = 16
REQUEST_TIMEOUT_SECONDS = 180.0

CONTROLLED_REQUEST_FIELDS = {
    "model",
    "messages",
    "stream",
    "max_tokens",
    "chat_template_kwargs",
    "structured_outputs",
}


def _post_json(
    url: str,
    *,
    json: dict[str, Any],
    timeout: float,
) -> Mapping[str, Any]:
    response = httpx.post(
        url,
        json=json,
        timeout=timeout,
    )
    response.raise_for_status()

    response_data = response.json()

    if not isinstance(response_data, Mapping):
        raise TypeError("vLLM response must be a JSON object")

    return response_data


def _get_message(
    response: Mapping[str, Any],
) -> Mapping[str, Any]:
    choices = response.get("choices")

    if not isinstance(choices, list) or not choices:
        raise ValueError("vLLM response contains no choices")

    choice = choices[0]

    if not isinstance(choice, Mapping):
        raise TypeError("vLLM response choice must be an object")

    message = choice.get("message")

    if not isinstance(message, Mapping):
        raise TypeError("vLLM response contains no message")

    return message


def _get_usage(
    response: Mapping[str, Any],
) -> Mapping[str, Any]:
    usage = response.get("usage")

    if not isinstance(usage, Mapping):
        raise TypeError("vLLM response contains no token usage")

    return usage


def _get_usage_count(
    response: Mapping[str, Any],
    field: str,
) -> int:
    return int(_get_usage(response).get(field, 0) or 0)


def _get_reasoning_token_count(
    response: Mapping[str, Any],
) -> int:
    details = _get_usage(response).get(
        "completion_tokens_details",
    )

    if not isinstance(details, Mapping):
        return 0

    return int(details.get("reasoning_tokens", 0) or 0)


def _get_reasoning_text(
    response: Mapping[str, Any],
) -> str:
    reasoning = _get_message(response).get("reasoning")
    reasoning_text = str(reasoning or "").strip()

    if not reasoning_text:
        raise ValueError("vLLM response contains no reasoning text")

    return reasoning_text


def _get_answer_text(
    response: Mapping[str, Any],
) -> str:
    content = _get_message(response).get("content")
    answer = str(content or "").strip().upper()

    if answer not in ANSWER_CHOICES:
        raise ValueError("vLLM structured answer must be one of A, B, C, or D")

    return rf"\boxed{{{answer}}}"


def _build_answer_payload(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        **options,
        "stream": False,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
        "structured_outputs": {
            "choice": ANSWER_CHOICES,
        },
    }


def run_vllm_generation(
    query: Query,
    configuration: ModelConfiguration,
    *,
    base_url: str,
    run_index: int = 0,
    request_options: Mapping[str, Any] | None = None,
    final_answer_max_tokens: int = FINAL_ANSWER_MAX_TOKENS,
    request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    post_function: VLLMPostFunction = _post_json,
) -> GenerationResult:
    """Run one token-budget configuration through a vLLM server."""

    if not base_url.strip():
        raise ValueError("base_url cannot be empty")

    if run_index < 0:
        raise ValueError("run_index cannot be negative")

    if final_answer_max_tokens <= 0:
        raise ValueError("final_answer_max_tokens must be positive")

    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")

    budget = configuration.reasoning_budget

    if not isinstance(budget, TokenBudget):
        raise TypeError("The vLLM runner supports TokenBudget configurations only.")

    additional_options = dict(request_options or {})
    controlled_overrides = CONTROLLED_REQUEST_FIELDS & additional_options.keys()

    if controlled_overrides:
        raise ValueError(
            "request_options cannot override controlled fields: "
            + ", ".join(sorted(controlled_overrides))
        )

    reasoning_options = {
        **VLLM_THINKING_OPTIONS,
        **additional_options,
    }
    answer_options = {
        **VLLM_ANSWER_OPTIONS,
        **additional_options,
    }

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    model_name = configuration.model_spec.model_id
    question = format_multiple_choice_question(query)

    answer_instruction = "Return only the correct option letter: A, B, C, or D."

    start_time = perf_counter()

    if budget.value == 0:
        answer_response = post_function(
            endpoint,
            json=_build_answer_payload(
                model_name=model_name,
                prompt=f"{question}\n{answer_instruction}",
                max_tokens=final_answer_max_tokens,
                options=answer_options,
            ),
            timeout=request_timeout_seconds,
        )

        reasoning_tokens = _get_reasoning_token_count(answer_response)

        if reasoning_tokens != 0:
            raise ValueError("vLLM used reasoning tokens for a zero-budget request")

        response_text = _get_answer_text(answer_response)
        reasoning_text = None
        prompt_tokens = _get_usage_count(
            answer_response,
            "prompt_tokens",
        )
        completion_tokens = _get_usage_count(
            answer_response,
            "completion_tokens",
        )

    else:
        reasoning_response = post_function(
            endpoint,
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": format_multiple_choice_prompt(query),
                    }
                ],
                **reasoning_options,
                "stream": False,
                "max_tokens": budget.value,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                },
            },
            timeout=request_timeout_seconds,
        )

        reasoning_text = _get_reasoning_text(reasoning_response)

        answer_response = post_function(
            endpoint,
            json=_build_answer_payload(
                model_name=model_name,
                prompt=(
                    f"{question}\n\n"
                    "Use the following prior reasoning:\n"
                    f"{reasoning_text}\n\n"
                    f"{answer_instruction}"
                ),
                max_tokens=final_answer_max_tokens,
                options=answer_options,
            ),
            timeout=request_timeout_seconds,
        )

        response_text = _get_answer_text(answer_response)

        prompt_tokens = _get_usage_count(
            reasoning_response,
            "prompt_tokens",
        ) + _get_usage_count(
            answer_response,
            "prompt_tokens",
        )

        reasoning_completion_tokens = _get_usage_count(
            reasoning_response,
            "completion_tokens",
        )
        answer_completion_tokens = _get_usage_count(
            answer_response,
            "completion_tokens",
        )

        reasoning_tokens = _get_reasoning_token_count(
            reasoning_response
        ) + _get_reasoning_token_count(answer_response)

        completion_tokens = (
            reasoning_completion_tokens + answer_completion_tokens - reasoning_tokens
        )

        if completion_tokens < 0:
            raise ValueError("vLLM reasoning tokens exceed completion tokens")

    latency_seconds = perf_counter() - start_time

    return GenerationResult(
        generation_id=(
            f"{query.query_id}__{configuration.configuration_id}__run-{run_index}"
        ),
        query_id=query.query_id,
        configuration_id=configuration.configuration_id,
        response_text=response_text,
        reasoning_text=reasoning_text,
        token_usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=completion_tokens,
        ),
        latency_seconds=latency_seconds,
        run_index=run_index,
    )
