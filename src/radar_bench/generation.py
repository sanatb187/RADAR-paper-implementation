from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

from litellm import completion  # type: ignore[import-untyped]

from radar_bench.evaluators.multiple_choice import (
    format_multiple_choice_prompt,
)
from radar_bench.schemas import (
    EffortBudget,
    GenerationResult,
    ModelConfiguration,
    Query,
    TokenUsage,
)

CompletionFunction = Callable[..., Any]

RESERVED_REQUEST_FIELDS = {
    "model",
    "messages",
    "reasoning_effort",
}


def _get_field(
    value: Any,
    field: str,
    default: Any = None,
) -> Any:
    if isinstance(value, Mapping):
        return value.get(field, default)

    return getattr(value, field, default)


def run_litellm_generation(
    query: Query,
    configuration: ModelConfiguration,
    *,
    run_index: int = 0,
    request_kwargs: Mapping[str, Any] | None = None,
    completion_function: CompletionFunction = completion,
) -> GenerationResult:
    """Run one effort-based configuration through LiteLLM."""

    if run_index < 0:
        raise ValueError("run_index cannot be negative")

    budget = configuration.reasoning_budget

    if not isinstance(budget, EffortBudget):
        raise TypeError(
            "The LiteLLM runner currently supports EffortBudget only. "
            "TokenBudget requires a provider-specific runner."
        )

    additional_kwargs = dict(request_kwargs or {})

    invalid_fields = RESERVED_REQUEST_FIELDS & additional_kwargs.keys()

    if invalid_fields:
        raise ValueError(
            "request_kwargs cannot override reserved fields: "
            + ", ".join(sorted(invalid_fields))
        )

    prompt = format_multiple_choice_prompt(query)

    completion_kwargs: dict[str, Any] = {
        "model": configuration.model_spec.litellm_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "reasoning_effort": budget.value,
        **additional_kwargs,
    }

    start_time = perf_counter()
    response = completion_function(**completion_kwargs)
    latency_seconds = perf_counter() - start_time

    choices = _get_field(response, "choices")

    if not choices:
        raise ValueError("LiteLLM response contains no choices")

    message = _get_field(choices[0], "message")

    if message is None:
        raise ValueError("LiteLLM response contains no message")

    response_text = _get_field(message, "content", "") or ""
    reasoning_text = _get_field(
        message,
        "reasoning_content",
    )

    if reasoning_text is None:
        reasoning_text = _get_field(message, "reasoning")

    usage = _get_field(response, "usage")

    if usage is None:
        raise ValueError("LiteLLM response contains no token usage")

    prompt_tokens = int(_get_field(usage, "prompt_tokens", 0) or 0)
    total_completion_tokens = int(_get_field(usage, "completion_tokens", 0) or 0)

    completion_details = _get_field(
        usage,
        "completion_tokens_details",
    )

    reasoning_tokens = int(
        _get_field(
            completion_details,
            "reasoning_tokens",
            0,
        )
        or 0
    )

    visible_completion_tokens = max(
        total_completion_tokens - reasoning_tokens,
        0,
    )

    return GenerationResult(
        generation_id=(
            f"{query.query_id}__{configuration.configuration_id}__run-{run_index}"
        ),
        query_id=query.query_id,
        configuration_id=configuration.configuration_id,
        response_text=str(response_text),
        reasoning_text=(str(reasoning_text) if reasoning_text is not None else None),
        token_usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=visible_completion_tokens,
        ),
        latency_seconds=latency_seconds,
        run_index=run_index,
    )
