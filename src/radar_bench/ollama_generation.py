from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, Literal

from ollama import chat

from radar_bench.evaluators.multiple_choice import (
    format_multiple_choice_prompt,
    format_multiple_choice_question,
)
from radar_bench.schemas import (
    GenerationResult,
    ModelConfiguration,
    Query,
    RADARSchema,
    TokenBudget,
    TokenUsage,
)

OllamaChatFunction = Callable[..., Any]

QWEN_THINKING_OPTIONS: dict[str, float | int] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}

QWEN_ANSWER_OPTIONS: dict[str, float | int] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
}

FINAL_ANSWER_MAX_TOKENS = 256


class OllamaMultipleChoiceAnswer(RADARSchema):
    answer: Literal["A", "B", "C", "D"]


def _get_field(
    value: Any,
    field: str,
    default: Any = None,
) -> Any:
    if isinstance(value, Mapping):
        return value.get(field, default)

    return getattr(value, field, default)


def _get_message_text(
    response: Any,
    field: str,
) -> str:
    message = _get_field(response, "message")

    if message is None:
        raise ValueError("Ollama response contains no message")

    return str(_get_field(message, field, "") or "")


def _get_structured_answer(response: Any) -> str:
    response_text = _get_message_text(
        response,
        "content",
    )

    answer = OllamaMultipleChoiceAnswer.model_validate_json(
        response_text,
    )

    return rf"\boxed{{{answer.answer}}}"


def _get_token_count(
    response: Any,
    field: str,
) -> int:
    return int(_get_field(response, field, 0) or 0)


def _ollama_model_name(configuration: ModelConfiguration) -> str:
    model_name = configuration.model_spec.litellm_model

    if model_name.startswith("ollama/"):
        return model_name.removeprefix("ollama/")

    return model_name


def run_ollama_generation(
    query: Query,
    configuration: ModelConfiguration,
    *,
    run_index: int = 0,
    request_options: Mapping[str, Any] | None = None,
    final_answer_max_tokens: int = FINAL_ANSWER_MAX_TOKENS,
    chat_function: OllamaChatFunction = chat,
) -> GenerationResult:
    """Run one token-budget configuration through Ollama."""

    if run_index < 0:
        raise ValueError("run_index cannot be negative")

    if final_answer_max_tokens <= 0:
        raise ValueError("final_answer_max_tokens must be positive")

    budget = configuration.reasoning_budget

    if not isinstance(budget, TokenBudget):
        raise TypeError("The Ollama runner supports TokenBudget configurations only.")

    additional_options = dict(request_options or {})

    if "num_predict" in additional_options:
        raise ValueError(
            "request_options cannot override the controlled num_predict field"
        )
    reasoning_options = {
        **QWEN_THINKING_OPTIONS,
        **additional_options,
    }

    answer_options = {
        **QWEN_ANSWER_OPTIONS,
        **additional_options,
    }

    model_name = _ollama_model_name(configuration)
    prompt = format_multiple_choice_prompt(query)
    question = format_multiple_choice_question(query)
    reasoning_prompt = f"{prompt}\n/think"

    answer_instruction = (
        "Return a JSON object containing only the field "
        '"answer". Its value must be one of "A", "B", "C", or "D".'
        "\n/no_think"
    )

    start_time = perf_counter()

    if budget.value == 0:
        response = chat_function(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"{question}\n{answer_instruction}",
                }
            ],
            think=False,
            stream=False,
            format=OllamaMultipleChoiceAnswer.model_json_schema(),
            options={
                **answer_options,
                "num_predict": final_answer_max_tokens,
            },
        )

        response_text = _get_structured_answer(response)
        reasoning_text = None

        prompt_tokens = _get_token_count(
            response,
            "prompt_eval_count",
        )
        reasoning_tokens = 0
        completion_tokens = _get_token_count(
            response,
            "eval_count",
        )

    else:
        reasoning_response = chat_function(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": reasoning_prompt,
                }
            ],
            think=True,
            stream=False,
            options={
                **reasoning_options,
                "num_predict": budget.value,
            },
        )

        reasoning_text = _get_message_text(
            reasoning_response,
            "thinking",
        )

        final_prompt = (
            f"{question}\n\n"
            "Use the following prior reasoning:\n"
            f"{reasoning_text}\n\n"
            f"{answer_instruction}"
        )

        answer_response = chat_function(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": final_prompt,
                }
            ],
            think=False,
            stream=False,
            format=OllamaMultipleChoiceAnswer.model_json_schema(),
            options={
                **answer_options,
                "num_predict": final_answer_max_tokens,
            },
        )

        response_text = _get_structured_answer(
            answer_response,
        )

        prompt_tokens = _get_token_count(
            reasoning_response,
            "prompt_eval_count",
        ) + _get_token_count(
            answer_response,
            "prompt_eval_count",
        )

        # Ollama reports one generated-token count for the entire
        # first response, rather than separate thinking/content counts.
        reasoning_tokens = _get_token_count(
            reasoning_response,
            "eval_count",
        )
        completion_tokens = _get_token_count(
            answer_response,
            "eval_count",
        )

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
