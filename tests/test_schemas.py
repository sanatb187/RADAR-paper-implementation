from datetime import date

import pytest
from pydantic import ValidationError

from radar_bench.schemas import (
    EvaluationRecord,
    GenerationResult,
    ModelConfiguration,
    ModelSpec,
    Pricing,
    Query,
    TokenBudget,
    TokenUsage,
)


def test_query_uses_train_as_default_split() -> None:
    query = Query(
        query_id="gpqa-001",
        prompt="Which answer is correct?",
        choices=("First", "Second", "Third", "Fourth"),
        gold_answer="B",
        dataset="gpqa",
    )

    assert query.query_id == "gpqa-001"
    assert query.split == "train"
    assert len(query.choices) == 4


def test_token_budget_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        TokenBudget(value=-1)


def test_model_configuration_accepts_token_budget() -> None:
    model = ModelSpec(
        model_id="qwen3-4b",
        litellm_model="openai/Qwen/Qwen3-4B",
    )

    configuration = ModelConfiguration(
        configuration_id="qwen3-4b__tokens-512",
        model_spec=model,
        reasoning_budget=TokenBudget(value=512),
    )

    assert configuration.model_spec.model_id == "qwen3-4b"
    assert configuration.reasoning_budget.kind == "tokens"
    assert configuration.reasoning_budget.value == 512


def test_token_usage_calculates_totals() -> None:
    usage = TokenUsage(
        prompt_tokens=100,
        reasoning_tokens=500,
        completion_tokens=20,
    )

    assert usage.output_tokens == 520
    assert usage.total_tokens == 620


def test_evaluation_record_serializes_nested_models() -> None:
    generation = GenerationResult(
        generation_id="gpqa-001__qwen3-4b__512__run-0",
        query_id="gpqa-001",
        configuration_id="qwen3-4b__tokens-512",
        response_text="The correct answer is B.",
        reasoning_text="After comparing the four options...",
        token_usage=TokenUsage(
            prompt_tokens=100,
            reasoning_tokens=500,
            completion_tokens=20,
        ),
        latency_seconds=2.4,
    )

    evaluation = EvaluationRecord(
        generation=generation,
        parsed_answer="B",
        correct=True,
    )

    serialized = evaluation.model_dump()

    assert serialized["parsed_answer"] == "B"
    assert serialized["correct"] is True
    assert serialized["generation"]["query_id"] == "gpqa-001"
    assert serialized["generation"]["token_usage"]["reasoning_tokens"] == 500


def test_pricing_parses_effective_date() -> None:
    pricing = Pricing.model_validate(
        {
            "model_id": "qwen3-4b",
            "input_price_per_million_tokens": 1.0,
            "output_price_per_million_tokens": 2.0,
            "source": "https://provider.example/pricing",
            "effective_date": "2026-09-01",
        }
    )

    assert pricing.currency == "USD"
    assert pricing.effective_date == date(2026, 9, 1)


def test_pricing_rejects_negative_prices() -> None:
    with pytest.raises(ValidationError):
        Pricing.model_validate(
            {
                "model_id": "qwen3-4b",
                "input_price_per_million_tokens": -1.0,
                "output_price_per_million_tokens": 2.0,
                "source": "https://provider.example/pricing",
                "effective_date": "2026-09-01",
            }
        )
