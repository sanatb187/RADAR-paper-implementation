from datetime import date
from pathlib import Path

import pytest

from radar_bench.cost import (
    calculate_generation_cost,
    estimate_configuration_costs,
    estimate_configuration_output_token_costs,
    estimate_routing_costs,
    load_pricing_file,
    normalize_costs,
)
from radar_bench.schemas import (
    EvaluationRecord,
    GenerationResult,
    ModelConfiguration,
    ModelSpec,
    Pricing,
    TokenBudget,
    TokenUsage,
)


def make_configuration(
    configuration_id: str,
    model_id: str = "qwen3-4b",
) -> ModelConfiguration:
    return ModelConfiguration(
        configuration_id=configuration_id,
        model_spec=ModelSpec(
            model_id=model_id,
            litellm_model=f"openai/{model_id}",
        ),
        reasoning_budget=TokenBudget(value=512),
    )


def make_record(
    configuration_id: str,
    query_id: str,
    reasoning_tokens: int,
    completion_tokens: int,
    prompt_tokens: int = 100,
    latency_seconds: float = 1.0,
) -> EvaluationRecord:
    return EvaluationRecord(
        generation=GenerationResult(
            generation_id=f"{query_id}__{configuration_id}",
            query_id=query_id,
            configuration_id=configuration_id,
            response_text="B",
            reasoning_text="Reasoning",
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                reasoning_tokens=reasoning_tokens,
                completion_tokens=completion_tokens,
            ),
            latency_seconds=latency_seconds,
            run_index=0,
        ),
        parsed_answer="B",
        correct=True,
    )


def make_pricing(
    model_id: str = "qwen3-4b",
    output_price: float = 10.0,
) -> Pricing:
    return Pricing(
        model_id=model_id,
        input_price_per_million_tokens=2.0,
        output_price_per_million_tokens=output_price,
        currency="USD",
        source="test",
        effective_date=date(2026, 1, 1),
    )


def test_calculate_generation_cost_uses_output_tokens() -> None:
    token_usage = TokenUsage(
        prompt_tokens=1_000_000,
        reasoning_tokens=500,
        completion_tokens=20,
    )

    cost = calculate_generation_cost(
        token_usage=token_usage,
        pricing=make_pricing(output_price=10.0),
    )

    assert cost == pytest.approx(0.0052)


def test_estimate_configuration_costs_uses_average() -> None:
    configuration = make_configuration("config-a")

    records = [
        make_record(
            configuration_id="config-a",
            query_id="query-1",
            reasoning_tokens=100,
            completion_tokens=50,
        ),
        make_record(
            configuration_id="config-a",
            query_id="query-2",
            reasoning_tokens=200,
            completion_tokens=50,
        ),
    ]

    costs = estimate_configuration_costs(
        evaluation_records=records,
        configurations=[configuration],
        pricing_by_model_id={"qwen3-4b": make_pricing(output_price=10.0)},
    )

    # Average output tokens = (150 + 250) / 2 = 200.
    # Cost = 200 / 1,000,000 * $10 = $0.002.
    assert costs["config-a"] == pytest.approx(0.002)


def test_normalize_costs() -> None:
    normalized = normalize_costs(
        {
            "cheap": 1.0,
            "middle": 3.0,
            "expensive": 5.0,
        }
    )

    assert normalized == pytest.approx(
        {
            "cheap": 0.0,
            "middle": 0.5,
            "expensive": 1.0,
        }
    )


def test_equal_costs_normalize_to_zero() -> None:
    normalized = normalize_costs(
        {
            "config-a": 2.0,
            "config-b": 2.0,
        }
    )

    assert normalized == {
        "config-a": 0.0,
        "config-b": 0.0,
    }


def test_missing_pricing_is_rejected() -> None:
    configuration = make_configuration("config-a")

    records = [
        make_record(
            configuration_id="config-a",
            query_id="query-1",
            reasoning_tokens=100,
            completion_tokens=20,
        )
    ]

    with pytest.raises(
        ValueError,
        match="No pricing found",
    ):
        estimate_configuration_costs(
            evaluation_records=records,
            configurations=[configuration],
            pricing_by_model_id={},
        )


def test_unknown_configuration_is_rejected() -> None:
    configuration = make_configuration("config-a")

    records = [
        make_record(
            configuration_id="unknown",
            query_id="query-1",
            reasoning_tokens=100,
            completion_tokens=20,
        )
    ]

    with pytest.raises(
        ValueError,
        match="Unknown configuration ID",
    ):
        estimate_configuration_costs(
            evaluation_records=records,
            configurations=[configuration],
            pricing_by_model_id={"qwen3-4b": make_pricing()},
        )


def test_configuration_without_records_is_rejected() -> None:
    configurations = [
        make_configuration("config-a"),
        make_configuration("config-b"),
    ]

    records = [
        make_record(
            configuration_id="config-a",
            query_id="query-1",
            reasoning_tokens=100,
            completion_tokens=20,
        )
    ]

    with pytest.raises(
        ValueError,
        match="No evaluation records found",
    ):
        estimate_configuration_costs(
            evaluation_records=records,
            configurations=configurations,
            pricing_by_model_id={"qwen3-4b": make_pricing()},
        )


def test_negative_cost_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        normalize_costs(
            {
                "config-a": -1.0,
                "config-b": 2.0,
            }
        )


def test_estimates_average_output_token_costs() -> None:
    records = [
        make_record(
            "configuration-a",
            "query-1",
            reasoning_tokens=100,
            completion_tokens=20,
        ),
        make_record(
            "configuration-a",
            "query-2",
            reasoning_tokens=200,
            completion_tokens=40,
        ),
    ]

    costs = estimate_configuration_output_token_costs(
        records,
        ["configuration-a"],
    )

    assert costs == {
        "configuration-a": 180.0,
    }


def test_selects_latency_routing_cost() -> None:
    records = [
        make_record(
            "configuration-a",
            "query-1",
            reasoning_tokens=100,
            completion_tokens=20,
            latency_seconds=2.5,
        ),
    ]

    costs = estimate_routing_costs(
        records,
        ["configuration-a"],
        metric="latency",
    )

    assert costs == {
        "configuration-a": 2.5,
    }


def test_selects_output_token_routing_cost() -> None:
    records = [
        make_record(
            "configuration-a",
            "query-1",
            reasoning_tokens=100,
            completion_tokens=20,
        ),
    ]

    costs = estimate_routing_costs(
        records,
        ["configuration-a"],
        metric="output-tokens",
    )

    assert costs == {
        "configuration-a": 120.0,
    }


def test_loads_pricing_file(
    tmp_path: Path,
) -> None:
    pricing_path = tmp_path / "pricing.json"

    pricing_path.write_text(
        """
        [
          {
            "model_id": "qwen3-4b",
            "input_price_per_million_tokens": 0.10,
            "output_price_per_million_tokens": 0.20,
            "currency": "USD",
            "source": "test pricing",
            "effective_date": "2026-09-05"
          }
        ]
        """,
        encoding="utf-8",
    )

    pricing = load_pricing_file(pricing_path)

    assert set(pricing) == {"qwen3-4b"}
    assert pricing["qwen3-4b"].output_price_per_million_tokens == 0.20
