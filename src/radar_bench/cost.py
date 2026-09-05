import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter

from radar_bench.schemas import (
    EvaluationRecord,
    ModelConfiguration,
    Pricing,
    TokenUsage,
)

CostMetric = Literal[
    "latency",
    "output-tokens",
    "token-price",
]

TOKENS_PER_MILLION = 1_000_000

PRICING_LIST_ADAPTER = TypeAdapter(list[Pricing])


def calculate_generation_cost(
    token_usage: TokenUsage,
    pricing: Pricing,
) -> float:
    """Calculate the output-token cost of one generation."""

    return (
        token_usage.output_tokens
        / TOKENS_PER_MILLION
        * pricing.output_price_per_million_tokens
    )


def estimate_configuration_costs(
    evaluation_records: Sequence[EvaluationRecord],
    configurations: Sequence[ModelConfiguration],
    pricing_by_model_id: Mapping[str, Pricing],
) -> dict[str, float]:
    """Calculate the average observed cost of each configuration."""

    if not evaluation_records:
        raise ValueError("evaluation_records cannot be empty")

    if not configurations:
        raise ValueError("configurations cannot be empty")

    configurations_by_id = {
        configuration.configuration_id: configuration
        for configuration in configurations
    }

    if len(configurations_by_id) != len(configurations):
        raise ValueError("configuration IDs must be unique")

    total_costs: dict[str, float] = defaultdict(float)
    record_counts: dict[str, int] = defaultdict(int)

    for record in evaluation_records:
        configuration_id = record.generation.configuration_id

        if configuration_id not in configurations_by_id:
            raise ValueError(f"Unknown configuration ID: {configuration_id}")

        configuration = configurations_by_id[configuration_id]
        model_id = configuration.model_spec.model_id

        pricing = pricing_by_model_id.get(model_id)

        if pricing is None:
            raise ValueError(f"No pricing found for model ID: {model_id}")

        generation_cost = calculate_generation_cost(
            token_usage=record.generation.token_usage,
            pricing=pricing,
        )

        total_costs[configuration_id] += generation_cost
        record_counts[configuration_id] += 1

    average_costs: dict[str, float] = {}

    for configuration in configurations:
        configuration_id = configuration.configuration_id

        if record_counts[configuration_id] == 0:
            raise ValueError(
                f"No evaluation records found for configuration ID: {configuration_id}"
            )

        average_costs[configuration_id] = (
            total_costs[configuration_id] / record_counts[configuration_id]
        )

    return average_costs


def normalize_costs(
    costs_by_configuration: Mapping[str, float],
) -> dict[str, float]:
    """Min-max normalize configuration costs to the range [0, 1]."""

    if not costs_by_configuration:
        raise ValueError("costs_by_configuration cannot be empty")

    if any(cost < 0 for cost in costs_by_configuration.values()):
        raise ValueError("configuration costs cannot be negative")

    minimum_cost = min(costs_by_configuration.values())
    maximum_cost = max(costs_by_configuration.values())

    if minimum_cost == maximum_cost:
        return {configuration_id: 0.0 for configuration_id in costs_by_configuration}

    cost_range = maximum_cost - minimum_cost

    return {
        configuration_id: (cost - minimum_cost) / cost_range
        for configuration_id, cost in costs_by_configuration.items()
    }


def estimate_configuration_latency_costs(
    evaluation_records: Sequence[EvaluationRecord],
    configuration_ids: Sequence[str],
) -> dict[str, float]:
    """Calculate average observed latency for each configuration."""

    if not evaluation_records:
        raise ValueError("evaluation_records cannot be empty")

    if not configuration_ids:
        raise ValueError("configuration_ids cannot be empty")

    if len(set(configuration_ids)) != len(configuration_ids):
        raise ValueError("configuration_ids must be unique")

    expected_configuration_ids = set(configuration_ids)
    total_latencies: dict[str, float] = defaultdict(float)
    record_counts: dict[str, int] = defaultdict(int)

    for record in evaluation_records:
        configuration_id = record.generation.configuration_id

        if configuration_id not in (expected_configuration_ids):
            raise ValueError(f"Unknown configuration ID: {configuration_id}")

        total_latencies[configuration_id] += record.generation.latency_seconds
        record_counts[configuration_id] += 1

    missing_configuration_ids = [
        configuration_id
        for configuration_id in configuration_ids
        if record_counts[configuration_id] == 0
    ]

    if missing_configuration_ids:
        raise ValueError(
            "No evaluation records found for configurations: "
            + ", ".join(missing_configuration_ids)
        )

    return {
        configuration_id: (
            total_latencies[configuration_id] / record_counts[configuration_id]
        )
        for configuration_id in configuration_ids
    }


def estimate_configuration_output_token_costs(
    evaluation_records: Sequence[EvaluationRecord],
    configuration_ids: Sequence[str],
) -> dict[str, float]:
    """Estimate average generated-token usage for each configuration."""

    requested_ids = tuple(configuration_ids)

    if not requested_ids:
        raise ValueError("configuration_ids cannot be empty")

    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("configuration_ids must be unique")

    requested_id_set = set(requested_ids)

    total_tokens = {configuration_id: 0.0 for configuration_id in requested_ids}
    record_counts = {configuration_id: 0 for configuration_id in requested_ids}

    for record in evaluation_records:
        configuration_id = record.generation.configuration_id

        if configuration_id not in requested_id_set:
            continue

        total_tokens[configuration_id] += record.generation.token_usage.output_tokens
        record_counts[configuration_id] += 1

    average_tokens: dict[str, float] = {}

    for configuration_id in requested_ids:
        if record_counts[configuration_id] == 0:
            raise ValueError(
                f"No evaluation records found for configuration ID: {configuration_id}"
            )

        average_tokens[configuration_id] = (
            total_tokens[configuration_id] / record_counts[configuration_id]
        )

    return average_tokens


def estimate_routing_costs(
    evaluation_records: Sequence[EvaluationRecord],
    configuration_ids: Sequence[str],
    *,
    metric: CostMetric,
    configurations: Sequence[ModelConfiguration] | None = None,
    pricing_by_model_id: Mapping[str, Pricing] | None = None,
) -> dict[str, float]:
    """Estimate routing costs using the selected cost metric."""

    if metric == "latency":
        return estimate_configuration_latency_costs(
            evaluation_records,
            configuration_ids,
        )

    if metric == "output-tokens":
        return estimate_configuration_output_token_costs(
            evaluation_records,
            configuration_ids,
        )

    if metric == "token-price":
        if configurations is None:
            raise ValueError("configurations are required for token-price cost")

        if pricing_by_model_id is None:
            raise ValueError("pricing_by_model_id is required for token-price cost")

        configurations_by_id = {
            configuration.configuration_id: configuration
            for configuration in configurations
        }

        missing_configuration_ids = [
            configuration_id
            for configuration_id in configuration_ids
            if configuration_id not in configurations_by_id
        ]

        if missing_configuration_ids:
            raise ValueError(
                "Missing configurations: " + ", ".join(missing_configuration_ids)
            )

        selected_configurations = [
            configurations_by_id[configuration_id]
            for configuration_id in configuration_ids
        ]

        return estimate_configuration_costs(
            evaluation_records,
            selected_configurations,
            pricing_by_model_id,
        )

    raise ValueError(f"Unsupported cost metric: {metric}")


def load_pricing_file(
    path: Path,
) -> dict[str, Pricing]:
    """Load a versioned model-pricing table from JSON."""

    raw_pricing = json.loads(path.read_text(encoding="utf-8"))
    pricing_entries = PRICING_LIST_ADAPTER.validate_python(raw_pricing)

    if not pricing_entries:
        raise ValueError("Pricing file cannot be empty")

    pricing_by_model_id = {pricing.model_id: pricing for pricing in pricing_entries}

    if len(pricing_by_model_id) != len(pricing_entries):
        raise ValueError("Pricing file contains duplicate model IDs")

    return pricing_by_model_id
