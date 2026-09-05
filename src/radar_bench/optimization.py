from collections.abc import Mapping, Sequence

import torch


def _prepare_objectives(
    predicted_probabilities: torch.Tensor,
    configuration_ids: Sequence[str],
    normalized_costs: Mapping[str, float],
    performance_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted_probabilities.ndim != 1:
        raise ValueError("predicted_probabilities must be a 1D tensor")

    if len(configuration_ids) == 0:
        raise ValueError("configuration_ids cannot be empty")

    if len(configuration_ids) != predicted_probabilities.shape[0]:
        raise ValueError(
            "The number of configuration IDs must match the "
            "number of predicted probabilities"
        )

    if len(set(configuration_ids)) != len(configuration_ids):
        raise ValueError("configuration_ids must be unique")

    if not 0.0 <= performance_weight <= 1.0:
        raise ValueError("performance_weight must be between 0 and 1")

    if not torch.is_floating_point(predicted_probabilities):
        predicted_probabilities = predicted_probabilities.to(dtype=torch.float32)

    if not torch.isfinite(predicted_probabilities).all():
        raise ValueError("predicted_probabilities must contain finite values")

    if torch.any(predicted_probabilities < 0) or torch.any(predicted_probabilities > 1):
        raise ValueError("predicted_probabilities must be between 0 and 1")

    missing_costs = [
        configuration_id
        for configuration_id in configuration_ids
        if configuration_id not in normalized_costs
    ]

    if missing_costs:
        raise ValueError(
            "Missing normalized costs for configurations: " + ", ".join(missing_costs)
        )

    configuration_costs = [
        normalized_costs[configuration_id] for configuration_id in configuration_ids
    ]

    if any(cost < 0.0 or cost > 1.0 for cost in configuration_costs):
        raise ValueError("Normalized costs must be between 0 and 1")

    cost_tensor = predicted_probabilities.new_tensor(configuration_costs)

    return predicted_probabilities, cost_tensor


def calculate_linear_scores(
    predicted_probabilities: torch.Tensor,
    configuration_ids: Sequence[str],
    normalized_costs: Mapping[str, float],
    performance_weight: float,
) -> torch.Tensor:
    """Calculate the linear scalarization score."""

    probabilities, costs = _prepare_objectives(
        predicted_probabilities,
        configuration_ids,
        normalized_costs,
        performance_weight,
    )

    cost_weight = 1.0 - performance_weight

    return performance_weight * probabilities - cost_weight * costs


def calculate_chebyshev_penalties(
    predicted_probabilities: torch.Tensor,
    configuration_ids: Sequence[str],
    normalized_costs: Mapping[str, float],
    performance_weight: float,
) -> torch.Tensor:
    """Calculate distance from the ideal performance-cost point."""

    probabilities, costs = _prepare_objectives(
        predicted_probabilities,
        configuration_ids,
        normalized_costs,
        performance_weight,
    )

    performance_penalties = performance_weight * (1.0 - probabilities)
    cost_penalties = (1.0 - performance_weight) * costs

    return torch.maximum(
        performance_penalties,
        cost_penalties,
    )


def select_configuration(
    predicted_probabilities: torch.Tensor,
    configuration_ids: Sequence[str],
    normalized_costs: Mapping[str, float],
    performance_weight: float,
) -> str:
    """Select a configuration using linear scalarization."""

    scores = calculate_linear_scores(
        predicted_probabilities=predicted_probabilities,
        configuration_ids=configuration_ids,
        normalized_costs=normalized_costs,
        performance_weight=performance_weight,
    )

    selected_index = int(torch.argmax(scores).item())

    return configuration_ids[selected_index]


def select_configuration_chebyshev(
    predicted_probabilities: torch.Tensor,
    configuration_ids: Sequence[str],
    normalized_costs: Mapping[str, float],
    performance_weight: float,
) -> str:
    """Select a configuration using Chebyshev scalarization."""

    penalties = calculate_chebyshev_penalties(
        predicted_probabilities=predicted_probabilities,
        configuration_ids=configuration_ids,
        normalized_costs=normalized_costs,
        performance_weight=performance_weight,
    )

    selected_index = int(torch.argmin(penalties).item())

    return configuration_ids[selected_index]
