from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from radar_bench.routing_evaluation import RoutingEvaluation


@dataclass(frozen=True)
class PerformanceCostPoint:
    strategy: str
    accuracy: float
    normalized_cost: float


def _validate_probability_targets(
    predicted_probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if predicted_probabilities.shape != targets.shape:
        raise ValueError("predicted probability and target shapes must match")

    if predicted_probabilities.numel() == 0:
        raise ValueError("predicted probabilities cannot be empty")

    if not torch.isfinite(predicted_probabilities).all():
        raise ValueError("predicted probabilities must be finite")

    if torch.any((predicted_probabilities < 0.0) | (predicted_probabilities > 1.0)):
        raise ValueError("predicted probabilities must be between 0 and 1")

    if not torch.isfinite(targets).all():
        raise ValueError("targets must be finite")

    if torch.any((targets != 0) & (targets != 1)):
        raise ValueError("targets must be binary")


def calculate_brier_score(
    predicted_probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Calculate mean squared error between probabilities and binary outcomes."""

    _validate_probability_targets(
        predicted_probabilities,
        targets,
    )

    probabilities = predicted_probabilities.to(dtype=torch.float64)
    binary_targets = targets.to(dtype=torch.float64)

    return float(
        torch.mean(
            (probabilities - binary_targets) ** 2,
        ).item()
    )


def calculate_expected_calibration_error(
    predicted_probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_bins: int = 10,
) -> float:
    """Calculate equal-width expected calibration error."""

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    _validate_probability_targets(
        predicted_probabilities,
        targets,
    )

    probabilities = predicted_probabilities.to(dtype=torch.float64).flatten()
    binary_targets = targets.to(dtype=torch.float64).flatten()

    bin_indices = torch.clamp(
        (probabilities * num_bins).to(dtype=torch.long),
        max=num_bins - 1,
    )

    calibration_error = 0.0
    observation_count = probabilities.numel()

    for bin_index in range(num_bins):
        bin_mask = bin_indices == bin_index
        bin_count = int(bin_mask.sum().item())

        if bin_count == 0:
            continue

        mean_probability = float(probabilities[bin_mask].mean().item())
        mean_outcome = float(binary_targets[bin_mask].mean().item())

        calibration_error += (
            bin_count / observation_count * abs(mean_probability - mean_outcome)
        )

    return calibration_error


def calculate_cost_at_performance_threshold(
    points: Sequence[PerformanceCostPoint],
    *,
    reference_accuracy: float,
    performance_fraction: float = 0.9,
) -> float | None:
    """
    Return the lowest reference-relative cost that reaches the target accuracy.

    Point costs must be expressed as fractions of the reference configuration's
    raw cost. A return value of 0.25 therefore means 25% of the reference cost.
    """

    if not points:
        raise ValueError("points cannot be empty")

    if reference_accuracy <= 0.0 or reference_accuracy > 1.0:
        raise ValueError("reference_accuracy must be greater than 0 and at most 1")

    if performance_fraction <= 0.0 or performance_fraction > 1.0:
        raise ValueError("performance_fraction must be greater than 0 and at most 1")

    if any(point.accuracy < 0.0 or point.accuracy > 1.0 for point in points):
        raise ValueError("Point accuracy must be between 0 and 1")

    if any(point.normalized_cost < 0.0 for point in points):
        raise ValueError("Point cost cannot be negative")

    target_accuracy = reference_accuracy * performance_fraction

    qualifying_costs = [
        point.normalized_cost
        for point in points
        if point.accuracy + 1e-12 >= target_accuracy
    ]

    if not qualifying_costs:
        return None

    return min(qualifying_costs)


def build_performance_cost_points(
    results: Sequence[RoutingEvaluation],
    normalized_costs: Mapping[str, float],
) -> tuple[PerformanceCostPoint, ...]:
    """Convert routing results into performance-cost points."""

    points: list[PerformanceCostPoint] = []

    for result in results:
        selected_configuration_ids = result.selected_configuration_ids

        if not selected_configuration_ids:
            raise ValueError(f"Strategy has no selections: {result.strategy}")

        missing_configuration_ids = sorted(
            set(selected_configuration_ids) - normalized_costs.keys()
        )

        if missing_configuration_ids:
            raise ValueError(
                "Missing normalized costs for configurations: "
                + ", ".join(missing_configuration_ids)
            )

        average_cost = sum(
            normalized_costs[configuration_id]
            for configuration_id in selected_configuration_ids
        ) / len(selected_configuration_ids)

        points.append(
            PerformanceCostPoint(
                strategy=result.strategy,
                accuracy=result.accuracy,
                normalized_cost=average_cost,
            )
        )

    return tuple(points)


def calculate_hypervolume(
    points: Sequence[PerformanceCostPoint],
) -> float:
    """Calculate dominated area using reference point (cost=1, accuracy=0)."""

    if not points:
        raise ValueError("points cannot be empty")

    for point in points:
        if point.accuracy < 0.0 or point.accuracy > 1.0:
            raise ValueError("Point accuracy must be between 0 and 1")

        if point.normalized_cost < 0.0 or point.normalized_cost > 1.0:
            raise ValueError("Point normalized cost must be between 0 and 1")

    best_accuracy_by_cost: dict[float, float] = {}

    for point in points:
        current_best = best_accuracy_by_cost.get(
            point.normalized_cost,
            0.0,
        )
        best_accuracy_by_cost[point.normalized_cost] = max(
            current_best,
            point.accuracy,
        )

    ordered_costs = sorted(best_accuracy_by_cost)

    hypervolume = 0.0
    best_accuracy = 0.0

    for index, cost in enumerate(ordered_costs):
        best_accuracy = max(
            best_accuracy,
            best_accuracy_by_cost[cost],
        )

        next_cost = ordered_costs[index + 1] if index + 1 < len(ordered_costs) else 1.0

        hypervolume += (next_cost - cost) * best_accuracy

    return hypervolume


def calculate_area_under_risk_coverage(
    confidences: torch.Tensor,
    correct: torch.Tensor,
) -> float:
    """Calculate discrete area under the selective risk-coverage curve."""

    _validate_probability_targets(
        confidences,
        correct,
    )

    flattened_confidences = confidences.to(dtype=torch.float64).flatten()
    flattened_correct = correct.to(dtype=torch.float64).flatten()

    confidence_order = torch.argsort(
        flattened_confidences,
        descending=True,
        stable=True,
    )
    ordered_correct = flattened_correct[confidence_order]

    cumulative_errors = torch.cumsum(
        1.0 - ordered_correct,
        dim=0,
    )
    evaluated_counts = torch.arange(
        1,
        ordered_correct.numel() + 1,
        dtype=torch.float64,
        device=ordered_correct.device,
    )

    risks = cumulative_errors / evaluated_counts

    return float(risks.mean().item())
