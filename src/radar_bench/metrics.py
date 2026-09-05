from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from radar_bench.routing_evaluation import RoutingEvaluation


@dataclass(frozen=True)
class PerformanceCostPoint:
    strategy: str
    accuracy: float
    normalized_cost: float


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
