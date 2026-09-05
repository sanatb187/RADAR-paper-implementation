import pytest

from radar_bench.metrics import (
    PerformanceCostPoint,
    build_performance_cost_points,
    calculate_hypervolume,
)
from radar_bench.routing_evaluation import RoutingEvaluation


def make_result(
    strategy: str,
    accuracy: float,
    selected_configuration_ids: tuple[str, ...],
) -> RoutingEvaluation:
    return RoutingEvaluation(
        strategy=strategy,
        accuracy=accuracy,
        average_latency_seconds=1.0,
        selected_configuration_ids=selected_configuration_ids,
    )


def test_builds_performance_cost_points() -> None:
    results = [
        make_result(
            "radar:0.5",
            0.75,
            (
                "config-a",
                "config-b",
                "config-b",
            ),
        )
    ]

    points = build_performance_cost_points(
        results,
        {
            "config-a": 0.0,
            "config-b": 1.0,
        },
    )

    assert len(points) == 1
    assert points[0].strategy == "radar:0.5"
    assert points[0].accuracy == 0.75
    assert points[0].normalized_cost == pytest.approx(2.0 / 3.0)


def test_calculates_hypervolume() -> None:
    points = [
        PerformanceCostPoint(
            strategy="low-cost",
            accuracy=0.4,
            normalized_cost=0.0,
        ),
        PerformanceCostPoint(
            strategy="balanced",
            accuracy=0.7,
            normalized_cost=0.5,
        ),
        PerformanceCostPoint(
            strategy="high-cost",
            accuracy=0.9,
            normalized_cost=1.0,
        ),
    ]

    hypervolume = calculate_hypervolume(points)

    assert hypervolume == pytest.approx(0.55)


def test_dominated_point_does_not_increase_hypervolume() -> None:
    points = [
        PerformanceCostPoint(
            strategy="dominant",
            accuracy=0.4,
            normalized_cost=0.0,
        ),
        PerformanceCostPoint(
            strategy="dominated",
            accuracy=0.3,
            normalized_cost=0.5,
        ),
    ]

    hypervolume = calculate_hypervolume(points)

    assert hypervolume == pytest.approx(0.4)


def test_rejects_missing_configuration_cost() -> None:
    results = [
        make_result(
            "radar:0.5",
            0.5,
            ("missing-config",),
        )
    ]

    with pytest.raises(
        ValueError,
        match="Missing normalized costs",
    ):
        build_performance_cost_points(
            results,
            {},
        )


@pytest.mark.parametrize(
    ("accuracy", "normalized_cost"),
    [
        (-0.1, 0.5),
        (1.1, 0.5),
        (0.5, -0.1),
        (0.5, 1.1),
    ],
)
def test_rejects_invalid_point(
    accuracy: float,
    normalized_cost: float,
) -> None:
    point = PerformanceCostPoint(
        strategy="invalid",
        accuracy=accuracy,
        normalized_cost=normalized_cost,
    )

    with pytest.raises(
        ValueError,
        match="must be between 0 and 1",
    ):
        calculate_hypervolume([point])
