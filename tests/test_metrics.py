import pytest
import torch

from radar_bench.metrics import (
    PerformanceCostPoint,
    build_performance_cost_points,
    calculate_area_under_risk_coverage,
    calculate_brier_score,
    calculate_cost_at_performance_threshold,
    calculate_expected_calibration_error,
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


def test_calculates_brier_score() -> None:
    predicted_probabilities = torch.tensor(
        [
            0.1,
            0.4,
            0.8,
            0.9,
        ]
    )
    targets = torch.tensor(
        [
            0.0,
            0.0,
            1.0,
            1.0,
        ]
    )

    score = calculate_brier_score(
        predicted_probabilities,
        targets,
    )

    assert score == pytest.approx(0.055)


def test_calculates_expected_calibration_error() -> None:
    predicted_probabilities = torch.tensor(
        [
            0.1,
            0.4,
            0.8,
            0.9,
        ]
    )
    targets = torch.tensor(
        [
            0.0,
            0.0,
            1.0,
            1.0,
        ]
    )

    error = calculate_expected_calibration_error(
        predicted_probabilities,
        targets,
        num_bins=2,
    )

    assert error == pytest.approx(0.2)


def test_calibration_metrics_reject_shape_mismatch() -> None:
    predicted_probabilities = torch.tensor([0.2, 0.8])
    targets = torch.tensor([1.0])

    with pytest.raises(
        ValueError,
        match="shapes must match",
    ):
        calculate_brier_score(
            predicted_probabilities,
            targets,
        )

    with pytest.raises(
        ValueError,
        match="shapes must match",
    ):
        calculate_expected_calibration_error(
            predicted_probabilities,
            targets,
        )


def test_expected_calibration_error_rejects_invalid_bin_count() -> None:
    with pytest.raises(
        ValueError,
        match="num_bins must be positive",
    ):
        calculate_expected_calibration_error(
            torch.tensor([0.5]),
            torch.tensor([1.0]),
            num_bins=0,
        )


def test_calculates_area_under_risk_coverage() -> None:
    confidences = torch.tensor(
        [
            0.9,
            0.8,
            0.1,
        ]
    )
    correct = torch.tensor(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    area = calculate_area_under_risk_coverage(
        confidences,
        correct,
    )

    expected_area = (0.0 + 0.5 + (2.0 / 3.0)) / 3.0

    assert area == pytest.approx(expected_area)


def test_area_under_risk_coverage_rewards_correct_confidence_order() -> None:
    correct = torch.tensor(
        [
            1.0,
            0.0,
        ]
    )

    correctly_ordered_area = calculate_area_under_risk_coverage(
        torch.tensor([0.9, 0.1]),
        correct,
    )
    incorrectly_ordered_area = calculate_area_under_risk_coverage(
        torch.tensor([0.1, 0.9]),
        correct,
    )

    assert correctly_ordered_area < incorrectly_ordered_area


def test_area_under_risk_coverage_rejects_shape_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="shapes must match",
    ):
        calculate_area_under_risk_coverage(
            torch.tensor([0.2, 0.8]),
            torch.tensor([1.0]),
        )


def test_calculates_cost_at_performance_threshold() -> None:
    points = [
        PerformanceCostPoint(
            strategy="low-cost",
            accuracy=0.70,
            normalized_cost=0.10,
        ),
        PerformanceCostPoint(
            strategy="qualified",
            accuracy=0.81,
            normalized_cost=0.25,
        ),
        PerformanceCostPoint(
            strategy="expensive",
            accuracy=0.90,
            normalized_cost=1.00,
        ),
    ]

    cost_fraction = calculate_cost_at_performance_threshold(
        points,
        reference_accuracy=0.90,
        performance_fraction=0.90,
    )

    assert cost_fraction == pytest.approx(0.25)


def test_cost_at_performance_threshold_selects_lowest_qualifying_cost() -> None:
    points = [
        PerformanceCostPoint(
            strategy="expensive",
            accuracy=0.90,
            normalized_cost=0.80,
        ),
        PerformanceCostPoint(
            strategy="cheaper",
            accuracy=0.82,
            normalized_cost=0.30,
        ),
        PerformanceCostPoint(
            strategy="cheapest",
            accuracy=0.81,
            normalized_cost=0.20,
        ),
    ]

    cost_fraction = calculate_cost_at_performance_threshold(
        points,
        reference_accuracy=0.90,
        performance_fraction=0.90,
    )

    assert cost_fraction == pytest.approx(0.20)


def test_cost_at_performance_threshold_returns_none_when_unreachable() -> None:
    points = [
        PerformanceCostPoint(
            strategy="insufficient",
            accuracy=0.79,
            normalized_cost=0.10,
        )
    ]

    cost_fraction = calculate_cost_at_performance_threshold(
        points,
        reference_accuracy=0.90,
        performance_fraction=0.90,
    )

    assert cost_fraction is None
