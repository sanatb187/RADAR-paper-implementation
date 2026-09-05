import pytest
import torch

from radar_bench.optimization import (
    calculate_chebyshev_penalties,
    calculate_linear_scores,
    select_configuration,
    select_configuration_chebyshev,
)

CONFIGURATION_IDS = (
    "high-performance",
    "balanced",
    "low-cost",
)

PROBABILITIES = torch.tensor([0.9, 0.75, 0.4])

NORMALIZED_COSTS = {
    "high-performance": 1.0,
    "balanced": 0.3,
    "low-cost": 0.0,
}


def test_calculate_linear_scores() -> None:
    scores = calculate_linear_scores(
        predicted_probabilities=PROBABILITIES,
        configuration_ids=CONFIGURATION_IDS,
        normalized_costs=NORMALIZED_COSTS,
        performance_weight=0.5,
    )

    expected = torch.tensor(
        [
            -0.05,
            0.225,
            0.2,
        ]
    )

    assert torch.allclose(scores, expected)


def test_performance_only_selects_best_performance() -> None:
    selected = select_configuration(
        predicted_probabilities=PROBABILITIES,
        configuration_ids=CONFIGURATION_IDS,
        normalized_costs=NORMALIZED_COSTS,
        performance_weight=1.0,
    )

    assert selected == "high-performance"


def test_cost_only_selects_cheapest_configuration() -> None:
    selected = select_configuration(
        predicted_probabilities=PROBABILITIES,
        configuration_ids=CONFIGURATION_IDS,
        normalized_costs=NORMALIZED_COSTS,
        performance_weight=0.0,
    )

    assert selected == "low-cost"


def test_balanced_weight_selects_compromise() -> None:
    selected = select_configuration(
        predicted_probabilities=PROBABILITIES,
        configuration_ids=CONFIGURATION_IDS,
        normalized_costs=NORMALIZED_COSTS,
        performance_weight=0.5,
    )

    assert selected == "balanced"


@pytest.mark.parametrize(
    "performance_weight",
    [-0.1, 1.1],
)
def test_rejects_invalid_performance_weight(
    performance_weight: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="performance_weight must be between 0 and 1",
    ):
        select_configuration(
            predicted_probabilities=PROBABILITIES,
            configuration_ids=CONFIGURATION_IDS,
            normalized_costs=NORMALIZED_COSTS,
            performance_weight=performance_weight,
        )


def test_rejects_missing_cost() -> None:
    with pytest.raises(
        ValueError,
        match="Missing normalized costs",
    ):
        select_configuration(
            predicted_probabilities=PROBABILITIES,
            configuration_ids=CONFIGURATION_IDS,
            normalized_costs={
                "high-performance": 1.0,
                "balanced": 0.3,
            },
            performance_weight=0.5,
        )


def test_rejects_probability_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="number of configuration IDs",
    ):
        select_configuration(
            predicted_probabilities=torch.tensor([0.9, 0.7]),
            configuration_ids=CONFIGURATION_IDS,
            normalized_costs=NORMALIZED_COSTS,
            performance_weight=0.5,
        )


def test_rejects_probability_outside_valid_range() -> None:
    with pytest.raises(
        ValueError,
        match="must be between 0 and 1",
    ):
        select_configuration(
            predicted_probabilities=torch.tensor([1.2, 0.7, 0.4]),
            configuration_ids=CONFIGURATION_IDS,
            normalized_costs=NORMALIZED_COSTS,
            performance_weight=0.5,
        )


def test_calculate_chebyshev_penalties() -> None:
    penalties = calculate_chebyshev_penalties(
        predicted_probabilities=PROBABILITIES,
        configuration_ids=CONFIGURATION_IDS,
        normalized_costs=NORMALIZED_COSTS,
        performance_weight=0.5,
    )

    expected = torch.tensor(
        [
            0.5,
            0.15,
            0.3,
        ]
    )

    assert torch.allclose(
        penalties,
        expected,
    )


def test_chebyshev_selects_balanced_configuration() -> None:
    selected = select_configuration_chebyshev(
        predicted_probabilities=PROBABILITIES,
        configuration_ids=CONFIGURATION_IDS,
        normalized_costs=NORMALIZED_COSTS,
        performance_weight=0.5,
    )

    assert selected == "balanced"


@pytest.mark.parametrize(
    ("performance_weight", "expected_configuration"),
    [
        (0.0, "low-cost"),
        (1.0, "high-performance"),
    ],
)
def test_chebyshev_handles_objective_endpoints(
    performance_weight: float,
    expected_configuration: str,
) -> None:
    selected = select_configuration_chebyshev(
        predicted_probabilities=PROBABILITIES,
        configuration_ids=CONFIGURATION_IDS,
        normalized_costs=NORMALIZED_COSTS,
        performance_weight=performance_weight,
    )

    assert selected == expected_configuration


def test_chebyshev_can_recover_compromise_missed_by_linear() -> None:
    probabilities = torch.tensor(
        [
            0.9,
            0.6,
            0.4,
        ]
    )
    costs = {
        "high-performance": 1.0,
        "balanced": 0.4,
        "low-cost": 0.0,
    }

    linear_selection = select_configuration(
        predicted_probabilities=probabilities,
        configuration_ids=CONFIGURATION_IDS,
        normalized_costs=costs,
        performance_weight=0.5,
    )
    chebyshev_selection = select_configuration_chebyshev(
        predicted_probabilities=probabilities,
        configuration_ids=CONFIGURATION_IDS,
        normalized_costs=costs,
        performance_weight=0.5,
    )

    assert linear_selection == "low-cost"
    assert chebyshev_selection == "balanced"
