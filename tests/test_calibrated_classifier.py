import numpy as np
import pytest
import torch

from radar_bench.calibrated_classifier import (
    EmbeddingCorrectnessClassifier,
    split_fit_calibration_indices,
    train_calibrated_classifier,
)
from radar_bench.response_matrix import ResponseMatrix


def make_response_matrix() -> ResponseMatrix:
    return ResponseMatrix(
        values=np.array(
            [
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=np.int8,
        ),
        configuration_ids=(
            "config-a",
            "config-b",
        ),
        query_ids=tuple(f"query-{index}" for index in range(10)),
    )


def make_embeddings() -> torch.Tensor:
    return torch.tensor(
        [
            [-2.0, -1.0],
            [-1.5, 1.0],
            [-1.0, -1.0],
            [-0.5, 1.0],
            [-0.1, -1.0],
            [0.1, 1.0],
            [0.5, -1.0],
            [1.0, 1.0],
            [1.5, -1.0],
            [2.0, 1.0],
        ]
    )


def test_classifier_predicts_complete_probability_matrix() -> None:
    model = EmbeddingCorrectnessClassifier(
        num_configurations=3,
        embedding_dimension=4,
    )

    probabilities = model.predict_probabilities(
        torch.randn(5, 4),
    )

    assert probabilities.shape == (3, 5)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_fit_calibration_split_is_disjoint_and_reproducible() -> None:
    first_fit, first_calibration = split_fit_calibration_indices(
        query_count=10,
        calibration_fraction=0.2,
        random_seed=42,
    )
    second_fit, second_calibration = split_fit_calibration_indices(
        query_count=10,
        calibration_fraction=0.2,
        random_seed=42,
    )

    assert torch.equal(first_fit, second_fit)
    assert torch.equal(first_calibration, second_calibration)

    assert len(first_fit) == 8
    assert len(first_calibration) == 2

    fit_indices = set(first_fit.tolist())
    calibration_indices = set(first_calibration.tolist())

    assert fit_indices.isdisjoint(calibration_indices)
    assert fit_indices | calibration_indices == set(range(10))


def test_trains_reproducible_calibrated_classifier() -> None:
    response_matrix = make_response_matrix()
    embeddings = make_embeddings()

    first_model = train_calibrated_classifier(
        response_matrix,
        embeddings,
        num_epochs=50,
        calibration_epochs=50,
        learning_rate=0.05,
        batch_size=4,
        random_seed=7,
    )
    second_model = train_calibrated_classifier(
        response_matrix,
        embeddings,
        num_epochs=50,
        calibration_epochs=50,
        learning_rate=0.05,
        batch_size=4,
        random_seed=7,
    )

    first_model.eval()
    second_model.eval()

    with torch.no_grad():
        first_probabilities = first_model.predict_probabilities(
            embeddings,
        )
        second_probabilities = second_model.predict_probabilities(
            embeddings,
        )

    assert first_probabilities.shape == response_matrix.values.shape
    assert torch.allclose(
        first_probabilities,
        second_probabilities,
    )
    assert torch.isfinite(first_model.temperature)
    assert first_model.temperature.item() > 0.0


@pytest.mark.parametrize(
    "calibration_fraction",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_rejects_invalid_calibration_fraction(
    calibration_fraction: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="calibration_fraction must be between 0 and 1",
    ):
        split_fit_calibration_indices(
            query_count=10,
            calibration_fraction=calibration_fraction,
            random_seed=42,
        )


def test_rejects_too_few_queries_for_calibration() -> None:
    with pytest.raises(
        ValueError,
        match="at least two queries",
    ):
        split_fit_calibration_indices(
            query_count=1,
            calibration_fraction=0.2,
            random_seed=42,
        )


def test_training_rejects_query_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="number of query embeddings",
    ):
        train_calibrated_classifier(
            make_response_matrix(),
            torch.randn(9, 2),
        )
