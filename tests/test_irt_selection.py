from typing import Any

import numpy as np
import pytest
import torch

from radar_bench import irt_selection
from radar_bench.irt_selection import (
    IRTHyperparameters,
    IRTTrainingSplit,
    select_irt_hyperparameters,
    split_irt_training_data,
)
from radar_bench.response_matrix import ResponseMatrix


def make_response_matrix() -> ResponseMatrix:
    return ResponseMatrix(
        values=np.array(
            [
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 1],
            ],
            dtype=np.int8,
        ),
        configuration_ids=(
            "config-a",
            "config-b",
        ),
        query_ids=(
            "query-0",
            "query-1",
            "query-2",
            "query-3",
            "query-4",
        ),
    )


def make_embeddings() -> torch.Tensor:
    return torch.arange(
        10,
        dtype=torch.float32,
    ).reshape(5, 2)


def test_split_is_deterministic() -> None:
    matrix = make_response_matrix()
    embeddings = make_embeddings()

    first = split_irt_training_data(
        matrix,
        embeddings,
        validation_fraction=0.4,
        random_seed=42,
    )
    second = split_irt_training_data(
        matrix,
        embeddings,
        validation_fraction=0.4,
        random_seed=42,
    )

    assert first.fit_matrix.query_ids == second.fit_matrix.query_ids
    assert first.validation_matrix.query_ids == second.validation_matrix.query_ids
    assert torch.equal(
        first.fit_embeddings,
        second.fit_embeddings,
    )
    assert torch.equal(
        first.validation_embeddings,
        second.validation_embeddings,
    )


def test_split_has_no_overlap_and_preserves_all_queries() -> None:
    split = split_irt_training_data(
        make_response_matrix(),
        make_embeddings(),
        validation_fraction=0.4,
        random_seed=42,
    )

    fit_ids = set(split.fit_matrix.query_ids)
    validation_ids = set(split.validation_matrix.query_ids)

    assert fit_ids.isdisjoint(validation_ids)

    assert fit_ids | validation_ids == {
        "query-0",
        "query-1",
        "query-2",
        "query-3",
        "query-4",
    }

    assert len(fit_ids) == 3
    assert len(validation_ids) == 2


def test_split_preserves_embedding_alignment() -> None:
    embeddings = make_embeddings()

    split = split_irt_training_data(
        make_response_matrix(),
        embeddings,
        validation_fraction=0.4,
        random_seed=42,
    )

    for query_id, embedding in zip(
        split.fit_matrix.query_ids,
        split.fit_embeddings,
        strict=True,
    ):
        original_index = int(query_id.removeprefix("query-"))

        assert torch.equal(
            embedding,
            embeddings[original_index],
        )

    for query_id, embedding in zip(
        split.validation_matrix.query_ids,
        split.validation_embeddings,
        strict=True,
    ):
        original_index = int(query_id.removeprefix("query-"))

        assert torch.equal(
            embedding,
            embeddings[original_index],
        )


@pytest.mark.parametrize(
    "validation_fraction",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_rejects_invalid_validation_fraction(
    validation_fraction: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="validation_fraction",
    ):
        split_irt_training_data(
            make_response_matrix(),
            make_embeddings(),
            validation_fraction=validation_fraction,
        )


def test_rejects_embedding_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="number of query embeddings",
    ):
        split_irt_training_data(
            make_response_matrix(),
            torch.randn(4, 2),
        )


def test_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(
        ValueError,
        match="num_epochs",
    ):
        IRTHyperparameters(
            num_epochs=0,
            learning_rate=0.01,
        )

    with pytest.raises(
        ValueError,
        match="learning_rate",
    ):
        IRTHyperparameters(
            num_epochs=100,
            learning_rate=0.0,
        )


class FakeIRTModel(torch.nn.Module):
    def __init__(
        self,
        logit: float,
    ) -> None:
        super().__init__()
        self.logit = logit

    def forward(
        self,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full(
            (1, query_embeddings.shape[0]),
            self.logit,
            dtype=torch.float32,
            device=query_embeddings.device,
        )


def test_selects_candidate_with_lowest_validation_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_matrix = ResponseMatrix(
        values=np.array([[1, 0]], dtype=np.int8),
        configuration_ids=("config-a",),
        query_ids=("fit-1", "fit-2"),
    )
    validation_matrix = ResponseMatrix(
        values=np.array([[1, 1]], dtype=np.int8),
        configuration_ids=("config-a",),
        query_ids=("validation-1", "validation-2"),
    )

    training_split = IRTTrainingSplit(
        fit_matrix=fit_matrix,
        validation_matrix=validation_matrix,
        fit_embeddings=torch.randn(2, 3),
        validation_embeddings=torch.randn(2, 3),
    )

    def fake_train_irt_model(
        **keyword_arguments: Any,
    ) -> tuple[FakeIRTModel, list[float]]:
        learning_rate = keyword_arguments["learning_rate"]

        if learning_rate == 0.01:
            return FakeIRTModel(logit=5.0), [0.1]

        return FakeIRTModel(logit=-5.0), [0.9]

    monkeypatch.setattr(
        irt_selection,
        "train_irt_model",
        fake_train_irt_model,
    )

    selected = select_irt_hyperparameters(
        training_split,
        candidates=(
            IRTHyperparameters(
                num_epochs=100,
                learning_rate=5e-4,
            ),
            IRTHyperparameters(
                num_epochs=100,
                learning_rate=0.01,
            ),
        ),
    )

    assert selected.selected_hyperparameters == IRTHyperparameters(
        num_epochs=100,
        learning_rate=0.01,
    )
    assert selected.validation_loss < 0.01
    assert len(selected.scores) == 2


def test_hyperparameter_selection_is_deterministic() -> None:
    training_split = split_irt_training_data(
        make_response_matrix(),
        make_embeddings(),
        validation_fraction=0.4,
        random_seed=42,
    )

    candidates = (
        IRTHyperparameters(
            num_epochs=2,
            learning_rate=0.001,
        ),
        IRTHyperparameters(
            num_epochs=3,
            learning_rate=0.001,
        ),
    )

    first = select_irt_hyperparameters(
        training_split,
        candidates,
        random_seed=42,
    )
    second = select_irt_hyperparameters(
        training_split,
        candidates,
        random_seed=42,
    )

    assert first.selected_hyperparameters == (second.selected_hyperparameters)
    assert first.validation_loss == second.validation_loss
    assert first.scores == second.scores


def test_rejects_empty_candidate_list() -> None:
    training_split = split_irt_training_data(
        make_response_matrix(),
        make_embeddings(),
    )

    with pytest.raises(
        ValueError,
        match="candidates cannot be empty",
    ):
        select_irt_hyperparameters(
            training_split,
            candidates=(),
        )
