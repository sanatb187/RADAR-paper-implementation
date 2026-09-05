import numpy as np
import pytest
import torch

from radar_bench.irt import TwoPLIRT, train_irt_model
from radar_bench.response_matrix import ResponseMatrix


def test_output_shape() -> None:
    model = TwoPLIRT(
        num_configurations=3,
        embedding_dimension=4,
    )

    query_embeddings = torch.randn(5, 4)

    logits = model(query_embeddings)

    assert logits.shape == (3, 5)


def test_probabilities_are_between_zero_and_one() -> None:
    model = TwoPLIRT(
        num_configurations=2,
        embedding_dimension=3,
    )

    query_embeddings = torch.randn(4, 3)

    probabilities = model.predict_probabilities(query_embeddings)

    assert probabilities.shape == (2, 4)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)


def test_known_irt_calculation() -> None:
    model = TwoPLIRT(
        num_configurations=2,
        embedding_dimension=2,
    )

    with torch.no_grad():
        model.abilities.copy_(torch.tensor([1.0, 2.0]))
        model.discrimination_weights.copy_(torch.tensor([1.0, 0.0]))
        model.difficulty_weights.copy_(torch.tensor([0.0, 1.0]))

    query_embeddings = torch.tensor(
        [
            [2.0, 0.5],
            [1.0, 1.5],
        ]
    )

    logits = model(query_embeddings)

    expected = torch.tensor(
        [
            [1.0, -0.5],
            [3.0, 0.5],
        ]
    )

    assert torch.allclose(logits, expected)


def test_all_parameters_receive_gradients() -> None:
    model = TwoPLIRT(
        num_configurations=2,
        embedding_dimension=3,
    )

    query_embeddings = torch.randn(4, 3)
    targets = torch.randint(
        low=0,
        high=2,
        size=(2, 4),
    ).float()

    logits = model(query_embeddings)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
    )

    loss.backward()

    assert model.abilities.grad is not None
    assert model.discrimination_weights.grad is not None
    assert model.difficulty_weights.grad is not None

    assert torch.any(model.abilities.grad != 0)
    assert torch.any(model.discrimination_weights.grad != 0)
    assert torch.any(model.difficulty_weights.grad != 0)


def test_rejects_wrong_embedding_dimension() -> None:
    model = TwoPLIRT(
        num_configurations=2,
        embedding_dimension=3,
    )

    query_embeddings = torch.randn(4, 5)

    with pytest.raises(
        ValueError,
        match="Expected query embeddings with dimension 3",
    ):
        model(query_embeddings)


def test_rejects_non_matrix_embeddings() -> None:
    model = TwoPLIRT(
        num_configurations=2,
        embedding_dimension=3,
    )

    query_embeddings = torch.randn(3)

    with pytest.raises(
        ValueError,
        match="query_embeddings must be a 2D tensor",
    ):
        model(query_embeddings)


@pytest.mark.parametrize(
    ("num_configurations", "embedding_dimension"),
    [
        (0, 3),
        (2, 0),
        (-1, 3),
        (2, -1),
    ],
)
def test_rejects_invalid_model_dimensions(
    num_configurations: int,
    embedding_dimension: int,
) -> None:
    with pytest.raises(ValueError):
        TwoPLIRT(
            num_configurations=num_configurations,
            embedding_dimension=embedding_dimension,
        )


def test_training_rejects_invalid_batch_size() -> None:
    response_matrix = ResponseMatrix(
        values=np.array([[1, 0]], dtype=np.int8),
        configuration_ids=("config-a",),
        query_ids=("query-1", "query-2"),
    )

    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        train_irt_model(
            response_matrix=response_matrix,
            query_embeddings=torch.randn(2, 4),
            batch_size=0,
        )


def test_training_rejects_invalid_gradient_norm() -> None:
    response_matrix = ResponseMatrix(
        values=np.array([[1, 0]], dtype=np.int8),
        configuration_ids=("config-a",),
        query_ids=("query-1", "query-2"),
    )

    with pytest.raises(
        ValueError,
        match="max_gradient_norm",
    ):
        train_irt_model(
            response_matrix=response_matrix,
            query_embeddings=torch.randn(2, 4),
            max_gradient_norm=0.0,
        )


def test_training_reduces_loss() -> None:
    torch.manual_seed(42)

    response_matrix = ResponseMatrix(
        values=np.array(
            [
                [1, 0, 1, 0],
                [1, 1, 1, 0],
            ],
            dtype=np.int8,
        ),
        configuration_ids=("config-a", "config-b"),
        query_ids=("query-1", "query-2", "query-3", "query-4"),
    )

    query_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    _, loss_history = train_irt_model(
        response_matrix=response_matrix,
        query_embeddings=query_embeddings,
        num_epochs=300,
        learning_rate=0.05,
    )

    assert len(loss_history) == 300
    assert loss_history[-1] < loss_history[0]


def test_trained_model_predicts_complete_matrix() -> None:
    torch.manual_seed(42)

    response_matrix = ResponseMatrix(
        values=np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.int8,
        ),
        configuration_ids=("config-a", "config-b"),
        query_ids=("query-1", "query-2", "query-3"),
    )

    query_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    model, _ = train_irt_model(
        response_matrix=response_matrix,
        query_embeddings=query_embeddings,
        num_epochs=100,
        learning_rate=0.05,
    )

    model.eval()

    with torch.no_grad():
        probabilities = model.predict_probabilities(query_embeddings)

    assert probabilities.shape == response_matrix.values.shape
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)


def test_training_rejects_query_count_mismatch() -> None:
    response_matrix = ResponseMatrix(
        values=np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.int8,
        ),
        configuration_ids=("config-a", "config-b"),
        query_ids=("query-1", "query-2", "query-3"),
    )

    query_embeddings = torch.randn(2, 4)

    with pytest.raises(
        ValueError,
        match="number of query embeddings",
    ):
        train_irt_model(
            response_matrix=response_matrix,
            query_embeddings=query_embeddings,
        )
