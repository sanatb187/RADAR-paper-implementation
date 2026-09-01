import pytest
import torch

from radar_bench.irt import TwoPLIRT


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
        model.abilities.copy_(
            torch.tensor([1.0, 2.0])
        )
        model.discrimination_weights.copy_(
            torch.tensor([1.0, 0.0])
        )
        model.difficulty_weights.copy_(
            torch.tensor([0.0, 1.0])
        )

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