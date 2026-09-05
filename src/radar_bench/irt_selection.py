from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from radar_bench.irt import TwoPLIRT, train_irt_model
from radar_bench.response_matrix import ResponseMatrix


@dataclass(frozen=True)
class IRTHyperparameters:
    num_epochs: int
    learning_rate: float

    def __post_init__(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be greater than zero")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")


SMALL_DATA_IRT_CANDIDATES: tuple[IRTHyperparameters, ...] = tuple(
    IRTHyperparameters(num_epochs=num_epochs, learning_rate=learning_rate)
    for num_epochs in (
        25,
        50,
        100,
        200,
        500,
    )
    for learning_rate in (
        5e-4,
        1e-3,
        5e-3,
        1e-2,
    )
)


@dataclass(frozen=True)
class IRTTrainingSplit:
    fit_matrix: ResponseMatrix
    validation_matrix: ResponseMatrix
    fit_embeddings: torch.Tensor
    validation_embeddings: torch.Tensor


@dataclass(frozen=True)
class IRTValidationScore:
    hyperparameters: IRTHyperparameters
    validation_loss: float


@dataclass(frozen=True)
class IRTSelectionResult:
    selected_hyperparameters: IRTHyperparameters
    validation_loss: float
    scores: tuple[IRTValidationScore, ...]


@dataclass(frozen=True)
class IRTSelectedModel:
    model: TwoPLIRT
    loss_history: tuple[float, ...]
    selection: IRTSelectionResult


def split_irt_training_data(
    response_matrix: ResponseMatrix,
    query_embeddings: torch.Tensor,
    *,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
) -> IRTTrainingSplit:
    """Split training queries into deterministic fit and validation sets."""

    if query_embeddings.ndim != 2:
        raise ValueError("query_embeddings must be a 2D tensor")

    query_count = len(response_matrix.query_ids)

    if response_matrix.values.shape[1] != query_count:
        raise ValueError("response matrix values must match query IDs")

    if query_embeddings.shape[0] != query_count:
        raise ValueError(
            "The number of query embeddings must match the response matrix"
        )

    if query_count < 2:
        raise ValueError("At least two queries are required for validation")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    validation_count = round(query_count * validation_fraction)
    validation_count = max(1, validation_count)
    validation_count = min(query_count - 1, validation_count)

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    shuffled_indices = torch.randperm(
        query_count,
        generator=generator,
    ).tolist()

    validation_indices = sorted(shuffled_indices[:validation_count])
    fit_indices = sorted(shuffled_indices[validation_count:])

    fit_tensor_indices = torch.tensor(
        fit_indices,
        dtype=torch.long,
        device=query_embeddings.device,
    )
    validation_tensor_indices = torch.tensor(
        validation_indices,
        dtype=torch.long,
        device=query_embeddings.device,
    )

    fit_matrix = ResponseMatrix(
        values=response_matrix.values[:, fit_indices].copy(),
        configuration_ids=response_matrix.configuration_ids,
        query_ids=tuple(response_matrix.query_ids[index] for index in fit_indices),
    )

    validation_matrix = ResponseMatrix(
        values=response_matrix.values[:, validation_indices].copy(),
        configuration_ids=response_matrix.configuration_ids,
        query_ids=tuple(
            response_matrix.query_ids[index] for index in validation_indices
        ),
    )

    return IRTTrainingSplit(
        fit_matrix=fit_matrix,
        validation_matrix=validation_matrix,
        fit_embeddings=query_embeddings.index_select(
            0,
            fit_tensor_indices,
        ),
        validation_embeddings=query_embeddings.index_select(
            0,
            validation_tensor_indices,
        ),
    )


def select_irt_hyperparameters(
    training_split: IRTTrainingSplit,
    candidates: Sequence[IRTHyperparameters],
    *,
    batch_size: int = 32,
    max_gradient_norm: float = 1.0,
    random_seed: int = 42,
) -> IRTSelectionResult:
    """Select IRT hyperparameters using validation BCE."""

    if not candidates:
        raise ValueError("candidates cannot be empty")

    validation_targets = torch.as_tensor(
        training_split.validation_matrix.values,
        dtype=torch.float32,
        device=training_split.validation_embeddings.device,
    )

    scores: list[IRTValidationScore] = []

    for hyperparameters in candidates:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)

            model, _ = train_irt_model(
                response_matrix=training_split.fit_matrix,
                query_embeddings=training_split.fit_embeddings,
                num_epochs=hyperparameters.num_epochs,
                learning_rate=hyperparameters.learning_rate,
                batch_size=batch_size,
                max_gradient_norm=max_gradient_norm,
            )

        model.eval()

        with torch.no_grad():
            validation_logits = model(training_split.validation_embeddings)
            validation_loss = float(
                F.binary_cross_entropy_with_logits(
                    validation_logits,
                    validation_targets,
                ).item()
            )

        scores.append(
            IRTValidationScore(
                hyperparameters=hyperparameters,
                validation_loss=validation_loss,
            )
        )

    selected_score = min(
        scores,
        key=lambda score: (
            score.validation_loss,
            score.hyperparameters.num_epochs,
            score.hyperparameters.learning_rate,
        ),
    )

    return IRTSelectionResult(
        selected_hyperparameters=selected_score.hyperparameters,
        validation_loss=selected_score.validation_loss,
        scores=tuple(scores),
    )


def train_irt_model_with_selection(
    response_matrix: ResponseMatrix,
    query_embeddings: torch.Tensor,
    candidates: Sequence[IRTHyperparameters],
    *,
    validation_fraction: float = 0.2,
    batch_size: int = 32,
    max_gradient_norm: float = 1.0,
    random_seed: int = 42,
) -> IRTSelectedModel:
    """Select hyperparameters, then retrain on all training queries."""

    training_split = split_irt_training_data(
        response_matrix,
        query_embeddings,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )

    selection = select_irt_hyperparameters(
        training_split,
        candidates,
        batch_size=batch_size,
        max_gradient_norm=max_gradient_norm,
        random_seed=random_seed,
    )

    selected = selection.selected_hyperparameters

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)

        model, loss_history = train_irt_model(
            response_matrix=response_matrix,
            query_embeddings=query_embeddings,
            num_epochs=selected.num_epochs,
            learning_rate=selected.learning_rate,
            batch_size=batch_size,
            max_gradient_norm=max_gradient_norm,
        )

    return IRTSelectedModel(
        model=model,
        loss_history=tuple(loss_history),
        selection=selection,
    )
