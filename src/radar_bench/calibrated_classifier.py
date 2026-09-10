import torch
from torch import nn

from radar_bench.response_matrix import ResponseMatrix


class EmbeddingCorrectnessClassifier(nn.Module):
    """Predict configuration correctness directly from query embeddings."""

    temperature: torch.Tensor

    def __init__(
        self,
        num_configurations: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__()

        if num_configurations <= 0:
            raise ValueError("num_configurations must be greater than zero")

        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than zero")

        self.num_configurations = num_configurations
        self.embedding_dimension = embedding_dimension
        self.linear = nn.Linear(
            embedding_dimension,
            num_configurations,
        )
        self.register_buffer(
            "temperature",
            torch.ones((), dtype=torch.float32),
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return uncalibrated correctness logits.

        The output has shape [num_configurations, num_queries].
        """

        if query_embeddings.ndim != 2:
            raise ValueError(
                "query_embeddings must be a 2D tensor with shape "
                "[num_queries, embedding_dimension]"
            )

        if query_embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                "Expected query embeddings with dimension "
                f"{self.embedding_dimension}, but received "
                f"{query_embeddings.shape[1]}"
            )

        return self.linear(query_embeddings).transpose(0, 1)

    def predict_probabilities(
        self,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return temperature-calibrated correctness probabilities."""

        logits = self(query_embeddings)
        return torch.sigmoid(logits / self.temperature)


def split_fit_calibration_indices(
    *,
    query_count: int,
    calibration_fraction: float,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic, disjoint fit and calibration query indices."""

    if query_count < 2:
        raise ValueError("at least two queries are required for calibration")

    if calibration_fraction <= 0.0 or calibration_fraction >= 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    calibration_count = round(query_count * calibration_fraction)
    calibration_count = max(
        1,
        min(
            calibration_count,
            query_count - 1,
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    permutation = torch.randperm(
        query_count,
        generator=generator,
    )

    calibration_indices = permutation[:calibration_count]
    fit_indices = permutation[calibration_count:]

    return fit_indices, calibration_indices


def train_calibrated_classifier(
    response_matrix: ResponseMatrix,
    query_embeddings: torch.Tensor,
    *,
    num_epochs: int = 100,
    calibration_epochs: int = 100,
    learning_rate: float = 5e-4,
    batch_size: int = 32,
    max_gradient_norm: float = 1.0,
    calibration_fraction: float = 0.2,
    random_seed: int = 42,
) -> EmbeddingCorrectnessClassifier:
    """Train a direct classifier and calibrate it on held-out queries."""

    if query_embeddings.ndim != 2:
        raise ValueError(
            "query_embeddings must be a 2D tensor with shape "
            "[num_queries, embedding_dimension]"
        )

    if response_matrix.values.shape[1] != query_embeddings.shape[0]:
        raise ValueError(
            "The number of query embeddings must match the number "
            "of response-matrix columns"
        )

    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than zero")

    if calibration_epochs <= 0:
        raise ValueError("calibration_epochs must be greater than zero")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be greater than zero")

    training_embeddings = query_embeddings.to(dtype=torch.float32)
    targets = torch.as_tensor(
        response_matrix.values,
        dtype=torch.float32,
        device=training_embeddings.device,
    )

    fit_indices, calibration_indices = split_fit_calibration_indices(
        query_count=training_embeddings.shape[0],
        calibration_fraction=calibration_fraction,
        random_seed=random_seed,
    )

    fit_indices = fit_indices.to(training_embeddings.device)
    calibration_indices = calibration_indices.to(training_embeddings.device)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)

        model = EmbeddingCorrectnessClassifier(
            num_configurations=response_matrix.values.shape[0],
            embedding_dimension=training_embeddings.shape[1],
        ).to(training_embeddings.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
        loss_function = nn.BCEWithLogitsLoss()

        fit_embeddings = training_embeddings[fit_indices]
        fit_targets = targets[:, fit_indices]
        fit_query_count = fit_embeddings.shape[0]

        model.train()

        for _ in range(num_epochs):
            permutation = torch.randperm(
                fit_query_count,
                device=training_embeddings.device,
            )

            for start in range(0, fit_query_count, batch_size):
                batch_indices = permutation[start : start + batch_size]

                optimizer.zero_grad()

                logits = model(fit_embeddings[batch_indices])
                loss = loss_function(
                    logits,
                    fit_targets[:, batch_indices],
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_gradient_norm,
                )

                optimizer.step()

        model.eval()

        with torch.no_grad():
            calibration_logits = model(
                training_embeddings[calibration_indices]
            ).detach()

        calibration_targets = targets[:, calibration_indices]

        log_temperature = nn.Parameter(
            torch.zeros(
                (),
                dtype=torch.float32,
                device=training_embeddings.device,
            )
        )
        calibration_optimizer = torch.optim.Adam(
            [log_temperature],
            lr=learning_rate,
        )

        for _ in range(calibration_epochs):
            calibration_optimizer.zero_grad()

            temperature = torch.exp(log_temperature)
            calibration_loss = loss_function(
                calibration_logits / temperature,
                calibration_targets,
            )

            calibration_loss.backward()
            calibration_optimizer.step()

            with torch.no_grad():
                log_temperature.clamp_(
                    min=-5.0,
                    max=5.0,
                )

            with torch.no_grad():
                calibrated_temperature = log_temperature.detach().exp()
                model.temperature.copy_(calibrated_temperature)

    model.eval()
    return model
