import torch
from torch import nn
from radar_bench.response_matrix import ResponseMatrix


class TwoPLIRT(nn.Module):
    """Two-parameter logistic IRT model used by RADAR."""

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

        # theta_i: ability of each model configuration
        self.abilities = nn.Parameter(
            torch.empty(num_configurations)
        )

        # w_a: converts a query embedding into discrimination a_j
        self.discrimination_weights = nn.Parameter(
            torch.empty(embedding_dimension)
        )

        # w_b: converts a query embedding into difficulty b_j
        self.difficulty_weights = nn.Parameter(
            torch.empty(embedding_dimension)
        )

        nn.init.normal_(self.abilities, mean=0.0, std=0.1)
        nn.init.normal_(
            self.discrimination_weights,
            mean=0.0,
            std=0.1,
        )
        nn.init.normal_(
            self.difficulty_weights,
            mean=0.0,
            std=0.1,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return correctness logits.

        Args:
            query_embeddings:
                Tensor with shape
                [num_queries, embedding_dimension].

        Returns:
            Tensor with shape
            [num_configurations, num_queries].
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

        discrimination = (
            query_embeddings @ self.discrimination_weights
        )
        difficulty = query_embeddings @ self.difficulty_weights

        logits = discrimination.unsqueeze(0) * (
            self.abilities.unsqueeze(1)
            - difficulty.unsqueeze(0)
        )

        return logits

    def predict_probabilities(
        self,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted probabilities of correct responses."""
        return torch.sigmoid(self(query_embeddings))


def train_irt_model(
    response_matrix: ResponseMatrix,
    query_embeddings: torch.Tensor,
    num_epochs: int = 500,
    learning_rate: float = 0.01,
) -> tuple[TwoPLIRT, list[float]]:
    """Train a 2PL IRT model against a response matrix."""

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

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")

    training_embeddings = query_embeddings.to(dtype=torch.float32)

    targets = torch.as_tensor(
        response_matrix.values,
        dtype=torch.float32,
        device=training_embeddings.device,
    )

    model = TwoPLIRT(
        num_configurations=response_matrix.values.shape[0],
        embedding_dimension=training_embeddings.shape[1],
    ).to(training_embeddings.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    loss_function = nn.BCEWithLogitsLoss()

    loss_history: list[float] = []

    model.train()

    for _ in range(num_epochs):
        optimizer.zero_grad()

        logits = model(training_embeddings)
        loss = loss_function(logits, targets)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return model, loss_history