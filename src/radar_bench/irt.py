import torch
from torch import nn


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
            torch.zeros(num_configurations)
        )

        # w_a: converts a query embedding into discrimination a_j
        self.discrimination_weights = nn.Parameter(
            torch.zeros(embedding_dimension)
        )

        # w_b: converts a query embedding into difficulty b_j
        self.difficulty_weights = nn.Parameter(
            torch.zeros(embedding_dimension)
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