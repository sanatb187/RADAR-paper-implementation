from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from ollama import embed

from radar_bench.evaluators.multiple_choice import (
    format_multiple_choice_prompt,
)
from radar_bench.schemas import Query

EmbeddingFunction = Callable[..., Any]

DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"


def _get_field(
    value: Any,
    field: str,
    default: Any = None,
) -> Any:
    if isinstance(value, Mapping):
        return value.get(field, default)

    return getattr(value, field, default)


def embed_queries(
    queries: Sequence[Query],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    embed_function: EmbeddingFunction = embed,
) -> torch.Tensor:
    """Embed multiple-choice queries using Ollama."""

    if not queries:
        raise ValueError("queries cannot be empty")

    if not model:
        raise ValueError("model cannot be empty")

    prompts = [format_multiple_choice_prompt(query) for query in queries]

    response = embed_function(
        model=model,
        input=prompts,
    )

    embeddings = _get_field(response, "embeddings")

    if embeddings is None:
        raise ValueError("Ollama embedding response contains no embeddings")

    if len(embeddings) != len(queries):
        raise ValueError(
            "The number of embeddings does not match the number of queries"
        )

    try:
        embedding_tensor = torch.tensor(
            embeddings,
            dtype=torch.float32,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Ollama returned invalid embeddings") from error

    if embedding_tensor.ndim != 2:
        raise ValueError("Ollama embeddings must form a 2D tensor")

    if not torch.isfinite(embedding_tensor).all():
        raise ValueError("Ollama embeddings must contain finite values")

    return embedding_tensor
