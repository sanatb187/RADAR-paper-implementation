from typing import Any

import pytest
import torch

from radar_bench.embeddings import embed_queries
from radar_bench.schemas import Query


def make_query(index: int) -> Query:
    return Query(
        query_id=f"query-{index}",
        prompt=f"Question {index}?",
        choices=("One", "Two", "Three", "Four"),
        gold_answer="B",
        dataset="test",
        split="test",
    )


def test_embeds_queries() -> None:
    captured_arguments: dict[str, Any] = {}

    def fake_embed(**kwargs: Any) -> dict[str, Any]:
        captured_arguments.update(kwargs)

        return {
            "embeddings": [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        }

    embeddings = embed_queries(
        [make_query(0), make_query(1)],
        embed_function=fake_embed,
    )

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == torch.float32
    assert captured_arguments["model"] == ("qwen3-embedding:0.6b")

    prompts = captured_arguments["input"]

    assert len(prompts) == 2
    assert "Question 0?" in prompts[0]
    assert "A) One" in prompts[0]


def test_rejects_empty_queries() -> None:
    with pytest.raises(
        ValueError,
        match="queries cannot be empty",
    ):
        embed_queries([])


def test_rejects_missing_embeddings() -> None:
    def fake_embed(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {}

    with pytest.raises(
        ValueError,
        match="contains no embeddings",
    ):
        embed_queries(
            [make_query(0)],
            embed_function=fake_embed,
        )


def test_rejects_wrong_embedding_count() -> None:
    def fake_embed(**kwargs: Any) -> dict[str, Any]:
        del kwargs

        return {
            "embeddings": [
                [1.0, 2.0],
            ]
        }

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        embed_queries(
            [make_query(0), make_query(1)],
            embed_function=fake_embed,
        )


def test_rejects_nonfinite_embeddings() -> None:
    def fake_embed(**kwargs: Any) -> dict[str, Any]:
        del kwargs

        return {
            "embeddings": [
                [1.0, float("nan")],
            ]
        }

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        embed_queries(
            [make_query(0)],
            embed_function=fake_embed,
        )
