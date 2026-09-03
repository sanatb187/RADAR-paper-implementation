import torch

from radar_bench.datasets.gpqa import load_gpqa_splits
from radar_bench.embeddings import embed_queries
from radar_bench.experiment import select_query_subset


def main() -> None:
    splits = load_gpqa_splits(seed=42)

    train_queries = select_query_subset(
        splits.train,
        count=1,
        seed=42,
    )
    test_queries = select_query_subset(
        splits.test,
        count=1,
        seed=42,
    )

    train_embeddings = embed_queries(train_queries)
    test_embeddings = embed_queries(test_queries)

    assert train_embeddings.ndim == 2
    assert test_embeddings.ndim == 2
    assert train_embeddings.shape[0] == 1
    assert test_embeddings.shape[0] == 1
    assert train_embeddings.shape[1] == (test_embeddings.shape[1])
    assert torch.isfinite(train_embeddings).all()
    assert torch.isfinite(test_embeddings).all()

    print("Embedding smoke test passed")
    print(f"train_shape: {tuple(train_embeddings.shape)}")
    print(f"test_shape: {tuple(test_embeddings.shape)}")
    print(
        "train_norm:",
        torch.linalg.vector_norm(train_embeddings[0]).item(),
    )


if __name__ == "__main__":
    main()
