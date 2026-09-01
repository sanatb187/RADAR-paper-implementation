import pytest

from radar_bench.datasets import gpqa
from radar_bench.datasets.gpqa import convert_gpqa_row

SAMPLE_ROW = {
    "Record ID": "sample-001",
    "Question": "Which option is correct?",
    "Correct Answer": "Correct",
    "Incorrect Answer 1": "Incorrect one",
    "Incorrect Answer 2": "Incorrect two",
    "Incorrect Answer 3": "Incorrect three",
}


def test_convert_gpqa_row() -> None:
    query = convert_gpqa_row(SAMPLE_ROW, seed=42)

    assert query.query_id == "gpqa_diamond::sample-001"
    assert query.prompt == "Which option is correct?"
    assert query.dataset == "gpqa_diamond"
    assert query.split == "train"

    assert set(query.choices) == {
        "Correct",
        "Incorrect one",
        "Incorrect two",
        "Incorrect three",
    }

    correct_index = ord(query.gold_answer) - ord("A")

    assert query.choices[correct_index] == "Correct"


def test_choice_order_is_deterministic() -> None:
    first = convert_gpqa_row(SAMPLE_ROW, seed=42)
    second = convert_gpqa_row(SAMPLE_ROW, seed=42)

    assert first.choices == second.choices
    assert first.gold_answer == second.gold_answer


def test_missing_required_field_raises_error() -> None:
    incomplete_row = {
        key: value for key, value in SAMPLE_ROW.items() if key != "Correct Answer"
    }

    with pytest.raises(
        ValueError,
        match="Correct Answer",
    ):
        convert_gpqa_row(incomplete_row)


def test_load_gpqa_diamond(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_arguments: dict[str, object] = {}

    def fake_load_dataset(
        dataset_id: str,
        config_name: str,
        *,
        split: str,
        revision: str | None,
    ) -> list[dict[str, str]]:
        captured_arguments.update(
            {
                "dataset_id": dataset_id,
                "config_name": config_name,
                "split": split,
                "revision": revision,
            }
        )

        return [SAMPLE_ROW]

    monkeypatch.setattr(
        gpqa,
        "load_dataset",
        fake_load_dataset,
    )

    queries = gpqa.load_gpqa_diamond(
        seed=42,
        revision="test-revision",
    )

    assert len(queries) == 1
    assert queries[0].query_id == "gpqa_diamond::sample-001"

    assert captured_arguments == {
        "dataset_id": "Idavidrein/gpqa",
        "config_name": "gpqa_diamond",
        "split": "train",
        "revision": "test-revision",
    }
