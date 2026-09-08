from datetime import UTC, datetime
from pathlib import Path

import pytest

from radar_bench.provenance import (
    complete_experiment_manifest,
    initialize_experiment_manifest,
    load_experiment_manifest,
    load_vllm_runtime_provenance,
    save_experiment_manifest,
)
from radar_bench.schemas import (
    ExperimentManifest,
    ModelConfiguration,
    ModelSpec,
    TokenBudget,
    VLLMRuntimeProvenance,
)


def make_manifest() -> ExperimentManifest:
    configuration = ModelConfiguration(
        configuration_id="qwen3-4b-awq__tokens-256",
        model_spec=ModelSpec(
            model_id="qwen3-4b-awq",
            litellm_model="openai/qwen3-4b-awq",
        ),
        reasoning_budget=TokenBudget(value=256),
    )

    return ExperimentManifest(
        status="planned",
        created_at=datetime(
            2026,
            9,
            5,
            12,
            0,
            tzinfo=UTC,
        ),
        git_commit="abc123",
        git_dirty=False,
        dataset_id="Idavidrein/gpqa",
        dataset_config="gpqa_diamond",
        dataset_revision="revision-1",
        split_strategy="diamond-80-20",
        seed=42,
        train_query_ids=("query-1",),
        test_query_ids=("query-2",),
        configurations=(configuration,),
        runtime=VLLMRuntimeProvenance(
            served_model_name="qwen3-4b-awq",
            source_model="Qwen/Qwen3-4B-AWQ",
            model_revision="model-revision-1",
            quantization="awq",
            dtype="float16",
            max_model_length=4096,
            tensor_parallel_size=1,
            vllm_version="0.28.0",
            pytorch_version="2.13.0+cu132",
            cuda_version="13.2",
            gpu_model="Tesla T4",
            server_arguments={
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 1,
                "reasoning_parser": "qwen3",
            },
        ),
        base_url="http://127.0.0.1:8000/v1",
        request_timeout_seconds=300.0,
        request_options={
            "seed": 42,
        },
        train_output="train_n-158.jsonl",
        test_output="test_n-40.jsonl",
        train_record_count=0,
        test_record_count=0,
    )


def test_saves_and_loads_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    manifest = make_manifest()

    save_experiment_manifest(
        path,
        manifest,
    )

    assert load_experiment_manifest(path) == manifest
    assert not path.with_suffix(".json.tmp").exists()


def test_initializes_new_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    manifest = make_manifest()

    initialized = initialize_experiment_manifest(
        path,
        manifest,
    )

    assert initialized == manifest
    assert path.exists()


def test_resume_accepts_matching_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    manifest = make_manifest()

    save_experiment_manifest(
        path,
        manifest,
    )

    matching_data = manifest.model_dump()
    matching_data.update(
        {
            "status": "completed",
            "completed_at": datetime(
                2026,
                9,
                5,
                12,
                30,
                tzinfo=UTC,
            ),
            "train_record_count": 12,
            "test_record_count": 6,
        }
    )
    existing_manifest = ExperimentManifest.model_validate(matching_data)

    save_experiment_manifest(
        path,
        existing_manifest,
    )

    resumed = initialize_experiment_manifest(
        path,
        manifest,
    )

    assert resumed == existing_manifest


def test_resume_rejects_mismatched_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    manifest = make_manifest()

    save_experiment_manifest(
        path,
        manifest,
    )

    changed_data = manifest.model_dump()
    changed_data["seed"] = 7
    changed_manifest = ExperimentManifest.model_validate(changed_data)

    with pytest.raises(
        ValueError,
        match="Mismatched fields: seed",
    ):
        initialize_experiment_manifest(
            path,
            changed_manifest,
        )


def test_completes_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    manifest = make_manifest()
    completed_at = datetime(
        2026,
        9,
        5,
        12,
        30,
        tzinfo=UTC,
    )

    completed = complete_experiment_manifest(
        path,
        manifest,
        completed_at=completed_at,
        train_record_count=12,
        test_record_count=6,
    )

    assert completed.status == "completed"
    assert completed.completed_at == completed_at
    assert completed.train_record_count == 12
    assert completed.test_record_count == 6
    assert load_experiment_manifest(path) == completed


def test_loads_vllm_runtime_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.json"
    runtime = make_manifest().runtime

    path.write_text(
        runtime.model_dump_json(indent=2),
        encoding="utf-8",
    )

    assert load_vllm_runtime_provenance(path) == runtime


def test_rejects_invalid_vllm_runtime_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid vLLM runtime metadata",
    ):
        load_vllm_runtime_provenance(path)


def test_rejects_invalid_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        "not valid JSON\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid experiment manifest",
    ):
        load_experiment_manifest(path)
