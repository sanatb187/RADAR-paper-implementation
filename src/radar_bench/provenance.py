from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from radar_bench.schemas import (
    EvaluationRecord,
    ExperimentManifest,
    ModelConfiguration,
    TokenBudget,
    VLLMRuntimeProvenance,
)

MUTABLE_MANIFEST_FIELDS = {
    "status",
    "created_at",
    "completed_at",
    "train_record_count",
    "test_record_count",
}


def load_experiment_manifest(
    path: Path,
) -> ExperimentManifest:
    """Load and validate an experiment manifest."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Could not read experiment manifest: {path}") from error

    try:
        return ExperimentManifest.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(f"Invalid experiment manifest: {path}") from error


def build_manifest_filename(
    model_id: str,
    configurations: Sequence[ModelConfiguration],
) -> str:
    budgets = sorted(
        configuration.reasoning_budget.value
        for configuration in configurations
        if isinstance(
            configuration.reasoning_budget,
            TokenBudget,
        )
    )
    budget_label = "-".join(str(budget) for budget in budgets)

    return f"manifest_{model_id}_budgets-{budget_label}.json"


def count_configuration_records(
    records: Sequence[EvaluationRecord],
    configuration_ids: set[str],
) -> int:
    return sum(
        record.generation.configuration_id in configuration_ids for record in records
    )


def validate_runtime_model(
    runtime: VLLMRuntimeProvenance,
    model_id: str,
) -> None:
    if runtime.served_model_name != model_id:
        raise ValueError(
            "Runtime served_model_name does not match --model: "
            f"{runtime.served_model_name!r} != {model_id!r}"
        )


def load_vllm_runtime_provenance(
    path: Path,
) -> VLLMRuntimeProvenance:
    """Load metadata describing the remote vLLM runtime."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Could not read vLLM runtime metadata: {path}") from error

    try:
        return VLLMRuntimeProvenance.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(f"Invalid vLLM runtime metadata: {path}") from error


def save_experiment_manifest(
    path: Path,
    manifest: ExperimentManifest,
) -> None:
    """Atomically save an experiment manifest."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        temporary_path.write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def initialize_experiment_manifest(
    path: Path,
    expected_manifest: ExperimentManifest,
) -> ExperimentManifest:
    """Create a manifest or validate a resumed experiment."""

    if not path.exists():
        save_experiment_manifest(
            path,
            expected_manifest,
        )
        return expected_manifest

    existing_manifest = load_experiment_manifest(path)

    mismatched_fields = [
        field_name
        for field_name in ExperimentManifest.model_fields
        if field_name not in MUTABLE_MANIFEST_FIELDS
        and getattr(existing_manifest, field_name)
        != getattr(expected_manifest, field_name)
    ]

    if mismatched_fields:
        raise ValueError(
            "Existing experiment manifest does not match "
            "the requested run. Mismatched fields: " + ", ".join(mismatched_fields)
        )

    return existing_manifest


def complete_experiment_manifest(
    path: Path,
    manifest: ExperimentManifest,
    *,
    completed_at: datetime,
    train_record_count: int,
    test_record_count: int,
) -> ExperimentManifest:
    """Mark an experiment manifest as completed."""

    manifest_data = manifest.model_dump()
    manifest_data.update(
        {
            "status": "completed",
            "completed_at": completed_at,
            "train_record_count": train_record_count,
            "test_record_count": test_record_count,
        }
    )

    completed_manifest = ExperimentManifest.model_validate(manifest_data)

    save_experiment_manifest(
        path,
        completed_manifest,
    )

    return completed_manifest
