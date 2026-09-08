from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from radar_bench.schemas import (
    EvaluationRecord,
    ExperimentManifest,
    GenerationResult,
    ModelConfiguration,
    ModelSpec,
    Pricing,
    Query,
    TokenBudget,
    TokenUsage,
    VLLMRuntimeProvenance,
)


def test_query_uses_train_as_default_split() -> None:
    query = Query(
        query_id="gpqa-001",
        prompt="Which answer is correct?",
        choices=("First", "Second", "Third", "Fourth"),
        gold_answer="B",
        dataset="gpqa",
    )

    assert query.query_id == "gpqa-001"
    assert query.split == "train"
    assert len(query.choices) == 4


def test_token_budget_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        TokenBudget(value=-1)


def test_model_configuration_accepts_token_budget() -> None:
    model = ModelSpec(
        model_id="qwen3-4b",
        litellm_model="openai/Qwen/Qwen3-4B",
    )

    configuration = ModelConfiguration(
        configuration_id="qwen3-4b__tokens-512",
        model_spec=model,
        reasoning_budget=TokenBudget(value=512),
    )

    assert configuration.model_spec.model_id == "qwen3-4b"
    assert configuration.reasoning_budget.kind == "tokens"
    assert configuration.reasoning_budget.value == 512


def test_token_usage_calculates_totals() -> None:
    usage = TokenUsage(
        prompt_tokens=100,
        reasoning_tokens=500,
        completion_tokens=20,
    )

    assert usage.output_tokens == 520
    assert usage.total_tokens == 620


def test_evaluation_record_serializes_nested_models() -> None:
    generation = GenerationResult(
        generation_id="gpqa-001__qwen3-4b__512__run-0",
        query_id="gpqa-001",
        configuration_id="qwen3-4b__tokens-512",
        response_text="The correct answer is B.",
        reasoning_text="After comparing the four options...",
        token_usage=TokenUsage(
            prompt_tokens=100,
            reasoning_tokens=500,
            completion_tokens=20,
        ),
        latency_seconds=2.4,
    )

    evaluation = EvaluationRecord(
        generation=generation,
        parsed_answer="B",
        correct=True,
    )

    serialized = evaluation.model_dump()

    assert serialized["parsed_answer"] == "B"
    assert serialized["correct"] is True
    assert serialized["generation"]["query_id"] == "gpqa-001"
    assert serialized["generation"]["token_usage"]["reasoning_tokens"] == 500


def test_pricing_parses_effective_date() -> None:
    pricing = Pricing.model_validate(
        {
            "model_id": "qwen3-4b",
            "input_price_per_million_tokens": 1.0,
            "output_price_per_million_tokens": 2.0,
            "source": "https://provider.example/pricing",
            "effective_date": "2026-09-01",
        }
    )

    assert pricing.currency == "USD"
    assert pricing.effective_date == date(2026, 9, 1)


def test_pricing_rejects_negative_prices() -> None:
    with pytest.raises(ValidationError):
        Pricing.model_validate(
            {
                "model_id": "qwen3-4b",
                "input_price_per_million_tokens": -1.0,
                "output_price_per_million_tokens": 2.0,
                "source": "https://provider.example/pricing",
                "effective_date": "2026-09-01",
            }
        )


def test_experiment_manifest_round_trip() -> None:
    configuration = ModelConfiguration(
        configuration_id="qwen3-4b-awq__tokens-256",
        model_spec=ModelSpec(
            model_id="qwen3-4b-awq",
            litellm_model="openai/qwen3-4b-awq",
        ),
        reasoning_budget=TokenBudget(value=256),
    )

    manifest = ExperimentManifest(
        status="planned",
        created_at=datetime(
            2026,
            9,
            5,
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
            vllm_version="0.28.0",
            pytorch_version="2.13.0+cu132",
            cuda_version="13.2",
            gpu_model="Tesla T4",
            dtype="float16",
            max_model_length=4096,
            tensor_parallel_size=1,
            server_arguments={
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 1,
                "reasoning_parser": "qwen3",
            },
        ),
        base_url="http://127.0.0.1:8000/v1",
        request_timeout_seconds=300.0,
        request_options={"seed": 42},
        train_output="train_n-158.jsonl",
        test_output="test_n-40.jsonl",
        train_record_count=0,
        test_record_count=0,
    )

    restored = ExperimentManifest.model_validate_json(manifest.model_dump_json())

    assert restored == manifest
    assert restored.schema_version == 1
    assert restored.status == "planned"
    assert restored.runtime.quantization == "awq"
