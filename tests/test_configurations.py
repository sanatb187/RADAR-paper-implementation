from radar_bench.configurations import (
    QWEN3_OLLAMA_MODELS,
    QWEN3_REASONING_BUDGETS,
    QWEN3_VLLM_MODELS,
    build_qwen3_ollama_configurations,
    build_qwen3_vllm_configurations,
)
from radar_bench.schemas import TokenBudget


def test_builds_all_qwen3_configurations() -> None:
    configurations = build_qwen3_ollama_configurations()

    assert len(configurations) == 32
    assert len(QWEN3_OLLAMA_MODELS) == 4
    assert len(QWEN3_REASONING_BUDGETS) == 8

    configuration_ids = {
        configuration.configuration_id for configuration in configurations
    }

    assert len(configuration_ids) == 32
    assert "qwen3-0.6b__tokens-0" in configuration_ids
    assert "qwen3-8b__tokens-16384" in configuration_ids


def test_each_model_has_every_budget() -> None:
    configurations = build_qwen3_ollama_configurations()

    for model in QWEN3_OLLAMA_MODELS:
        model_configurations = [
            configuration
            for configuration in configurations
            if configuration.model_spec.model_id == model.model_id
        ]

        budgets = {
            configuration.reasoning_budget.value
            for configuration in model_configurations
            if isinstance(
                configuration.reasoning_budget,
                TokenBudget,
            )
        }

        assert budgets == set(QWEN3_REASONING_BUDGETS)


def test_uses_ollama_model_names() -> None:
    configurations = build_qwen3_ollama_configurations()

    assert all(
        configuration.model_spec.litellm_model.startswith("ollama/qwen3:")
        for configuration in configurations
    )


def test_builds_qwen3_vllm_configurations() -> None:
    configurations = build_qwen3_vllm_configurations()

    assert len(QWEN3_VLLM_MODELS) == 2
    assert len(configurations) == (
        len(QWEN3_VLLM_MODELS) * len(QWEN3_REASONING_BUDGETS)
    )

    configuration_ids = {
        configuration.configuration_id for configuration in configurations
    }

    assert "qwen3-4b-awq__tokens-0" in configuration_ids
    assert "qwen3-8b-awq__tokens-16384" in configuration_ids


def test_vllm_models_use_served_model_names() -> None:
    configurations = build_qwen3_vllm_configurations()

    assert {configuration.model_spec.model_id for configuration in configurations} == {
        "qwen3-4b-awq",
        "qwen3-8b-awq",
    }

    assert all(
        configuration.model_spec.litellm_model
        == f"openai/{configuration.model_spec.model_id}"
        for configuration in configurations
    )
