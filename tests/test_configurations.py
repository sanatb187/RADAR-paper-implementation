from radar_bench.configurations import (
    QWEN3_OLLAMA_MODELS,
    QWEN3_REASONING_BUDGETS,
    build_qwen3_ollama_configurations,
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
