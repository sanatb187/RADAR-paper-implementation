from radar_bench.schemas import (
    ModelConfiguration,
    ModelSpec,
    TokenBudget,
)

QWEN3_OLLAMA_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="qwen3-0.6b",
        litellm_model="ollama/qwen3:0.6b",
    ),
    ModelSpec(
        model_id="qwen3-1.7b",
        litellm_model="ollama/qwen3:1.7b",
    ),
    ModelSpec(
        model_id="qwen3-4b",
        litellm_model="ollama/qwen3:4b",
    ),
    ModelSpec(
        model_id="qwen3-8b",
        litellm_model="ollama/qwen3:8b",
    ),
)

QWEN3_REASONING_BUDGETS: tuple[int, ...] = (
    0,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
)


def build_qwen3_ollama_configurations() -> list[ModelConfiguration]:
    """Build the 32 Qwen model-budget configurations."""

    return [
        ModelConfiguration(
            configuration_id=(f"{model.model_id}__tokens-{budget}"),
            model_spec=model,
            reasoning_budget=TokenBudget(value=budget),
        )
        for model in QWEN3_OLLAMA_MODELS
        for budget in QWEN3_REASONING_BUDGETS
    ]
