from radar_bench.evaluators.multiple_choice import (
    evaluate_multiple_choice_generation,
)
from radar_bench.ollama_generation import run_ollama_generation
from radar_bench.schemas import (
    ModelConfiguration,
    ModelSpec,
    Query,
    TokenBudget,
)


def main() -> None:
    query = Query(
        query_id="smoke-query-1",
        prompt="What is 2 + 2?",
        choices=("3", "4", "5", "6"),
        gold_answer="B",
        dataset="smoke",
        split="test",
    )

    configuration = ModelConfiguration(
        configuration_id="qwen3-0.6b__tokens-256",
        model_spec=ModelSpec(
            model_id="qwen3-0.6b",
            litellm_model="ollama/qwen3:0.6b",
        ),
        reasoning_budget=TokenBudget(value=256),
    )

    generation = run_ollama_generation(
        query=query,
        configuration=configuration,
        request_options={
            "temperature": 0.0,
            "seed": 42,
            "num_ctx": 4096,
        },
    )

    evaluation = evaluate_multiple_choice_generation(
        query=query,
        generation=generation,
    )

    print(evaluation.model_dump_json(indent=2))

    assert evaluation.parsed_answer == "B"
    assert evaluation.correct
    assert generation.reasoning_text
    assert generation.token_usage.reasoning_tokens > 0
    assert generation.token_usage.completion_tokens > 0

    print("Ollama generation smoke test passed")


if __name__ == "__main__":
    main()
