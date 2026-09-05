import argparse
import json

from radar_bench.configurations import (
    QWEN3_REASONING_BUDGETS,
    QWEN3_VLLM_MODELS,
    build_qwen3_vllm_configurations,
)
from radar_bench.evaluators.multiple_choice import (
    evaluate_multiple_choice_generation,
)
from radar_bench.schemas import Query, TokenBudget
from radar_bench.vllm_generation import run_vllm_generation

MODEL_IDS = tuple(model.model_id for model in QWEN3_VLLM_MODELS)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test generation through a vLLM server."
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_IDS,
        default="qwen3-4b-awq",
    )
    parser.add_argument(
        "--budget",
        type=int,
        choices=QWEN3_REASONING_BUDGETS,
        default=256,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    configuration = next(
        configuration
        for configuration in build_qwen3_vllm_configurations()
        if configuration.model_spec.model_id == arguments.model
        and isinstance(
            configuration.reasoning_budget,
            TokenBudget,
        )
        and configuration.reasoning_budget.value == arguments.budget
    )

    query = Query(
        query_id="vllm-smoke-query-1",
        prompt="What is 2 + 2?",
        choices=("3", "4", "5", "6"),
        gold_answer="B",
        dataset="smoke",
        split="test",
    )

    generation = run_vllm_generation(
        query=query,
        configuration=configuration,
        base_url=arguments.base_url,
        request_options={
            "seed": 42,
        },
    )

    evaluation = evaluate_multiple_choice_generation(
        query,
        generation,
    )

    print(
        json.dumps(
            {
                "generation": generation.model_dump(
                    mode="json",
                ),
                "parsed_answer": evaluation.parsed_answer,
                "correct": evaluation.correct,
            },
            indent=2,
        )
    )

    if not evaluation.correct:
        raise RuntimeError("vLLM generation returned an incorrect answer")

    if arguments.budget > 0 and not generation.reasoning_text:
        raise RuntimeError("vLLM returned no reasoning text")

    print("vLLM generation smoke test passed")


if __name__ == "__main__":
    main()
