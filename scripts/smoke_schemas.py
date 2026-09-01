from radar_bench.schemas import (
    EvaluationRecord,
    GenerationResult,
    ModelConfiguration,
    ModelSpec,
    Pricing,
    Query,
    TokenBudget,
    TokenUsage,
)

query = Query(
    query_id="gpqa-001",
    prompt="Which answer is correct?",
    choices=("Option A", "Option B", "Option C", "Option D"),
    gold_answer="B",
    dataset="gpqa",
)
print(query.prompt)
print(query.split)

qwen3 = ModelSpec(
    model_id="qwen3-4b",
    litellm_model="openai/Qwen/Qwen3-4B",
)

configuration = ModelConfiguration(
    configuration_id="qwen3-4b__tokens-512",
    model_spec=qwen3,
    reasoning_budget=TokenBudget(value=512),
)
print(configuration.model_spec.litellm_model)
print(configuration.reasoning_budget.value)

usage = TokenUsage(
    prompt_tokens=100,
    reasoning_tokens=500,
    completion_tokens=20,
)

print(usage.output_tokens)  # 520
print(usage.total_tokens)   # 620

generation = GenerationResult(
    generation_id="gpqa-001__qwen3-4b__512__run-0",
    query_id="gpqa-001",
    configuration_id="qwen3-4b__tokens-512",
    response_text="The correct answer is B.",
    reasoning_text="After comparing the four options...",
    token_usage=usage,
    latency_seconds=2.4,
)

evaluation = EvaluationRecord(
    generation=generation,
    parsed_answer="B",
    correct=True,
)
print(evaluation.model_dump())
print(evaluation.model_dump_json(indent=2))

pricing = Pricing(
    model_id="qwen3-4b",
    input_price_per_million_tokens=1.0,
    output_price_per_million_tokens=2.0,
    source="https://provider.example/pricing",
    effective_date="2026-09-01",
)

print(pricing.model_dump())
print(type(pricing.effective_date))