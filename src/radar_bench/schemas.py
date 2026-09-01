from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class RADARSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class Query(RADARSchema):
    query_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    choices: tuple[str, ...] = Field(min_length=4, max_length=4)
    gold_answer: Literal["A", "B", "C", "D"]
    dataset: str = Field(min_length=1)
    split: Literal["train", "validation", "test"] = "train"


class ModelSpec(RADARSchema):
    model_id: str
    litellm_model: str
    revision: str | None = None


class TokenBudget(RADARSchema):
    kind: Literal["tokens"] = "tokens"
    value: int = Field(ge=0)


class EffortBudget(RADARSchema):
    kind: Literal["effort"] = "effort"
    value: Literal["low", "medium", "high"]


ReasoningBudget = Annotated[TokenBudget | EffortBudget, Field(discriminator="kind")]


class ModelConfiguration(RADARSchema):
    configuration_id: str
    model_spec: ModelSpec
    reasoning_budget: ReasoningBudget


class TokenUsage(RADARSchema):
    prompt_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)

    @property
    def output_tokens(self) -> int:
        return self.reasoning_tokens + self.completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


class GenerationResult(RADARSchema):
    generation_id: str
    query_id: str
    configuration_id: str
    response_text: str
    reasoning_text: str | None = None
    token_usage: TokenUsage
    latency_seconds: float = Field(ge=0)
    run_index: int = Field(default=0, ge=0)


class EvaluationRecord(RADARSchema):
    generation: GenerationResult
    parsed_answer: str | None
    correct: bool


class Pricing(RADARSchema):
    model_id: str
    input_price_per_million_tokens: float = Field(ge=0)
    output_price_per_million_tokens: float = Field(ge=0)
    currency: Literal["USD"] = "USD"
    source: str
    effective_date: date
