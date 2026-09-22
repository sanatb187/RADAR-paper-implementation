# GPQA paper-aligned reproduction evidence

This directory contains the review artifacts for the scoped GPQA reproduction
described in
[vLLM Semantic Router issue #1166](https://github.com/vllm-project/semantic-router/issues/1166).

Raw GPQA prompts, responses, and dataset files are not committed because GPQA
is gated. The completed manifests and evaluator report contain the provenance
and aggregate results needed to review the run.

## Immutable provenance

- Generation implementation:
  [`04c06897d4e4d36dc941f1b02c4f012b36507ef0`](https://github.com/sanatb187/RADAR-paper-implementation/commit/04c06897d4e4d36dc941f1b02c4f012b36507ef0)
- Merged evaluation implementation:
  [`3ee8ec2d2294216f4c2454990895181af9848336`](https://github.com/sanatb187/RADAR-paper-implementation/commit/3ee8ec2d2294216f4c2454990895181af9848336)
- GPQA dataset revision:
  `633f5ee89ab8ad4522a9f850766b73f62147ffdd`
- Split seed: `42`
- Training split: GPQA Main excluding Diamond, 250 queries
- Test split: full GPQA-Diamond, 198 queries
- Configurations: Qwen3 4B/8B AWQ with budgets 0, 256, and 1024
- Embedding model: `qwen3-embedding:8b`
- Scalarization: Chebyshev
- Cost metric: output tokens

Each model manifest contains 750 training records and 594 test records. The
combined evaluation therefore contains 1,500 training records and 1,188 test
records across six configurations.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| `manifest_qwen3-4b-awq_budgets-0-256-1024.json` | `383b2345c87bf7f7a2055ecc913e4286971b2647d9af4db0b4497910b4f32d1c` |
| `manifest_qwen3-8b-awq_budgets-0-256-1024.json` | `ab36684d43b0c988a88cf06f27736f580bc905fb30aade20565db08f02a98972` |
| `evaluation_chebyshev_output-tokens.txt` | `0b34c494d4e97713cca123fa0c43668897e2b917b72a9f4b784fbbaa4cb7e98f` |

The evaluator output includes:

- Fixed, RADAR, calibrated-classifier, routing-sampling, and Random-Pair results
- Hypervolume comparisons
- IRT and classifier calibration metrics
- Per-strategy area under the risk-coverage curve
- Proxy CPT results
- Query-level routing diagnostics

## Headline results

| Strategy | Hypervolume | Difference from fixed |
|---|---:|---:|
| Fixed configurations | 0.347359 | — |
| RADAR | 0.330996 | -0.016363 |
| Calibrated classifier | 0.352483 | +0.005124 |
| Random-Pair | 0.340875 | -0.006484 |

Additional results:

- Best fixed accuracy: `0.419`
- Best RADAR accuracy: `0.338`
- Oracle accuracy: `0.682`
- IRT training loss: `0.394328`
- IRT test loss: `0.699372`
- IRT test Brier score: `0.240454`
- IRT test expected calibration error: `0.120617`
- Classifier test Brier score: `0.227512`
- Classifier test expected calibration error: `0.028726`

Best reported AURC values within each strategy family:

| Strategy family | Best AURC |
|---|---:|
| Fixed configurations | 0.495806 |
| Routing sampling | 0.511346 |
| RADAR | 0.584680 |
| Calibrated classifier | 0.534413 |

Lower AURC is better. Full per-strategy values are available in the evaluator
output.

## Evaluation command

```bash
RESULT_DIR="outputs/gpqa-vllm-paper-reproduction/paper-split_revision-633f5ee89ab8_seed-42"

uv run python scripts/evaluate_gpqa_experiment.py \
  --train-records "$RESULT_DIR/train_n-250.jsonl" \
  --test-records "$RESULT_DIR/test_n-198.jsonl" \
  --paper-split \
  --embedding-model qwen3-embedding:8b \
  --scalarization chebyshev \
  --cost-metric output-tokens
```

## Scope limitation
This is a scoped reproduction rather than the paper's complete experimental
suite. It uses one benchmark, six self-hosted AWQ configurations, and
output-token cost instead of provider pricing.