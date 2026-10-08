# Evaluation report: full-agent-judge

| Field | Value |
|---|---|
| Generated | 2026-10-08T07:54:43Z |
| Mode | full |
| Dataset | evals\datasets\baseline.jsonl (79 samples) |
| LLM | gemini / gemini-3.5-flash-lite |
| Embeddings | intfloat/multilingual-e5-large |
| Retrieval score threshold | 0.8 |
| Engine | agent (prompt v3) |
| Per-question time limit | 240 s |
| Rescored from | no (graded live) |
| Router | llm |
| LLM judge | yes |
| Langfuse tracing | yes (session `full-agent-judge-20261008-075443`) |

## Retrieval

Evaluated on 36 answerable questions (gate disabled).

| Metric | Value |
|---|---|
| hit@1 | 100.0% |
| hit@3 | 100.0% |
| hit@6 | 100.0% |
| MRR | 1.000 |
| section hit@1 | 91.7% |
| section hit@3 | 100.0% |
| section hit@6 | 100.0% |
| cross-language hit@1 (other-language docs only) | 94.4% |
| cross-language hit@3 (other-language docs only) | 100.0% |

| Question language | hit@1 | hit@6 | mrr |
|---|---|---|---|
| en | 100.0% | 100.0% | 1.000 |
| vi | 100.0% | 100.0% | 1.000 |

### Score-threshold calibration

36 answerable and 14 unanswerable questions (not covered by the policies, off-topic, small talk).

|  | Value |
|---|---|
| Configured threshold | 0.8 |
| Answerable questions passing it (recall) | 100.0% |
| Unanswerable questions rejected (specificity) | 85.7% |
| Lowest score of an answerable question | 0.813 |
| Highest score of an unanswerable question | 0.812 |
| **Recommended threshold** | 0.813 |

| Closest to the gate | Question id | Best score |
|---|---|---|
| answerable (lowest) | `policy-en-004` | 0.813 |
| answerable (lowest) | `policy-vi-004` | 0.821 |
| answerable (lowest) | `combined-en-002` | 0.823 |
| answerable (lowest) | `combined-vi-001` | 0.826 |
| answerable (lowest) | `combined-vi-002` | 0.829 |
| unanswerable (highest) | `trap-vi-012` | 0.812 |
| unanswerable (highest) | `trap-vi-010` | 0.800 |
| unanswerable (highest) | `trap-vi-009` | 0.791 |
| unanswerable (highest) | `trap-en-013` | 0.783 |
| unanswerable (highest) | `trap-vi-008` | 0.780 |

| threshold | recall | specificity |
|---|---|---|
| 0.70 | 100.0% | 0.0% |
| 0.72 | 100.0% | 14.3% |
| 0.74 | 100.0% | 14.3% |
| 0.76 | 100.0% | 42.9% |
| 0.78 | 100.0% | 64.3% |
| 0.80 | 100.0% | 85.7% |
| 0.82 | 97.2% | 100.0% |
| 0.84 | 77.8% | 100.0% |
| 0.86 | 50.0% | 100.0% |
| 0.88 | 19.4% | 100.0% |
| 0.90 | 0.0% | 100.0% |

## Routing

Source: **pipeline**. Accuracy 100.0% over 79 samples.

| Type | Accuracy |
|---|---|
| combined | 100.0% |
| personal | 100.0% |
| policy | 100.0% |
| trap | 100.0% |

| expected \ actual | chitchat | combined | out_of_scope | personal | policy |
|---|---|---|---|---|---|
| chitchat | 4 | 0 | 0 | 0 | 0 |
| combined | 0 | 13 | 0 | 0 | 0 |
| out_of_scope | 0 | 0 | 6 | 0 | 0 |
| personal | 0 | 0 | 0 | 28 | 0 |
| policy | 0 | 0 | 0 | 0 | 28 |

## Pipeline

| Metric | Value |
|---|---|
| routing | 100.0% |
| trajectory | 100.0% |
| answer | 100.0% |
| business | 100.0% |
| safety | 100.0% |
| language | 100.0% |

| Type | routing | trajectory | answer | language |
|---|---|---|---|---|
| combined | 100.0% | 100.0% | 100.0% | 100.0% |
| personal | 100.0% | 100.0% | 100.0% | 100.0% |
| policy | 100.0% | 100.0% | 100.0% | 100.0% |
| trap | 100.0% | 100.0% | 100.0% | 100.0% |

| Language | routing | trajectory | answer | language |
|---|---|---|---|---|
| en | 100.0% | 100.0% | 100.0% | 100.0% |
| vi | 100.0% | 100.0% | 100.0% | 100.0% |

Tokens: 398716 in / 9273 out. Latency p50 14280 ms, p95 18890 ms. Judged by LLM: 54.

## Comparison with `baseline-full` (pipeline engine): no regression

| Metric | Baseline | This run | Change |
|---|---|---|---|
| routing | 100.0% | 100.0% | +0.0 pts |
| trajectory | 100.0% | 100.0% | +0.0 pts |
| answer | 100.0% | 100.0% | +0.0 pts |
| business | 100.0% | 100.0% | +0.0 pts |
| safety | 100.0% | 100.0% | +0.0 pts |
| language | 100.0% | 100.0% | +0.0 pts |

## Thresholds

| Metric | Value | Required | Result |
|---|---|---|---|
| routing | 100.0% | >= 92.0% | PASS |
| trajectory | 100.0% | >= 85.0% | PASS |
| answer | 100.0% | >= 85.0% | PASS |
| business | 100.0% | >= 100.0% | PASS |
| safety | 100.0% | >= 100.0% | PASS |
