# Evaluation report: aftersales-v8-gemini

| Field | Value |
|---|---|
| Generated | 2026-10-08T11:17:17Z |
| Mode | full |
| Dataset | evals\datasets\aftersales.jsonl (63 samples) |
| LLM | gemini / gemini-3.5-flash-lite |
| Embeddings | intfloat/multilingual-e5-large |
| Retrieval score threshold | 0.8 |
| Engine | agent (prompt v8) |
| Per-question time limit | 300 s |
| Rescored from | no (graded live) |
| Router | llm |
| LLM judge | no |
| Langfuse tracing | yes (session `aftersales-v8-gemini-20261008-111717`) |

## Retrieval

Evaluated on 3 answerable questions (gate disabled).

| Metric | Value |
|---|---|
| hit@1 | 100.0% |
| hit@3 | 100.0% |
| hit@6 | 100.0% |
| MRR | 1.000 |
| section hit@1 | 100.0% |
| section hit@3 | 100.0% |
| section hit@6 | 100.0% |
| cross-language hit@1 (other-language docs only) | 100.0% |
| cross-language hit@3 (other-language docs only) | 100.0% |

| Question language | hit@1 | hit@6 | mrr |
|---|---|---|---|
| en | 100.0% | 100.0% | 1.000 |
| vi | 100.0% | 100.0% | 1.000 |

### Score-threshold calibration

3 answerable and 0 unanswerable questions (not covered by the policies, off-topic, small talk).

|  | Value |
|---|---|
| Configured threshold | 0.8 |
| Answerable questions passing it (recall) | 100.0% |
| Unanswerable questions rejected (specificity) | n/a |
| Lowest score of an answerable question | 0.845 |
| Highest score of an unanswerable question | n/a |
| **Recommended threshold** | n/a |

| Closest to the gate | Question id | Best score |
|---|---|---|
| answerable (lowest) | `aft-en-034` | 0.845 |
| answerable (lowest) | `aft-vi-028` | 0.871 |
| answerable (lowest) | `aft-en-032` | 0.903 |

| threshold | recall | specificity |
|---|---|---|
| 0.70 | 100.0% | 100.0% |
| 0.72 | 100.0% | 100.0% |
| 0.74 | 100.0% | 100.0% |
| 0.76 | 100.0% | 100.0% |
| 0.78 | 100.0% | 100.0% |
| 0.80 | 100.0% | 100.0% |
| 0.82 | 100.0% | 100.0% |
| 0.84 | 100.0% | 100.0% |
| 0.86 | 66.7% | 100.0% |
| 0.88 | 33.3% | 100.0% |
| 0.90 | 33.3% | 100.0% |

## Routing

Source: **pipeline**. Accuracy 100.0% over 3 samples.

| Type | Accuracy |
|---|---|
| aftersales | 100.0% |

| expected \ actual | policy |
|---|---|
| policy | 3 |

## Pipeline

| Metric | Value |
|---|---|
| routing | 100.0% |
| trajectory | 98.4% |
| answer | 98.4% |
| business | 98.4% |
| safety | 100.0% |
| language | 100.0% |

| Type | routing | trajectory | answer | language |
|---|---|---|---|---|
| aftersales | 100.0% | 98.4% | 98.4% | 100.0% |

| Language | routing | trajectory | answer | language |
|---|---|---|---|---|
| en | 100.0% | 97.1% | 97.1% | 100.0% |
| vi | 100.0% | 100.0% | 100.0% | 100.0% |

Tokens: 716020 in / 13164 out. Latency p50 18368 ms, p95 30428 ms. Judged by LLM: 0.

### Failures (1)

- `aft-en-014`: tools ['get_order', 'check_warranty_eligibility'] vs expected ['propose_draft']; facts 0/1; request: expected warranty (decision approve), got no confirmation, 0 draft(s) created

## Thresholds

| Metric | Value | Required | Result |
|---|---|---|---|
| routing | 100.0% | >= 92.0% | PASS |
| trajectory | 98.4% | >= 85.0% | PASS |
| answer | 98.4% | >= 85.0% | PASS |
| business | 98.4% | >= 100.0% | **FAIL** |
| safety | 100.0% | >= 100.0% | PASS |
