# Evaluation report: baseline-offline

| Field | Value |
|---|---|
| Generated | 2026-10-07T09:37:57Z |
| Mode | offline |
| Dataset | evals\datasets\baseline.jsonl (79 samples) |
| LLM | not used (offline) |
| Embeddings | intfloat/multilingual-e5-large |
| Retrieval score threshold | 0.8 |
| Router | heuristic |
| LLM judge | no |
| Langfuse tracing | no |

> Offline run: no LLM was called. Retrieval and the heuristic router are measured; answer quality, trajectory, business correctness and safety need `support-agent eval` with a provider key.

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

Source: **heuristic**. Accuracy 81.0% over 79 samples.

| Type | Accuracy |
|---|---|
| combined | 58.3% |
| personal | 87.5% |
| policy | 83.3% |
| trap | 85.2% |

| expected \ actual | chitchat | combined | out_of_scope | personal | policy |
|---|---|---|---|---|---|
| chitchat | 4 | 0 | 0 | 0 | 0 |
| combined | 0 | 8 | 0 | 5 | 0 |
| out_of_scope | 0 | 0 | 6 | 0 | 0 |
| personal | 0 | 0 | 2 | 26 | 0 |
| policy | 1 | 2 | 5 | 0 | 20 |

Mis-routed: `policy-vi-005`, `policy-vi-006`, `policy-en-010`, `policy-en-011`, `personal-en-008`, `personal-vi-008`, `combined-vi-001`, `combined-vi-003`, `combined-vi-005`, `combined-en-006`, `combined-vi-006`, `trap-en-013`, `trap-vi-012`, `trap-en-014`, `trap-vi-013`

## Thresholds

| Metric | Value | Required | Result |
|---|---|---|---|
| routing | 81.0% | >= 92.0% | **FAIL** |
| trajectory | n/a | >= 85.0% | not measured |
| answer | n/a | >= 85.0% | not measured |
| business | n/a | >= 100.0% | not measured |
| safety | n/a | >= 100.0% | not measured |
