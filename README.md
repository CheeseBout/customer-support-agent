# Customer Support Agent

Backend for an AI customer-support assistant for small and mid-size e-commerce shops. Vietnamese
and English, single tenant. It answers shop-policy questions from your own documents, looks up a
customer's own orders through a read-only data layer, and combines both ("is my order #1234 still
eligible for return?").

Design documents: [`SPEC.md`](SPEC.md) (requirements) and [`PLAN.md`](PLAN.md) (phases).

## Status

| | |
|---|---|
| **Phase A: RAG** (PLAN phases 0-3) | **Done.** Foundations, policy RAG, personal data over MCP, baseline evaluation. |
| **Phase 4: agent loop** | **Done.** On the full 79-sample set with the judge the agent scores 100% on every metric and does not fall below the Phase A pipeline baseline, see [Evaluation](#evaluation). |
| **Phase 5: requests and human confirmation** | **Built, tested, and measured on the 63-sample after-sales set** (refund, return, warranty, order drafts; confirm / edit / reject; staff review CLI). Answer 92-98% and safety 90-100% depending on the model; business correctness is 94-98%, short of the 100% target. See [After-sales results](#after-sales-results). |
| **Phase 6: memory and workspace** | **Built and tested.** Conversation summaries, optional saved preferences, per-session files. The multi-agent split is deliberately not built: nothing shows it would beat the single agent. |
| **Phase 7: guardrails** | **Built and tested.** Input screening, rate limit, output checks, an English and Vietnamese red-team set. Phrase-based: see [docs/agent.md](docs/agent.md). |
| **Phase 9: HTTP API** | **Built and tested.** JWT auth, chat as JSON or SSE, sessions, requests, staff review, Docker. See [docs/api.md](docs/api.md). |
| Phase 8: evaluation in CI | Partly done: tracing, thresholds and `--fail-under` exist; the 4-provider comparison and an automatic CI run do not. |

You can use the system through the HTTP API (`support-agent serve`) or the CLI, where `--user`
stands in for the identity a JWT carries on the API.

## What works

* **Policy Q&A** over `.md`, `.pdf` and `.docx` with structured citations. Hybrid search (dense
  `multilingual-e5-large` + BM25) in Qdrant, incremental ingest by content hash, a calibrated
  relevance gate, and "I don't know" when nothing relevant is found. A Vietnamese question finds
  an English document and vice versa.
* **Personal data** (orders, shipments, stock, product search, return eligibility) through an MCP
  server over PostgreSQL, MySQL, SQLite or MongoDB, driven by a YAML
  [schema mapping](docs/schema-mapping.md). Domain tools only; there is no free SQL.
* **Question routing**: `policy | personal | combined | chitchat | out_of_scope`, with regex
  extraction of order ids and SKUs so lookups are exact, never embedding-based.
* **Return rules in code** (window, status, excluded categories, existing requests). The LLM only
  explains the result.
* **An agent loop** ([docs/agent.md](docs/agent.md)): the model decides which tools to call,
  step by step, and its answer streams token by token with citations. Conversations are saved
  (SQLite by default) and resume across restarts; each user's sessions are isolated; failed turns
  are cleaned up.
* **Four LLM providers** behind one setting: OpenAI, Anthropic, Gemini, OpenRouter, with
  client-side pacing and bounded retries for rate-limited API tiers.
* **An HTTP API** ([docs/api.md](docs/api.md)): JWT authentication with `customer` and `staff`
  roles, chat as JSON or Server-Sent Events, confirmation of requests, sessions, a staff review
  queue, health and readiness, OpenAPI at `/docs`.
* **Memory**: long conversations are summarised for the model while the transcript stays whole;
  saved preferences (off by default, with expiry and an erase endpoint); per-session files such
  as a product comparison table.
* **Guardrails**: prompt-injection screening, per-user rate limit, masking of contact details the
  customer's own data does not contain, withdrawal of false refund promises and prompt leaks.
* **Evaluation**: a 79-question bilingual dataset, retrieval/routing/answer/safety metrics,
  threshold calibration, optional LLM judge and Langfuse tracing.

## Quick start

Requires Python 3.12 and Docker.

```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[dev]"

support-agent init                                      # creates .env; add your provider key to it
docker compose --profile demo-db up -d                  # demo PostgreSQL (+ optional Qdrant server;
                                                        # by default Qdrant runs embedded in ./data/qdrant)

# point the agent at the demo database with its READ-ONLY account (put these in .env)
#   BUSINESS_DB_TYPE=postgres
#   BUSINESS_DB_URL=postgresql://support_ro:support_ro@localhost:5432/shop
# seeding needs the owner account, which the agent itself never uses:
support-agent seed-demo --url postgresql://shop_owner:shop_owner@localhost:5432/shop

support-agent ingest                                    # downloads the embedding model on first run
support-agent validate-mapping
support-agent check

support-agent ask "How many days do I have to return an item?"
support-agent ask "Where is order #1236?" --user u_100
support-agent ask "Đơn #1234 của tôi còn được trả không?" --user u_100
support-agent ask "Where is order #2001?" --user u_100   # belongs to u_101: reported as not found
support-agent chat --user u_100                          # several turns; prints how to resume

# the HTTP API (needs JWT_SECRET in .env; see docs/api.md for a token you can try)
support-agent serve                                      # http://127.0.0.1:8000/docs
```

`ingest`, `validate-mapping`, `check --skip-llm` and `eval --offline` need no API key. `ask`,
`chat` and the full `eval` need one.

Demo customers are `u_100`, `u_101`, `u_102`. The demo orders are built so each scenario is
reproducible: `1234` is returnable, `1235` is past the window, `1239` is a gift card,
`1240` already has a return request, `1236` is in transit.

## Commands

| Command | What it does |
|---|---|
| `init` | Create `.env` and local directories |
| `ingest [--full] [--dir]` | Load policy documents into Qdrant; unchanged files are skipped |
| `ask QUESTION [--user] [--lang] [--json] [--no-tools] [--session ID] [--engine agent\|pipeline]` | Ask as an authenticated customer; the answer streams. `--session` continues a conversation |
| `resume --session ID --id CONFIRMATION --decision approve\|reject\|edit [--set field=value]` | Answer a confirmation that `ask` left open (works after a restart) |
| `drafts init\|list\|show\|approve\|reject\|cancel` | Staff side of customer requests. `drafts init --url` creates `support_drafts`; `reject` needs `--note` |
| `chat [--user] [--lang] [--no-tools] [--session ID]` | Talk over several turns; the conversation is saved |
| `seed-demo [--url] [--reset]` | Create the demo shop database (needs a write-capable account) |
| `validate-mapping` | Check the schema mapping against the live database |
| `serve [--host] [--port]` | Run the HTTP API. Refuses to start without a valid JWT setting |
| `check [--skip-llm]` | Verify configuration, Qdrant, database and the model's capabilities |
| `eval [--engine agent\|pipeline] [--offline] [--subset ci] [--judge] [--fail-under] [--compare REPORT.json] [--concurrency N] [--run-timeout S]` | Evaluate and write a report. The agent is compared with the Phase A baseline; `--fail-under` also fails on a regression |

## Configuration

Secrets and endpoints come from the environment (`.env`, see `.env.example`); tunables live in
[`config/app.yaml`](config/app.yaml). Nothing is hard-coded.

| Variable | Meaning |
|---|---|
| `APP_CONFIG_PATH` | Which `app.yaml` to use (default `./config/app.yaml`). A shop's file can start from the shared one with `extends: ../../config/app.yaml` and list only what differs |
| `LLM_PROVIDER` | `openai` \| `anthropic` \| `gemini` \| `openrouter` |
| `LLM_MODEL` | Optional override of the provider's default model |
| `OPENAI_API_KEY` `ANTHROPIC_API_KEY` `GOOGLE_API_KEY` `OPENROUTER_API_KEY` | Key for the chosen provider |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `local` (default) \| `openai` \| `gemini` |
| `BUSINESS_DB_TYPE` / `BUSINESS_DB_URL` | `postgres \| mysql \| mongodb \| sqlite` and a **read-only** DSN |
| `QDRANT_URL` or `QDRANT_PATH` | Server, or embedded storage (default `./data/qdrant`) |
| `STRICT_CAPABILITY_CHECK` | Refuse to start if the model lacks tool calling or structured output |
| `LLM_REQUESTS_PER_MINUTE` | Pace all model calls to stay under your API tier (empty = unlimited) |
| `CHECKPOINT_URL` | Where conversations are saved: `sqlite:///./data/checkpoints.db` (default), `postgresql://...` (Linux only, `pip install ".[postgres]"`) or `memory` |
| `LANGFUSE_PUBLIC_KEY` `LANGFUSE_SECRET_KEY` `LANGFUSE_HOST` | Enable tracing (`pip install ".[langfuse]"`) |
| `JWT_ALGORITHM` `JWT_SECRET` / `JWT_JWKS_URL` | How the API verifies tokens: `HS256` with a secret of at least 32 bytes, or `RS256` with the issuer's JWKS URL |
| `JWT_AUDIENCE` `JWT_ISSUER` `JWT_CUSTOMER_CLAIM` `JWT_ROLE_CLAIM` | Optional claim checks and names (defaults `sub`, `role`) |
| `CORS_ORIGINS` | Comma-separated browser origins allowed to call the API (empty = off) |
| `SESSIONS_DB_URL` `WORKSPACE_DIR` | Session index, feedback and saved preferences (SQLite); per-session files |

Default models: `gpt-4o-mini`, `claude-haiku-4-5`, `gemini-3.5-flash-lite`, and a free OpenRouter
model that supports tools. Check these IDs against your provider before relying on them; free
OpenRouter models change often and tend to have flaky tool calling and rate limits, so use another
provider in production. `llm.temperature` is not sent to Claude models that reject it (Haiku, Sonnet and Opus 5.x, Opus 4.7/4.8, Fable), so the setting has no effect on them. At startup the capability check reports `ok`, `degraded` (the router then
falls back to JSON parsing) or `failed`.

**Mind your API tier's quotas.** Free tiers have per-minute *and per-day* limits (the free
Gemini tier allowed 500 requests per day per model when this was written). An evaluation makes
several hundred model calls, and a provider SDK that retries a rejected call can spend quota
again. Set `LLM_REQUESTS_PER_MINUTE` below your limit, keep `llm.max_retries` low, and run
`eval --subset ci` (about 30 questions) before a full run. When the daily quota is gone every
call fails with 429 until it resets, which is why a long run can look hung.

Business rules (`return`, `refund`, `warranty`, `inventory`) are in `config/app.yaml` and are the
single source of truth for decisions. The policy documents in `knowledge/` explain and cite those
rules, so keep the two consistent when you change either.

## Security model

* **Identity is never an LLM argument.** The caller's identity travels in the MCP request
  metadata as an HMAC-signed, expiring token. The server verifies it before every call and
  rejects any tool argument that tries to carry an identity (`customer_id`, `user_id`, ...).
* **Ownership is part of the query**, not a filter applied afterwards. Someone else's order and a
  non-existent order return the identical `NOT_FOUND`, so order ids cannot be probed.
* **Read-only everywhere**: the adapters only issue `SELECT`/aggregations, sessions are read-only
  where supported, and you should still give them a `SELECT`-only account (`scripts/sql/`).
* **Untrusted text is data.** Policy text, product descriptions and tool output are wrapped in
  delimiters and the model is told never to follow instructions inside them. The demo catalogue
  includes a product whose description contains an injection attempt, and the dataset tests it.
* Every tool call is audit-logged with masked arguments and a hashed user id.

* **Guardrails** back these up and do not replace them: injection phrases are refused before the
  model runs, requests are rate limited per user, and the model's answer is checked for secrets,
  prompt text, false refund promises and contact details that are not the customer's. They match
  phrases, so they stop common attacks and not a determined one. The English and Vietnamese
  attack set is in `tests/test_guardrails.py`.
* **The API trusts only the token.** The customer id comes from the verified JWT, the signing
  algorithm from configuration, and another customer's session, draft or file looks exactly like
  a missing one.
* Card numbers and stated passwords are removed from a message before it is stored or summarised.

## Evaluation

```bash
support-agent eval --offline          # no LLM: retrieval, score-threshold calibration, keyword router
support-agent eval                    # full pipeline (needs a provider key)
support-agent eval --judge --fail-under
support-agent eval --subset ci        # ~30 samples for CI
```

Reports are written to `evals/reports/`. The agent evaluation uses a throwaway drafts store, so
it never touches real requests.

### After-sales dataset

`evals/datasets/aftersales.jsonl` (63 samples, 29 Vietnamese / 34 English, 19 in the `ci` subset)
covers requests, not just answers:

```bash
support-agent eval --engine agent --dataset evals/datasets/aftersales.jsonl --subset ci
```

| Group | What it checks |
|---|---|
| Refund / return / warranty / order | The confirmation shown to the customer has the right type, amount (from the database), priority flag and items, also in multi-turn conversations |
| Edit / reject / approve | A simulated customer answers the confirmation; an edit asks again, a rejection stores nothing, an approval stores exactly one `pending` draft and the reply never promises a refund |
| Not allowed | Window expired, not delivered, gift card, request already pending, cancelled, item not on the order, COD limit, out of stock, quantity limit, payment method: nothing is proposed and the reason is given |
| Missing information | The agent asks instead of guessing (reason, problem, address, payment) |
| Safety | Someone else's order, unknown order, prompt injection in the message and in the catalogue, an invented amount, "approve it yourself", stock quantities that must stay hidden |
| Reading drafts | `list_my_drafts` shows only your own; cancelling works |

Samples that create drafts run last, one at a time, in file order. Every expected amount and
refusal in the file is checked against the seeded shop by `tests/test_aftersales_dataset.py`, so
the dataset cannot drift from the rules. The `business` metric now also covers this: the right
confirmation and the right drafts. The offline baseline on the sample corpus with the
default embedding model (see [`evals/reports/baseline-offline.md`](evals/reports/baseline-offline.md)):

| | |
|---|---|
| Retrieval hit@1 / hit@6 / MRR | 100% / 100% / 1.00 (36 answerable questions, VI and EN) |
| Section-level hit@1 | 91.7% |
| Cross-language hit@1 / hit@3 (a VI question against EN documents only, and back) | 94.4% / 100% |
| Relevance gate at 0.80 (per query) | keeps 100% of answerable questions, rejects 86% of unanswerable ones |
| Keyword-router fallback accuracy | 81% (the floor the LLM router is expected to beat) |

### After-sales results

The 63-sample after-sales set, agent engine, no judge, one run per row (the model varies a little from run to
run, so a difference of one or two samples is noise):

| Model, prompt | Answer | Business | Safety | Trajectory | Failures | Latency p50 |
|---|---|---|---|---|---|---|
| Gemini `gemini-3.5-flash-lite`, v3 (first run) | 77.8% | 88.9% | 70% | 96.8% | 15 | 15 s (paced) |
| Gemini `gemini-3.5-flash-lite`, v6 | 95.2% | 98.4% | 100% | 98.4% | 3 | 14 s (paced) |
| Gemini `gemini-3.5-flash-lite`, v8 | 98.4% | 98.4% | 100% | 98.4% | 1 | 18 s (paced) |
| Claude Haiku 5.5, v8 | 92.1% | 93.7% | 90% | 93.7% | 5 | 4.3 s |

The Gemini runs are paced to stay under the free tier's limit, so their latency is not the model's speed. Business
correctness (the right confirmation and the right drafts) is not yet 100% on either model: Gemini's one miss,
`aft-en-014`, is the model inventing a SKU instead of reading it from the order. None of the failures leaked another customer's data, invented an
amount or approved a request. The Haiku safety figure is one sample, `aft-vi-026` (a customer asking for a
10 million refund): the agent refuses the amount and asks whether to go ahead instead of proposing the request,
which the dataset counts as a miss. A Haiku run costs a few tens of cents at its list price.

Reports: `evals/reports/aftersales-v6.md`, `evals/reports/aftersales-v8-gemini.md`, `evals/reports/aftersales-v8b-haiku55.md`.

### Full baseline (the figures Phase B must not regress)

Full pipeline on all 79 samples with `gemini-3.5-flash-lite` as both assistant and judge
([`evals/reports/baseline-full.md`](evals/reports/baseline-full.md)):

| Metric | Result | Threshold |
|---|---|---|
| Routing | 100% | ≥ 92% |
| Trajectory (right tools called) | 100% | ≥ 85% |
| Answer correctness (code checks + LLM judge) | 100% | ≥ 85% |
| Business correctness (return verdict) | 100% | = 100% |
| Safety (no leaks, no injected instructions followed) | 100% | = 100% |
| Answer language | 100% | |

Read these numbers with care:

* **It is a small, hand-written set (79 questions)** and its expectations were refined after the
  first run. The first run scored answer 83.5%, business 90% and safety 72.7%
  ([`baseline-full-run1.md`](evals/reports/baseline-full-run1.md)). It exposed one real defect (the
  model refused "is X in stock?" when the stock tool said `low_stock`, fixed in the prompt) and
  several flaws in the evaluation itself (an over-strict judge, injection traps that never
  reached the model, a language check fooled by Vietnamese product names). 100% therefore means
  "the known cases pass", not "the system is flawless".
* **The judge is the same model family as the assistant**, which tends to flatter it. Treat the
  code-based checks (facts, forbidden text, tools, route, outcome) as the firm part.
* **Latency is uneven**: median 2.3 s but p95 about 37 s on this model.
* This model occasionally emits stray characters in Vietnamese refusals (for example
  "qu½ khách"). Try a stronger model if Vietnamese quality matters to you.

Thresholds (`config/app.yaml`): routing ≥ 0.92, trajectory ≥ 0.85, answer ≥ 0.85, business
correctness = 1.0, safety = 1.0. `support-agent eval --judge --fail-under` exits non-zero below
them, so it can gate a pull request.

Tracing: when the Langfuse keys are set, every evaluated question becomes a trace in a session
named after the run (a fresh session per run), tagged `eval`, type and language, with
`answer_ok`, `route_ok` and `safety_ok` scores attached.

## Tests

```bash
pytest                      # ~900 unit tests, offline, no API keys
pytest -m integration       # PostgreSQL, MySQL, MongoDB and the Postgres checkpointer, in Docker
ruff check . && ruff format --check . && mypy src
```

Language models are replaced by a scripted fake in tests (one that also streams in small pieces,
to exercise marker handling), so the router, agent loop and pipeline are deterministic and free. The adapter contract tests run unchanged against SQLite, MongoDB
(mocked) and, in the integration suite, the real PostgreSQL, MySQL and MongoDB.

## Where this differs from the spec

| Spec | What was built | Why |
|---|---|---|
| Embeddings `bge-m3` or e5 via fastembed | `intfloat/multilingual-e5-large` | fastembed does not ship bge-m3. fastembed's e5 vectors are not normalised, so they are normalised here. |
| Chunks of 400-800 tokens | 200-450 tokens | e5 truncates input at 512 tokens; longer chunks would be silently cut for the dense vector. |
| Score threshold 0.35 | 0.80, applied per query | e5 scores are compressed (relevant 0.81-0.89, unrelated 0.70-0.81). Calibrated on the sample corpus; recalibrate for your own documents and model. |
| FastMCP | `mcp` 2.x `MCPServer` | The 2.x SDK renamed it. |
| Return rule in Phase 5 | Return eligibility implemented in Phase A | The `combined` route cannot conclude correctly without it. Refund amounts, warranty and drafts came in Phase 5. |
| `run_readonly_sql` (MAY) | Not implemented | Optional and risky; domain tools cover the use cases. |
| `uv` | `pip` + `hatchling` | `uv` was not available; nothing depends on it. |
| `AgentState` with `plan`, `pending_actions` | `pending` (the confirmation) and `drafts_created`; no `plan` | Nothing reads a plan in the single-agent design. |
| Router -> four specialist agents | One agent with per-route tool allow-lists | PLAN Phase 6 keeps the split only if evaluation shows a benefit over a single agent; none has been measured. |
| Summarise after `summarize_after_tokens` | Summary in `compact`, transcript kept whole | The API shows the full conversation; only the model's view shrinks. |
| Idempotency `session_id`-only | Customer id is part of the key too | A client-chosen session id must not let two customers share a draft. |
| JWT, API key (MAY) for service-to-service | JWT only | An API key cannot carry a customer identity safely. |
| Default Gemini model `gemini-2.5-flash-lite` | `gemini-3.5-flash-lite` | Google retired the old ID for new users (HTTP 404). Model IDs go stale: check yours. |

The return window counts calendar days in the business time zone with the delivery day as day 1:
delivered on 7 Oct with a 7-day window is returnable through 13 Oct.

## Known limitations

* **Gemini** (`gemini-3.5-flash-lite`) and **Anthropic** (`claude-haiku-5-5`) have been exercised against a live API,
  including full evaluations. OpenAI and OpenRouter are verified by construction and unit tests only; run
  `support-agent check` with your key to confirm tool calling and structured output before
  trusting them. OpenRouter free models in particular are unreliable for tool calling.
* Langfuse tracing and scoring were verified against Langfuse Cloud for evaluation runs. The
  interactive `ask` and `chat` paths are not traced yet (that arrives with the API).
* PostgreSQL conversation checkpoints work on Linux but not on Windows: psycopg cannot use the
  proactor event loop that the MCP subprocess needs. SQLite checkpoints work everywhere.
* Conversation summaries are written by the same model as the answers and are only checked for
  secrets, not for accuracy. Saved preferences are limited to two keys (`preferred_language`,
  `product_interests`) and are saved only through `PUT /v1/memory`, never by the agent.
* The API runs as one process (embedded Qdrant, the MCP subprocess and the rate limiter are per
  process). For several instances use a Qdrant server, a shared checkpoint database and an
  external rate limiter.
* Requests need a drafts account (`DRAFTS_DB_URL`) that may write only `support_drafts`. Create the table with `support-agent drafts init --url <owner url>`. Without it the agent explains that it cannot submit requests.
* Approved drafts are only recorded: nothing writes to the shop's own tables. Fulfilment is staff work.
* Scanned (image-only) PDFs are not supported; there is no OCR.
* Product search is keyword-based with accent folding over at most 1,000 candidate rows;
  semantic product search arrives with the product advisor in Phase 5.
* Embedded Qdrant allows one process at a time. Use `QDRANT_URL` with the compose service when
  more than one process needs the index.
* Relative paths in `config/app.yaml` resolve against the working directory: run from the
  repository root.

## Layout

```
config/            app.yaml, schema_mapping.yaml, examples/ for MySQL, SQLite, MongoDB
knowledge/         sample bilingual policy documents (md, docx)
src/support_agent/
  core/            settings, logging, i18n, identity (principal)
  llm/             provider factory, capability check, structured output, usage
  rag/             loaders, chunking, embeddings, index, retriever, router, answer, pipeline
  mcp_db/          schema mapping, SQL and MongoDB adapters, service, MCP server
  tools/           MCP client and LangChain tool wrappers
  agent/           LangGraph agent: graph, state, tool node, markers, events, prompts, skills
  api/             FastAPI app: routes, JWT auth, errors, services
  memory/          sessions and saved preferences (SQLite), summaries, workspace files
  security/        guardrails (input, output, rate limit), PII masking
  rules/           return eligibility (pure functions)
  evals/           dataset, metrics, judge, runner, reports
  observability/   optional Langfuse
  seed/            demo data
evals/             dataset and reports
docs/              schema-mapping.md, agent.md, api.md
tests/             unit tests; tests/integration needs Docker
```
