# Customer Support Agent

Backend for an AI customer-support assistant for small and mid-size e-commerce shops. Vietnamese
and English, single tenant. It answers shop-policy questions from your own documents, looks up a
customer's own orders through a read-only data layer, and combines both ("is my order #1234 still
eligible for return?").

Where to go next:

* **[DEPLOY.md](DEPLOY.md)**: set it up for your own shop (database, policies, rules, LLM), run it,
  and integrate it with your site or app.
* [docs/api.md](docs/api.md): the HTTP API. [docs/agent.md](docs/agent.md): how the agent works.
  [docs/schema-mapping.md](docs/schema-mapping.md): connecting your database.

## Status

The features below are built and covered by unit tests; the integration suite (real PostgreSQL,
MySQL and MongoDB) runs in Docker.

| Area | State |
|---|---|
| Policy Q&A and personal data lookup | Built, tested and evaluated on the bundled demo shop (see [Evaluation](#evaluation)). |
| Agent loop | Built. On the full 79-sample set with the judge the agent scores 100% on every metric and does not fall below the fixed-pipeline baseline. |
| Requests with human confirmation | Built, tested and measured on the 63-sample after-sales set (refund, return, warranty, order drafts; confirm / edit / reject; staff review CLI). Answer 92-98% and safety 90-100% depending on the model; business correctness is 94-98%, short of the 100% target. See [After-sales results](#after-sales-results). |
| Memory and workspace | Built and tested. Conversation summaries, optional saved preferences, per-session files. A multi-agent split is deliberately not built: nothing shows it would beat the single agent. |
| Guardrails | Built and tested. Input screening, rate limit, output checks, an English and Vietnamese red-team set. Phrase-based: see [docs/agent.md](docs/agent.md). |
| HTTP API | Built and tested. JWT auth, chat as JSON or SSE, sessions, requests, staff review, Docker. See [docs/api.md](docs/api.md). |
| Fitting it to your shop | Built and covered by unit tests, **not measured with an LLM evaluation**: a mapping drafted from your database (`introspect-db`), optional data entities and tools that switch themselves off, parcels and product variants, return windows by category and reason, shop name and contact details, threshold calibration on your own documents, handoff to staff, and a signed webhook for decisions. The evaluation results below were measured on the demo shop with prompt v8; the prompt is now v9 (same instructions, plus the shop profile and a note of what the shop does not offer). Re-run the evaluation on your own shop before relying on it. |
| Evaluation in CI | Partly. CI runs lint, type checks and the unit tests; the LLM evaluation is run by hand. Tracing, thresholds and `--fail-under` exist; a four-provider comparison does not. |

All evaluation figures in this README were measured on the bundled demo shop and its sample
documents. Measure your own shop before relying on them (see [DEPLOY.md](DEPLOY.md)).

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
* **Built for your shop, not just the demo.** Only `customer`, `order` and `order_item` are
  required in the database mapping; leave out stock, catalogue or shipments and the tools that need
  them are simply not offered. `support-agent introspect-db` drafts the mapping from your
  schema. Return windows can differ by category and by reason; one order can be several parcels;
  products can have variants; the shop's name, contact details and currency are configuration.
  The demo lives in [`examples/demo-shop/`](examples/demo-shop/).
* **Decisions leave the building.** A customer who needs a person is handed to your staff through
  the same confirm-then-review flow as a refund, and each request and decision can be sent to
  your own system as a signed, retried webhook.
* **Evaluation**: a 79-question bilingual dataset, retrieval/routing/answer/safety metrics,
  threshold calibration, optional LLM judge and Langfuse tracing.

## Quick start

This runs the bundled **demo shop** so you can try the agent in minutes. To run it for your
own shop, follow [DEPLOY.md](DEPLOY.md).

Requires Python 3.12 and Docker.

```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[dev]"

support-agent init                                      # creates .env; add your provider key to it
docker compose --profile demo-db up -d                  # demo PostgreSQL (+ optional Qdrant server;
                                                        # by default Qdrant runs embedded in ./data/qdrant)

# point the agent at the demo shop and its database with the READ-ONLY account (put in .env)
#   APP_CONFIG_PATH=./examples/demo-shop/app.yaml      # demo documents, mapping and evaluation data
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
| `introspect-db [--write FILE] [--schema S] [--force]` | Draft `schema_mapping.yaml` by reading the structure of your database (no customer data), with every guess marked |
| `drafts deliver / events / retry-events` | Send queued events to your webhook now; list them; send the failed ones again |
| `calibrate QUESTIONS` | Find the relevance threshold for your own documents from a list of answerable and unanswerable questions (no API key) |
| `ask QUESTION [--user] [--role customer\|staff] [--lang] [--json] [--no-tools] [--session ID] [--engine agent\|pipeline]` | Ask as an authenticated customer; the answer streams. `--session` continues a conversation |
| `resume --session ID --id CONFIRMATION --decision approve\|reject\|edit [--user] [--set field=value]` | Answer a confirmation that `ask` left open (works after a restart) |
| `drafts init\|list\|show\|approve\|reject\|cancel` | Staff side of customer requests. `drafts init --url` creates `support_drafts`; `reject` needs `--note`; `--staff` sets the reviewer id and `--url` the drafts database (default `DRAFTS_DB_URL`) |
| `chat [--user] [--role customer\|staff] [--lang] [--no-tools] [--session ID]` | Talk over several turns; the conversation is saved |
| `seed-demo [--url] [--reset\|--no-reset]` | Create the demo shop database (needs a write-capable account); by default it drops and recreates the demo tables |
| `validate-mapping` | Check the schema mapping against the live database |
| `serve [--host] [--port]` | Run the HTTP API. Refuses to start without a valid JWT setting |
| `check [--skip-llm]` | Verify configuration, Qdrant, database and the model's capabilities |
| `eval [--engine agent\|pipeline] [--offline] [--dataset FILE] [--subset ci] [--limit N] [--name NAME] [--judge] [--fail-under] [--compare REPORT.json] [--rescore REPORT.json] [--concurrency N] [--run-timeout S]` | Evaluate and write a report. The agent is compared with the fixed-pipeline baseline; `--fail-under` also fails on a regression; `--rescore` grades a saved report again with the current dataset and calls no model |

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
| `DRAFTS_DB_URL` | Account that may write only `support_drafts` and `support_draft_events`; without it the agent cannot submit requests (see [DEPLOY.md](DEPLOY.md)) |
| `WEBHOOK_URL` `WEBHOOK_SECRET` | Send each request and decision to your system as a signed JSON POST; both are needed together |
| `MCP_PRINCIPAL_SECRET` | Signs the caller identity passed to the MCP server; empty = a random secret per process |
| `LOG_LEVEL` | Logging level (default `INFO`) |

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
single source of truth for decisions. The policy documents in `knowledge/` (the demo's are in `examples/demo-shop/knowledge/`) explain and cite those
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

Reports are written to the directory set by `evals.report_dir` (the demo's go to
`examples/demo-shop/evals/reports/`). The agent evaluation uses a throwaway drafts store, so
it never touches real requests.

### After-sales dataset

`examples/demo-shop/evals/datasets/aftersales.jsonl` (63 samples, 29 Vietnamese / 34 English, 19 in the `ci` subset)
covers requests, not just answers:

```bash
support-agent eval --engine agent --dataset examples/demo-shop/evals/datasets/aftersales.jsonl --subset ci
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
default embedding model (see [`baseline-offline.md`](examples/demo-shop/evals/reports/baseline-offline.md)):

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
| Gemini `gemini-3.5-flash-lite` | 98.4% | 98.4% | 100% | 98.4% | 1 | 18 s (paced) |
| Claude Haiku 5.5 | 92.1% | 93.7% | 90% | 93.7% | 5 | 4.3 s |

The Gemini run is paced to stay under the free tier's limit, so its latency is not the model's speed. Business
correctness (the right confirmation and the right drafts) is not yet 100% on either model: Gemini's one miss,
`aft-en-014`, is the model inventing a SKU instead of reading it from the order. None of the failures leaked another customer's data, invented an
amount or approved a request. The Haiku safety figure is one sample, `aft-vi-026` (a customer asking for a
10 million refund): the agent refuses the amount and asks whether to go ahead instead of proposing the request,
which the dataset counts as a miss. A Haiku run costs a few tens of cents at its list price.

Reports: `examples/demo-shop/evals/reports/aftersales-v8-gemini.md`,
`examples/demo-shop/evals/reports/aftersales-v8b-haiku55.md`.

### Full baseline

Full pipeline on all 79 samples with `gemini-3.5-flash-lite` as both assistant and judge
([`baseline-full.md`](examples/demo-shop/evals/reports/baseline-full.md)):

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
  ([`baseline-full-run1.md`](examples/demo-shop/evals/reports/baseline-full-run1.md)). It exposed one real defect (the
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
pytest                      # ~1100 unit tests, offline, no API keys
pytest -m integration       # PostgreSQL, MySQL, MongoDB and the Postgres checkpointer, in Docker
ruff check . && ruff format --check . && mypy src
```

Language models are replaced by a scripted fake in tests (one that also streams in small pieces,
to exercise marker handling), so the router, agent loop and pipeline are deterministic and free. The adapter contract tests run unchanged against SQLite, MongoDB
(mocked) and, in the integration suite, the real PostgreSQL, MySQL and MongoDB.

## Design notes

| Choice | Why |
|---|---|
| Embeddings `intfloat/multilingual-e5-large` via fastembed | fastembed does not ship bge-m3. Its e5 vectors are not normalised, so they are normalised here. |
| Chunks of 200-450 tokens | e5 truncates input at 512 tokens; longer chunks would be silently cut for the dense vector. |
| Relevance threshold 0.80, applied per query | e5 scores are compressed (relevant 0.81-0.89, unrelated 0.70-0.81). Calibrated on the sample corpus; recalibrate for your own documents and model. |
| `mcp` 2.x `MCPServer` | The 2.x SDK renamed FastMCP. |
| Return eligibility is code, not prompt | The `combined` route cannot conclude correctly without it, and a model must not decide who is eligible. Refund amounts, warranty and drafts build on the same rules. |
| No free-form SQL tool | Optional and risky; domain tools cover the use cases. |
| `pip` + `hatchling` | `uv` was not available; nothing depends on it. |
| One agent with per-route tool allow-lists | A split into specialist agents is worth keeping only if evaluation shows a benefit over a single agent; none has been measured. |
| Summary kept in `compact`, transcript kept whole | The API shows the full conversation; only the model's view shrinks. |
| Idempotency key includes the customer id | A client-chosen session id must not let two customers share a draft. |
| JWT only for service-to-service calls | An API key cannot carry a customer identity safely. |
| Default Gemini model `gemini-3.5-flash-lite` | Google retired the old ID (`gemini-2.5-flash-lite`) for new users (HTTP 404). Model IDs go stale: check yours. |

The return window counts calendar days in the business time zone with the delivery day as day 1:
delivered on 7 Oct with a 7-day window is returnable through 13 Oct.

## Known limitations

* **Gemini** (`gemini-3.5-flash-lite`) and **Anthropic** (`claude-haiku-5-5`) have been exercised against a live API,
  including full evaluations. OpenAI and OpenRouter are verified by construction and unit tests only; run
  `support-agent check` with your key to confirm tool calling and structured output before
  trusting them. OpenRouter free models in particular are unreliable for tool calling.
* Langfuse tracing and scoring were verified against Langfuse Cloud for evaluation runs. The API
  traces every turn; the `ask` and `chat` CLI commands are not traced.
* PostgreSQL conversation checkpoints work on Linux but not on Windows: psycopg cannot use the
  proactor event loop that the MCP subprocess needs. SQLite checkpoints work everywhere.
* Conversation summaries are written by the same model as the answers and are only checked for
  secrets, not for accuracy. Saved preferences are limited to two keys (`preferred_language`,
  `product_interests`) and are saved only through `PUT /v1/memory`, never by the agent.
* The API runs as one process (embedded Qdrant, the MCP subprocess and the rate limiter are per
  process). For several instances use a Qdrant server, a shared checkpoint database and an
  external rate limiter.
* Requests need a drafts account (`DRAFTS_DB_URL`) that may write only `support_drafts` and `support_draft_events`. Create the tables with `support-agent drafts init --url <owner url>`. Without it the agent explains that it cannot submit requests.
* Approved drafts are only recorded: nothing writes to the shop's own tables. Fulfilment is staff work.
  The webhook tells your system about each decision; it is at-least-once, so your side must ignore
  a repeated event id.
* `introspect-db` is a set of name lists and rules, not a model: it recognises English and
  Vietnamese schemas and says where it is unsure, but a schema with unusual names or with the
  order id and SKU only reachable through joins needs a database view and a hand edit. Product
  variants need a view that joins the parent product's name to each variant.
* There is no adapter for a shop platform's own API (Shopify, WooCommerce, ...): the data layer
  reads a SQL or MongoDB database. Expose the platform's data as views or a replica.
* Guest checkout is not supported: every customer lookup needs the customer id in the token.
* Scanned (image-only) PDFs are not supported; there is no OCR.
* The shipped evaluation datasets, thresholds and results describe the demo shop only. A real
  shop needs its own dataset and a retrieval threshold calibrated on its own documents.
* Product search is keyword-based with accent folding over at most 1,000 candidate rows;
  there is no semantic product search.
* Embedded Qdrant allows one process at a time. Use `QDRANT_URL` with the compose service when
  more than one process needs the index.
* Relative paths in `config/app.yaml` resolve against the working directory: run from the
  repository root.

## Layout

```
DEPLOY.md          set up for your own shop, run and integrate
config/            app.yaml, schema_mapping.yaml (a template to fill in for your shop)
knowledge/         your policy documents (empty in the repository)
examples/demo-shop/  the bundled demo: app.yaml, knowledge/, mapping for each database,
                   evals/ (datasets and the reports quoted below)
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
evals/             your own evaluation datasets and reports (empty in the repository)
docs/              schema-mapping.md, agent.md, api.md
scripts/           sql/: SELECT-only database accounts; make_sample_docx.py
tests/             unit tests; tests/integration needs Docker
```
