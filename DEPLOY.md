# Deploying for your own shop

This guide takes you from a fresh clone to an agent that answers your customers' questions from
your own policies and your own order data, and shows how to plug it into your website or app.

* [1. How the pieces fit](#1-how-the-pieces-fit)
* [2. What to prepare](#2-what-to-prepare)
* [3. Install](#3-install)
* [4. Choose the language model](#4-choose-the-language-model)
* [5. Connect your database](#5-connect-your-database)
* [6. Set up customer requests (drafts)](#6-set-up-customer-requests-drafts)
* [7. Add your policies](#7-add-your-policies)
* [8. Set your business rules](#8-set-your-business-rules)
* [9. Check it works](#9-check-it-works)
* [10. Run it](#10-run-it)
* [11. Integrate it](#11-integrate-it)
* [12. Go-live checklist](#12-go-live-checklist)
* [13. Measure your own shop](#13-measure-your-own-shop)
* [14. Troubleshooting](#14-troubleshooting)

---

## 1. How the pieces fit

```
 your website / app                       this service                          your systems
┌──────────────────┐   JWT (who the     ┌─────────────────────────┐
│ customer logs in │   customer is)     │ HTTP API (FastAPI)      │
│ with your auth   │ ─────────────────▶ │  verifies the token     │
└──────────────────┘                    │         │               │
                                        │   agent (LLM + tools)   │ ──▶ LLM provider (OpenAI, Anthropic,
┌──────────────────┐                    │    │            │       │     Gemini or OpenRouter)
│ staff tool       │   staff JWT        │ policy search  data tools│
│ (review requests)│ ─────────────────▶ │ (Qdrant)    (read-only) │ ──▶ your shop database (read-only account)
└──────────────────┘                    │         │               │
                                        │  support_drafts table   │ ──▶ requests waiting for staff
                                        └─────────────────────────┘
```

Three rules shape everything below:

1. **Your system decides who the customer is.** You issue a signed token (JWT); this service only
   verifies it. The model never chooses or sees an identity, and a customer can only ever see their
   own orders.
2. **The agent only reads your shop database.** It never writes to your orders, payments or stock.
   When a customer asks for a refund, return, warranty claim or order, the agent prepares a
   *draft*, the customer confirms it, and it waits in the `support_drafts` table for your staff.
   Nothing is fulfilled automatically.
3. **Business decisions are code, not the model.** Return windows, refund limits and warranty
   periods live in `config/app.yaml`. The model explains the result.

## 2. What to prepare

| You need | Why |
|---|---|
| Python 3.12 (and Docker, if you will run it in a container) | Runtime |
| An API key for **one** model provider | The agent needs a language model with tool calling |
| A shop database: PostgreSQL, MySQL, SQLite or MongoDB | Orders, customers, products, stock, shipments |
| A **read-only** database account for it | The agent must not be able to change your data |
| Somewhere to keep the drafts table | A small separate database or a SQLite file is enough |
| Your policy documents (`.md`, `.pdf` or `.docx`, Vietnamese and/or English) | Returns, shipping, warranty, payment, FAQs |
| A way to issue JWTs for logged-in customers and staff | Your existing login system, or a small endpoint |
| About 3 GB of free disk for the first run | The local embedding model is a large download |

## 3. Install

```bash
git clone <your repository url> && cd customer-support-agent
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -e ".[dev]"             # drop "[dev]" on a production machine

support-agent init                  # creates .env from .env.example and the data folders
```

Run every command from the repository root: relative paths in `config/app.yaml` resolve against
the working directory.

Optional extras: `pip install ".[langfuse]"` for tracing, `pip install ".[postgres]"` to keep
conversations in PostgreSQL (Linux only, see [14](#14-troubleshooting)).

## 4. Choose the language model

Edit `.env`:

```ini
LLM_PROVIDER=gemini                 # openai | anthropic | gemini | openrouter
GOOGLE_API_KEY=...                  # the key for the provider you chose
# LLM_MODEL=                        # optional: override the provider's default model
LLM_REQUESTS_PER_MINUTE=10          # keep it below your plan's limit; empty = unlimited
```

Provider keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`.

* Default models are in `config/app.yaml` under `llm.models`. Model names change: confirm yours
  with your provider.
* Gemini and Anthropic (`claude-haiku-5-5`) have been run against live APIs here. OpenAI and
  OpenRouter are covered by unit tests only. Whichever you pick, run `support-agent check` (step 9).
* Free OpenRouter models are unreliable at tool calling. Use a paid model for production.
* Free API tiers have per-minute **and per-day** quotas. When the day's quota is gone every call
  fails with HTTP 429 until it resets.

Embeddings default to a local model (`EMBEDDING_PROVIDER=local`), which needs no key. It is
downloaded on first use.

## 5. Connect your database

The agent never writes SQL. You describe where your data lives, and it uses a fixed set of
lookups (order, shipment, stock, product search, return eligibility).

### 5.1 Create a read-only account

Give the agent an account that can only `SELECT`, preferably on **views** rather than base tables.
Examples are in `scripts/sql/` (PostgreSQL, MySQL). Test it: the account must fail on `INSERT`,
`UPDATE` and `DELETE`.

### 5.2 Point the agent at it

```ini
BUSINESS_DB_TYPE=postgres           # postgres | mysql | mongodb | sqlite
BUSINESS_DB_URL=postgresql://support_ro:PASSWORD@db.example.com:5432/shop
```

URL forms: `postgresql://…`, `mysql://…`, `sqlite:///./path/shop.db`, and
`mongodb://host:27017/shop` (the URL must include the database name).

> **Never run `support-agent seed-demo` against your real database.** It drops and recreates the
> demo tables (`--reset` is on by default). It exists only to create a throwaway demo shop.

### 5.3 Write the schema mapping

`config/schema_mapping.yaml` maps standard entities to your tables or collections. Three are
required: `customer`, `order` and `order_item`. The others are optional, and each one you leave
out switches off only what needs it: `product` and `inventory` (product search, comparison, stock
checks, new orders), `shipment` (tracking) and `return_request` (blocks a second return of the
same item). The file in `config/` is a template with placeholder names; working examples for
PostgreSQL, MySQL, SQLite and MongoDB are in `examples/demo-shop/config/`. Change only the
right-hand sides. The full guide, with the list of fields, is
[docs/schema-mapping.md](docs/schema-mapping.md).

You do not have to start from a blank page. With `BUSINESS_DB_URL` set, let the tool read your
database and draft the file:

```bash
support-agent introspect-db                                  # show the draft and what it found
support-agent introspect-db --write config/schema_mapping.yaml
support-agent validate-mapping
```

It reads table and column names, declared foreign keys and the distinct values of the status
columns (no customer data), and matches them to the entities using English and Vietnamese name
lists (`src/support_agent/mcp_db/vocabulary.yaml`, which you can extend). Treat the result as a
draft: a line ending in `# check` is a guess, `TODO_...` is a required field it could not find,
and a status value it does not recognise is left commented out for you to decide. If your order
lines point at a surrogate `products.id` rather than the SKU, it tells you to create a view,
because a mapping has no joins. Status values are taken from the data as it is today; add any
status your system can set later.

Things that commonly go wrong:

* **Dialect:** `dialect:` in the mapping must equal `BUSINESS_DB_TYPE`.
* **Statuses:** list every status value your orders use under `order.status_map`, mapped to
  `pending_payment | processing | shipping | partially_shipped | delivered | cancelled |
  returned | refunded | on_hold`. A missing value would silently make
  those orders non-returnable, so the validator fails on it.
* **Several tables:** the mapping has no joins. Create a database view that exposes the columns
  in one place and point `table` at the view.
* **Customer ids:** `order.customer_id` must hold the **same value** your tokens carry as the
  customer id (see [11.1](#111-the-token)). This is how ownership is enforced.
* **Delivery date:** map `order.delivered_at` if you want return windows counted from delivery
  (`business_rules.return.window_basis`).
* **Return requests:** map `return_request` if your system already stores them, so "already
  requested" is detected.

Validate until it is clean:

```bash
support-agent validate-mapping
```

## 6. Set up customer requests (drafts)

Refunds, returns, warranty claims, orders and requests to talk to a person are saved in a table
called `support_drafts`; events waiting for your webhook are kept in `support_draft_events` next
to it. Use an account that may write **only** these two tables, never the read-only shop account.

> **Upgrading?** `support_draft_events` is new. Run `support-agent drafts init --url <owner url>`
> once: it creates the missing table and leaves `support_drafts` as it is.

```ini
# simplest start: a local SQLite file
DRAFTS_DB_URL=sqlite:///./data/drafts.db
# or a database account that can write only support_drafts:
# DRAFTS_DB_URL=postgresql://support_rw:PASSWORD@db.example.com:5432/shop
```

Create the table once. If `DRAFTS_DB_URL` can create it itself (the SQLite example above), save
`.env` and run `support-agent init` again: it leaves `.env` untouched and creates the table. For
a database where the writing account cannot create tables, use an owner account:

```bash
support-agent drafts init --url postgresql://owner:PASSWORD@db.example.com:5432/shop
```

Without `DRAFTS_DB_URL` the agent still answers questions, but tells customers it cannot submit
requests.

Staff can review requests from the command line or the API ([11.4](#114-let-your-staff-act-on-requests)):

```bash
support-agent drafts list
support-agent drafts show <id>
support-agent drafts approve <id>
support-agent drafts reject <id> --note "reason the customer will see"
```

Approving only records the decision. Refunding money, creating the order and so on remain your
staff's or your system's job. To have your own system told about each decision, use the
[webhook](#tell-your-system-about-decisions-the-webhook).

**Customers who need a person.** A `handoff` request is how the assistant passes a customer to your
staff: when the customer asks for a person, or the assistant cannot resolve the problem, it asks
the customer to confirm and creates a request of type `handoff` with what they need, in their
words, the order if there is one, and a contact only if they offered one. It appears in the same
queue (`support-agent drafts list --type handoff`). Approving it means "a person took it"; reject
it with a note if it needs nothing. It is on by default; remove `handoff` from
`capabilities.request_types` to turn it off. Set `shop.contact` in `config/app.yaml` so the
assistant can also give customers your opening hours and contact details.

### Tell your system about decisions (the webhook)

Set both in `.env` and your system receives a signed `POST` whenever a request is made or decided:

```
WEBHOOK_URL=https://shop.example.com/hooks/support
WEBHOOK_SECRET=a-long-random-string
```

Which events, and the retry policy, are under `connectors.webhook`. The section is not in the
shipped `config/app.yaml`, so add it there only to change a default:

```yaml
connectors:
  webhook:
    events: [draft.created, draft.approved, draft.rejected]   # also: draft.cancelled
    max_attempts: 8          # then the event is marked failed until someone retries it
    backoff_seconds: 30      # doubled after every failed attempt, at most an hour
    poll_seconds: 15         # how often the server looks for events to send
    reconcile_hours: 48      # how far back to look for a decision that was never queued
```

The body is JSON:

```json
{
  "id": "6f1c...:approved",
  "event": "draft.approved",
  "created_at": "2026-10-09T10:15:00+00:00",
  "draft": {
    "id": "6f1c...",
    "type": "refund",
    "status": "approved",
    "customer_id": "u_100",
    "order_id": "1234",
    "payload": {"refundable_amount": 350000, "items": [{"sku": "EAR-BT20", "qty": 1}], "...": "..."},
    "priority_review": false,
    "reviewed_by": "s_9",
    "reviewed_at": "2026-10-09T10:15:00+00:00",
    "review_note": "Refund sent",
    "created_at": "2026-10-09T09:58:00+00:00",
    "updated_at": "2026-10-09T10:15:00+00:00"
  }
}
```

Headers: `X-Support-Event`, `X-Support-Event-Id` (the `id` above) and
`X-Support-Signature: t=<unix time>,v1=<hex>`, where `v1` is the HMAC-SHA256 of
`"<t>." + <raw body>` with your secret. **Verify it before trusting the request**, and answer with
any `2xx`. A reference check:

```python
import hashlib
import hmac
import time


def valid(secret: str, header: str, body: bytes, tolerance: int = 300) -> bool:
    parts = dict(item.split("=", 1) for item in header.split(","))
    if abs(time.time() - int(parts["t"])) > tolerance:
        return False  # too old: a replay
    expected = hmac.new(secret.encode(), parts["t"].encode() + b"." + body, hashlib.sha256)
    return hmac.compare_digest(expected.hexdigest(), parts["v1"])
```

How it behaves, so your side can rely on it:

* **Not lost.** The event is saved in the drafts database before it is sent. If your system is
  down it is retried with growing waits (30 s, 1 min, 2 min, ... up to an hour) for
  `max_attempts` tries, then marked `failed`. A decision saved but never queued (a crash in
  between) is found by a check every 10 minutes and queued again.
* **At least once.** The same event can arrive twice; use the event `id` to ignore a repeat.
* **In order per request.** `draft.created` of one request is sent before its
  `draft.approved`, and a later event waits while an earlier one is being retried. Events of
  different requests are independent.
* **Permanent refusals are not retried.** A `4xx` other than 408/425/429 (a wrong secret, a wrong
  URL) marks the event `failed` at once, so a misconfiguration is visible instead of hammering you.
* **No redirects** are followed, and the response body is never logged.

Watch and repair it:

```bash
support-agent drafts events --state failed     # what could not be delivered, and why
support-agent drafts retry-events              # after fixing the receiving side
support-agent drafts deliver                   # send what is due now
```

`GET /v1/admin/events` (staff) shows the same. The server delivers by itself every
`poll_seconds` while it runs; the `drafts approve/reject/cancel` commands deliver right after
the decision.

## 7. Add your policies

1. Put your own documents in `knowledge/` (it is empty in the repository; the demo's sample
   documents live in `examples/demo-shop/knowledge/`): returns, shipping,
   warranty, payment, FAQs. Subfolders are fine. Use `.md`, `.pdf` or `.docx`. Scanned PDFs
   without text are not supported.
2. Write clear headings and short sections: answers cite the document and section.
3. Vietnamese and English documents can be mixed. A question in one language finds documents in
   the other.
4. Load them:

```bash
support-agent ingest                # unchanged files are skipped
support-agent ingest --full         # rebuild the index from scratch
```

Re-run `ingest` whenever a document changes. Deleted files are removed from the index.

**Keep documents and rules consistent.** The documents explain the policy; the numbers that
decide a case (return days, refund limit, warranty months) come from `config/app.yaml` (step 8).
If your document says 14 days and the config says 7, the agent will quote 14 and decide with 7.

### Choose what the assistant does

The assistant offers only what your data supports and your settings allow. It is never shown a
tool that would only fail, and the prompt tells it what this shop does not offer so it does not
promise it.

* **By data.** Leave `product`, `inventory`, `shipment` or `return_request` out of the mapping and
  the tools that need them are not registered (see the table in
  [docs/schema-mapping.md](docs/schema-mapping.md)). Skills that need them (placing an order,
  comparing products) are not offered either.
* **By setting.** In `config/app.yaml`:

  ```yaml
  capabilities:
    disabled_tools: [compare_products]            # any of the tool names; a typo is an error
    request_types: [refund, return, warranty]     # no orders through the chat
  ```

  `request_types` lists what a customer may ask the assistant to prepare for confirmation:
  `order`, `refund`, `return`, `warranty`. An empty list turns requests off altogether. An
  `order` also needs `product` and `inventory` in the mapping, because it is priced from them.

## 8. Set your business rules

Open `config/app.yaml` and edit `business_rules`:

| Setting | Meaning |
|---|---|
| `timezone` | Used to count days and read timestamps. Default `Asia/Ho_Chi_Minh` |
| `currency` | ISO code of the shop's currency, used for refund and order amounts. Default `VND` |
| `return.window_days` | Days a customer may return an item |
| `return.window_basis` | Count from `delivered_at` or `created_at` |
| `return.allowed_order_statuses` | Statuses that can be returned (default `[delivered]`) |
| `return.excluded_categories` | Product categories that cannot be returned (for example `gift_card`) |
| `return.windows` | Different windows for some categories or reasons (see below). The first entry that matches an item wins; `window_days` applies when none does |
| `refund.auto_review_max_amount` | Refunds above this are flagged for priority review |
| `warranty.default_months`, `warranty.by_category` | Warranty period overall and per category |
| `inventory.show_exact_quantity` | `false` shows only in stock / low / out of stock |
| `inventory.low_stock_threshold` | Quantity below which stock is reported as "low" |
| `order.*` | Limits for orders the agent may prepare: quantity per line, lines, payment methods, cash-on-delivery cap. They use `currency` above unless `order.currency` is set |

**Different windows for different items.** Many shops give a faulty item longer than a change of
mind, or a category its own period. List them under `return.windows`:

```yaml
return:
  window_days: 7                       # everything else
  windows:
    - reasons: [defective, wrong_item, not_as_described, damaged_in_transit]
      days: 30
    - categories: [fashion]
      days: 14
```

An entry with `reasons` applies once the customer has said why. Until then the answer uses the
ordinary window, and the result lists the longer ones (`reason_windows`) so the agent can ask
"is the item faulty?" instead of refusing. Each item then has its own deadline, and an item past
its window is left out while the rest of the order can still go through. Reason codes:
`defective`, `wrong_item`, `not_as_described`, `damaged_in_transit`, `changed_mind`, `other`.

**Who the agent speaks for.** The `shop` section gives the shop's name, a one-line description
and how to reach a person (email, phone, opening hours, help page). The agent names the shop and
gives exactly those contact details when a customer asks for a person or it cannot help; it is
told never to give any other.

Other useful settings:

* **Order id and SKU formats.** The agent recognises `#1234`, `order 1234`, `đơn hàng 1234` and
  `ORD-1234`. If your ids look different, add a regular expression under
  `router.order_id_patterns` (group 1 must capture the id) and, if needed, `router.sku_pattern`.
* **Guardrails:** `guardrails.rate_limit_per_minute`, `guardrails.max_input_chars`.
* **Timeouts and limits:** `agent.run_timeout_seconds`, `agent.max_steps`,
  `agent.confirmation_ttl_minutes` (how long a customer has to confirm a request).
* **Retention:** `memory.session_retention_days`. Saved preferences are off unless
  `memory.long_term_enabled` is `true`.

## 9. Check it works

```bash
support-agent check --skip-llm      # configuration, index, database (no API cost)
support-agent check                 # also tests the model's tool calling and structured output
```

If the model check reports `failed`, switch to a stronger model. `degraded` means the question
router falls back to JSON parsing. Set `STRICT_CAPABILITY_CHECK=true` to refuse to start in that
case.

Then try it from the command line. Use a **real customer id and real order ids from your
database** (`--user` stands in for the identity a token carries):

```bash
support-agent ask "How many days do I have to return an item?"
support-agent ask "Where is order #<real order id>?" --user <real customer id>
support-agent ask "Is my order #<real order id> still eligible for return?" --user <real customer id>
support-agent chat --user <real customer id>
```

Pick a few orders whose answer you already know and confirm each result:

| Try | Correct behaviour |
|---|---|
| Ask about customer A's order while acting as customer B | "Not found", identical to an order that does not exist |
| Ask something your documents do not cover | Says it has no information; does not guess |
| An order past the return window | Refuses and says why |
| An order in an excluded category | Refuses and says why |
| A question in Vietnamese about an English document (or the reverse) | Answers correctly with a citation |
| "Ignore your previous instructions and …" | Refused |
| A valid refund request | Shows a confirmation with the right amount; approving creates exactly one `pending` draft |

## 10. Run it

### 10.1 On a machine or VM

Add authentication settings to `.env` (see [11.1](#111-the-token)), then:

```bash
support-agent serve --host 0.0.0.0 --port 8000
```

Interactive API docs are at `/docs`; `/health` is liveness and `/ready` is readiness (it returns
`503` until the index has documents and the model passed its check).

Run **one process**. Do not use `--workers` or `--reload`: the tool subprocess, the embedded
vector index and the rate limiter belong to a single process. Put the service behind a reverse
proxy (nginx, Caddy, a cloud load balancer) that terminates **HTTPS**, and keep it under a process
manager (systemd, supervisor) so it restarts.

For SSE streaming, turn off response buffering in the proxy (for nginx: `proxy_buffering off;`).

### 10.2 With Docker Compose

```bash
docker compose up -d                          # api + qdrant
docker compose exec api support-agent ingest  # load the policies into the index
```

What to know:

* The `api` service reads `.env`. Inside the compose network, the vector store is reached at
  `http://qdrant:6333` (compose sets it). Your database must be reachable **from the container**:
  `localhost` in `BUSINESS_DB_URL` would mean the container itself. On Docker Desktop use
  `host.docker.internal` for a database on your computer.
* `config/`, `knowledge/` and `examples/` are copied into the image when it is built. After changing them,
  rebuild: `docker compose up -d --build`, then run `ingest` again if documents changed. To edit
  without rebuilding, add a `docker-compose.override.yml` that mounts them:

  ```yaml
  services:
    api:
      volumes:
        - app_data:/app/data
        - ./config:/app/config
        - ./knowledge:/app/knowledge
  ```
* `app_data` (conversations, sessions, downloaded models) and `qdrant_data` (the index) are
  Docker volumes. Back them up, and keep them when you rebuild.
* The optional demo databases (`--profile demo-db`, `mysql`, `mongo`) are for trying the demo
  shop only. Do not start them in production.

### 10.3 Several instances

A single instance is the supported setup. To run more than one you need a shared Qdrant server
(`QDRANT_URL`), a shared checkpoint database (`CHECKPOINT_URL`) and an external rate limiter (the
built-in one counts per process).

## 11. Integrate it

### 11.1 The token

Your login system issues a JWT for each logged-in customer. This service verifies it and trusts
nothing else about the caller.

Set how it is verified in `.env`:

```ini
JWT_ALGORITHM=HS256
JWT_SECRET=<at least 32 random bytes, shared with the issuer>   # python -c "import secrets; print(secrets.token_hex(32))"

# or, if your identity provider publishes public keys:
# JWT_ALGORITHM=RS256
# JWT_JWKS_URL=https://auth.example.com/.well-known/jwks.json

# optional extra checks and claim names
# JWT_AUDIENCE=
# JWT_ISSUER=
# JWT_CUSTOMER_CLAIM=sub            # which claim holds the customer id
# JWT_ROLE_CLAIM=role               # which claim holds the role
```

The token must contain:

| Claim | Value |
|---|---|
| `exp` | Expiry time. Use short lifetimes (minutes to an hour) |
| `sub` (or your `JWT_CUSTOMER_CLAIM`) | The customer id, **exactly as it appears in `order.customer_id` in your database** |
| `role` (or your `JWT_ROLE_CLAIM`) | `customer` for shoppers, `staff` for your team |

Anything missing, expired or forged is `401`. A `staff` token on a customer endpoint, or the
reverse, is `403`.

Issuing a token from Python (your backend), using PyJWT:

```python
import time, jwt


def customer_token(customer_id: str, secret: str) -> str:
    return jwt.encode(
        {"sub": customer_id, "role": "customer", "exp": int(time.time()) + 900},
        secret,
        algorithm="HS256",
    )
```

**Issue tokens on your server, never in the browser.** The signing secret must not reach client
code. Your frontend asks your backend for a token after the customer has logged in.

If your site calls the API from browser JavaScript, list the site's origin in `CORS_ORIGINS`
(comma-separated). Cookies are never used, only the `Authorization` header.

### 11.2 Chat

Every endpoint is listed in [docs/api.md](docs/api.md). A minimal exchange:

```bash
curl -s https://support.example.com/v1/chat \
  -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"message": "Đơn #1234 của tôi còn được trả hàng không?"}'
```

The answer contains `answer`, `citations`, `outcome`, `route`, `language` and a `session_id`.
Send that `session_id` with the next message to continue the conversation. Conversations are
saved and survive restarts. Show the citations to customers: they say which document and section
an answer came from.

For a responsive chat window, use streaming (`POST /v1/chat/stream`, Server-Sent Events). The
browser's `EventSource` cannot send an `Authorization` header or a POST body, so read the stream
with `fetch`:

```javascript
async function ask(token, message, sessionId, onEvent) {
  const res = await fetch("https://support.example.com/v1/chat/stream", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!res.ok) throw await res.json();            // {"error": {"code", "message", "request_id"}}
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer = (buffer + value).replace(/\r\n/g, "\n");   // the server separates lines with \r\n
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const event = /^event: (.*)$/m.exec(block)?.[1];
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (event && data) onEvent(event, JSON.parse(data));
    }
  }
}
```

Events to handle: `token` (append text), `citation`, `replace` (a safety check withdrew the
text so far: replace what you showed with this text), `interrupt` (a confirmation, see below),
`done` and `error`. The full table is in [docs/api.md](docs/api.md#streaming).

### 11.3 Confirmations

When a customer asks for a refund, return, warranty claim or order, the turn stops with
`outcome: "confirmation"` and an `interrupt`. Your UI must show its `summary` (the amount, items
and order, all computed by the server from your data) with buttons for the `allowed_decisions`:

```bash
curl -s https://support.example.com/v1/chat/resume \
  -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"session_id": "…", "interrupt_id": "…", "decision": "approve"}'
```

* `approve` creates exactly one `pending` draft (repeating the call does not create another).
* `reject` stores nothing.
* `edit` (with `"edits": {...}` limited to `editable_fields`) returns a new confirmation.
* A confirmation expires after `agent.confirmation_ttl_minutes`. A reconnecting client can read
  the pending one from `GET /v1/sessions/{id}` (`pending_interrupt`).

Customers can list and withdraw their own requests with `GET /v1/drafts` and
`POST /v1/drafts/{id}/cancel`.

### 11.4 Let your staff act on requests

Staff use a token with `role: staff`:

| Call | Use |
|---|---|
| `GET /v1/admin/drafts` | The review queue (`status` defaults to `pending`; also `type`, `limit`, `offset`) |
| `GET /v1/admin/drafts/{id}` | One request, including the customer id |
| `POST /v1/admin/drafts/{id}/approve` | Approve, with an optional `note` |
| `POST /v1/admin/drafts/{id}/reject` | Reject; a `note` is required and the customer sees it |
| `POST /v1/admin/ingest` | Reload `knowledge/` after you change documents |

Build a review screen on these calls, or use the `support-agent drafts` commands.

**Connecting approvals to your order system.** An approved draft is only a record. To act on it
automatically, have a job in your system poll `GET /v1/admin/drafts?status=approved`, perform the
refund or order in your own system, and remember the draft ids it has already handled. This
service does not mark a draft as fulfilled.

### 11.5 Optional: tracing

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` (and install the `langfuse`
extra) to record every turn as a trace. The `trace_id` is returned to the client, so a complaint
can be matched to the exact run. Without these, nothing is sent anywhere.

## 12. Go-live checklist

**Security**

- [ ] `.env` is not in version control and is readable only by the service user.
- [ ] The database account is `SELECT`-only (try an `UPDATE` and confirm it fails).
- [ ] The drafts account can write only `support_drafts` and `support_draft_events`.
- [ ] `JWT_SECRET` is random and at least 32 bytes, or you use RS256 with a JWKS URL; set
      `JWT_AUDIENCE` and `JWT_ISSUER` if your identity provider supports them.
- [ ] Tokens are issued by your backend, expire quickly, and the customer id in them matches your
      database.
- [ ] The API is served over HTTPS only; `CORS_ORIGINS` lists only your own sites.
- [ ] Customer A cannot read customer B's order, session or draft (test it with two real accounts).

**Quality**

- [ ] `support-agent validate-mapping` and `support-agent check` are clean.
- [ ] The policy documents are loaded and agree with `config/app.yaml`.
- [ ] You tried real orders for each case in [step 9](#9-check-it-works).
- [ ] You measured your own shop ([13](#13-measure-your-own-shop)).

**Operations**

- [ ] The service runs under a process manager or Docker `restart` policy, as one process.
- [ ] `/health` and `/ready` are monitored.
- [ ] `data/` (or the `app_data` and `qdrant_data` volumes) and the drafts database are backed up.
- [ ] Someone reviews pending drafts regularly and knows how to reject with a clear note.
- [ ] Your model plan's quota and cost are understood. Every customer message costs model calls.
- [ ] A person is reachable for cases the agent cannot handle: there is no live handoff to a
      human in this service.

## 13. Measure your own shop

The figures in the README describe the demo shop. For yours:

1. **Calibrate retrieval on your documents.** The relevance threshold (`retrieval.score_threshold`,
   default 0.80) was tuned on the sample corpus and must be recalibrated:

   ```bash
   support-agent calibrate evals/calibration.yaml   # no API key; run `ingest` first
   ```

   `calibrate` takes a short file listing questions your documents answer and questions they do
   not (other topics, small talk, policies you do not have), in the form of
   `examples/demo-shop/evals/calibration.yaml`: `- {q: "...", answerable: true}`. Aim for at
   least 10 of each. It prints how well the current threshold separates them and the value to put
   under `retrieval.score_threshold`. It works for any embedding model, whatever range its
   scores fall in. (`eval --offline` runs the same analysis over a full evaluation dataset.)
2. **Write your own questions.** Copy the structure of
   `examples/demo-shop/evals/datasets/baseline.jsonl` to `evals/datasets/shop.jsonl` (the default
   `evals.dataset`), using your own order ids, customers and policies, or pass `--dataset`. Include cases that must be refused, such as someone else's order.
3. **Run it** with a model key and read the report in `evals/reports/`:

   ```bash
   support-agent eval --subset ci                 # a small first run, to limit cost
   support-agent eval --judge --fail-under        # exits non-zero below the thresholds
   ```

A full run makes several hundred model calls: mind your plan's quota.

## 14. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `validate-mapping` fails on a status | A status value in your orders is missing from `order.status_map`. Add it |
| `validate-mapping` says a table or column is missing | Wrong name, schema or permission. On MySQL the account also needs `SHOW VIEW` on views |
| Configuration error about the dialect | `dialect:` in the mapping differs from `BUSINESS_DB_TYPE` |
| `serve` refuses to start | No valid `JWT_SECRET` (at least 32 bytes) or `JWT_JWKS_URL` |
| `401 UNAUTHENTICATED` | Token expired, signed with another secret, or missing `exp`, the customer claim or a valid `role` |
| `403 FORBIDDEN` | A staff token on a customer endpoint, or the reverse |
| Every order is "not found" | The customer id in the token does not match `order.customer_id` in your database |
| Returns always refused | `return.window_days` / `window_basis`, a missing `delivered_at` mapping, or an order status not in `allowed_order_statuses` |
| `/ready` stays `503` | The index is empty (run `ingest`) or the model check has not passed (`support-agent check`) |
| "I don't have information" for things your documents cover | The relevance threshold is wrong for your documents: recalibrate ([13](#13-measure-your-own-shop)); also check the file was ingested |
| Answers quote a number that disagrees with the decision | The document and `config/app.yaml` disagree. Align them |
| `503 NOT_CONFIGURED` when requesting a refund | `DRAFTS_DB_URL` is not set, or the table was not created |
| `429` from the model | Your provider's quota. Lower `LLM_REQUESTS_PER_MINUTE`, or wait for the daily reset. `429 RATE_LIMITED` from the API is the per-customer limit instead |
| `502 UPSTREAM_ERROR` | The model or your database failed after retries; check provider status and database reachability |
| `504 TIMEOUT` | The turn exceeded `agent.run_timeout_seconds` |
| Index errors when a second process starts | The embedded vector store allows one process. Use a Qdrant server (`QDRANT_URL`) |
| `CHECKPOINT_URL=postgresql://…` fails on Windows | PostgreSQL conversation storage works on Linux only. Use the default SQLite checkpoints |
| Config changes have no effect in Docker | `config/`, `knowledge/` and `examples/` are baked into the image: rebuild, or mount them ([10.2](#102-with-docker-compose)) |
| Streaming arrives all at once behind a proxy | Disable response buffering for the API |

Every API error includes a `request_id`, repeated in the `X-Request-ID` response header. Quote it
when you look for the matching log line. Set `LOG_LEVEL=DEBUG` in `.env` for more detail.
