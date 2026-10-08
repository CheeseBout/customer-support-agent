# The HTTP API

```bash
support-agent serve                  # http://127.0.0.1:8000, interactive docs at /docs
docker compose up -d                 # api + qdrant (see README)
```

The OpenAPI document is at `/openapi.json` and always matches the code. This page explains what
it cannot: how the pieces fit together and why.

## Authentication

Your own system issues the tokens; this service only verifies them (SPEC 14.1).

| Setting | Meaning |
|---|---|
| `JWT_ALGORITHM` | `HS256` (shared secret) or `RS256` (public keys from a JWKS endpoint). The algorithm comes from this setting, never from the token, so a token cannot choose a weaker one or `none` |
| `JWT_SECRET` | HS256 only. At least 32 bytes; the server will not start without it |
| `JWT_JWKS_URL` | RS256 only. Keys are cached after the first request |
| `JWT_AUDIENCE`, `JWT_ISSUER` | Checked when set |
| `JWT_CUSTOMER_CLAIM` | Claim holding the customer id (default `sub`) |
| `JWT_ROLE_CLAIM` | Claim holding the role (default `role`) |

A token must carry `exp`, the customer claim, and a role claim equal to `customer` or `staff`.
Anything else is `401`. The customer id from the token is the **only** identity the system uses:
it scopes the conversation, every data lookup and every request. Nothing in a request body can
change it.

* `customer` tokens use chat, sessions, drafts, feedback and memory.
* `staff` tokens use `/v1/admin/*`. A staff token on a customer endpoint, or the reverse, is `403`.
* `/health` and `/ready` need no token.

To try it locally:

```bash
export JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
TOKEN=$(python -c "import jwt,time,os; print(jwt.encode({'sub':'u_100','role':'customer','exp':int(time.time())+3600}, os.environ['JWT_SECRET'], algorithm='HS256'))")
curl -s localhost:8000/v1/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
     -d '{"message": "How many days do I have to return an item?"}'
```

## Endpoints

| Method | Path | Role | |
|---|---|---|---|
| POST | `/v1/chat` | customer | Send a message, wait for the whole answer |
| POST | `/v1/chat/stream` | customer | Same, as Server-Sent Events |
| POST | `/v1/chat/resume` | customer | Approve, reject or edit a pending confirmation |
| GET | `/v1/sessions` | customer | Your conversations, newest first (`limit`, `offset`) |
| GET | `/v1/sessions/{id}` | customer | Messages, pending confirmation and file names |
| DELETE | `/v1/sessions/{id}` | customer | Forget the conversation and its files |
| GET | `/v1/sessions/{id}/artifacts/{name}` | customer | `comparison_table.md` or `request_summary.md` |
| GET | `/v1/drafts` | customer | Your requests (`status` filter) |
| POST | `/v1/drafts/{id}/cancel` | customer | Withdraw a `pending` request |
| POST | `/v1/feedback` | customer | `up` or `down` on an answer |
| GET, PUT, DELETE | `/v1/memory` | customer | Saved preferences (off unless `memory.long_term_enabled`) |
| GET | `/v1/admin/drafts` | staff | Review queue (`status` default `pending`, `type`, `limit`, `offset`) |
| GET | `/v1/admin/drafts/{id}` | staff | One request, including the customer id |
| POST | `/v1/admin/drafts/{id}/approve` | staff | Optional `note` |
| POST | `/v1/admin/drafts/{id}/reject` | staff | `note` required; the customer sees it |
| POST | `/v1/admin/ingest` | staff | Reload `knowledge/` (`full` rebuilds the index) |
| GET | `/health` | public | Liveness |
| GET | `/ready` | public | `200` when the index has documents and the model passed its capability check; `503` otherwise, with the reason per check |

## A chat turn

```json
POST /v1/chat
{"session_id": "optional-8-to-64-chars", "message": "Đơn #1234 của tôi còn được trả hàng không?", "language": "auto"}
```

Leave `session_id` out to start a conversation; the response returns the id to reuse. The answer:

```json
{
  "session_id": "…", "message_id": "…",
  "answer": "…", "outcome": "answered", "language": "vi", "route": "combined",
  "citations": [{"source": "return-policy.vi.md", "section": "…"}],
  "interrupt": null, "drafts": [],
  "usage": {"input_tokens": 4374, "output_tokens": 76}, "trace_id": "…"
}
```

`outcome` is `answered`, `no_info`, `not_found`, `clarify`, `refused`, `confirmation` or `error`
(a controlled apology, for example when the step limit was hit).

### Confirmations

When the customer asks for a refund, return, warranty claim or order, the turn stops with
`outcome: "confirmation"` and an `interrupt` the client must show:

```json
"interrupt": {
  "id": "…", "kind": "confirm_draft", "draft_type": "refund",
  "summary": {"order_id": "1234", "amount": 350000, "currency": "VND", "…": "…"},
  "priority_review": false,
  "allowed_decisions": ["approve", "reject", "edit"],
  "editable_fields": ["reason_code", "reason_text"],
  "expires_at": "2026-10-08T06:15:00+00:00"
}
```

Everything in `summary` was computed by the server from the shop's data. Answer it with:

```json
POST /v1/chat/resume
{"session_id": "…", "interrupt_id": "…", "decision": "approve"}
{"session_id": "…", "interrupt_id": "…", "decision": "edit", "edits": {"reason_text": "left bud is dead"}}
```

`approve` creates one `pending` draft (repeating the call does not create a second). `edit`
returns a new confirmation with a new id. An old or wrong id is `409 CONFLICT`. The state is
saved, so a confirmation survives a restart. `GET /v1/sessions/{id}` returns `pending_interrupt`
for a client that reconnects.

### Streaming

`POST /v1/chat/stream` takes the same body and answers `text/event-stream`:

| Event | Data |
|---|---|
| `session` | `session_id`, `message_id` (first) |
| `route` | `route`, `language` |
| `tool_start` | `call_id`, `tool`, `args_summary` (personal data masked) |
| `tool_end` | `call_id`, `tool`, `ok`, `duration_ms` (never the raw result) |
| `token` | `text` |
| `citation` | `source`, `section` |
| `replace` | `text`: the text so far was withdrawn by a guardrail; show this instead |
| `interrupt` | as above plus `session_id`, `message_id`; the stream ends and waits for `/v1/chat/resume` |
| `done` | `outcome`, `route`, `language`, `drafts`, `usage`, `trace_id`, … |
| `error` | `code`, `message`; the stream ends |

A `: ping` comment is sent every 15 seconds. If the client disconnects the run is cancelled.
Problems found before the stream starts (token, validation, rate limit, guardrail) are ordinary
JSON errors with the status codes below, not events. Joining every `token` gives the answer
unless a `replace` came after it.

## Errors

Always `{"error": {"code": "…", "message": "…", "request_id": "…"}}`. The same `request_id` is
in the `X-Request-ID` response header (send your own `X-Request-ID` to correlate; it is kept if
it is 1 to 64 letters, digits, `.`, `_` or `-`).

| HTTP | `code` | When |
|---|---|---|
| 400 | `INVALID_REQUEST` | Bad body, empty or over-long message (`guardrails.max_input_chars`), bad session id |
| 401 | `UNAUTHENTICATED` | Missing, expired, forged or incomplete token |
| 403 | `FORBIDDEN` | Wrong role |
| 404 | `NOT_FOUND` | Missing, **or not yours** (indistinguishable on purpose) |
| 409 | `CONFLICT` | Stale confirmation, or a draft that is no longer `pending` |
| 422 | `GUARDRAIL_BLOCKED` | The message tried to override the rules or reveal the prompt |
| 429 | `RATE_LIMITED` | Over `guardrails.rate_limit_per_minute`; see `Retry-After` |
| 502 | `UPSTREAM_ERROR` | Model or shop database failed after retries |
| 503 | `NOT_CONFIGURED` | Requests need `DRAFTS_DB_URL`; ingest unavailable |
| 504 | `TIMEOUT` | The turn exceeded `agent.run_timeout_seconds` |

## Running it

* One process only. The MCP tool subprocess and embedded Qdrant belong to a process, so do not
  use `--workers` or `--reload`. For several instances use `QDRANT_URL` and a shared database
  for `CHECKPOINT_URL`, and remember the rate limiter is per process.
* `CORS_ORIGINS` lists the browser origins allowed to call it (empty = off). Tokens travel in the
  `Authorization` header, so credentials/cookies are never allowed.
* Startup runs the model capability check in the background (`STRICT_CAPABILITY_CHECK=true`
  waits for it and refuses to start on `failed`). `/ready` stays `503` until it finishes.
* Sessions, feedback and saved preferences are in `SESSIONS_DB_URL` (SQLite by default). Sessions
  idle longer than `memory.session_retention_days` are deleted every six hours.
* In Docker, mount a volume on `/app/data` (compose does) so models, the index, conversations
  and sessions survive restarts. Inside the compose network the database host is its service name.
* Tracing: with the Langfuse keys set, every turn is a trace in a session named after the
  conversation, and `trace_id` is returned to the client.
