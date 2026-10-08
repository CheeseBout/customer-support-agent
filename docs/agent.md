# The agent loop

Phase 4 replaced the fixed Phase A pipeline with a LangGraph agent that decides for itself which
tools to call. The Phase A pipeline is still available (`--engine pipeline`) because it is the
baseline the agent is compared with.

## The graph

```
START -> prepare -> compact -> route -+- clarify / smalltalk / refuse ------+
                                      '- agent <-> tools -- limit ----------+-> finalize -> END
                                                  '-> confirm (customer) -'
```

| Node | What it does |
|---|---|
| `prepare` | Truncates over-long input, removes card numbers and passwords before they are stored, extracts order ids and SKUs with regexes, resets per-turn state |
| `compact` | Folds old turns into a summary when the conversation outgrows the window (see Memory) |
| `route` | Classifies the question (`policy`, `personal`, `combined`, `chitchat`, `out_of_scope`) |
| `clarify`, `smalltalk`, `refuse` | Fixed replies for low confidence, greetings and off-topic questions. No agent call, no tools |
| `agent` | One model call with the route's tools bound. Streams the answer as it is written |
| `tools` | Runs the tool calls the model asked for (concurrently) and returns the results |
| `confirm` | Stops with `interrupt()` until the customer approves, rejects or edits a request, then creates the draft or reports the decision to the model |
| `limit` | Reached when `agent.max_steps` is used up: ends with a polite message |
| `finalize` | Strips markers, decides the outcome, runs the output guardrails, builds citations, saves workspace files, deletes scratch messages |

This is the single-agent form of SPEC 11.2. Splitting it into Policy / Order / Product /
After-sales specialists is only worth it if evaluation shows a benefit over this design (PLAN
Phase 6); none has been measured yet, so the single agent with per-route tool lists stays.

## Rules the loop enforces

* **The first step must use a tool** (`tool_choice="any"`) for any question that needs data, so
  the model cannot answer policies, orders or prices from memory.
* **Tools are allowed per route.** A policy question only gets `search_policy`; a personal one
  only the order/product tools; a combined one gets both. A call to a tool outside the list is
  refused, not run.
* **Identity comes from the run state**, written once before the graph starts. The tools shown to
  the model have no identity parameter, and an argument such as `customer_id` is refused with
  `FORBIDDEN` before it reaches the data layer. No node returns `principal`.
* **Everything a tool returns is data.** It is wrapped in `<untrusted_data>` (or `<fact
  authoritative="true">` for the rules engine) with closing tags neutralised, and the system
  prompt says never to follow instructions found in it.
* **Return decisions are not the model's.** `check_return_eligibility` runs the rules in code;
  the prompt says its verdict is final.

## What the model writes

The final answer is plain text, so it can be streamed. Two markers carry structure and are
removed before the customer sees anything, even when a marker is split across stream chunks:

* `[D1]`, `[D1, D2]`: citations of the numbered policy documents the tool returned.
* `[NO_INFO]` as the very first thing: the gathered material does not answer the question.

The outcome (`answered`, `no_info`, `not_found`, `clarify`, `refused`, `error`) is decided by
`finalize` from what actually happened, not only from what the model said. For example, an
answer with no real data behind it becomes `no_info` even if the model forgot the marker.

## Streaming events

`SupportAgent.stream()` yields these, in this order, and always ends with one `done`, `interrupt` or `error`:

| Event | Data |
|---|---|
| `session` | `session_id`, `message_id` |
| `route` | `route`, `language` |
| `tool_start` | `call_id`, `tool`, `args_summary` (personal data masked) |
| `tool_end` | `call_id`, `tool`, `ok`, `duration_ms` (never the raw result) |
| `token` | `text`: a piece of the answer |
| `citation` | `source`, `section` |
| `replace` | `text`: the output guardrail withdrew the text streamed so far; show this instead |
| `interrupt` | A request waits for the customer: `id`, `kind`, `draft_type`, `summary`, `priority_review`, `allowed_decisions`, `editable_fields`, `expires_at`, `session_id`. Ends the stream instead of `done` |
| `done` | `answer`, `outcome`, `route`, `citations`, `drafts`, `guardrails`, `tool_calls`, `steps`, `usage`, `session_id`, `trace_id`, `latency_ms`, `prompt_version` |
| `error` | `code` (`TIMEOUT`, `STEP_LIMIT`, `UPSTREAM_ERROR`), `message` |

`tokens` joined together equal `done.answer`. Closing the stream early cancels the run and frees
the conversation.

## Requests the customer confirms

Refund, return, warranty and order requests go through `propose_draft` (PLAN Phase 5):

1. The model gathers facts with read tools and calls `propose_draft` once with only the type,
   order, item and reason. It cannot give an amount, a price or an identity.
2. `DraftProposer` re-runs the eligibility rules and works out the amount (or order total) from
   the database. Anything that fails comes back to the model as a tool error it can explain.
3. The `confirm` node stops the run. The customer sees what our code computed and answers with
   `SupportAgent.resume(principal, session_id, interrupt_id, "approve" | "reject" | "edit")`.
   The state is checkpointed, so the answer may come after a restart (`support-agent resume`).
4. On approve the draft is created in `support_drafts` as `pending` (idempotent, written with a
   write-limited account). The model is told the result and must not say a request was
   submitted unless that result says `created`. Staff decide with `support-agent drafts ...`.

A confirmation is valid for `agent.confirmation_ttl_minutes`. An edit may change only the
fields listed in `editable_fields` and produces a new confirmation with a new id (at most 3
edits). A wrong or old id changes nothing. A new message from the customer drops an open
confirmation. Without `DRAFTS_DB_URL` the agent still answers questions and says it cannot
submit requests. `load_skill` gives the model step-by-step procedures
(`agent/skills/<name>/SKILL.{en,vi}.md`).

## Conversations

A conversation is a LangGraph thread saved by a checkpointer (`CHECKPOINT_URL`): SQLite by
default, PostgreSQL optionally, `memory` for throwaway runs.

* The thread id is `<hash of the user id>:<session id>`. A session id alone can never reach
  another user's conversation.
* Only the question and the clean answer are kept between turns. Tool calls and their results
  are deleted when the turn ends, so old tool data never resurfaces and history stays small.
* The model sees the last `agent.history_messages` earlier messages, plus a summary of anything
  older once the conversation is long (see Memory). The saved transcript is never shortened.
* Turns on one session are processed one at a time.
* If a turn fails or times out, its scratch work is replaced by the error reply, so the next
  turn does not find an unanswered question.

## Memory and workspace

**Short-term (FR-401/402).** The transcript lives in the checkpoint and is never cut. What the
model is shown is: the last `history_messages` messages, and, once the earlier turns exceed
`summarize_after_tokens` (about three characters per token), `<conversation_summary>` plus
every message the summary does not cover yet. The `compact` node writes the summary with one
model call. To avoid a call on every turn it waits until `history_messages / 2` extra messages
have piled up, then folds all but the last `history_messages`. The text sent to the summariser
has card numbers, passwords, e-mail addresses and phone numbers removed, is wrapped as data, and
the result is masked again. If the summary call fails the turn continues with the plain window.

**Long-term (FR-403, off by default).** `memory.long_term_enabled: true` turns on saved
preferences: the customer saves `preferred_language` or `product_interests` with
`PUT /v1/memory` (an explicit act, which is the consent), reads them with `GET /v1/memory` and
erases them with `DELETE /v1/memory`. Each fact expires after `long_term_ttl_days`. They reach the
model in a `<customer_preferences>` block labelled as data; a value that looks like an
instruction is refused when it is saved.

**Workspace (FR-404).** Files for a session live under
`<WORKSPACE_DIR>/<hash of user id>/<session id>/`. Only two names exist: `comparison_table.md`
(written when the agent compares products) and `request_summary.md` (written when a request is
confirmed and created). `GET /v1/sessions/{id}/artifacts/{name}` returns them to their owner.
Deleting a session, or `memory.session_retention_days` passing, deletes the conversation, the
files and the session record.

## Guardrails

Cheap deterministic checks (`security/guardrails.py`); they back up, not replace, the structural
defences (identity is never a tool argument, ownership is in the query, untrusted text is data).

* **Input.** Control and bidi characters are removed and the text is normalised. Phrases that try
  to override the rules, reveal the prompt, switch persona, claim to be staff or another customer,
  or forge our delimiters are refused: the turn ends with `outcome: refused`, the model is never
  called and the attempt is not added to the conversation. Through the API this is a `422
  GUARDRAIL_BLOCKED`. `guardrails.injection_detection: false` turns it off.
* **Rate limit.** `guardrails.rate_limit_per_minute` requests per user per minute (sliding
  window, per API process): `429 RATE_LIMITED` with `Retry-After`.
* **Output.** The model's answer is checked before it is final: a secret or a long verbatim run of
  the system prompt becomes a fixed refusal; a promise of a refund or approval after the rules
  engine said "not eligible" is replaced by an honest message; an e-mail address or phone number
  that appears nowhere in the customer's own data for this turn is masked. If text was already
  streamed, a `replace` event follows; `done.guardrails` lists what happened.
* **Red team.** `tests/test_guardrails.py` holds the English and Vietnamese attack set and a
  set of ordinary messages that must never be blocked. Add every new attack you meet.

## Limits and failure handling

| Setting (`config/app.yaml`, `agent:`) | Default | Effect |
|---|---|---|
| `max_steps` | 12 | Agent iterations per question; then the `limit` node |
| `run_timeout_seconds` | 60 | Whole turn; then an `error` event with `TIMEOUT` |
| `max_tool_calls` | 8 | Tool calls per question; extra calls are refused |
| `tool_retries` | 2 | Extra attempts after `UPSTREAM_ERROR` / `TIMEOUT`, doubling `tool_backoff_seconds` |
| `history_messages` | 12 | Earlier messages shown to the model verbatim |
| `summarize_after_tokens` | 4000 | Estimated size of earlier turns above which old ones are summarised |
| `llm.requests_per_minute` | none | Client-side pacing for rate-limited API tiers |

A model call that fails before showing anything is retried (also up to `tool_retries` times,
with the same backoff); one that fails after tokens were streamed is not, because they cannot be
taken back. A tool that keeps failing becomes an error result the
model can explain; if nothing at all came back, the outcome is `error`.

## Prompts

`src/support_agent/agent/prompts/v8/system.md` is the system prompt (earlier versions are recorded in `prompts/__init__.py`). The version is attached to
every trace and report, so a quality change can be traced to a prompt change. Add a new folder
and bump `PROMPT_VERSION` when you change it, and re-run `support-agent eval`.

## Known limits

* PostgreSQL checkpoints do not work on Windows (psycopg cannot use the proactor event loop that
  the MCP subprocess needs). Use SQLite there, or run on Linux.
* `AgentState` has no `plan`: nothing needs one in the single-agent design.
* The input and output checks are phrase-based. They catch the common attacks and the mistakes
  worth catching, not a determined attacker; do not rely on them alone.
* Only one request can wait at a time; a second `propose_draft` in the same step is refused.
* A model that writes text *and* calls a tool in the same step would show that text. The prompt
  forbids it; the answer in `done` is always the clean final one.
