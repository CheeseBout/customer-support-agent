"""The HTTP API (SPEC 14): chat (JSON and SSE), sessions, drafts, memory, admin, health."""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sse_starlette import EventSourceResponse, ServerSentEvent

from support_agent import __version__
from support_agent.agent.events import AgentEvent
from support_agent.api.auth import InvalidToken
from support_agent.api.errors import (
    ERROR_RESPONSES,
    ApiError,
    RequestIdMiddleware,
    install_error_handlers,
)
from support_agent.api.schemas import (
    AdminDraftList,
    AdminDraftOut,
    ApproveRequest,
    ChatRequest,
    ChatResponse,
    CreatedDraft,
    DraftList,
    DraftOut,
    EventList,
    EventOut,
    FactIn,
    FactOut,
    FeedbackRequest,
    Health,
    IngestRequest,
    IngestResponse,
    Interrupt,
    MemoryView,
    Message,
    Readiness,
    RejectRequest,
    ResumeRequest,
    SessionDetail,
    SessionList,
    SessionSummary,
    Usage,
)
from support_agent.api.services import Services, open_services
from support_agent.core.i18n import Lang, resolve_language, t
from support_agent.core.logging import bind_context, request_id_var
from support_agent.core.principal import Principal, hash_user_id
from support_agent.core.settings import Settings, get_settings
from support_agent.drafts.events import EventState
from support_agent.drafts.models import DraftStatus, DraftType
from support_agent.drafts.service import DraftError, DraftService
from support_agent.memory.workspace import ARTIFACT_NAMES, WorkspaceError
from support_agent.security.guardrails import check_input, clean_input

log = logging.getLogger(__name__)

bearer = HTTPBearer(auto_error=False, description="A JWT issued by the shop's own system.")

_DRAFT_ERROR_STATUS = {"NOT_FOUND": 404, "FORBIDDEN": 403, "CONFLICT": 409, "INVALID_ARGUMENT": 400}


# --- dependencies ---------------------------------------------------------------------


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


async def current_principal(
    services: ServicesDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)],
) -> Principal:
    unauthorized = {"WWW-Authenticate": "Bearer"}
    if credentials is None:
        raise ApiError(401, message="Missing bearer token.", headers=unauthorized)
    try:
        principal = await services.verifier.verify(credentials.credentials)
    except InvalidToken as exc:
        log.info("token rejected: %s", exc)
        raise ApiError(401, message="Invalid or expired token.", headers=unauthorized) from exc
    bind_context(user_hash=hash_user_id(principal.user_id))
    return principal


async def customer(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
    if principal.role != "customer":
        raise ApiError(403, message="This endpoint is for customers.")
    return principal


async def staff(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
    if principal.role != "staff":
        raise ApiError(403, message="This endpoint requires the staff role.")
    return principal


CustomerDep = Annotated[Principal, Depends(customer)]
StaffDep = Annotated[Principal, Depends(staff)]


def require_drafts(services: Services) -> DraftService:
    if services.drafts is None:
        raise ApiError(503, "NOT_CONFIGURED", "Requests are not configured (set DRAFTS_DB_URL).")
    return services.drafts


def draft_failure(exc: DraftError) -> ApiError:
    return ApiError(_DRAFT_ERROR_STATUS.get(exc.code, 400), message=exc.message)


# --- chat plumbing --------------------------------------------------------------------


def prepare_turn(
    services: Services, principal: Principal, message: str, language: str
) -> tuple[str, Lang]:
    """Rate limit, validate and screen a customer message. Raises `ApiError` to refuse it."""
    cfg = services.settings.app.guardrails
    wait = services.limiter.hit(principal.user_id)
    text = clean_input(message)
    lang = resolve_language(language, text)
    if wait:
        seconds = math.ceil(wait)
        raise ApiError(
            429,
            message=t("rate_limited", lang, seconds=seconds),
            headers={"Retry-After": str(seconds)},
        )
    if not text:
        raise ApiError(400, message="The message is empty.")
    if len(text) > cfg.max_input_chars:
        raise ApiError(400, message=f"The message is longer than {cfg.max_input_chars} characters.")
    verdict = check_input(text, cfg)
    if verdict.blocked:
        log.warning("input blocked by guardrails", extra={"reasons": ",".join(verdict.reasons)})
        raise ApiError(422, message=t("guardrail_blocked", lang))
    return text, lang


def trace_kwargs(services: Services, principal: Principal, session_id: str) -> dict[str, Any]:
    tracing = services.tracing
    if not tracing.enabled:
        return {}
    trace_id = tracing.trace_id(f"{request_id_var.get() or uuid.uuid4().hex}:{session_id}")
    handler = tracing.handler(trace_id)
    return {
        "callbacks": [handler] if handler else None,
        "trace_id": trace_id,
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": hash_user_id(principal.user_id),
        },
    }


async def preferences_for(services: Services, principal: Principal) -> list[tuple[str, str]]:
    """The customer's saved preferences, when long-term memory is on (SPEC 12.2)."""
    if not services.settings.app.memory.long_term_enabled:
        return []
    return [(f.key, f.value) for f in await services.store.list_facts(principal.user_id)]


def interrupt_model(stop: dict[str, Any]) -> Interrupt:
    known = Interrupt.model_fields.keys()
    return Interrupt(**{k: v for k, v in stop.items() if k in known})


def _usage(data: dict[str, Any]) -> Usage:
    return Usage(**(data.get("usage") or {}))


def _drafts(data: dict[str, Any]) -> list[CreatedDraft]:
    return [CreatedDraft(**d) for d in data.get("drafts", [])]


def done_response(data: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        session_id=data["session_id"],
        message_id=data["message_id"],
        answer=data["answer"],
        outcome=data["outcome"],
        language=data["language"],
        route=data.get("route"),
        citations=data.get("citations", []),
        drafts=_drafts(data),
        usage=_usage(data),
        trace_id=data.get("trace_id"),
    )


def interrupt_response(data: dict[str, Any], lang: Lang) -> ChatResponse:
    turn = data.get("turn") or {}
    return ChatResponse(
        session_id=data["session_id"],
        message_id=data["message_id"],
        answer=t("confirm_prompt", turn.get("language", lang)),
        outcome="confirmation",
        language=turn.get("language", lang),
        route=turn.get("route"),
        citations=turn.get("citations", []),
        interrupt=interrupt_model(data),
        usage=_usage(turn),
        trace_id=turn.get("trace_id"),
    )


_ERROR_STATUS = {"NO_PENDING_CONFIRMATION": 409, "TIMEOUT": 504, "UPSTREAM_ERROR": 502}


async def collect_response(events: AsyncIterator[AgentEvent], lang: Lang) -> ChatResponse:
    """Run a turn to its end and shape the result as the non-streaming response."""
    session_id = message_id = ""
    async with closing_events(events) as stream:
        async for event in stream:
            data = event.data
            if event.kind == "session":
                session_id, message_id = data["session_id"], data["message_id"]
            elif event.kind == "done":
                return done_response(data)
            elif event.kind == "interrupt":
                return interrupt_response(data, lang)
            elif event.kind == "error":
                status = _ERROR_STATUS.get(str(data["code"]))
                if status is None:  # e.g. STEP_LIMIT: a controlled apology, not a failure
                    return ChatResponse(
                        session_id=session_id,
                        message_id=message_id,
                        answer=str(data["message"]),
                        outcome="error",
                        language=lang,
                    )
                raise ApiError(status, message=str(data["message"]))
    raise ApiError(502, message=t("upstream_error", lang))


@asynccontextmanager
async def closing_events(
    events: AsyncIterator[AgentEvent],
) -> AsyncIterator[AsyncIterator[AgentEvent]]:
    """Make sure the agent's event stream is closed, whatever ends the loop that reads it."""
    try:
        yield events
    finally:
        aclose = getattr(events, "aclose", None)
        if aclose is not None:
            await aclose()


def sse_frame(event: AgentEvent) -> ServerSentEvent:
    """One agent event as an SSE frame (SPEC 14.5). Raw tool data never goes out."""
    data = dict(event.data)
    if event.kind == "interrupt":
        turn = data.pop("turn", {}) or {}
        data = interrupt_model(data).model_dump(mode="json") | {
            "session_id": data["session_id"],
            "message_id": data["message_id"],
            "usage": _usage(turn).model_dump(),
            "trace_id": turn.get("trace_id"),
        }
    elif event.kind == "done":
        data = {
            "session_id": data["session_id"],
            "message_id": data["message_id"],
            "outcome": data["outcome"],
            "route": data.get("route"),
            "language": data["language"],
            "drafts": data.get("drafts", []),
            "usage": data.get("usage", {}),
            "trace_id": data.get("trace_id"),
        }
    return ServerSentEvent(event=event.kind, data=json.dumps(data, ensure_ascii=False, default=str))


async def sse_stream(events: AsyncIterator[AgentEvent]) -> AsyncIterator[ServerSentEvent]:
    # Closing the agent's stream when this generator is cancelled (a dropped client) releases the
    # conversation lock and stops the graph.
    async with closing_events(events) as stream:
        async for event in stream:
            yield sse_frame(event)


# --- routers --------------------------------------------------------------------------

chat = APIRouter(prefix="/v1", tags=["chat"], responses=ERROR_RESPONSES)


@chat.post("/chat", response_model=ChatResponse, summary="Send a message and wait for the answer")
async def post_chat(
    body: ChatRequest, principal: CustomerDep, services: ServicesDep
) -> ChatResponse:
    text, lang = prepare_turn(services, principal, body.message, body.language)
    session_id = body.session_id or uuid.uuid4().hex
    bind_context(session_id=session_id)
    await services.store.touch_session(principal.user_id, session_id, text)
    events = services.agent.stream(
        text,
        principal,
        session_id=session_id,
        language=body.language,
        preferences=await preferences_for(services, principal),
        **trace_kwargs(services, principal, session_id),
    )
    return await collect_response(events, lang)


@chat.post(
    "/chat/stream",
    summary="Send a message and stream the answer (Server-Sent Events)",
    response_class=EventSourceResponse,
    responses={
        200: {
            "description": (
                "`text/event-stream`. Events: `session`, `route`, `tool_start`, `tool_end`, "
                "`token`, `citation`, `replace` (the text so far is withdrawn: show this "
                "instead), `interrupt` (a confirmation is needed; the stream ends), `done`, "
                "`error`. A `: ping` comment is sent every 15 seconds."
            ),
            "content": {"text/event-stream": {}},
        }
    },
)
async def post_chat_stream(
    body: ChatRequest, principal: CustomerDep, services: ServicesDep
) -> EventSourceResponse:
    text, _ = prepare_turn(services, principal, body.message, body.language)
    session_id = body.session_id or uuid.uuid4().hex
    bind_context(session_id=session_id)
    await services.store.touch_session(principal.user_id, session_id, text)
    events = services.agent.stream(
        text,
        principal,
        session_id=session_id,
        language=body.language,
        preferences=await preferences_for(services, principal),
        **trace_kwargs(services, principal, session_id),
    )
    return EventSourceResponse(sse_stream(events), ping=15)


@chat.post(
    "/chat/resume",
    response_model=ChatResponse,
    summary="Answer a confirmation (approve, reject or edit)",
)
async def post_resume(
    body: ResumeRequest, principal: CustomerDep, services: ServicesDep
) -> ChatResponse:
    wait = services.limiter.hit(principal.user_id)
    if wait:
        seconds = math.ceil(wait)
        raise ApiError(
            429,
            message=t("rate_limited", "en", seconds=seconds),
            headers={"Retry-After": str(seconds)},
        )
    if body.decision != "edit" and body.edits:
        raise ApiError(400, message="`edits` is only valid with decision=edit.")
    if not await services.store.owns_session(principal.user_id, body.session_id):
        raise ApiError(409, message=t("stale_confirmation", "en"))
    bind_context(session_id=body.session_id)
    await services.store.touch_session(principal.user_id, body.session_id)
    events = services.agent.resume(
        principal,
        body.session_id,
        body.interrupt_id,
        body.decision,
        edits=body.edits,
        **trace_kwargs(services, principal, body.session_id),
    )
    return await collect_response(events, "en")


sessions = APIRouter(prefix="/v1/sessions", tags=["sessions"], responses=ERROR_RESPONSES)


async def _owned_session(services: Services, principal: Principal, session_id: str) -> None:
    if not await services.store.owns_session(principal.user_id, session_id):
        raise ApiError(404, message="Session not found.")


@sessions.get("", response_model=SessionList, summary="Your conversations, newest first")
async def list_sessions(
    principal: CustomerDep,
    services: ServicesDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SessionList:
    found = await services.store.list_sessions(principal.user_id, limit=limit, offset=offset)
    return SessionList(sessions=[SessionSummary(**s.model_dump()) for s in found])


@sessions.get("/{session_id}", response_model=SessionDetail, summary="One conversation")
async def get_session(
    session_id: str, principal: CustomerDep, services: ServicesDep
) -> SessionDetail:
    info = await services.store.get_session(principal.user_id, session_id)
    if info is None:
        raise ApiError(404, message="Session not found.")
    history = await services.agent.history(principal, session_id)
    pending = await services.agent.pending_confirmation(principal, session_id)
    try:
        artifacts = services.workspace.names(principal.user_id, session_id)
    except WorkspaceError:
        artifacts = []
    return SessionDetail(
        **info.model_dump(),
        messages=[Message.model_validate(m) for m in history],
        pending_interrupt=interrupt_model(pending) if pending else None,
        artifacts=artifacts,
    )


@sessions.delete("/{session_id}", status_code=204, summary="Delete a conversation and its files")
async def delete_session(
    session_id: str, principal: CustomerDep, services: ServicesDep
) -> Response:
    await _owned_session(services, principal, session_id)
    await services.agent.delete_session(principal, session_id)
    services.workspace.delete_session(principal.user_id, session_id)
    await services.store.delete_session(principal.user_id, session_id)
    return Response(status_code=204)


@sessions.get(
    "/{session_id}/artifacts/{name}",
    response_class=PlainTextResponse,
    summary="A file the agent produced in this conversation",
    responses={200: {"content": {"text/markdown": {}}}},
)
async def get_artifact(
    session_id: str, name: str, principal: CustomerDep, services: ServicesDep
) -> PlainTextResponse:
    await _owned_session(services, principal, session_id)
    if name not in ARTIFACT_NAMES:
        raise ApiError(404, message="Artifact not found.")
    content = services.workspace.read(principal.user_id, session_id, name)
    if content is None:
        raise ApiError(404, message="Artifact not found.")
    return PlainTextResponse(content, media_type="text/markdown; charset=utf-8")


drafts_router = APIRouter(prefix="/v1/drafts", tags=["drafts"], responses=ERROR_RESPONSES)


@drafts_router.get("", response_model=DraftList, summary="Your requests")
async def list_drafts(
    principal: CustomerDep,
    services: ServicesDep,
    status: DraftStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> DraftList:
    drafts = require_drafts(services)
    found = await drafts.list_for(principal, statuses=[status] if status else None, limit=limit)
    return DraftList(drafts=[DraftOut.of(d) for d in found])


@drafts_router.post(
    "/{draft_id}/cancel", response_model=DraftOut, summary="Withdraw a pending request"
)
async def cancel_draft(draft_id: str, principal: CustomerDep, services: ServicesDep) -> DraftOut:
    try:
        return DraftOut.of(await require_drafts(services).cancel(principal, draft_id))
    except DraftError as exc:
        raise draft_failure(exc) from exc


feedback_router = APIRouter(prefix="/v1", tags=["feedback", "memory"], responses=ERROR_RESPONSES)


@feedback_router.post("/feedback", status_code=204, summary="Rate an answer")
async def post_feedback(
    body: FeedbackRequest, principal: CustomerDep, services: ServicesDep
) -> Response:
    await _owned_session(services, principal, body.session_id)
    await services.store.add_feedback(
        principal.user_id,
        body.session_id,
        body.message_id,
        1 if body.rating == "up" else -1,
        clean_input(body.comment),
    )
    return Response(status_code=204)


def _require_memory(services: Services) -> None:
    if not services.settings.app.memory.long_term_enabled:
        raise ApiError(409, message="Long-term memory is turned off.")


@feedback_router.get("/memory", response_model=MemoryView, summary="What the assistant remembers")
async def get_memory(principal: CustomerDep, services: ServicesDep) -> MemoryView:
    enabled = services.settings.app.memory.long_term_enabled
    facts = await services.store.list_facts(principal.user_id) if enabled else []
    return MemoryView(enabled=enabled, facts=[FactOut(**f.model_dump()) for f in facts])


@feedback_router.put("/memory", response_model=FactOut, summary="Ask the assistant to remember")
async def put_memory(body: FactIn, principal: CustomerDep, services: ServicesDep) -> FactOut:
    """Saving is an explicit act of the customer (consent). The value is shown to the model."""
    _require_memory(services)
    value = clean_input(body.value)
    if body.key == "preferred_language" and value not in ("vi", "en"):
        raise ApiError(400, message="preferred_language must be 'vi' or 'en'.")
    if check_input(value, services.settings.app.guardrails).blocked:
        raise ApiError(422, message=t("guardrail_blocked", "en"))
    fact = await services.store.set_fact(
        principal.user_id,
        body.key,
        value,
        ttl_days=services.settings.app.memory.long_term_ttl_days,
    )
    return FactOut(**fact.model_dump())


@feedback_router.delete("/memory", status_code=204, summary="Forget everything (or one item)")
async def delete_memory(
    principal: CustomerDep, services: ServicesDep, key: str | None = None
) -> Response:
    await services.store.delete_facts(principal.user_id, key)
    return Response(status_code=204)


admin = APIRouter(prefix="/v1/admin", tags=["admin"], responses=ERROR_RESPONSES)


@admin.get("/drafts", response_model=AdminDraftList, summary="The review queue")
async def admin_list_drafts(
    principal: StaffDep,
    services: ServicesDep,
    status: Annotated[DraftStatus | None, Query()] = "pending",
    type: Annotated[DraftType | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AdminDraftList:
    try:
        found = await require_drafts(services).queue(
            principal,
            statuses=[status] if status else None,
            draft_type=type,
            limit=limit,
            offset=offset,
        )
    except DraftError as exc:
        raise draft_failure(exc) from exc
    return AdminDraftList(drafts=[AdminDraftOut.of(d) for d in found], limit=limit, offset=offset)


@admin.get("/drafts/{draft_id}", response_model=AdminDraftOut, summary="One request")
async def admin_get_draft(
    draft_id: str, principal: StaffDep, services: ServicesDep
) -> AdminDraftOut:
    try:
        return AdminDraftOut.of(await require_drafts(services).get_for(principal, draft_id))
    except DraftError as exc:
        raise draft_failure(exc) from exc


@admin.post("/drafts/{draft_id}/approve", response_model=AdminDraftOut, summary="Approve")
async def admin_approve(
    draft_id: str, body: ApproveRequest, principal: StaffDep, services: ServicesDep
) -> AdminDraftOut:
    try:
        draft = await require_drafts(services).review(principal, draft_id, "approve", body.note)
    except DraftError as exc:
        raise draft_failure(exc) from exc
    log.info("draft approved", extra={"draft_id": draft_id})
    return AdminDraftOut.of(draft)


@admin.post("/drafts/{draft_id}/reject", response_model=AdminDraftOut, summary="Reject")
async def admin_reject(
    draft_id: str, body: RejectRequest, principal: StaffDep, services: ServicesDep
) -> AdminDraftOut:
    try:
        draft = await require_drafts(services).review(principal, draft_id, "reject", body.note)
    except DraftError as exc:
        raise draft_failure(exc) from exc
    log.info("draft rejected", extra={"draft_id": draft_id})
    return AdminDraftOut.of(draft)


@admin.get("/events", response_model=EventList, summary="Webhook deliveries")
async def admin_events(
    principal: StaffDep,
    services: ServicesDep,
    state: Annotated[EventState | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> EventList:
    """What was queued for the shop system, newest first. Look here for `failed` events."""
    drafts = require_drafts(services)
    found = await drafts.repo.list_events(state=state, limit=limit)
    return EventList(
        configured=bool(services.settings.webhook_url),
        events=[EventOut(**e.model_dump(exclude={"body"})) for e in found],
    )


@admin.post("/ingest", response_model=IngestResponse, summary="Reload the policy documents")
async def admin_ingest(
    body: IngestRequest, principal: StaffDep, services: ServicesDep
) -> IngestResponse:
    if services.ingest is None:
        raise ApiError(503, "NOT_CONFIGURED", "Ingest is not available.")
    try:
        report = await services.ingest(body.full)
    except FileNotFoundError as exc:
        raise ApiError(400, message=str(exc)) from exc
    log.info("ingest finished: %s", report.summary())
    return IngestResponse(
        added=report.added,
        updated=report.updated,
        unchanged=report.unchanged,
        removed=report.removed,
        failed=report.failed,
        chunks_written=report.chunks_written,
    )


health = APIRouter(tags=["health"])


@health.get("/health", response_model=Health, summary="Liveness")
async def get_health() -> Health:
    return Health()


@health.get(
    "/ready",
    response_model=Readiness,
    summary="Readiness (index, model capabilities)",
    responses={503: {"model": Readiness, "description": "Something is not ready"}},
)
async def get_ready(services: ServicesDep) -> Response:
    checks = await services.readiness.run()
    ready = services.readiness.ready(checks)
    body = Readiness(status="ready" if ready else "not_ready", checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if ready else 503)


# --- application ----------------------------------------------------------------------

DESCRIPTION = """\
Customer-support assistant for an e-commerce shop (Vietnamese and English).

Every endpoint except `/health` and `/ready` needs `Authorization: Bearer <JWT>`. The caller's
identity comes from the token (`sub` by default); the `role` claim is `customer` or `staff`.
Errors always have the shape `{"error": {"code", "message", "request_id"}}`.
"""


def create_app(settings: Settings | None = None, *, services: Services | None = None) -> FastAPI:
    """`services` lets tests inject fakes; in production the lifespan builds the real ones."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is None:
            async with open_services(settings) as built:
                app.state.services = built
                yield
        else:
            yield

    app = FastAPI(
        title="Customer Support Agent",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "chat", "description": "Talk to the assistant (JSON or SSE)."},
            {"name": "sessions", "description": "Your conversations and their files."},
            {"name": "drafts", "description": "Requests you made through the assistant."},
            {"name": "feedback", "description": "Rate answers."},
            {"name": "memory", "description": "Optional long-term preferences."},
            {"name": "admin", "description": "Staff only: review queue and document reload."},
            {"name": "health", "description": "Liveness and readiness."},
        ],
    )
    if services is not None:
        app.state.services = services  # set now: a test client may skip the lifespan
    install_error_handlers(app)
    origins = settings.cors_origin_list
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
        )
    app.add_middleware(RequestIdMiddleware)
    for router in (chat, sessions, drafts_router, feedback_router, admin, health):
        app.include_router(router)
    return app
