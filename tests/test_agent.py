from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from support_agent.agent.agent import SupportAgent
from support_agent.agent.checkpoint import open_checkpointer, sqlite_path
from support_agent.agent.events import AgentEvent
from support_agent.agent.graph import AgentDeps, build_graph
from support_agent.core.principal import Principal, session_namespace
from support_agent.core.settings import AppConfig, RetrievalConfig
from support_agent.rag.index import VectorStore
from support_agent.rag.ingest import ingest
from support_agent.rag.retriever import Retriever
from support_agent.tools.client import DomainToolClient
from tests.conftest import ALICE, BOB, DEMO
from tests.fakes import AgentFakeLLM, FakeSparse, HashingEmbeddings

RETURN_Q = "How many days do I have to return an item within delivery?"
SEARCH_RETURN = {"query": "return window days delivery"}


@pytest.fixture
def fast_config(app_config: AppConfig) -> AppConfig:
    cfg = app_config.model_copy(deep=True)
    cfg.agent.tool_backoff_seconds = 0.0  # retries must not slow the tests down
    return cfg


@pytest.fixture
def retriever(store: VectorStore) -> Retriever:
    cfg = RetrievalConfig(top_k=4, score_threshold=0.25)
    ingest(
        DEMO / "knowledge", store=store, embeddings=HashingEmbeddings(), sparse=FakeSparse(),
        embedding_model="fake", cfg=cfg,
    )  # fmt: skip
    return Retriever(store, HashingEmbeddings(), cfg, FakeSparse())


@pytest_asyncio.fixture
async def make_agent(
    retriever: Retriever, tool_client: DomainToolClient, fast_config: AppConfig
) -> AsyncIterator[Callable[..., SupportAgent]]:
    def build(
        llm: AgentFakeLLM, *, config: AppConfig | None = None, checkpointer: Any = None
    ) -> SupportAgent:
        cfg = config or fast_config
        deps = AgentDeps(model=llm.model, retriever=retriever, client=tool_client, config=cfg)
        graph = build_graph(deps, checkpointer or InMemorySaver())
        return SupportAgent(graph=graph, retriever=retriever, config=cfg)

    yield build


async def collect(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in stream]


def kinds(events: list[AgentEvent]) -> list[str]:
    return [e.kind for e in events]


def tokens(events: list[AgentEvent]) -> str:
    return "".join(e.data["text"] for e in events if e.kind == "token")


def done(events: list[AgentEvent]) -> dict[str, Any]:
    assert events[-1].kind == "done", [e.kind for e in events]
    return events[-1].data


def tool_text(llm: AgentFakeLLM, call_index: int) -> str:
    """Everything the model saw as tool output in its `call_index`-th agent call."""
    return "\n".join(str(m.content) for m in llm.agent_calls()[call_index] if m.type == "tool")


# --- a policy question end to end ---------------------------------------------------------------------


async def test_policy_question_streams_events_in_order_and_cites_a_source(make_agent):
    llm = AgentFakeLLM(
        [
            {"tools": [("search_policy", SEARCH_RETURN)]},
            {"text": "You have 7 days to return it [D1]."},
        ],
        route="policy",
    )
    events = await collect(make_agent(llm).stream(RETURN_Q, ALICE))

    k = kinds(events)
    assert k[:2] == ["session", "route"] and k[-1] == "done"
    assert (
        k.index("tool_start")
        < k.index("tool_end")
        < k.index("token")
        < k.index("citation")
        < len(k) - 1
    )
    assert events[0].data["session_id"] and events[0].data["message_id"]
    assert events[1].data == {"route": "policy", "language": "en"}

    result = done(events)
    assert (
        tokens(events) == result["answer"] == "You have 7 days to return it."
    )  # markers never shown
    assert result["outcome"] == "answered" and result["tool_calls"] == ["search_policy"]
    assert result["citations"] and result["citations"][0]["source"].endswith(".md")
    assert result["retrieved"] and result["steps"] == 2 and result["prompt_version"] == "v9"
    assert sum(1 for e in events if e.kind == "citation") == len(result["citations"])


async def test_the_first_step_is_forced_to_use_a_tool_and_later_steps_are_not(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("search_policy", SEARCH_RETURN)]}, {"text": "Seven days [D1]."}],
        route="policy",
    )
    await make_agent(llm).answer(RETURN_Q, ALICE)
    first, second = llm.agent_binds()
    assert first["kwargs"] == {"tool_choice": "any"} and second["kwargs"] == {}
    assert set(first["tools"]) == {
        "search_policy",
        "load_skill",
    }  # a policy question is only offered policy tools


async def test_token_usage_is_summed_over_router_and_agent_calls(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("search_policy", SEARCH_RETURN)]}, {"text": "Seven days [D1]."}],
        route="policy",
    )
    answer = await make_agent(llm).answer(RETURN_Q, ALICE)
    assert (answer.usage.input_tokens, answer.usage.output_tokens) == (10 + 20 + 20, 5 + 8 + 8)


async def test_answer_returns_the_same_shape_as_the_phase_a_pipeline(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("get_order", {"order_id": "1236"}), ("get_shipment_status", {"order_id": "1236"})]},
         {"text": "It is in transit with GHTK."}],
        route="personal",
    )  # fmt: skip
    r = await make_agent(llm).answer("Where is order #1236?", ALICE)
    assert (r.route, r.outcome, r.sufficient) == ("personal", "answered", True)
    assert r.order_ids == ["1236"] and r.tool_calls == ["get_order", "get_shipment_status"]
    assert r.citations == [] and r.steps == 2 and r.latency_ms >= 0 and r.language == "en"


async def test_personal_route_cannot_use_the_policy_tool(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("search_policy", SEARCH_RETURN)]}, {"text": "Sorry."}], route="personal"
    )
    r = await make_agent(llm).answer("Where is order #1236?", ALICE)
    assert '"FORBIDDEN"' in tool_text(llm, 1) and r.outcome == "no_info"
    assert llm.agent_binds()[0]["tools"] and "search_policy" not in llm.agent_binds()[0]["tools"]


async def test_combined_question_gets_policy_and_an_authoritative_rule_result(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("get_order", {"order_id": "1234"}), ("check_return_eligibility", {"order_id": "1234"}),
                    ("search_policy", SEARCH_RETURN)]},
         {"text": "Yes, 3 days left [D1]."}],
        route="combined",
    )  # fmt: skip
    r = await make_agent(llm).answer(
        "Is my order #1234 still eligible for return within delivery days?", ALICE
    )
    assert r.outcome == "answered" and r.route == "combined" and r.citations
    seen = tool_text(llm, 1)
    assert 'authoritative="true"' in seen and '"eligible":' in seen.replace(" ", "")
    assert "<documents>" in seen
    system = str(llm.agent_calls()[0][0].content)
    assert (
        "order id(s) 1234" in system and "call check_return_eligibility" in system
    )  # code-derived hints


# --- not found, no info, isolation ---------------------------------------------------------------------


async def test_someone_elses_order_is_reported_as_not_found_and_its_data_never_reaches_the_model(
    make_agent,
):
    llm = AgentFakeLLM(
        [{"tools": [("get_order", {"order_id": "2001"})]}, {"text": "I could not find order 2001 in your account."}],
        route="personal",
    )  # fmt: skip
    r = await make_agent(llm).answer("Where is order #2001?", ALICE)  # 2001 belongs to BOB
    assert r.outcome == "not_found"
    leak = " ".join(str(m.content) for call in llm.model.calls for m in call)
    assert "Nova X100" not in leak and "VNP2001" not in leak and "Trần Hưng Đạo" not in leak


async def test_the_same_order_is_visible_to_its_owner(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("get_order", {"order_id": "2001"})]}, {"text": "It contains a phone."}],
        route="personal",
    )
    r = await make_agent(llm).answer("Show order #2001", BOB)
    assert r.outcome == "answered" and "Nova X100" in tool_text(llm, 1)


async def test_the_model_cannot_smuggle_an_identity_into_a_tool(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("get_order", {"order_id": "2001", "customer_id": "u_101"})]}, {"text": "I could not look that up."}],
        route="personal",
    )  # fmt: skip
    r = await make_agent(llm).answer("Show my order 2001", ALICE)
    assert '"FORBIDDEN"' in tool_text(llm, 1) and "Nova X100" not in tool_text(llm, 1)
    assert r.outcome == "no_info"  # nothing it was told came from real data


async def test_no_info_sentinel_becomes_an_outcome_and_is_never_shown(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("search_policy", SEARCH_RETURN)]}, {"text": "[NO_INFO] Sorry, I do not have that information."}],
        route="policy",
    )  # fmt: skip
    events = await collect(make_agent(llm).stream(RETURN_Q, ALICE))
    result = done(events)
    assert result["outcome"] == "no_info" and result["citations"] == []
    assert result["answer"] == "Sorry, I do not have that information." == tokens(events)
    assert "NO_INFO" not in tokens(events)


async def test_an_answer_with_no_grounding_is_no_info_even_without_the_sentinel(make_agent):
    llm = AgentFakeLLM(
        [
            {"tools": [("search_policy", {"query": "zzqx wvvk plorb"})]},
            {"text": "I cannot help with that."},
        ],
        route="policy",
    )
    r = await make_agent(llm).answer("zzqx wvvk plorb?", ALICE)
    assert (
        "No policy text is relevant" in tool_text(llm, 1)
        and r.outcome == "no_info"
        and r.citations == []
    )


async def test_a_policy_answer_without_citation_markers_still_carries_a_source(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("search_policy", SEARCH_RETURN)]}, {"text": "Seven days from delivery."}],
        route="policy",
    )
    r = await make_agent(llm).answer(RETURN_Q, ALICE)
    assert r.outcome == "answered" and len(r.citations) == 1  # FR-002 fallback: the top document


async def test_invented_citation_ids_are_ignored(make_agent):
    llm = AgentFakeLLM(
        [
            {"tools": [("search_policy", SEARCH_RETURN)]},
            {"text": "Seven days [D1] and more [D99]."},
        ],
        route="policy",
    )
    r = await make_agent(llm).answer(RETURN_Q, ALICE)
    assert len(r.citations) == 1 and "D99" not in r.answer


# --- canned routes ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route,confidence,outcome,fragment",
    [
        ("chitchat", 0.95, "refused", "How can I help"),
        ("out_of_scope", 0.95, "refused", "only help with shopping"),
        ("policy", 0.2, "clarify", "clarify"),
    ],
)
async def test_non_data_routes_answer_without_the_agent_loop(
    make_agent, route, confidence, outcome, fragment
):
    llm = AgentFakeLLM([], route=route, confidence=confidence)
    events = await collect(make_agent(llm).stream("hello", ALICE))
    result = done(events)
    assert result["outcome"] == outcome and fragment in result["answer"]
    assert tokens(events) == result["answer"] and result["tool_calls"] == []
    assert llm.agent_calls() == []  # the agent model was never called


# --- conversations -----------------------------------------------------------------------------------------


async def test_follow_up_turns_see_earlier_dialogue_but_no_old_tool_data(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("get_order", {"order_id": "1236"})]}, {"text": "Order 1236 is in transit."},
         {"tools": [("search_policy", SEARCH_RETURN)]}, {"text": "Seven days [D1]."}],
        route="personal",
    )  # fmt: skip
    agent = make_agent(llm)
    first = done(await collect(agent.stream("Where is order #1236?", ALICE)))
    sid = first["session_id"]
    llm.route = "policy"
    second = done(await collect(agent.stream(RETURN_Q, ALICE, session_id=sid)))
    assert second["session_id"] == sid

    third_call = llm.agent_calls()[-1 - 1]  # the agent call that opened turn 2
    shown = [(m.type, str(m.content)) for m in third_call if m.type != "system"]
    assert ("human", "Where is order #1236?") in shown and (
        "ai",
        "Order 1236 is in transit.",
    ) in shown
    assert not any(m.type == "tool" for m in third_call)  # last turn's tool output is gone

    history = await agent.history(ALICE, sid)
    assert [h["role"] for h in history] == ["user", "assistant", "user", "assistant"]
    assert history[3]["content"] == "Seven days." and "[D1]" not in history[3]["content"]


async def test_scratch_messages_never_persist(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("get_order", {"order_id": "1236"})]}, {"text": "In transit."}],
        route="personal",
    )
    agent = make_agent(llm)
    sid = done(await collect(agent.stream("Where is order #1236?", ALICE)))["session_id"]
    thread = {"configurable": {"thread_id": f"{session_namespace('u_100')}:{sid}"}}
    state = await agent.graph.aget_state(thread)
    assert [m.type for m in state.values["messages"]] == ["human", "ai"]


async def test_a_session_id_cannot_reach_another_users_conversation(make_agent):
    llm = AgentFakeLLM([], route="chitchat")
    agent = make_agent(llm)
    sid = done(await collect(agent.stream("Hello, I am Alice with a secret question", ALICE)))[
        "session_id"
    ]

    assert len(await agent.history(ALICE, sid)) == 2
    assert await agent.history(BOB, sid) == []  # same session id, different user: nothing

    llm.model.calls.clear()
    await agent.answer("Hello", BOB, session_id=sid)
    shown_to_model = " ".join(str(m.content) for call in llm.model.calls for m in call)
    assert "secret question" not in shown_to_model  # Alice's turn is not in Bob's context
    assert [h["content"] for h in await agent.history(BOB, sid) if h["role"] == "user"] == ["Hello"]
    assert len(await agent.history(ALICE, sid)) == 2  # and Alice's history is untouched


async def test_turns_on_one_session_are_serialised(make_agent):
    llm = AgentFakeLLM([], route="chitchat")
    agent = make_agent(llm)
    sid = "shared"
    await asyncio.gather(
        *(agent.answer(f"Hello number {i}", ALICE, session_id=sid) for i in range(4))
    )
    roles = [h["role"] for h in await agent.history(ALICE, sid)]
    assert roles == ["user", "assistant"] * 4  # never user, user, assistant...


async def test_the_reply_language_follows_the_question(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("search_policy", SEARCH_RETURN)]}, {"text": "Bảy ngày [D1]."}], route="policy"
    )
    r = await make_agent(llm).answer(
        "Tôi được đổi trả trong bao nhiêu ngày kể từ khi nhận hàng?", ALICE
    )
    assert r.language == "vi" and "Reply in Vietnamese" in str(llm.agent_calls()[0][0].content)


async def test_overlong_questions_are_truncated_before_the_model_sees_them(make_agent, fast_config):
    llm = AgentFakeLLM([], route="chitchat")
    await make_agent(llm).answer("hello " * 2000, ALICE)
    assert len(str(llm.model.calls[0][-1].content)) <= fast_config.guardrails.max_input_chars


async def test_older_dialogue_is_limited_to_the_configured_window(make_agent, fast_config):
    fast_config.agent.history_messages = 2
    llm = AgentFakeLLM([], route="chitchat")
    agent = make_agent(llm, config=fast_config)
    sid = "s"
    for i in range(4):
        await agent.answer(f"question number {i}", ALICE, session_id=sid)
    router_input = " ".join(str(m.content) for m in llm.model.calls[-1])
    assert "question number 3" in router_input and "question number 0" not in router_input


# --- limits and failures ------------------------------------------------------------------------------------


async def test_the_loop_stops_at_max_steps_with_a_friendly_message(make_agent, fast_config):
    fast_config.agent.max_steps = 2
    llm = AgentFakeLLM([{"tools": [("get_order", {"order_id": "1236"})]}] * 5, route="personal")
    events = await collect(
        make_agent(llm, config=fast_config).stream("Where is order #1236?", ALICE)
    )
    result = done(events)
    assert result["outcome"] == "error" and "allowed number of steps" in result["answer"]
    assert result["steps"] == 2 and tokens(events) == result["answer"]


async def test_a_flaky_model_call_is_retried_before_anything_is_shown(make_agent):
    llm = AgentFakeLLM(
        [{"error": RuntimeError("503")}, {"tools": [("search_policy", SEARCH_RETURN)]}, {"text": "Seven days [D1]."}],
        route="policy",
    )  # fmt: skip
    r = await make_agent(llm).answer(RETURN_Q, ALICE)
    assert r.outcome == "answered" and r.answer == "Seven days."


async def test_a_dead_model_ends_the_turn_with_an_error_and_the_session_stays_usable(make_agent):
    llm = AgentFakeLLM([{"error": RuntimeError("down")}] * 3, route="policy")
    agent = make_agent(llm)
    events = await collect(agent.stream(RETURN_Q, ALICE, session_id="s1"))
    assert events[-1].kind == "error" and events[-1].data["code"] == "UPSTREAM_ERROR"
    assert "down" not in str(events[-1].data["message"])  # provider detail never leaks

    llm.add_steps({"tools": [("search_policy", SEARCH_RETURN)]}, {"text": "Seven days [D1]."})
    retry = await agent.answer(RETURN_Q, ALICE, session_id="s1")
    assert retry.outcome == "answered"
    history = await agent.history(ALICE, "s1")
    assert [h["role"] for h in history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]  # no orphan question


async def test_an_empty_model_answer_is_an_error_not_silence(make_agent):
    llm = AgentFakeLLM(
        [{"tools": [("search_policy", SEARCH_RETURN)]}, {"text": ""}], route="policy"
    )
    events = await collect(make_agent(llm).stream(RETURN_Q, ALICE))
    result = done(events)
    assert result["outcome"] == "error" and result["answer"] and tokens(events) == result["answer"]


async def test_a_backend_outage_after_retries_is_an_error_outcome(make_agent, service, monkeypatch):
    from support_agent.mcp_db.adapters.base import AdapterError

    async def down(*a: Any, **k: Any) -> None:
        raise AdapterError("db down")

    monkeypatch.setattr(service.adapter, "get_order", down)
    llm = AgentFakeLLM(
        [
            {"tools": [("get_order", {"order_id": "1236"})]},
            {"text": "The order system is unavailable."},
        ],
        route="personal",
    )
    r = await make_agent(llm).answer("Where is order #1236?", ALICE)
    assert r.outcome == "error" and "db down" not in r.answer


async def test_a_slow_turn_times_out_and_is_cleaned_up(make_agent, fast_config):
    fast_config.agent.run_timeout_seconds = 0.3
    llm = AgentFakeLLM([{"tools": [("search_policy", SEARCH_RETURN)], "delay": 5}], route="policy")
    agent = make_agent(llm, config=fast_config)
    started = time.perf_counter()
    events = await collect(agent.stream(RETURN_Q, ALICE, session_id="slow"))
    assert time.perf_counter() - started < 2
    assert events[-1].kind == "error" and events[-1].data["code"] == "TIMEOUT"
    roles = [h["role"] for h in await agent.history(ALICE, "slow")]
    assert roles == ["user", "assistant"]


async def test_recursion_errors_map_to_the_step_limit_message(make_agent, monkeypatch):
    llm = AgentFakeLLM([], route="chitchat")
    agent = make_agent(llm)

    def explode(*a: Any, **k: Any):
        raise GraphRecursionError("too deep")

    monkeypatch.setattr(agent.graph, "astream", explode)
    events = await collect(agent.stream("hello", ALICE))
    assert events[-1].kind == "error" and events[-1].data["code"] == "STEP_LIMIT"


async def test_closing_the_stream_early_releases_the_conversation(make_agent):
    llm = AgentFakeLLM([], route="chitchat")
    agent = make_agent(llm)
    stream = agent.stream("hello", ALICE, session_id="c")
    await stream.__anext__()  # session event, then the client disconnects
    await stream.aclose()
    r = await asyncio.wait_for(agent.answer("hello again", ALICE, session_id="c"), timeout=5)
    assert r.outcome == "refused"


# --- durable state (FR-205) ---------------------------------------------------------------------------------


async def test_conversations_survive_a_restart_with_the_sqlite_checkpointer(
    make_agent, tmp_path: Path
):
    url = f"sqlite:///{(tmp_path / 'cp.db').as_posix()}"
    llm = AgentFakeLLM([], route="chitchat")
    async with open_checkpointer(url) as cp:
        sid = done(await collect(make_agent(llm, checkpointer=cp).stream("hello", ALICE)))[
            "session_id"
        ]

    async with open_checkpointer(url) as cp:  # a fresh process would do exactly this
        restarted = make_agent(AgentFakeLLM([], route="chitchat"), checkpointer=cp)
        history = await restarted.history(ALICE, sid)
        assert [h["role"] for h in history] == ["user", "assistant"]
        await restarted.answer("hello again", ALICE, session_id=sid)
        assert len(await restarted.history(ALICE, sid)) == 4


@pytest.mark.parametrize(
    "url,path",
    [
        ("sqlite:///./data/x.db", "./data/x.db"),
        ("sqlite+aiosqlite:///C:/tmp/x.db", "C:/tmp/x.db"),
        ("plain/path.db", "plain/path.db"),
    ],
)
def test_sqlite_urls_are_turned_into_paths(url: str, path: str):
    assert sqlite_path(url) == path


async def test_memory_checkpointer_needs_no_files():
    async with open_checkpointer("memory") as cp:
        assert isinstance(cp, InMemorySaver)


def test_principal_is_not_a_field_any_node_returns():
    """The identity is written once, before the graph starts; nodes must never return it."""
    import inspect

    from support_agent.agent import graph as graph_module

    source = inspect.getsource(graph_module.build_graph)
    assert '"principal":' not in source and "'principal':" not in source
    assert Principal(user_id="u_1").user_id == "u_1"


@pytest.mark.skipif(sys.platform != "win32", reason="the proactor loop exists only on Windows")
async def test_postgres_checkpoints_explain_why_they_cannot_run_on_windows():
    with pytest.raises(RuntimeError, match="Windows"):
        async with open_checkpointer("postgresql://user:pw@localhost/db"):
            pass


async def test_loading_a_skill_does_not_hide_a_missing_order(make_agent):
    llm = AgentFakeLLM(
        [
            {"tools": [("load_skill", {"name": "track-order"})]},
            {"tools": [("get_order", {"order_id": "9999"})]},
            {"text": "I could not find order 9999 in your account."},
        ],
        route="personal",
    )
    result = await make_agent(llm).answer("Where is order #9999?", ALICE)
    assert result.outcome == "not_found"


async def test_a_missing_order_stays_not_found_when_the_model_also_searches_the_catalogue(
    make_agent,
):
    llm = AgentFakeLLM(
        [
            {"tools": [("get_order", {"order_id": "9999"})]},
            {"tools": [("search_products", {"query": "headphones"})]},
            {"text": "I could not find order 9999 in your account."},
        ],
        route="personal",
    )
    result = await make_agent(llm).answer("Please return everything in order 9999.", ALICE)
    assert result.outcome == "not_found"


async def test_a_missing_order_with_a_policy_answer_is_answered(make_agent):
    llm = AgentFakeLLM(
        [
            {"tools": [("get_order", {"order_id": "9999"}), ("search_policy", {"query": "return window days"})]},
            {"text": "I could not find order 9999, but returns are accepted within 7 days [D1]."},
        ],
        route="combined",
    )  # fmt: skip
    result = await make_agent(llm).answer(
        "Can I return order 9999 and what is the return window?", ALICE
    )
    assert result.outcome == "answered"


async def test_an_out_of_stock_refusal_is_an_answer_not_no_info(make_agent):
    llm = AgentFakeLLM(
        [
            {"tools": [("prepare_order_draft", {
                "items": [{"sku": "CHG-65W", "qty": 1}],
                "shipping_address": "12 Lê Lợi, Quận 1, TP.HCM",
                "payment_method": "momo",
            })]},
            {"text": "Sorry, the 65W charger is out of stock."},
        ],
        route="personal",
    )  # fmt: skip
    result = await make_agent(llm).answer(
        "Order one 65W charger, pay with MoMo, deliver to 12 Lê Lợi.", ALICE
    )
    assert result.outcome == "answered"


async def test_exceeding_the_quantity_limit_is_an_answer_not_no_info(make_agent):
    llm = AgentFakeLLM(
        [
            {"tools": [("prepare_order_draft", {
                "items": [{"sku": "MOU-M10", "qty": 25}],
                "shipping_address": "9 Nguyễn Huệ, Đà Nẵng",
                "payment_method": "bank_transfer",
            })]},
            {"text": "At most 20 of one product can be ordered."},
        ],
        route="personal",
    )  # fmt: skip
    result = await make_agent(llm).answer(
        "Order 25 M10 mice to 9 Nguyễn Huệ, Đà Nẵng, bank transfer.", ALICE
    )
    assert result.outcome == "answered"


async def test_a_no_from_the_rules_wins_over_a_model_that_says_no_info(make_agent):
    llm = AgentFakeLLM(
        [
            {"tools": [("prepare_order_draft", {
                "items": [{"sku": "MOU-M10", "qty": 25}],
                "shipping_address": "9 Nguyễn Huệ, Đà Nẵng",
                "payment_method": "bank_transfer",
            })]},
            {"text": "[NO_INFO] At most 20 of one product can be ordered."},
        ],
        route="personal",
    )  # fmt: skip
    result = await make_agent(llm).answer(
        "Order 25 M10 mice to 9 Nguyễn Huệ, Đà Nẵng, bank transfer.", ALICE
    )
    assert result.outcome == "answered"
