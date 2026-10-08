from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from support_agent.core.settings import AppConfig, RetrievalConfig
from support_agent.mcp_db.adapters.base import AdapterError
from support_agent.rag.index import VectorStore
from support_agent.rag.ingest import ingest
from support_agent.rag.pipeline import SupportRAG
from support_agent.rag.retriever import Retriever
from support_agent.tools.client import DomainToolClient
from tests.conftest import ALICE, BOB, ROOT
from tests.fakes import FakeSparse, HashingEmbeddings, ScriptedChatModel, SupportFakeLLM

CFG = RetrievalConfig(top_k=4, score_threshold=0.25)


@pytest.fixture
def indexed_store(store: VectorStore) -> VectorStore:
    """The real bilingual knowledge base, indexed with the offline fakes."""
    ingest(
        ROOT / "knowledge",
        store=store,
        embeddings=HashingEmbeddings(),
        sparse=FakeSparse(),
        embedding_model="fake",
        cfg=CFG,
    )
    return store


def build(
    llm: SupportFakeLLM | ScriptedChatModel | None,
    store: VectorStore,
    tools: DomainToolClient | None,
    config: AppConfig,
) -> SupportRAG:
    model = llm.model if isinstance(llm, SupportFakeLLM) else llm
    return SupportRAG(
        model=model,
        retriever=Retriever(store, HashingEmbeddings(), CFG, FakeSparse()),
        tools=tools,
        config=config,
    )


def answer_prompts(llm: SupportFakeLLM) -> list[str]:
    """Human messages of the calls that were answer-generation calls."""
    return [
        str(call[-1].content)
        for call in llm.model.calls
        if call and str(call[0].content).startswith("You are the customer support assistant")
    ]


# --- policy ------------------------------------------------------------------------------------------


async def test_policy_question_is_answered_from_documents_with_citations(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="policy", answer="You have 7 days.", used_sources=[1])
    rag = build(llm, indexed_store, tool_client, app_config)

    r = await rag.answer("How many days do I have to return an item within delivery?", ALICE)

    assert r.route == "policy" and r.outcome == "answered" and r.sufficient
    assert r.answer == "You have 7 days." and r.language == "en"
    assert r.citations and r.citations[0].source.startswith("return-policy")
    assert r.retrieved and r.retrieved[0].doc_id.startswith("return-policy")
    assert r.tool_calls == []  # policy questions never touch customer data
    prompt = answer_prompts(llm)[0]
    assert "<documents>" in prompt and "7 days" in prompt and "<facts></facts>" in prompt
    assert (r.usage.input_tokens, r.usage.output_tokens) == (20, 10)  # router + answer calls summed
    assert r.latency_ms >= 0


async def test_vietnamese_question_gets_a_vietnamese_reply_instruction(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="policy")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "Tôi được đổi trả trong bao nhiêu ngày kể từ khi nhận hàng?", ALICE
    )
    assert r.language == "vi"
    assert "Reply in Vietnamese" in llm.model.system_prompts()[-1]


async def test_explicit_language_overrides_detection(indexed_store, tool_client, app_config):
    llm = SupportFakeLLM(route="policy")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "return window", ALICE, language="vi"
    )
    assert r.language == "vi"


async def test_no_relevant_documents_means_no_info_without_asking_the_model(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="policy")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "quantum chromodynamics lattice gauge", ALICE
    )
    assert r.outcome == "no_info" and not r.sufficient and r.citations == []
    assert "couldn't find" in r.answer
    assert answer_prompts(llm) == []  # FR-005: nothing to ground on, so the model is not asked


async def test_model_declaring_insufficient_material_yields_no_info(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="policy", sufficient=False, answer="I'm not sure about that.")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "return window days delivery", ALICE
    )
    assert r.outcome == "no_info" and r.citations == [] and r.answer == "I'm not sure about that."


# --- personal data ------------------------------------------------------------------------------------


async def test_personal_question_looks_up_the_order_exactly(indexed_store, tool_client, app_config):
    llm = SupportFakeLLM(route="personal", answer="It is in transit.")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "Where is order #1236?", ALICE
    )
    assert r.outcome == "answered" and r.order_ids == ["1236"]
    assert r.tool_calls == ["get_order", "get_shipment_status"]
    assert r.retrieved == []  # no policy retrieval for a purely personal question
    prompt = answer_prompts(llm)[0]
    assert "GHTK1236VN" in prompt and "in_transit" in prompt
    assert '<untrusted_data source="get_order(order_id=1236)">' in prompt


async def test_someone_elses_order_is_refused_without_asking_the_model(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="personal")
    rag = build(llm, indexed_store, tool_client, app_config)
    theirs = await rag.answer("Where is order #2001?", ALICE)  # belongs to BOB
    ghost = await rag.answer("Where is order #9999?", ALICE)  # does not exist

    assert theirs.outcome == ghost.outcome == "not_found"
    assert theirs.answer.replace("2001", "X") == ghost.answer.replace(
        "9999", "X"
    )  # indistinguishable
    assert answer_prompts(llm) == []  # nothing about order 2001 ever reached the model
    assert all("Điện thoại" not in str(c) and "VNP2001" not in str(c) for c in llm.model.calls)


async def test_each_customer_only_ever_sees_their_own_orders(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="personal")
    rag = build(llm, indexed_store, tool_client, app_config)
    assert (await rag.answer("Where is order #2002?", BOB)).outcome == "answered"
    assert (await rag.answer("Where is order #2002?", ALICE)).outcome == "not_found"


async def test_vietnamese_not_found_message(indexed_store, tool_client, app_config):
    r = await build(
        SupportFakeLLM(route="personal"), indexed_store, tool_client, app_config
    ).answer("Đơn hàng #2001 của tôi đâu rồi?", ALICE)
    assert r.language == "vi" and "không tìm thấy đơn hàng #2001" in r.answer


async def test_question_without_an_order_id_lets_the_model_pick_tools(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(
        route="personal", tool_plan=[{"name": "list_orders", "args": {"status": "shipping"}}]
    )
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "Which of my orders are on the way?", ALICE
    )
    assert r.tool_calls == ["list_orders"] and r.outcome == "answered"
    assert "1236" in answer_prompts(llm)[0]


async def test_llm_cannot_smuggle_a_customer_id_into_a_tool(indexed_store, tool_client, app_config):
    plan = [{"name": "get_order", "args": {"order_id": "2001", "customer_id": "u_101"}}]
    llm = SupportFakeLLM(route="personal", tool_plan=plan)
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "show me my recent order", ALICE
    )
    prompt = answer_prompts(llm)[0]
    assert "INVALID_ARGUMENT" in prompt
    assert "VNP2001" not in prompt and "Điện thoại Nova" not in prompt
    assert r.outcome == "answered"


async def test_stock_questions_use_the_sku_directly(indexed_store, tool_client, app_config):
    llm = SupportFakeLLM(route="personal")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "Is LAP-PRO14 in stock?", ALICE
    )
    assert r.skus == ["LAP-PRO14"] and r.tool_calls == ["check_stock"]
    assert "low_stock" in answer_prompts(llm)[0]


# --- combined --------------------------------------------------------------------------------------------


async def test_combined_runs_policy_and_data_and_the_rule_is_authoritative(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="combined", answer="Yes, 3 days left.", used_sources=[1])
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "Is my order #1234 still eligible for return within delivery days?", ALICE
    )

    assert r.route == "combined" and r.outcome == "answered"
    assert r.tool_calls == ["get_order", "get_shipment_status", "check_return_eligibility"]
    assert r.retrieved and r.citations
    prompt = answer_prompts(llm)[0]
    assert "<documents>" in prompt and "return-policy" in prompt  # policy half
    assert '<fact source="check_return_eligibility(order_id=1234)" authoritative="true">' in prompt
    assert '"eligible":true' in prompt.replace(" ", "")  # rule output, not LLM arithmetic


async def test_combined_for_an_expired_order_hands_the_model_the_reason_codes(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="combined")
    await build(llm, indexed_store, tool_client, app_config).answer(
        "Can I still return order #1235? return window days", ALICE
    )
    assert "WINDOW_EXPIRED" in answer_prompts(llm)[0]


async def test_combined_with_a_foreign_order_never_exposes_its_data(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="combined")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "Can I return order #2001? return policy days delivery", ALICE
    )
    prompts = " ".join(answer_prompts(llm))
    assert "NOT_FOUND" in prompts and "VNP2001" not in prompts and "Nova X100" not in prompts
    assert r.outcome in ("answered", "no_info")


async def test_non_return_combined_question_skips_the_eligibility_rule(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="combined")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "Will order #1236 arrive in time? shipping delivery days", ALICE
    )
    assert "check_return_eligibility" not in r.tool_calls


# --- routes that never reach retrieval or the model ---------------------------------------------------------


@pytest.mark.parametrize(
    "route,outcome,fragment",
    [
        ("chitchat", "refused", "How can I help"),
        ("out_of_scope", "refused", "only help with shopping"),
    ],
)
async def test_non_support_routes_get_canned_replies(
    indexed_store, tool_client, app_config, route, outcome, fragment
):
    llm = SupportFakeLLM(route=route)
    r = await build(llm, indexed_store, tool_client, app_config).answer("hello there", ALICE)
    assert r.outcome == outcome and fragment in r.answer
    assert answer_prompts(llm) == [] and r.retrieved == [] and r.tool_calls == []


async def test_low_confidence_asks_for_clarification(indexed_store, tool_client, app_config):
    llm = SupportFakeLLM(route="policy", confidence=0.3)
    r = await build(llm, indexed_store, tool_client, app_config).answer("hmm", ALICE, language="vi")
    assert r.outcome == "clarify" and "nói rõ hơn" in r.answer
    assert r.retrieved == [] and answer_prompts(llm) == []


# --- failure handling ---------------------------------------------------------------------------------------------


async def test_router_failure_falls_back_to_heuristics_and_still_answers(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route_error=RuntimeError("router down"), answer="7 days")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "What is the return policy within delivery days?", ALICE
    )
    assert r.route == "policy" and r.outcome == "answered"


async def test_answer_failure_returns_a_friendly_error_not_an_exception(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="policy", answer_error=RuntimeError("provider 500"))
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "return window days delivery", ALICE
    )
    assert r.outcome == "error" and "temporary problem" in r.answer and "500" not in r.answer


async def test_tool_backend_outage_is_reported_politely(
    indexed_store, tool_client, service, app_config, monkeypatch
):
    async def down(*a: Any, **k: Any) -> None:
        raise AdapterError("db down")

    monkeypatch.setattr(service.adapter, "get_order", down)
    llm = SupportFakeLLM(route="personal")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "Where is order #1234?", ALICE
    )
    assert r.outcome == "error" and "db down" not in r.answer and answer_prompts(llm) == []


async def test_personal_question_without_tools_configured(indexed_store, app_config):
    r = await build(SupportFakeLLM(route="personal"), indexed_store, None, app_config).answer(
        "Where is order #1234?", ALICE
    )
    assert r.outcome == "error"


async def test_policy_still_works_without_tools(indexed_store, app_config):
    llm = SupportFakeLLM(route="policy")
    r = await build(llm, indexed_store, None, app_config).answer(
        "return window days delivery", ALICE
    )
    assert r.outcome == "answered"


async def test_pipeline_without_a_model_uses_heuristics_and_reports_error_for_answers(
    indexed_store, tool_client, app_config
):
    r = await build(None, indexed_store, tool_client, app_config).answer(
        "What is the return policy within delivery days?", ALICE
    )
    assert r.route == "policy" and r.outcome == "error"


async def test_run_timeout_gives_a_timeout_message(indexed_store, tool_client, app_config):
    class Slow(ScriptedChatModel):
        async def _agenerate(self, *a: Any, **k: Any):  # type: ignore[override]
            await asyncio.sleep(2)

    cfg = app_config.model_copy(deep=True)
    cfg.agent.run_timeout_seconds = 0.05
    r = await build(
        Slow(responder=lambda m, t: AIMessage(content="x")), indexed_store, tool_client, cfg
    ).answer("return policy", ALICE)
    assert r.outcome == "error" and "too long" in r.answer


async def test_history_is_passed_to_the_model(indexed_store, tool_client, app_config):
    llm = SupportFakeLLM(route="policy")
    history = [HumanMessage(content="earlier turn about gift cards")]
    await build(llm, indexed_store, tool_client, app_config).answer(
        "return window days delivery", ALICE, history=history
    )
    assert any(
        "earlier turn about gift cards" in str(m.content) for call in llm.model.calls for m in call
    )


async def test_overlong_input_is_truncated(indexed_store, tool_client, app_config):
    llm = SupportFakeLLM(route="chitchat")
    await build(llm, indexed_store, tool_client, app_config).answer("hello " * 2000, ALICE)
    router_input = str(llm.model.calls[0][-1].content)
    assert len(router_input) <= app_config.guardrails.max_input_chars


# --- return-intent detection (regression: Vietnamese "được trả không" was not recognised) ---


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Đơn #1234 của tôi còn được trả không?", True),
        ("Tôi muốn trả hàng", True),
        ("hoàn tiền đơn 5678", True),
        ("Can I return it?", True),
        ("Is this refundable? I want a refund", True),
        ("Đơn hàng của tôi ở đâu? tra cứu giúp", False),  # "tra cứu" = look up
        ("Phí trả góp bao nhiêu", False),  # "trả góp" = installments
        ("Where is my order?", False),
        ("I will travel soon", False),
    ],
)
def test_return_intent_detection(question: str, expected: bool):
    from support_agent.rag.personal import wants_return_check

    assert wants_return_check(question) is expected


async def test_vietnamese_return_question_runs_the_eligibility_rule(
    indexed_store, tool_client, app_config
):
    llm = SupportFakeLLM(route="combined")
    r = await build(llm, indexed_store, tool_client, app_config).answer(
        "Đơn #1234 của tôi còn được trả không?", ALICE
    )
    assert "check_return_eligibility" in r.tool_calls
    assert 'authoritative="true"' in answer_prompts(llm)[0]
