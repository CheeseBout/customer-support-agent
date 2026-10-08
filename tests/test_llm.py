from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from support_agent.core.settings import OPENROUTER_BASE_URL, Settings
from support_agent.llm.capabilities import CapabilityFailure, check_capabilities, enforce
from support_agent.llm.factory import MissingCredentials, get_chat_model, get_embeddings
from support_agent.llm.structured import extract_json, message_text, structured_invoke
from support_agent.llm.usage import UsageTracker
from tests.conftest import ROOT
from tests.fakes import ScriptedChatModel, ai_json


def make_settings(provider: str, **kw) -> Settings:
    return Settings(
        _env_file=None, llm_provider=provider, app_config_path=ROOT / "config" / "app.yaml", **kw
    )


# --- provider factory (PLAN Phase 0 acceptance: switching LLM_PROVIDER selects each provider) ---


def test_factory_builds_each_provider():
    openai = get_chat_model(settings=make_settings("openai", openai_api_key="k"))
    assert isinstance(openai, ChatOpenAI) and openai.model_name == "gpt-4o-mini"
    assert openai.openai_api_base is None

    anthropic = get_chat_model(settings=make_settings("anthropic", anthropic_api_key="k"))
    assert isinstance(anthropic, ChatAnthropic) and anthropic.model == "claude-haiku-4-5"

    gemini = get_chat_model(settings=make_settings("gemini", google_api_key="k"))
    assert isinstance(gemini, ChatGoogleGenerativeAI) and gemini.model == "gemini-3.5-flash-lite"

    router = get_chat_model(settings=make_settings("openrouter", openrouter_api_key="k"))
    assert isinstance(router, ChatOpenAI)
    assert router.openai_api_base == OPENROUTER_BASE_URL
    assert router.model_name.endswith(":free")


def test_factory_applies_generation_config_and_model_override():
    m = get_chat_model(settings=make_settings("openai", openai_api_key="k", llm_model="gpt-x"))
    assert m.model_name == "gpt-x"
    assert m.temperature == 0.2 and m.max_tokens == 1024


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "openrouter"])
def test_factory_reports_missing_key_with_env_name(provider: str):
    with pytest.raises(MissingCredentials, match="API_KEY"):
        get_chat_model(settings=make_settings(provider))


def test_embeddings_factory():
    from langchain_openai import OpenAIEmbeddings

    from support_agent.rag.embeddings import FastEmbedEmbeddings

    local = get_embeddings(make_settings("openai"))
    assert (
        isinstance(local, FastEmbedEmbeddings)
        and local.model_name == "intfloat/multilingual-e5-large"
    )
    oa = get_embeddings(make_settings("openai", embedding_provider="openai", openai_api_key="k"))
    assert isinstance(oa, OpenAIEmbeddings)
    with pytest.raises(MissingCredentials):
        get_embeddings(make_settings("openai", embedding_provider="gemini"))


# --- capability check --------------------------------------------------------------------------


class _Probe(BaseModel):
    city: str
    temperature_c: int


def _full_model() -> ScriptedChatModel:
    def respond(messages, tools):
        text = str(messages[-1].content)
        if tools:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_probe_temperature", "args": {"city": "Hanoi"}, "id": "1"}
                ],
            )
        if "21 degrees" in text:
            return ai_json({"city": "Hanoi", "temperature_c": 21})
        return AIMessage(content="pong")

    return ScriptedChatModel(responder=respond)


async def test_capabilities_ok():
    report = await check_capabilities(_full_model())
    assert report.status == "ok" and report.tool_calling and report.structured_output
    enforce(report, strict=True)


async def test_capabilities_degraded_without_tool_calling(monkeypatch: pytest.MonkeyPatch):
    model = _full_model()

    def no_tools(self, tools, **kw):
        raise NotImplementedError("no tool support")

    monkeypatch.setattr(ScriptedChatModel, "bind_tools", no_tools)  # undone automatically
    report = await check_capabilities(model)
    assert report.status == "degraded" and not report.tool_calling
    assert any("tool calling" in w for w in report.warnings)
    enforce(report, strict=False)  # tolerated
    with pytest.raises(CapabilityFailure):
        enforce(report, strict=True)  # STRICT_CAPABILITY_CHECK=true refuses to start


async def test_capabilities_failed_when_model_errors():
    def boom(messages, tools):
        raise RuntimeError("401 unauthorized")

    report = await check_capabilities(ScriptedChatModel(responder=boom))
    assert report.status == "failed" and "401" in report.warnings[0]
    with pytest.raises(CapabilityFailure):
        enforce(report, strict=False)


# --- structured output + fallback ----------------------------------------------------------------


def test_extract_json_handles_fences_prose_and_nesting():
    assert extract_json('Sure! ```json\n{"a": {"b": [1, 2]}}\n``` done') == {"a": {"b": [1, 2]}}
    assert extract_json('prefix {"a": "}"} suffix') == {"a": "}"}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json('{"unterminated": 1')


def test_message_text_flattens_content_blocks():
    msg = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "tool_use"}, "b"])
    assert message_text(msg) == "ab"


class _Out(BaseModel):
    route: str
    confidence: float


async def test_structured_invoke_native_path():
    model = ScriptedChatModel(
        responder=lambda m, t: ai_json({"route": "policy", "confidence": 0.9})
    )
    out = await structured_invoke(model, _Out, [HumanMessage(content="hi")])
    assert out == _Out(route="policy", confidence=0.9)


async def test_structured_invoke_falls_back_to_json_prompt():
    seen: list[str] = []

    class NoStructured(ScriptedChatModel):
        def with_structured_output(self, *a, **k):  # type: ignore[override]
            raise NotImplementedError

    def respond(messages, tools):
        seen.append(str(messages[-1].content))
        return AIMessage(content='Here: {"route": "personal", "confidence": 0.7}')

    out = await structured_invoke(
        NoStructured(responder=respond), _Out, [HumanMessage(content="hi")]
    )
    assert out.route == "personal"
    assert "JSON schema" in seen[0]


async def test_usage_tracker_sums_provider_usage():
    tracker = UsageTracker()
    model = ScriptedChatModel(
        responder=lambda m, t: ai_json({"route": "x", "confidence": 1}, tokens=(7, 3))
    )
    await model.ainvoke([HumanMessage(content="a")], config={"callbacks": [tracker]})
    await model.ainvoke([HumanMessage(content="b")], config={"callbacks": [tracker]})
    assert (tracker.usage.input_tokens, tracker.usage.output_tokens) == (14, 6)


# --- client-side rate limiting -------------------------------------------------------------------


def test_every_model_in_a_process_shares_one_rate_limiter():
    from support_agent.llm.factory import rate_limiter_for

    s = make_settings("gemini", google_api_key="k", llm_requests_per_minute=12)
    first, second = get_chat_model(settings=s), get_chat_model(streaming=False, settings=s)
    assert first.rate_limiter is second.rate_limiter is rate_limiter_for(s)  # one budget in total
    assert get_chat_model(settings=make_settings("gemini", google_api_key="k")).rate_limiter is None


def test_the_limit_can_come_from_the_yaml_or_the_environment_setting():
    from support_agent.llm.factory import rate_limiter_for

    from_yaml = make_settings("openai", openai_api_key="k")
    from_yaml.app.llm.requests_per_minute = 30
    assert rate_limiter_for(from_yaml) is not None
    override = make_settings("openai", openai_api_key="k", llm_requests_per_minute=7)
    assert rate_limiter_for(override) is not rate_limiter_for(from_yaml)  # a different rate
    assert get_chat_model(settings=override).rate_limiter is rate_limiter_for(override)


async def test_a_rate_limited_model_spaces_its_calls_out():
    import time

    from langchain_core.rate_limiters import InMemoryRateLimiter

    limiter = InMemoryRateLimiter(
        requests_per_second=10, check_every_n_seconds=0.01, max_bucket_size=1
    )
    model = ScriptedChatModel(responder=lambda m, t: AIMessage(content="ok"), rate_limiter=limiter)
    started = time.perf_counter()
    for _ in range(4):
        await model.ainvoke([HumanMessage(content="hi")])
    assert time.perf_counter() - started >= 0.25  # 4 calls at 10/s cannot all be instant


@pytest.mark.parametrize(
    "provider,key",
    [("openai", "openai_api_key"), ("anthropic", "anthropic_api_key"),
     ("gemini", "google_api_key"), ("openrouter", "openrouter_api_key")],
)  # fmt: skip
def test_provider_sdk_retries_are_bounded_by_config(provider: str, key: str):
    s = make_settings(provider, **{key: "k"})
    assert get_chat_model(settings=s).max_retries == 2
    s.app.llm.max_retries = 0
    assert get_chat_model(settings=s).max_retries == 0


@pytest.mark.parametrize(
    "model,sent",
    [
        ("claude-haiku-5-5", None),
        ("claude-sonnet-5-5", None),
        ("claude-opus-5-5", None),
        ("claude-opus-4-8", None),
        ("claude-opus-4-7", None),
        ("claude-fable-5-1", None),
        ("claude-haiku-4-5", 0.2),
        ("claude-haiku-4-5-20251001", 0.2),
        ("claude-sonnet-4-6", 0.2),
        ("claude-opus-4-6", 0.2),
    ],
)
def test_temperature_is_left_out_for_claude_models_that_reject_it(model: str, sent: float | None):
    from support_agent.llm.factory import anthropic_temperature

    assert anthropic_temperature(model, 0.2) == sent


def test_a_haiku_55_chat_model_is_built_without_a_temperature():
    settings = make_settings("anthropic", anthropic_api_key="k", llm_model="claude-haiku-5-5")
    model = get_chat_model(settings=settings)
    assert isinstance(model, ChatAnthropic) and model.temperature is None
