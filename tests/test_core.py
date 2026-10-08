from __future__ import annotations

import json
import logging
import time

import pytest

from support_agent.core.i18n import detect_language, resolve_language, t
from support_agent.core.logging import JsonFormatter, bind_context, new_request_id
from support_agent.core.principal import (
    InvalidPrincipal,
    Principal,
    hash_user_id,
    sign_principal,
    verify_principal,
)
from support_agent.core.settings import Settings
from support_agent.security.pii import mask_email, mask_phone, mask_text
from tests.conftest import ROOT

# --- settings ---------------------------------------------------------------------------------


def test_settings_loads_yaml_defaults(settings: Settings):
    assert settings.app.retrieval.top_k == 6
    assert settings.app.business_rules.return_.window_days == 7
    assert settings.app.business_rules.return_.excluded_categories == ["gift_card"]
    assert settings.app.business_rules.warranty.by_category == {"accessories": 6, "appliances": 24}


def test_settings_default_model_follows_provider():
    for provider, model in [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-haiku-4-5"),
        ("gemini", "gemini-3.5-flash-lite"),
    ]:
        s = Settings(
            _env_file=None, llm_provider=provider, app_config_path=ROOT / "config" / "app.yaml"
        )
        assert s.chat_model_name == model
    s = Settings(
        _env_file=None, llm_provider="openrouter", app_config_path=ROOT / "config" / "app.yaml"
    )
    assert s.chat_model_name.endswith(":free")


def test_settings_model_override_and_missing_yaml(tmp_path):
    s = Settings(_env_file=None, llm_model="my-model", app_config_path=tmp_path / "nope.yaml")
    assert s.chat_model_name == "my-model"
    assert s.app.retrieval.top_k == 6  # falls back to code defaults


def test_embedding_model_default_and_override(settings: Settings):
    assert settings.embedding_model_name == "intfloat/multilingual-e5-large"
    s = Settings(
        _env_file=None, embedding_provider="openai", app_config_path=ROOT / "config" / "app.yaml"
    )
    assert s.embedding_model_name == "text-embedding-3-small"


# --- i18n -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("How many days do I have to return an item?", "en"),
        ("Tôi được đổi trả trong bao nhiêu ngày?", "vi"),
        ("Đơn hàng #1234 của tôi đang ở đâu?", "vi"),
        ("don hang cua toi o dau", "vi"),  # no diacritics
        ("toi muon doi tra san pham", "vi"),
        ("Where is order #1234?", "en"),
        ("refund", "en"),
        ("", "en"),
        ("1234", "en"),
    ],
)
def test_detect_language(text: str, expected: str):
    assert detect_language(text) == expected


def test_resolve_language_respects_explicit_choice():
    assert resolve_language("vi", "Hello there") == "vi"
    assert resolve_language("en", "Xin chào") == "en"
    assert resolve_language("auto", "Xin chào bạn") == "vi"
    assert resolve_language(None, "Hello") == "en"


def test_message_catalog_has_both_languages():
    for key in ("no_info", "clarify", "out_of_scope", "chitchat", "upstream_error", "timeout"):
        assert t(key, "vi") != t(key, "en")
    assert "#1234" in t("order_not_found", "vi", order_id="#1234")
    assert "#1234" in t("order_not_found", "en", order_id="#1234")


# --- PII and logging -----------------------------------------------------------------------------


def test_mask_email_and_phone():
    assert mask_email("mail an.nguyen@example.com now") == "mail a***@example.com now"
    assert mask_phone("call 0901234567 please") == "call *******567 please"
    assert mask_phone("call +84 901 234 567") != "call +84 901 234 567"


def test_mask_text_hides_card_numbers_and_keeps_order_ids():
    masked = mask_text("card 4111111111111111 order #1234")
    assert "4111111111111111" not in masked
    assert masked.endswith("order #1234")
    assert masked.count("*") == 12


def test_json_log_has_request_context_and_masks_pii():
    bind_context(request_id="req-1", session_id="sess-1", user_hash=hash_user_id("u_100"))
    record = logging.LogRecord(
        "x", logging.INFO, __file__, 1, "email %s", ("a.b@example.com",), None
    )
    record.extra_field = "phone 0901234567"
    out = json.loads(JsonFormatter().format(record))
    assert out["request_id"] == "req-1" and out["session_id"] == "sess-1"
    assert out["user_id"] == hash_user_id("u_100") and "u_100" not in json.dumps(out)
    assert "a.b@example.com" not in out["message"]
    assert "0901234567" not in out["extra_field"]


def test_new_request_ids_are_unique():
    assert new_request_id() != new_request_id()


# --- principal signing -------------------------------------------------------------------------


def test_principal_roundtrip():
    token = sign_principal(Principal(user_id="u_1", role="staff"), b"k")
    assert verify_principal(token, b"k") == Principal(user_id="u_1", role="staff")


def test_principal_rejects_wrong_secret_tampering_and_garbage():
    token = sign_principal(Principal(user_id="u_1"), b"k")
    with pytest.raises(InvalidPrincipal):
        verify_principal(token, b"other")
    forged = {"body": token["body"].replace("u_1", "u_2"), "sig": token["sig"]}
    with pytest.raises(InvalidPrincipal):
        verify_principal(forged, b"k")
    for bad in (None, {}, {"body": 1, "sig": 2}, "x"):
        with pytest.raises(InvalidPrincipal):
            verify_principal(bad, b"k")


def test_principal_expires():
    token = sign_principal(Principal(user_id="u_1"), b"k", ttl_seconds=-1)
    with pytest.raises(InvalidPrincipal, match="expired"):
        verify_principal(token, b"k")
    assert time.time() > 0
