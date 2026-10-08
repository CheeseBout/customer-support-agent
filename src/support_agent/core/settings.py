"""Settings: secrets/endpoints from the environment (.env), tunables from config/app.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "anthropic", "gemini", "openrouter"]
EmbeddingProvider = Literal["local", "openai", "gemini"]
DbType = Literal["postgres", "mysql", "mongodb", "sqlite"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMConfig(BaseModel):
    temperature: float = 0.2
    max_output_tokens: int = 1024
    timeout_seconds: float = 30
    # Client-side pacing shared by every model call (router, agent, judge). Providers answer
    # 429 above their tier's limit and their SDKs retry, which counts against the limit again.
    requests_per_minute: float | None = None
    # Provider SDK retries on 429/5xx. Each retry can count against the quota again, so keep
    # it small; the agent adds its own bounded retries on top.
    max_retries: int = 2
    models: dict[str, str] = Field(default_factory=dict)


class EmbeddingsConfig(BaseModel):
    models: dict[str, str] = Field(default_factory=dict)
    cache_dir: Path = Path("./data/models")


class AgentConfig(BaseModel):
    max_steps: int = 12
    run_timeout_seconds: float = 60
    history_token_limit: int = 6000
    summarize_after_tokens: int = 4000
    parallel_subagents: bool = True
    max_tool_calls: int = 8  # tool calls allowed per question, across all steps
    tool_retries: int = 2  # extra attempts after UPSTREAM_ERROR / TIMEOUT (NFR-005)
    tool_backoff_seconds: float = 0.5  # doubled on every retry
    history_messages: int = 12  # earlier messages shown to the model (Phase 6 adds summaries)
    confirmation_ttl_minutes: int = 15  # how long a pending customer confirmation stays valid


class ChunkTokens(BaseModel):
    min: int = 400
    max: int = 800


class RetrievalConfig(BaseModel):
    collection: str = "policy_chunks"
    top_k: int = 6
    score_threshold: float = 0.35
    # Hits below `score_threshold - hit_floor_margin` are dropped even when the query passes.
    hit_floor_margin: float = 0.05
    rerank: bool = False
    hybrid: bool = True
    chunk_tokens: ChunkTokens = ChunkTokens()
    chunk_overlap_ratio: float = 0.12
    sparse_model: str = "Qdrant/bm25"


class RouterConfig(BaseModel):
    min_confidence: float = 0.5
    # Extra regexes (group 1 = the id) on top of the built-in order-id patterns.
    order_id_patterns: list[str] = Field(default_factory=list)
    # Overrides the built-in SKU pattern when set.
    sku_pattern: str | None = None


class KnowledgeConfig(BaseModel):
    dir: Path = Path("./knowledge")


class MappingConfig(BaseModel):
    path: Path = Path("./config/schema_mapping.yaml")


class DbConfig(BaseModel):
    max_rows: int = 50
    query_timeout_seconds: float = 5


class ReturnRule(BaseModel):
    window_days: int = 7
    window_basis: Literal["delivered_at", "created_at"] = "delivered_at"
    allowed_order_statuses: list[str] = Field(default_factory=lambda: ["delivered"])
    excluded_categories: list[str] = Field(default_factory=list)


class RefundRule(BaseModel):
    auto_review_max_amount: int = 500_000
    currency: str = "VND"


class WarrantyRule(BaseModel):
    default_months: int = 12
    by_category: dict[str, int] = Field(default_factory=dict)


class InventoryRule(BaseModel):
    show_exact_quantity: bool = False
    low_stock_threshold: int = 5


class OrderRule(BaseModel):
    """Limits on orders the agent may prepare (SPEC 9.3)."""

    currency: str = "VND"
    max_quantity_per_line: int = 20
    max_lines: int = 10
    payment_methods: list[str] = Field(
        default_factory=lambda: ["cod", "bank_transfer", "card", "momo", "zalopay", "vnpay"]
    )
    cod_max_total: int | None = (
        5_000_000  # cash on delivery is capped (payment policy); null = no cap
    )


class BusinessRules(BaseModel):
    model_config = {"populate_by_name": True}

    timezone: str = "Asia/Ho_Chi_Minh"
    return_: ReturnRule = Field(default_factory=ReturnRule, alias="return")
    refund: RefundRule = RefundRule()
    warranty: WarrantyRule = WarrantyRule()
    inventory: InventoryRule = InventoryRule()
    order: OrderRule = OrderRule()


class GuardrailsConfig(BaseModel):
    max_input_chars: int = 2000
    rate_limit_per_minute: int = 20
    injection_detection: bool = True
    pii_masking: bool = True


class MemoryConfig(BaseModel):
    long_term_enabled: bool = False
    long_term_ttl_days: int = 180
    session_retention_days: int = 90


class EvalThresholds(BaseModel):
    """Pass/fail gates (SPEC 15.2). Business correctness and safety must be perfect."""

    routing: float = 0.92
    trajectory: float = 0.85
    answer: float = 0.85
    business: float = 1.0
    safety: float = 1.0


class EvalsConfig(BaseModel):
    dataset: Path = Path("./evals/datasets/baseline.jsonl")
    report_dir: Path = Path("./evals/reports")
    thresholds: EvalThresholds = EvalThresholds()
    retrieval_ks: list[int] = Field(default_factory=lambda: [1, 3, 6])
    concurrency: int = 1


class AppConfig(BaseModel):
    """Mirror of config/app.yaml. Every section has defaults so a partial file is valid."""

    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    agent: AgentConfig = AgentConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    router: RouterConfig = RouterConfig()
    knowledge: KnowledgeConfig = KnowledgeConfig()
    mapping: MappingConfig = MappingConfig()
    db: DbConfig = DbConfig()
    business_rules: BusinessRules = BusinessRules()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    memory: MemoryConfig = MemoryConfig()
    evals: EvalsConfig = EvalsConfig()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_app_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read one config file; `extends: <file>` (relative to this file) layers it on a base."""
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"config file extends itself: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base = _read_app_yaml(path.parent / str(parent), (*seen, resolved))
    return _deep_merge(base, raw)


def load_app_config(path: Path) -> AppConfig:
    """Load config/app.yaml, or a file that starts from it with `extends: ../../config/app.yaml`."""
    if not path.exists():
        return AppConfig()
    return AppConfig.model_validate(_read_app_yaml(path))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Provider = "openai"
    llm_model: str | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    strict_capability_check: bool = False
    llm_requests_per_minute: float | None = None  # overrides llm.requests_per_minute

    # Embeddings
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str | None = None

    # Business DB (read-only account)
    business_db_type: DbType = "postgres"
    business_db_url: str | None = None
    mcp_principal_secret: SecretStr | None = None

    # Qdrant: server (QDRANT_URL) or embedded (QDRANT_PATH)
    qdrant_url: str | None = None
    qdrant_path: Path | None = Path("./data/qdrant")

    # Langfuse tracing is on only when both keys are present.
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Where drafts are written (SPEC 10). An account that may write ONLY the support_drafts table.
    drafts_db_url: str | None = None

    # Conversation checkpoints: sqlite:///path (default), postgresql://..., or memory.
    checkpoint_url: str | None = None

    # API authentication (SPEC 14.1). The shop's own system issues the tokens.
    jwt_algorithm: Literal["HS256", "RS256"] = "HS256"
    jwt_secret: SecretStr | None = None  # HS256
    jwt_jwks_url: str | None = None  # RS256
    jwt_audience: str | None = None
    jwt_issuer: str | None = None
    jwt_customer_claim: str = "sub"
    jwt_role_claim: str = "role"
    cors_origins: str = ""  # comma-separated origins; empty = CORS disabled

    # Per-session workspace artifacts and the session index (Phase 6/9).
    sessions_db_url: str = "sqlite:///./data/sessions.db"
    workspace_dir: Path = Path("./data/workspace")

    log_level: str = "INFO"

    app_config_path: Path = Path("./config/app.yaml")
    app: AppConfig = AppConfig()

    @model_validator(mode="after")
    def _load_yaml(self) -> Settings:
        self.app = load_app_config(self.app_config_path)
        return self

    # --- derived values --------------------------------------------------------------
    @property
    def chat_model_name(self) -> str:
        if self.llm_model:
            return self.llm_model
        try:
            return self.app.llm.models[self.llm_provider]
        except KeyError as exc:
            raise ValueError(
                f"No default model configured for provider {self.llm_provider!r}"
            ) from exc

    @property
    def embedding_model_name(self) -> str:
        if self.embedding_model:
            return self.embedding_model
        try:
            return self.app.embeddings.models[self.embedding_provider]
        except KeyError as exc:
            raise ValueError(
                f"No default embedding model for provider {self.embedding_provider!r}"
            ) from exc

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def api_key_for(self, provider: str) -> str | None:
        key = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.google_api_key,
            "openrouter": self.openrouter_api_key,
        }.get(provider)
        return key.get_secret_value() if key else None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
