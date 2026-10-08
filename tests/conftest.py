from __future__ import annotations

from pathlib import Path

import pytest

from support_agent.core.settings import AppConfig, Settings, load_app_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    return load_app_config(ROOT / "config" / "app.yaml")


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, app_config_path=ROOT / "config" / "app.yaml")
