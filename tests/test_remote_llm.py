import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.config import settings  # noqa: E402
from utils.remote_llm import get_remote_llm  # noqa: E402


def test_gemini_is_the_default_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-key")

    remote = get_remote_llm()

    assert remote.provider == "gemini"
    assert remote.model == settings.gemini_model
    assert remote.base_url == settings.gemini_base_url
    assert remote.chat_options == {}


def test_openai_can_be_selected(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "openai-key")

    remote = get_remote_llm()

    assert remote.provider == "openai"
    assert remote.model == "gpt-6-luna"
    assert remote.base_url is None
    assert remote.chat_options == {"reasoning_effort": "none"}


def test_selected_provider_requires_its_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_remote_llm()
