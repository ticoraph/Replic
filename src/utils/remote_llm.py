"""Configuration minimale du LLM distant, sélectionné par ``LLM_PROVIDER``."""

from dataclasses import dataclass

from openai import OpenAI

from utils.config import settings


@dataclass(frozen=True)
class RemoteLLM:
    provider: str
    api_key: str
    model: str
    base_url: str | None

    def client(self, *, base_url: str | None = None) -> OpenAI:
        kwargs = {"api_key": self.api_key}
        selected_base_url = self.base_url if base_url is None else base_url
        if selected_base_url:
            kwargs["base_url"] = selected_base_url
        return OpenAI(**kwargs)

    @property
    def chat_options(self) -> dict[str, str]:
        # GPT-6 Luna accepte temperature et JSON mode via Chat Completions avec
        # l'effort de raisonnement désactivé. Gemini conserve sa requête actuelle.
        return {"reasoning_effort": "none"} if self.provider == "openai" else {}


def get_remote_llm() -> RemoteLLM:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return RemoteLLM(
            provider="openai",
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=None,
        )
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
    return RemoteLLM(
        provider="gemini",
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        base_url=settings.gemini_base_url,
    )
