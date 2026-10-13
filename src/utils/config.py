"""Configuration centralisée du projet (chargée depuis .env)."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATASETS_DIR = REPO_ROOT / "datasets"
MODELS_DIR = REPO_ROOT / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql://pulpfiction:pulpfiction@localhost:5434/pulpfictionchatbot"

    hf_token: str = ""
    wandb_api_key: str = ""
    wandb_project: str = "pulpfictionchatbot"

    modal_inference_url: str = ""
    modal_inference_api_key: str = ""

    # LLM distant pour l'augmentation, la traduction et les évaluations.
    # Le chatbot principal reste le modèle Llama servi sur Modal.
    llm_provider: Literal["gemini", "openai"] = "gemini"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-3.5-flash-lite"
    openai_api_key: str = ""
    openai_model: str = "gpt-6-luna"

    api_key: str = "change-me"
    cors_origins: str = "http://localhost:5173"

    # Modèles
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    train_base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    adapter_repo: str = "TicoRaph/pulpfictionchatbot-llama31-8b-lora"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
