"""Tests des endpoints de l'API (sans Modal ni pgvector : chemins d'erreur et lecture)."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.main import app  # noqa: E402
from src.utils.config import settings  # noqa: E402

client = TestClient(app)
AUTH = {"X-API-Key": settings.api_key}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_characters():
    r = client.get("/characters")
    assert r.status_code == 200
    keys = {c["key"] for c in r.json()}
    assert {"jules", "vincent", "mia", "marsellus", "butch", "wolf"} <= keys


def test_chat_requires_api_key():
    r = client.post("/chat", json={"character": "jules", "message": "yo"})
    assert r.status_code == 401


def test_chat_unknown_character():
    r = client.post("/chat", headers=AUTH, json={"character": "zed", "message": "yo"})
    assert r.status_code == 404


def test_chat_without_inference_url_is_503():
    if settings.modal_inference_url:
        return  # environnement configuré : ce chemin ne s'applique pas
    r = client.post("/chat", headers=AUTH, json={"character": "jules", "message": "yo"})
    assert r.status_code == 503
