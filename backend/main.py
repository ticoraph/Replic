"""API FastAPI d'orchestration : RAG pgvector + appel de l'endpoint vLLM sur Modal.

Endpoints :
- GET  /health      — liveness
- GET  /characters  — personnages disponibles
- POST /chat        — SSE streaming ; body : {character, message, history?}

Auth : header X-API-Key (settings.api_key). CORS restreint (settings.cors_origins).

Lancement local : uv run --group rag uvicorn backend.main:app --reload --port 8080
"""

import json
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.prompting import build_messages  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402
from utils.characters import CHARACTERS  # noqa: E402
from utils.config import settings  # noqa: E402

app = FastAPI(title="PulpFiction Chatbot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def check_api_key(key: str = Security(_api_key_header)) -> None:
    if key != settings.api_key:
        raise HTTPException(status_code=401, detail="Clé API invalide")


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    character: str
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    top_k: int = Field(default=4, ge=0, le=10)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/characters")
def characters() -> list[dict]:
    return [
        {"key": c.key, "display_name": c.display_name, "traits": list(c.traits)}
        for c in CHARACTERS.values()
    ]


@app.post("/chat", dependencies=[Depends(check_api_key)])
async def chat(req: ChatRequest) -> StreamingResponse:
    if req.character not in CHARACTERS:
        raise HTTPException(status_code=404, detail=f"Personnage inconnu : {req.character}")
    if not settings.modal_inference_url:
        raise HTTPException(status_code=503, detail="Endpoint d'inférence non configuré")

    chunks = retrieve(req.message, top_k=req.top_k, character=req.character) if req.top_k else []
    messages = build_messages(
        req.character,
        req.message,
        chunks,
        history=[t.model_dump() for t in req.history],
    )

    client = AsyncOpenAI(
        base_url=settings.modal_inference_url,
        api_key=settings.modal_inference_api_key,
        timeout=180,  # cold start Modal possible (~1-2 min)
    )

    async def stream():
        sources = [
            {"source": c.metadata["source"], "title": c.metadata["title"], "score": c.score}
            for c in chunks
        ]
        yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"
        response = await client.chat.completions.create(
            model="pulpfiction",  # adapter LoRA servi par vLLM
            messages=messages,
            stream=True,
            temperature=0.8,
            max_tokens=512,
        )
        async for part in response:
            delta = part.choices[0].delta.content or ""
            if delta:
                yield f"data: {json.dumps({'token': delta}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
