"""Assemblage du prompt d'inférence : persona + contexte RAG + consignes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.retrieve import RetrievedChunk  # noqa: E402
from utils.characters import get_character  # noqa: E402

CONTEXT_TEMPLATE = """{persona}

## Contexte documentaire (analyses et script du film) :
{context}

Consignes : reste strictement dans le personnage. Appuie tes réponses factuelles sur le
contexte ci-dessus quand il est pertinent, mais reformule toujours avec ta voix. Si le
contexte ne couvre pas la question, réponds avec ta personnalité sans inventer de faits
précis. Réponds toujours en français."""


def build_system_prompt(character_key: str, chunks: list[RetrievedChunk]) -> str:
    char = get_character(character_key)
    context = "\n\n".join(
        f"[{c.metadata['source']}] {c.content}" for c in chunks
    ) or "(aucun contexte trouvé)"
    return CONTEXT_TEMPLATE.format(persona=char.system_prompt, context=context)


def build_messages(
    character_key: str,
    user_message: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": build_system_prompt(character_key, chunks)}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_message})
    return messages
