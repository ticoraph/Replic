"""Évaluation RAGAS de la qualité du RAG (faithfulness, relevancy, precision/recall).

Génère (ou charge) un jeu de ~50 questions en français, interroge le pipeline RAG local +
l'endpoint d'inférence, puis note avec le LLM distant sélectionné. Les embeddings RAGAS
restent ceux de Gemini. Résultats logués sur W&B.

Prérequis : pgvector indexé (src/rag/index.py), GEMINI_API_KEY, endpoint Modal configuré.
Usage : uv run --group rag --group eval python evaluations/ragas/run_ragas.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from rag.prompting import build_messages  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402
from utils.config import settings  # noqa: E402
from utils.remote_llm import get_remote_llm  # noqa: E402

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
JUDGE_EMBEDDINGS = "gemini-embedding-001"


def generate_answer(question: str, character: str = "jules") -> tuple[str, list[str]]:
    from openai import OpenAI

    chunks = retrieve(question, top_k=4)
    messages = build_messages(character, question, chunks)
    client = OpenAI(
        base_url=settings.modal_inference_url, api_key=settings.modal_inference_api_key
    )
    resp = client.chat.completions.create(
        model="pulpfiction", messages=messages, temperature=0.3, max_tokens=400
    )
    return resp.choices[0].message.content or "", [c.content for c in chunks]


def main() -> None:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    if not QUESTIONS_PATH.exists():
        sys.exit(
            f"{QUESTIONS_PATH} manquant. Format JSONL : "
            '{"question": "...", "ground_truth": "...", "character": "jules"}'
        )
    rows = [json.loads(line_) for line_ in QUESTIONS_PATH.read_text().splitlines()]

    records = []
    for i, row in enumerate(rows, 1):
        answer, contexts = generate_answer(row["question"], row.get("character", "jules"))
        records.append(
            {
                "question": row["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": row.get("ground_truth", ""),
            }
        )
        print(f"  {i}/{len(rows)}")

    remote_llm = get_remote_llm()
    judge_kwargs = {
        "model": remote_llm.model,
        "api_key": remote_llm.api_key,
    }
    if remote_llm.base_url:
        judge_kwargs["base_url"] = remote_llm.base_url
    judge_kwargs.update(remote_llm.chat_options)
    judge = ChatOpenAI(**judge_kwargs)
    result = evaluate(
        Dataset.from_list(records),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=OpenAIEmbeddings(
            model=JUDGE_EMBEDDINGS,
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
        ),
    )
    print(result)

    import wandb

    run = wandb.init(project=settings.wandb_project, job_type="eval-ragas")
    run.log(dict(result))
    run.finish()


if __name__ == "__main__":
    main()
