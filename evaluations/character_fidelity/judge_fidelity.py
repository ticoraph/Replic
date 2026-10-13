"""Évaluation de la fidélité de personnage par le LLM distant sélectionné.

Pour chaque question de qa_set.jsonl, interroge le chatbot (pipeline RAG + endpoint
vLLM) et note :
- in_character (1-5) : ton, tics de langage, personnalité
- factuality  (1-5) : cohérence avec les faits attendus (expected_facts)
- language_ok (bool) : réponse en français (le chatbot est exclusivement francophone)

Avec --baseline, interroge le modèle de base (sans adapter LoRA) pour mesurer
l'apport du fine-tuning. Résultats logués sur W&B.

Usage : uv run --group rag --group eval python evaluations/character_fidelity/judge_fidelity.py [--baseline]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from rag.prompting import build_messages  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402
from utils.characters import get_character  # noqa: E402
from utils.config import settings  # noqa: E402
from utils.remote_llm import get_remote_llm  # noqa: E402

QA_PATH = Path(__file__).parent / "qa_set.jsonl"
JUDGE_PROMPT = """Tu évalues un chatbot incarnant {name} (Pulp Fiction, 1994).

Question posée : {question}
Réponse du chatbot : {answer}
Faits attendus dans la réponse : {facts}
Traits du personnage : {traits}

Note la réponse et réponds UNIQUEMENT en JSON :
{{"in_character": <1-5, la réponse sonne-t-elle comme {name} ? ton, tics, personnalité>,
"factuality": <1-5, cohérence avec les faits attendus et l'univers du film>,
"language_ok": <true si la réponse est intégralement en français>,
"comment": "<une phrase>"}}"""


def ask_chatbot(question: str, character: str, model_name: str) -> str:
    from openai import OpenAI

    chunks = retrieve(question, top_k=4)
    messages = build_messages(character, question, chunks)
    client = OpenAI(
        base_url=settings.modal_inference_url, api_key=settings.modal_inference_api_key
    )
    resp = client.chat.completions.create(
        model=model_name, messages=messages, temperature=0.7, max_tokens=400
    )
    return resp.choices[0].message.content or ""


def judge(row: dict, answer: str):
    char = get_character(row["character"])
    remote_llm = get_remote_llm()
    client = remote_llm.client()
    resp = client.chat.completions.create(
        model=remote_llm.model,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    name=char.display_name,
                    question=row["question"],
                    answer=answer,
                    facts=", ".join(row["expected_facts"]),
                    traits="; ".join(char.traits),
                ),
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
        **remote_llm.chat_options,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline", action="store_true",
        help="Interroge la base sans adapter (nom de modèle vLLM = base) au lieu du LoRA",
    )
    args = ap.parse_args()
    model_name = settings.base_model if args.baseline else "pulpfiction"

    rows = [json.loads(line_) for line_ in QA_PATH.read_text(encoding="utf-8").splitlines()]
    results = []
    for i, row in enumerate(rows, 1):
        answer = ask_chatbot(row["question"], row["character"], model_name)
        scores = judge(row, answer)
        results.append({**row, "answer": answer, **scores})
        print(f"  {i}/{len(rows)} [{row['character']}] in_char={scores.get('in_character')} fact={scores.get('factuality')}")

    summary = {
        "model": model_name,
        "in_character_mean": statistics.mean(r["in_character"] for r in results),
        "factuality_mean": statistics.mean(r["factuality"] for r in results),
        "language_ok_rate": sum(r["language_ok"] for r in results) / len(results),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out = Path(__file__).parent / f"results_{'baseline' if args.baseline else 'lora'}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import wandb

    run = wandb.init(project=settings.wandb_project, job_type="eval-fidelity", config=summary)
    run.log(summary)
    run.finish()


if __name__ == "__main__":
    main()
