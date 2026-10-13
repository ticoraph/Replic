"""Traduit les dialogues parsés (EN) en français via le LLM distant, scène par scène.

Le chatbot est exclusivement francophone : tout l'aval (build_pairs, augment,
chunks RAG du scénario) consomme la sortie de ce script.

- Traduction scène par scène : le modèle voit toute la scène → cohérence du contexte,
  tutoiement et registre argotique constants.
- Glossaire imposé des répliques cultes de la VF (Royale Cheese, Ézéchiel 25:17…).
- Reprise sur erreur : sortie append par scène, les scènes déjà traduites sont
  sautées (utile avec les rate limits du free tier Gemini — relancer suffit).

Entrée : data/interim/dialogues.jsonl (produit par parse_screenplay.py)
Sortie : data/interim/dialogues_fr.jsonl (même schéma, text=FR + text_en conservé,
         scene=heading traduit + scene_en conservé)

Usage : uv run python src/preprocessing/translate_dialogues.py
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import DATA_INTERIM  # noqa: E402
from utils.remote_llm import get_remote_llm  # noqa: E402

IN_PATH = DATA_INTERIM / "dialogues.jsonl"
OUT_PATH = DATA_INTERIM / "dialogues_fr.jsonl"

SPEAKER_MAP = {
    "YOUNG MAN": "JEUNE HOMME",
    "YOUNG WOMAN": "JEUNE FEMME",
    "SPORTSCASTER": "COMMENTATEUR SPORTIF",
}


def frenchify_scene(scene: str) -> str:
    if scene.startswith("INT."):
        return "INTERIEUR" + scene[len("INT."):]
    if scene.startswith("EXT."):
        return "EXTERIEUR" + scene[len("EXT."):]
    return scene

GLOSSARY = """- "Royale with Cheese" → "Royale Cheese" (VF culte)
- "Quarter Pounder" → "Quarter Pounder" (inchangé, Vincent explique justement le nom US)
- Ézéchiel 25:17 → la version VF : "La marche du vertueux est semée d'obstacles…
  Et tu connaîtras que mon nom est l'Éternel quand sur toi s'abattra ma vengeance."
- "The Wolf" → "Le Loup" ; "I solve problems" → "Je résous les problèmes"
- "foot massage" → "massage de pieds" ; "the miracle" → "le miracle"
- Jurons : registre argotique VF naturel (putain, merde, enfoiré, bordel) — jamais
  de traduction littérale scolaire."""

TRANSLATE_PROMPT = """Tu traduis le script de Pulp Fiction (1994) en français, dans le style
de la VF du film : registre oral, argot des années 90, tutoiement entre les personnages,
rythme des dialogues de Tarantino.

Glossaire imposé :
{glossary}

Scène : {scene}

Répliques (JSON) :
{lines}

Réponds UNIQUEMENT en JSON strict :
{{"scene_fr": "<heading traduit, garder le préfixe INT./EXT. et le format>",
"lines_fr": ["<traduction réplique 1>", "<traduction réplique 2>", ...]}}

Contrainte : lines_fr doit contenir exactement {n} éléments, dans le même ordre."""


def translated_scene_ids() -> set[int]:
    if not OUT_PATH.exists():
        return set()
    return {
        json.loads(line_)["scene_id"]
        for line_ in OUT_PATH.read_text(encoding="utf-8").splitlines()
    }


def translate_scene(
    client: OpenAI,
    scene_lines: list[dict],
    *,
    model: str,
    request_options: dict | None = None,
) -> list[dict]:
    texts = [d["text"] for d in scene_lines]
    prompt = TRANSLATE_PROMPT.format(
        glossary=GLOSSARY,
        scene=scene_lines[0]["scene"],
        lines=json.dumps(
            [{"speaker": d["speaker"], "text": d["text"]} for d in scene_lines],
            ensure_ascii=False,
        ),
        n=len(texts),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        response_format={"type": "json_object"},
        **(request_options or {}),
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    lines_fr = data["lines_fr"]
    if len(lines_fr) != len(texts):
        raise ValueError(f"attendu {len(texts)} répliques, reçu {len(lines_fr)}")
    return [
        {
            **d,
            "scene": frenchify_scene(data.get("scene_fr", d["scene"])),
            "speaker": SPEAKER_MAP.get(d["speaker"], d["speaker"]),
            "scene_en": d["scene"],
            "text": fr.strip(),
            "text_en": d["text"],
        }
        for d, fr in zip(scene_lines, lines_fr)
    ]


def main() -> None:
    try:
        remote_llm = get_remote_llm()
    except RuntimeError as error:
        sys.exit(str(error))
    client = remote_llm.client()

    dialogues = [json.loads(line_) for line_ in IN_PATH.read_text(encoding="utf-8").splitlines()]
    scenes: dict[int, list[dict]] = {}
    for d in dialogues:
        scenes.setdefault(d["scene_id"], []).append(d)

    done = translated_scene_ids()
    todo = [sid for sid in sorted(scenes) if sid not in done]
    print(f"{len(todo)} scènes à traduire ({len(done)} déjà faites)")

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i, sid in enumerate(todo, 1):
            try:
                rows = translate_scene(
                    client,
                    scenes[sid],
                    model=remote_llm.model,
                    request_options=remote_llm.chat_options,
                )
            except Exception as e:  # noqa: BLE001 — reprise possible en relançant
                print(f"  scène {sid} : ÉCHEC ({e}) — relancer le script pour reprendre")
                continue
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {i}/{len(todo)} scène {sid} ({len(rows)} répliques)")

    total = len(translated_scene_ids())
    print(f"{total}/{len(scenes)} scènes traduites → {OUT_PATH}")


if __name__ == "__main__":
    main()
