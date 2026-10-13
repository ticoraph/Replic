"""Parse le script original de Pulp Fiction (format screenplay) en dialogues structurés.

Le fichier data/raw/script-pulp-fiction-original-en.md conserve la mise en page
screenplay par indentation :
- ~15 espaces : action / scene headings (INT./EXT. ...)
- ~25 espaces : répliques
- ~30+ espaces : nom du personnage (tout en majuscules), parenthéticals centrés

Sortie : data/interim/dialogues.jsonl — une ligne par réplique :
    {"scene_id": int, "scene": str, "speaker": str, "text": str, "line_idx": int}

Usage : uv run python src/preprocessing/parse_screenplay.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import DATA_INTERIM, DATA_RAW  # noqa: E402

SCRIPT_PATH = DATA_RAW / "script-pulp-fiction-original-en.md"
OUT_PATH = DATA_INTERIM / "dialogues.jsonl"

SCENE_RE = re.compile(r"^\s*(INT\.|EXT\.|INT/EXT|EXT/INT)", re.IGNORECASE)
# Nom de personnage : fortement indenté, tout en majuscules, court,
# suffixe optionnel (V.O.), (O.S.), (CONT'D)...
CUE_RE = re.compile(r"^ {30,}([A-Z][A-Z0-9 .,'&#\-]{0,35}?)(?:\s*\((?:V\.O\.|O\.S\.|CONT'D|CONT|INTO PHONE|ON PHONE)\.?\))?\s*$")
PAREN_RE = re.compile(r"^\s*\(.*\)?\s*$")


def clean(text: str) -> str:
    text = text.replace("�", "—").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse(script_text: str) -> list[dict]:
    dialogues: list[dict] = []
    scene_id, scene = 0, "OPENING"
    speaker: str | None = None
    buf: list[str] = []
    line_idx = 0

    def flush() -> None:
        nonlocal speaker, buf, line_idx
        if speaker and buf:
            text = clean(" ".join(buf))
            if text:
                line_idx += 1
                dialogues.append(
                    {
                        "scene_id": scene_id,
                        "scene": scene,
                        "speaker": speaker,
                        "text": text,
                        "line_idx": line_idx,
                    }
                )
        speaker, buf = None, []

    for raw in script_text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.strip().startswith("```"):
            continue

        if SCENE_RE.match(line) and indent(line) < 25:
            flush()
            scene_id += 1
            scene = clean(line)
            continue

        m = CUE_RE.match(line)
        if m:
            flush()
            speaker = clean(m.group(1))
            continue

        if speaker is not None:
            ind = indent(line)
            if PAREN_RE.match(line) and ind >= 20:
                continue  # didascalie dans le dialogue
            if 20 <= ind < 34:
                buf.append(line.strip())
                continue
            # indentation d'action (~15) → fin de la réplique
            flush()

    flush()
    return dialogues


def main() -> None:
    text = SCRIPT_PATH.read_text(encoding="utf-8", errors="replace")
    dialogues = parse(text)
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    from collections import Counter

    counts = Counter(d["speaker"] for d in dialogues)
    print(f"{len(dialogues)} répliques, {len(counts)} locuteurs, {dialogues[-1]['scene_id']} scènes")
    for name, n in counts.most_common(15):
        print(f"  {name:<20} {n}")
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
