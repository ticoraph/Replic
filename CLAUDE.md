# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A French-language chatbot embodying the Pulp Fiction characters (Jules, Vincent, Mia, Marsellus,
Butch, Winston Wolf): QLoRA fine-tuning (Unsloth + TRL) of Llama 3.1 8B Instruct for character
voice/style, RAG over pgvector for factual grounding, vLLM inference served serverless on Modal
(scale-to-zero GPU), a FastAPI orchestration layer deployed on AWS App Runner, W&B for
experiment tracking, and the trained LoRA adapter hosted on Hugging Face Hub (the cloud-neutral
handoff point). Private infra notes live in the gitignored `CLAUDE.local.md` and
`docs/internal/` — read them before proposing infra changes, and never copy their content
into tracked files (this repo is public).

**Repo name:** the project is called **Replic** (formerly `pulpfictionchatbot`). Deployed
infra identifiers intentionally keep the old name — do not rename them without migrating the
deployed resources.

**Data is not in the repo** (copyright): `data/raw/`, `data/interim/`, `data/processed/`,
`datasets/` and `wandb/` are gitignored. See `data/README.md`.

**All chatbot-facing content is French** — system prompts, RAG corpus, dataset Q/A, and the
`_BASE_RULES` persona instruction all mandate French output regardless of user input language.

## Commands

```bash
# Setup
cp .env.example .env                                # fill HF_TOKEN, WANDB_API_KEY, GEMINI_API_KEY…
uv sync                                              # core deps only
docker compose -f docker/docker-compose.yml up -d    # local pgvector (port 5434)

# Tests / lint
uv run pytest                                        # tests/ (api tests use FastAPI TestClient, no real DB/Modal needed)
uv run pytest tests/api/test_endpoints.py::test_health   # single test
uv run ruff check .
uv run ruff format .
```

Most scripts need a dependency group beyond core (`uv sync` alone won't install them) — use
`uv run --group <name> python ...`. Groups: `rag` (sentence-transformers, pgvector, tiktoken),
`data` (datasets, transformers, jinja2), `tracking` (wandb), `train` (unsloth, trl — CUDA only,
never install locally), `eval` (ragas, langchain-openai), `ui` (streamlit), `dev` (pytest, ruff).

### Pipeline, in order (see README.md for full detail)

```bash
# 1. Fine-tuning dataset (French-only chatbot)
uv run python src/preprocessing/parse_screenplay.py     # EN script -> dialogues.jsonl
uv run python src/preprocessing/translate_dialogues.py  # -> dialogues_fr.jsonl (Gemini)
uv run python src/datasets/build_pairs.py               # -> pairs_script.jsonl
uv run python src/datasets/augment.py --per-character 80   # synthetic Q/A (GEMINI_API_KEY)
uv run python src/datasets/split.py --push-wandb        # -> datasets/{train,val}.jsonl
uv run --group data python src/datasets/push_hub.py     # private HF Hub dataset

# 2. RAG
uv run --group rag python src/rag/index.py              # index data/raw/ into pgvector
uv run --group rag python src/rag/retrieve.py "Pourquoi Jules cite Ézéchiel ?"

# 3. Fine-tuning — cloud-only (no local GPU), via GitHub Actions workflow_dispatch:
#    train-sagemaker.yml (ml.g5.xlarge spot) or train-vertex.yml (L4)

# 4. Inference
modal deploy src/inference/modal_app.py                 # OpenAI-compatible vLLM endpoint
uv run --group rag uvicorn backend.main:app --port 8080
uv run --group ui streamlit run frontend/streamlit_app.py

# 5. Evaluation
uv run --group rag --group eval python evaluations/ragas/run_ragas.py
uv run --group rag --group eval python evaluations/character_fidelity/judge_fidelity.py [--baseline]
```

## Architecture

### Request flow (production path)

`backend/main.py` (FastAPI) is the only service actually deployed (App Runner). On
`POST /chat` it: (1) retrieves top-k chunks from pgvector via `src/rag/retrieve.py`, filterable
by `character` against `metadata->characters_mentioned`; (2) builds the system+history+user
message list via `src/rag/prompting.py` (persona from `src/utils/characters.py` + RAG context
injected into `CONTEXT_TEMPLATE`); (3) streams completions from the Modal vLLM endpoint (OpenAI
client pointed at `settings.modal_inference_url`, model name `"pulpfiction"` — the LoRA adapter
name registered via `--lora-modules` in `modal_app.py`) as SSE (`event: sources` then
`data: {token}` chunks, terminated by `data: [DONE]`). Auth is a static `X-API-Key` header
checked against `settings.api_key`; there is no per-user auth.

`backend/main.py` inserts `src/` onto `sys.path` at import time (`parents[1] / "src"`) rather
than relying on packaging — imports like `from rag.retrieve import retrieve` depend on this.
Same pattern is used inside `src/rag/*.py` files to import sibling `src/utils/*` modules.

### Character personas — single source of truth

`src/utils/characters.py`'s `CHARACTERS` dict is used by three different consumers: the FastAPI
`/characters` endpoint, dataset generation (`script_aliases` drive screenplay parsing,
`traits` drive synthetic Q/A augmentation), and RAG chunk tagging (`chunking.py` detects which
characters are mentioned in a chunk by matching `display_name`/aliases via regex). Adding a
character means updating this one file; everything downstream (chunking, retrieval filter,
dataset builder, API) picks it up automatically.

### RAG indexing is idempotent by content hash

`src/rag/index.py` hashes each chunk's content (sha256) and upserts by `content_hash` — re-running
it after adding new markdown source files only embeds the delta, and it prunes chunks whose hash
is no longer produced by `chunk_corpus()` (e.g. after excluding the English script). Embedding
model is `intfloat/multilingual-e5-base` (768 dims, CPU-only, no GPU needed for RAG) — note the
mandatory e5 prefix convention: `"passage: "` at index time, `"query: "` at retrieval time
(`src/rag/retrieve.py::embed_query`), used inconsistently will silently degrade retrieval quality
rather than error.

Corpus has two flavors, both French: markdown analysis files in `data/raw/*.md` (chunked by
markdown headers, `chunking.py::chunk_markdown`) and the film script itself, translated to
French dialogue-by-scene in `data/interim/dialogues_fr.jsonl` (produced by
`translate_dialogues.py`, chunked scene-by-scene in `chunk_screenplay_fr`). The original
English script markdown (`script-pulp-fiction-original-en.md`) is explicitly excluded from
indexing — only the FR-translated dialogues are indexed for the screenplay content.

### Training is cloud-only, HF Hub is the handoff point

`src/training/train_qlora.py` requires CUDA (Unsloth) and is never run locally — it's invoked
by `train-sagemaker.yml` or `train-vertex.yml` GitHub Actions (`workflow_dispatch`), both testable
independently to compare cost/quality since both are pure pay-per-second. Both paths push the
resulting LoRA adapter to the same HF Hub repo (`settings.adapter_repo`) and log to the same W&B
project — this is intentional: HF Hub is the cloud-neutral
handoff so `modal_app.py` doesn't care which cloud trained the adapter it loads.

### Inference cold starts

Modal's `serve()` function has `min_containers=0` (cost is zero when idle) — the tradeoff is a
1-2 minute cold start on the first chat message after idle, surfaced to users via the Streamlit
UI caption and a 180s client timeout in `backend/main.py`. Weights are pre-downloaded into a
Modal Volume (`download_weights()`, run once manually) to cut cold start from 3-5 min to ~60-120s.

The adapter's tokenizer ships its chat template as a separate `chat_template.jinja` file (recent
HF convention) rather than embedding it in `tokenizer_config.json` — vLLM 0.6.6 doesn't
auto-detect that file, so `modal_app.py::serve()` downloads it via `hf_hub_download` and passes
it explicitly with `--chat-template`. `transformers`/`tokenizers` are pinned in the Modal image
(`4.46.3`/`0.20.3`) to match what vLLM 0.6.6 expects.

## Deployment

- `.github/workflows/deploy-api.yml` builds/pushes `backend/Dockerfile` and triggers a deployment
  on push to `main` touching `backend/**`, `src/rag/**`, `src/utils/**`, or dependency files.
  `index-rag.yml` handles prod pgvector reindexing separately.
- This repo's `.env` is local-dev only; prod config is injected by the deployment platform.
- Infra details (accounts, IaC repo, secrets store) are in the gitignored `CLAUDE.local.md`.
