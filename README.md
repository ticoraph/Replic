# Replic

Chatbot IA incarnant les personnages de Pulp Fiction (Jules, Vincent, Mia, Marsellus,
Butch, Winston Wolf) : fine-tuning **QLoRA (Unsloth + TRL)** de Llama 3.1 8B Instruct
pour le style, **RAG pgvector** pour la factualité, inférence **vLLM sur Modal**
(GPU serverless scale-to-zero), API **FastAPI** sur App Runner, tracking **W&B**,
adapter sur **HF Hub**.

Les tâches distantes de traduction, d'augmentation et d'évaluation utilisent Gemini
par défaut. Elles peuvent utiliser GPT-6 Luna avec `LLM_PROVIDER=openai` et
`OPENAI_API_KEY`, sans modifier le modèle Llama local du chatbot.

> **Données non incluses.** Le corpus RAG (`data/raw/`), les dialogues du film et les
> datasets dérivés ne sont pas versionnés pour des raisons de droit d'auteur. Voir
> [data/README.md](data/README.md) pour le format attendu et fournir votre propre corpus.
> Ce projet est un travail de fan sans lien avec les ayants droit du film.

## Démarrage local

```bash
cp .env.example .env            # remplir les clés nécessaires, puis LLM_PROVIDER=gemini|openai
uv sync                          # deps core
docker compose -f docker/docker-compose.yml up -d   # pgvector local (port 5434)
```

## Pipeline (dans l'ordre)

```bash
# 1. Dataset de fine-tuning (chatbot 100 % francophone)
uv run python src/preprocessing/parse_screenplay.py     # script EN → dialogues.jsonl
uv run python src/preprocessing/translate_dialogues.py  # → dialogues_fr.jsonl (LLM distant, VF)
uv run python src/datasets/build_pairs.py               # → pairs_script.jsonl
uv run python src/datasets/augment.py --per-character 80   # Q/R synthétiques (Gemini ou OpenAI)
uv run python src/datasets/split.py --push-wandb      # → datasets/{train,val}.jsonl
uv run --group data python src/datasets/push_hub.py   # dataset privé HF Hub

# 2. RAG
uv run --group rag python src/rag/index.py            # indexe data/raw/ dans pgvector
uv run --group rag python src/rag/retrieve.py "Pourquoi Jules cite Ézéchiel ?"

# 3. Fine-tuning (jobs cloud, via GitHub Actions workflow_dispatch)
#    train-sagemaker.yml (ml.g5.xlarge spot ~1-2 $) ou train-vertex.yml (L4 ~0,50-1 $)
#    Prérequis : licence Meta Llama acceptée sur HF + quotas GPU demandés.

# 4. Inférence
modal deploy src/inference/modal_app.py               # endpoint vLLM OpenAI-compatible
uv run --group rag uvicorn backend.main:app --port 8080
uv run --group ui streamlit run frontend/streamlit_app.py

# 5. Évaluation
uv run --group rag --group eval python evaluations/ragas/run_ragas.py
uv run --group rag --group eval python evaluations/character_fidelity/judge_fidelity.py [--baseline]
```

## GitHub Actions

| Workflow | Déclencheur | Rôle |
| --- | --- | --- |
| [`deploy-api.yml`](.github/workflows/deploy-api.yml) | push sur `main` touchant `backend/**`, `src/rag/**`, `src/utils/**`, `pyproject.toml`, `uv.lock` (+ `workflow_dispatch`) | build/push l'image `backend/Dockerfile` vers ECR puis déclenche un déploiement App Runner |
| [`index-rag.yml`](.github/workflows/index-rag.yml) | `workflow_dispatch` uniquement | réindexe `data/raw/` dans le pgvector de prod (idempotent par hash de contenu) |
| [`train-sagemaker.yml`](.github/workflows/train-sagemaker.yml) | `workflow_dispatch` (inputs : `instance` défaut `ml.g5.xlarge`, `use_spot` défaut `true`, `epochs` défaut `2`) | lance un job de fine-tuning QLoRA sur SageMaker (spot, ~1-2 $) |
| [`train-vertex.yml`](.github/workflows/train-vertex.yml) | `workflow_dispatch` (inputs : `machine` défaut `a2-highgpu-1g`, `gpu` défaut `NVIDIA_TESLA_A100`, `use_spot` défaut `true`, `epochs` défaut `2`, `region` défaut `europe-west4`) | lance un job de fine-tuning QLoRA sur Vertex AI (A100 40 Go, préemptible par défaut) |

Les deux workflows de training poussent l'adapter LoRA résultant vers le même repo HF Hub
(`settings.adapter_repo`) et loggent dans le même projet W&B : le Hub est le point de
passage cloud-neutre entre entraînement et inférence.

### Quotas GPU (prérequis avant le premier training)

Les deux workflows échouent tant que le quota GPU correspondant est à 0 — à demander
tôt, le traitement peut prendre 1-2 jours.

- **AWS (SageMaker)** : console *Service Quotas* → service **Amazon SageMaker** → région
  `eu-central-1`. Le workflow lance un job **spot** par défaut (`use_spot: true`), donc
  c'est le quota **« ml.g5.xlarge for spot training job usage »** qu'il faut augmenter
  (demander 1). Si un autre quota **« ml.g5.xlarge for training job usage »** est déjà
  non nul sur le compte, un run **on-demand** est possible immédiatement en décochant
  `use_spot` au déclenchement du workflow (~1,41 $/h au lieu de ~0,50-0,60 $/h en spot).
  
- **GCP (Vertex AI)** : le quota A100 **on-demand** (« Custom model training Nvidia A100
  GPUs per region ») est à 0 par défaut sur un compte récent et sa demande d'augmentation
  est souvent **refusée automatiquement** (« valeur comprise entre 0 et 0 »). Le workflow
  utilise donc le quota A100 **préemptible** (« ... preemptible Nvidia A100 GPUs per
  region ») par défaut (`use_spot: true`), généralement déjà approuvé sans demande
  (vérifiable via `gcloud alpha quotas info list --service=aiplatform.googleapis.com`).
  Le seul risque du préemptible est une interruption en cours de run (job à relancer) —
  acceptable pour un run de 30-60 min. Si le quota préemptible est à 0 sur le projet :
  1. vérifier si une **autre région** l'a non nul (`us-central1`, `asia-southeast1` entre
     autres) ;
  2. demander l'augmentation via console *IAM & Admin → Quotas* → service **Vertex AI
     API** (parfois affiché **« Agent Platform API »**) ;
  3. sinon, essayer un GPU moins demandé — quota **« Custom model training Nvidia L4 GPUs
     per region »** (`machine: g2-standard-8`, `gpu: NVIDIA_L4`, ~0,85 $/h) ou **« ...
     Nvidia T4 GPUs per region »** (`machine: n1-standard-8`, `gpu: NVIDIA_TESLA_T4`) ;
  4. sinon, démarrer avec **SageMaker** (quota déjà disponible) et retenter la demande
     GCP après un peu d'usage facturé sur le projet — les refus automatiques se
     débloquent généralement dès qu'il y a un historique.

### Secrets GitHub requis

- AWS (deploy + SageMaker) : `AWS_ROLE_ARN`, `SAGEMAKER_ROLE_ARN`
- GCP (Vertex) : `GCP_WIF_PROVIDER`, `GCP_SA_EMAIL`, `GCP_PROJECT`, `GCP_STAGING_BUCKET`
- Communs : `HF_TOKEN`, `WANDB_API_KEY`
- RAG prod : `PROD_DATABASE_URL`

## Déploiement prod

L'API (`backend/Dockerfile`) se déploie sur n'importe quel hébergeur de conteneurs ; le
workflow `deploy-api.yml` cible AWS. Après `modal deploy`, renseigner `MODAL_INFERENCE_URL`
et `MODAL_INFERENCE_API_KEY` dans la configuration de l'API (voir `.env.example`).
