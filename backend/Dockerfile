# API d'orchestration (RAG + proxy inférence Modal) — cible App Runner
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv export --group rag --no-dev --no-emit-project > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY backend/ backend/

# Pré-télécharge le modèle d'embeddings (multilingual-e5-base) dans l'image
# pour éviter le download au démarrage d'App Runner.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('intfloat/multilingual-e5-base')"

EXPOSE 8080
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]

