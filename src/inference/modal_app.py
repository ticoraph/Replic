"""Serving vLLM sur Modal (GPU serverless scale-to-zero, OpenAI-compatible).

- Base : meta-llama/Llama-3.1-8B-Instruct (gated — le secret Modal `huggingface`
  doit contenir HF_TOKEN d'un compte approuvé).
- Adapter LoRA chargé depuis HF Hub, exposé comme modèle "pulpfiction" via
  --enable-lora / --lora-modules.
- Poids pré-téléchargés dans un Volume Modal (cold start ~60-120 s au lieu de 3-5 min).
- min_containers=0 : coût nul quand personne ne chatte ; L4 24 Go facturé à la seconde.

Déploiement :
    modal secret create huggingface HF_TOKEN=hf_xxx
    modal secret create inference-api API_KEY=<clé pour le backend>
    modal deploy src/inference/modal_app.py
    → URL OpenAI-compatible : https://<workspace>--pulpfictionchatbot-vllm-serve.modal.run/v1
"""

import modal

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "TicoRaph/pulpfictionchatbot-llama31-8b-lora"
SERVED_NAME = "pulpfiction"

VLLM_PORT = 8000
MINUTES = 60

hf_cache = modal.Volume.from_name("pulpfictionchatbot-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.6",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
)

app = modal.App("pulpfictionchatbot-vllm")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/hf-cache": hf_cache},
    timeout=60 * MINUTES,
)
def download_weights() -> None:
    """À lancer une fois (modal run src/inference/modal_app.py::download_weights)."""
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL)
    snapshot_download(ADAPTER_REPO)
    hf_cache.commit()
    print("Poids en cache dans le volume.")


@app.function(
    image=image,
    gpu="L4",
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("inference-api"),  # API_KEY
    ],
    volumes={"/hf-cache": hf_cache},
    scaledown_window=5 * MINUTES,
    min_containers=0,
    timeout=15 * MINUTES,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve() -> None:
    import os
    import subprocess

    from huggingface_hub import hf_hub_download

    # Le tokenizer de l'adapter LoRA ne définit pas de `chat_template` inline
    # (convention récente HF : fichier .jinja séparé) — vLLM 0.6.6 ne le
    # détecte pas automatiquement, on le passe donc explicitement.
    chat_template_path = hf_hub_download(ADAPTER_REPO, "chat_template.jinja")

    cmd = [
        "vllm", "serve", BASE_MODEL,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--api-key", os.environ["API_KEY"],
        "--enable-lora",
        "--lora-modules", f"{SERVED_NAME}={ADAPTER_REPO}",
        "--chat-template", chat_template_path,
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.92",
        "--quantization", "fp8",  # tient confortablement sur L4 24 Go
    ]
    subprocess.Popen(cmd)
