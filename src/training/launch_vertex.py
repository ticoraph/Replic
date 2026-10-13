"""Lance le fine-tuning comme Vertex AI Custom Training Job (appelé par GitHub Actions).

Machine : a2-highgpu-1g + 1× NVIDIA A100 40 Go (~3,67 $/h on-demand, moins cher en spot).
Run attendu 30-60 min. Le job utilise l'image PyTorch GPU officielle et installe
requirements.txt.

Quota : le quota A100 on-demand est à 0 par défaut sur un compte GCP sans historique
(demande d'augmentation nécessaire, cf. README) alors que le quota A100 **préemptible**
est généralement déjà approuvé (8 GPU sur ce projet en europe-west4/us-central1) — d'où
--use-spot par défaut.

Env requis : creds GCP (WIF), GCP_PROJECT, GCP_REGION, GCP_STAGING_BUCKET,
HF_TOKEN, WANDB_API_KEY.
Usage : python src/training/launch_vertex.py [--use-spot]
"""

import argparse
import os
import time

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types.custom_job import Scheduling


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="a2-highgpu-1g")
    ap.add_argument("--gpu", default="NVIDIA_TESLA_A100")
    ap.add_argument("--use-spot", action="store_true")
    ap.add_argument("--dataset-repo", default="TicoRaph/pulpfictionchatbot-sft")
    ap.add_argument("--adapter-repo", default="TicoRaph/pulpfictionchatbot-llama31-8b-lora")
    ap.add_argument("--epochs", default="2")
    args = ap.parse_args()

    job_name = f"pulpfictionchatbot-qlora-{time.strftime('%Y%m%d-%H%M%S')}"
    aiplatform.init(
        project=os.environ["GCP_PROJECT"],
        location=os.environ.get("GCP_REGION", "europe-west4"),
        staging_bucket=os.environ["GCP_STAGING_BUCKET"],
    )

    job = aiplatform.CustomJob.from_local_script(
        display_name=job_name,
        script_path="src/training/train_qlora.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
        requirements=open("src/training/requirements.txt").read().splitlines(),
        machine_type=args.machine,
        accelerator_type=args.gpu,
        accelerator_count=1,
        boot_disk_size_gb=100,
        environment_variables={
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "TRAINING_PLATFORM": "vertex",
        },
        args=[
            f"--dataset-repo={args.dataset_repo}",
            f"--adapter-repo={args.adapter_repo}",
            f"--epochs={args.epochs}",
            f"--run-name={job_name}",
        ],
    )
    job.run(
        sync=True,
        timeout=3 * 3600,
        scheduling_strategy=Scheduling.Strategy.SPOT if args.use_spot else None,
        restart_job_on_worker_restart=args.use_spot,
    )
    print(f"Job terminé : {job_name}")


if __name__ == "__main__":
    main()
