"""Lance le fine-tuning comme SageMaker Training Job (appelé par le workflow GitHub Actions).

Instance : ml.g5.xlarge (A10G 24 Go, ~1,41 $/h ; spot ≈ -60 %). Run attendu 30-60 min.
Le code (src/training/) est uploadé par le SDK ; le dataset est lu depuis HF Hub.

Env requis : AWS creds (OIDC), HF_TOKEN, WANDB_API_KEY, SAGEMAKER_ROLE_ARN.
Usage : python src/training/launch_sagemaker.py [--use-spot] [--instance ml.g5.xlarge]
"""

import argparse
import os
import time

from sagemaker.pytorch import PyTorch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="ml.g5.xlarge")
    ap.add_argument("--use-spot", action="store_true")
    ap.add_argument("--dataset-repo", default="TicoRaph/pulpfictionchatbot-sft")
    ap.add_argument("--adapter-repo", default="TicoRaph/pulpfictionchatbot-llama31-8b-lora")
    ap.add_argument("--epochs", default="2")
    args = ap.parse_args()

    job_name = f"pulpfictionchatbot-qlora-{time.strftime('%Y%m%d-%H%M%S')}"
    estimator = PyTorch(
        entry_point="train_qlora.py",
        source_dir="src/training",
        role=os.environ["SAGEMAKER_ROLE_ARN"],
        instance_type=args.instance,
        instance_count=1,
        framework_version="2.4",
        py_version="py311",
        volume_size=100,
        max_run=3 * 3600,
        use_spot_instances=args.use_spot,
        max_wait=6 * 3600 if args.use_spot else None,
        environment={
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "TRAINING_PLATFORM": "sagemaker",
        },
        hyperparameters={
            "dataset-repo": args.dataset_repo,
            "adapter-repo": args.adapter_repo,
            "epochs": args.epochs,
            "run-name": job_name,
        },
    )
    estimator.fit(job_name=job_name, wait=True)
    print(f"Job terminé : {job_name}")


if __name__ == "__main__":
    main()
