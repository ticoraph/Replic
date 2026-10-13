# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3>=1.34"]
# ///
"""Repère les régions AWS/GCP les plus susceptibles d'avoir de la capacité GPU
disponible pour les jobs de fine-tuning (train-sagemaker.yml / train-vertex.yml).

Contexte : les jobs SageMaker (ml.g5.xlarge spot, eu-central-1) et Vertex AI
(NVIDIA_L4, europe-west4) restent bloqués en "Starting"/"PENDING" faute de
capacité GPU dans ces régions précises, alors que le quota est approuvé. Ce
script aide à choisir une région alternative avant de relancer un job.

AWS : utilise `ec2:GetSpotPlacementScores`, le seul signal AWS qui reflète une
vraie probabilité de capacité (pas juste un quota approuvé) — score 1 à 10,
plus haut = plus de chances qu'une requête spot capacity-optimized aboutisse.
Croisé avec le quota SageMaker "for spot training job usage" par région (une
région peut avoir de la capacité EC2 mais un quota SageMaker à 0).

GCP : il n'existe pas d'équivalent du Spot Placement Score. On ne peut lister
que les régions où le type d'accélérateur est *proposé au catalogue*
(`gcloud compute accelerator-types list`) et le quota associé — ça ne garantit
PAS de capacité réelle à l'instant T, juste que la région vaut la peine d'être
essayée.

Prérequis : credentials AWS locaux valides (`aws sts get-caller-identity`) et
`gcloud auth login` + projet configuré. Lancer avec `uv run`, qui installe
boto3 automatiquement dans un environnement éphémère :

    uv run scripts/gpu_region_availability.py
    uv run scripts/gpu_region_availability.py --aws-instance-types g5.xlarge g6.xlarge --gcp-accelerator nvidia-l4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

import boto3
from botocore.exceptions import ClientError

# Quotas SageMaker pertinents pour un training job GPU spot/à la demande.
SAGEMAKER_SPOT_QUOTA_TEMPLATE = "ml.{instance_family} for spot training job usage"
SAGEMAKER_ONDEMAND_QUOTA_TEMPLATE = "ml.{instance_family} for training job usage"


def check_aws(instance_types: list[str], target_capacity: int) -> None:
    ec2 = boto3.client("ec2", region_name="us-east-1")
    try:
        regions = [r["RegionName"] for r in ec2.describe_regions()["Regions"] if r["RegionName"].startswith("eu-")]
    except ClientError as e:
        print(f"Impossible de lister les régions AWS : {e}")
        return

    for instance_type in instance_types:
        check_aws_instance_type(instance_type, target_capacity, regions)


def check_aws_instance_type(instance_type: str, target_capacity: int, regions: list[str]) -> None:
    print(f"\n=== AWS — capacité spot pour {instance_type} (target_capacity={target_capacity}) ===\n")

    ec2 = boto3.client("ec2", region_name="us-east-1")
    try:
        resp = ec2.get_spot_placement_scores(
            InstanceTypes=[instance_type],
            TargetCapacity=target_capacity,
            RegionNames=regions,
        )
    except ClientError as e:
        print(f"get_spot_placement_scores a échoué (permissions ec2:GetSpotPlacementScores ?) : {e}")
        return

    scores = {s["Region"]: s["Score"] for s in resp.get("SpotPlacementScores", [])}
    if not scores:
        print(f"Aucun score retourné — {instance_type} n'est peut-être proposé dans aucune région testée.")
        return

    # Quota SageMaker (spot training) par région pour les régions scorées,
    # pour ne pas suggérer une région où la capacité EC2 existe mais où le
    # quota SageMaker bloquerait quand même le job. Le code de quota (ex.
    # L-DE07CBDF) est le même dans toutes les régions pour un nom de quota
    # donné : on le cherche une seule fois (list, lent) puis on interroge
    # chaque région avec get_service_quota (rapide, 1 appel direct).
    quota_name = SAGEMAKER_SPOT_QUOTA_TEMPLATE.format(instance_family=instance_type)
    quota_code = None
    sq_probe = boto3.client("service-quotas", region_name="us-east-1")
    try:
        paginator = sq_probe.get_paginator("list_service_quotas")
        for page in paginator.paginate(ServiceCode="sagemaker"):
            for q in page["Quotas"]:
                if q["QuotaName"] == quota_name:
                    quota_code = q["QuotaCode"]
                    break
            if quota_code:
                break
    except ClientError:
        pass

    quotas: dict[str, float | None] = {}
    if quota_code:
        for region in scores:
            sq = boto3.client("service-quotas", region_name=region)
            try:
                quotas[region] = sq.get_service_quota(ServiceCode="sagemaker", QuotaCode=quota_code)["Quota"]["Value"]
            except ClientError:
                quotas[region] = None
    else:
        quotas = dict.fromkeys(scores, None)

    rows = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"{'Région':<15}{'Score spot (1-10)':<20}{'Quota SageMaker spot':<22}")
    for region, score in rows:
        quota_val = quotas.get(region)
        quota_str = "?" if quota_val is None else str(int(quota_val))
        flag = "  <-- quota SageMaker à 0, capacité EC2 inutile ici" if quota_val == 0 else ""
        print(f"{region:<15}{score:<20}{quota_str:<22}{flag}")

    print(
        "\nScore = probabilité qu'une requête spot capacity-optimized aboutisse "
        "(1=faible, 10=élevé). Un score élevé mais un quota à 0 nécessite une "
        "demande d'augmentation de quota SageMaker dans cette région avant usage."
    )


def check_gcp(accelerator: str, project: str | None) -> None:
    print(f"\n=== GCP — régions proposant {accelerator} ===\n")

    if shutil.which("gcloud") is None:
        print("`gcloud` n'est pas installé/trouvable dans le PATH — impossible de vérifier GCP.")
        return

    base_cmd = ["gcloud", "compute"]
    if project:
        base_cmd += ["--project", project]

    try:
        out = subprocess.run(
            # ":" fait un filtre substring côté gcloud (matche aussi les
            # variantes du genre "nvidia-l4-vws") — on filtre large ici et on
            # ne garde que le nom exact ensuite pour ne pas dupliquer les zones.
            [*base_cmd, "accelerator-types", "list", f"--filter=name:{accelerator}", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"`gcloud compute accelerator-types list` a échoué : {e.stderr.strip()}")
        return

    entries = [e for e in json.loads(out.stdout or "[]") if e.get("name") == accelerator]
    if not entries:
        print(f"Aucune zone ne propose {accelerator} (ou projet/auth gcloud mal configuré).")
        return

    zones_by_region: dict[str, list[str]] = {}
    for entry in entries:
        zone = entry["zone"].rsplit("/", 1)[-1]
        region = zone.rsplit("-", 1)[0]
        if not region.startswith("europe-"):
            continue
        zones_by_region.setdefault(region, []).append(zone)

    print(f"{'Région':<26}{'Zones':<8}{'Quota limite':<14}{'Quota utilisé':<14}")
    for region in sorted(zones_by_region):
        try:
            out = subprocess.run(
                [*base_cmd, "regions", "describe", region, "--format=json(quotas)"],
                capture_output=True,
                text=True,
                check=True,
            )
            quotas = json.loads(out.stdout).get("quotas", [])
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            quotas = []

        metric = accelerator.upper().replace("-", "_") + "_GPUS"
        limit = usage = None
        for q in quotas:
            if q.get("metric") == metric:
                limit, usage = q.get("limit"), q.get("usage")
                break

        n_zones = len(zones_by_region[region])
        limit_str = "?" if limit is None else str(int(limit))
        usage_str = "?" if usage is None else str(int(usage))
        print(f"{region:<26}{n_zones:<8}{limit_str:<14}{usage_str:<14}")

    print(
        "\nCeci liste les régions au catalogue (offre) + quota — pas un signal de "
        "capacité réelle comme le Spot Placement Score AWS. GCP n'expose aucune "
        "API publique de capacité live : un job peut rester PENDING même avec "
        "quota > 0. Reste de l'essai-erreur, mais en priorisant les régions "
        "avec le plus de zones listées."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--aws-instance-types",
        nargs="+",
        default=["g4dn.xlarge", "g5.xlarge", "g6.xlarge"],
        help="Types EC2 sous-jacents à comparer (défaut: g4dn.xlarge g5.xlarge g6.xlarge)",
    )
    ap.add_argument("--aws-target-capacity", type=int, default=1)
    ap.add_argument("--gcp-accelerator", default="nvidia-l4", help="Nom gcloud de l'accélérateur (défaut: nvidia-l4)")
    ap.add_argument("--gcp-project", default=None, help="Projet GCP (défaut: config gcloud active)")
    ap.add_argument("--skip-aws", action="store_true")
    ap.add_argument("--skip-gcp", action="store_true")
    args = ap.parse_args()

    if not args.skip_aws:
        check_aws(args.aws_instance_types, args.aws_target_capacity)
    if not args.skip_gcp:
        check_gcp(args.gcp_accelerator, args.gcp_project)


if __name__ == "__main__":
    sys.exit(main())
